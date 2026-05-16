"""
赛题2 SOTA V3冲刺实验 - 冲击90%+

基于当前最佳88.81%的优化策略：
1. ResNet34（更大模型容量）
2. 时空噪声注入（已验证有效，noise_strength=0.01最优）
3. Mixup(α=0.2) + Label Smoothing(0.1)
4. EMA指数移动平均
5. 300 epochs超长训练 + Warmup + Cosine LR
6. 梯度裁剪 + 梯度中心化
7. 更平缓的噪声退火策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet import get_model


class SpatiotemporalNoiseLayer(nn.Module):
    """时空相关噪声层 - 修复版"""
    def __init__(self, layer, layer_idx, total_layers, noise_config):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.noise_strength = noise_config['noise_strength']
        self.rho_temporal = noise_config.get('rho_temporal', 0.7)
        self.beta = noise_config.get('beta', 1.0)
        self.noise_schedule = noise_config.get('noise_schedule', 'cosine')
        self.epoch = 1
        self.total_epochs = 100
        self.register_buffer('temporal_state', None)
        
        # 计算层次化噪声强度 gamma_l = gamma_base * (1 + beta * l/L)
        self.gamma_l = self.noise_strength * (1.0 + self.beta * layer_idx / max(total_layers - 1, 1))

    def get_noise_schedule(self, initial_strength, min_strength, schedule_type='cosine'):
        if schedule_type == 'cosine':
            progress = self.epoch / self.total_epochs
            return min_strength + (initial_strength - min_strength) * (1 + np.cos(np.pi * progress)) / 2
        elif schedule_type == 'linear':
            progress = self.epoch / self.total_epochs
            return initial_strength - (initial_strength - min_strength) * progress
        else:
            return initial_strength

    def forward(self, x):
        # 评估时不注入噪声
        if not self.training:
            if isinstance(self.layer, nn.Conv2d):
                return F.conv2d(x, self.layer.weight, self.layer.bias,
                               self.layer.stride, self.layer.padding,
                               self.layer.dilation, self.layer.groups)
            elif isinstance(self.layer, nn.Linear):
                return F.linear(x, self.layer.weight, self.layer.bias)

        # 获取当前epoch的噪声强度
        current_strength = self.get_noise_schedule(self.gamma_l, self.gamma_l * 0.1, self.noise_schedule)
        
        # 时间相关性：AR(1)过程
        if self.temporal_state is None:
            self.temporal_state = torch.randn_like(self.layer.weight)
        
        white_noise = torch.randn_like(self.layer.weight)
        noise = self.rho_temporal * self.temporal_state + np.sqrt(1 - self.rho_temporal**2) * white_noise
        self.temporal_state = noise.detach()
        
        # 应用噪声
        noisy_weight = self.layer.weight + noise * current_strength
        
        if isinstance(self.layer, nn.Conv2d):
            return F.conv2d(x, noisy_weight, self.layer.bias,
                           self.layer.stride, self.layer.padding,
                           self.layer.dilation, self.layer.groups)
        elif isinstance(self.layer, nn.Linear):
            return F.linear(x, noisy_weight, self.layer.bias)


def inject_spatiotemporal_noise(model, noise_config):
    """将模型的所有Conv2d/Linear层替换为时空噪声层"""
    total_layers = sum(1 for _ in model.modules()
                      if isinstance(_, (nn.Conv2d, nn.Linear)))

    layer_idx = 0
    for name, child in model.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            noisy_layer = SpatiotemporalNoiseLayer(
                child, layer_idx, total_layers, noise_config
            )
            setattr(model, name, noisy_layer)
            layer_idx += 1
        else:
            inject_spatiotemporal_noise(child, noise_config)

    return model


class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.997):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_to_model(self, model):
        """将EMA权重应用到模型（仅用于评估，不修改原shadow）"""
        restored = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                restored[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return restored
    
    def restore_from(self, model, restored):
        """恢复原始权重"""
        for name, data in restored.items():
            model.state_dict()[name].copy_(data)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutout_mask(img_size, cutout_length=16):
    mask = torch.ones(img_size, img_size)
    y = np.random.randint(img_size)
    x = np.random.randint(img_size)
    y_min = np.clip(y - cutout_length // 2, 0, img_size)
    y_max = np.clip(y + cutout_length // 2, 0, img_size)
    x_min = np.clip(x - cutout_length // 2, 0, img_size)
    x_max = np.clip(x + cutout_length // 2, 0, img_size)
    mask[y_min:y_max, x_min:x_max] = 0
    return mask


def gradient_centralization(optimizer):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.grad.dim() > 1:
                mean = param.grad.mean(dim=tuple(range(1, param.grad.dim())), keepdim=True)
                param.grad.sub_(mean)


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, total_epochs,
                    noise_config=None, use_mixup=False, mixup_alpha=0.2, 
                    use_cutout=False, cutout_length=8, ema=None):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # 更新噪声层的epoch
    for module in model.modules():
        if hasattr(module, 'epoch'):
            module.epoch = epoch
            module.total_epochs = total_epochs

    pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixup
        if use_mixup and np.random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            use_mixup_batch = True
        else:
            use_mixup_batch = False

        # Cutout
        if use_cutout:
            cutout_masks = torch.stack([cutout_mask(32, cutout_length) for _ in range(inputs.size(0))]).to(device)
            inputs = inputs * cutout_masks.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)

        if use_mixup_batch:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        
        # 梯度中心化
        gradient_centralization(optimizer)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # EMA更新
        if ema is not None:
            ema.update(model)

        train_loss += loss.item()

        if use_mixup_batch:
            _, predicted = outputs.max(1)
            total += targets_a.size(0)
            correct += (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        else:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return train_loss / len(trainloader), 100.*correct/total


@torch.no_grad()
def evaluate(model, testloader, criterion, device):
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return test_loss / len(testloader), 100.*correct/total


@torch.no_grad()
def evaluate_with_ema(model, testloader, criterion, device, ema):
    """用EMA权重评估（不破坏原始训练权重）"""
    # 保存并应用EMA权重
    restored = ema.apply_to_model(model)
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # 恢复原始权重用于继续训练
    ema.restore_from(model, restored)
    model.train()  # 恢复训练模式

    return test_loss / len(testloader), 100.*correct/total


def run_experiment(config):
    device = config['device']
    model_name = config.get('model_name', 'resnet34')
    noise_strength = config.get('noise_strength', 0.01)
    rho_temporal = config.get('rho_temporal', 0.7)
    beta = config.get('beta', 1.0)
    noise_schedule = config.get('noise_schedule', 'cosine')
    epochs = config.get('epochs', 300)
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    seed = config.get('seed', 42)
    use_mixup = config.get('use_mixup', True)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    label_smoothing = config.get('label_smoothing', 0.1)
    ema_decay = config.get('ema_decay', 0.997)
    use_cutout = config.get('use_cutout', False)
    cutout_length = config.get('cutout_length', 8)

    print("=" * 70)
    print(f"实验: {config.get('name', 'unknown')}")
    print(f"模型: {model_name}, noise: {noise_strength}, epochs: {epochs}")
    print(f"Mixup: {use_mixup} (α={mixup_alpha}), LS: {label_smoothing}, EMA: {ema_decay}")
    print(f"Cutout: {use_cutout} (len={cutout_length}), Noise Schedule: {noise_schedule}")
    print("=" * 70)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=True,
        download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=False,
        download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = get_model(name=model_name, num_classes=10, pretrained=True)
    
    # 注入时空噪声
    noise_config = {
        'noise_strength': noise_strength,
        'rho_temporal': rho_temporal,
        'beta': beta,
        'noise_schedule': noise_schedule,
    }
    model = inject_spatiotemporal_noise(model, noise_config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=0.9,
        weight_decay=config.get('weight_decay', 1e-3),
        nesterov=True
    )

    warmup_epochs = config.get('warmup_epochs', 15)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    ema = EMA(model, decay=ema_decay)
    # EMA不再需要单独的模型

    history = {
        'train_loss': [], 'train_acc': [],
        'test_acc': [], 'test_acc_ema': []
    }
    best_acc = 0.0
    best_acc_ema = 0.0
    best_epoch = 0
    best_epoch_ema = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch, epochs,
            noise_config, use_mixup, mixup_alpha, use_cutout, cutout_length, ema
        )
        scheduler.step()

        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        test_loss_ema, test_acc_ema = evaluate_with_ema(model, testloader, criterion, device, ema)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_acc_ema'].append(test_acc_ema)

        print(f'Epoch {epoch}/{epochs}: '
              f'Train Acc={train_acc:.2f}%, '
              f'Test Acc={test_acc:.2f}%, '
              f'Test Acc(EMA)={test_acc_ema:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print(f'  -> 新最佳(原始): {best_acc:.2f}% (Epoch {best_epoch})')

        if test_acc_ema > best_acc_ema:
            best_acc_ema = test_acc_ema
            best_epoch_ema = epoch
            print(f'  -> 新最佳(EMA): {best_acc_ema:.2f}% (Epoch {best_epoch_ema})')

        # 保存检查点
        if (test_acc_ema > best_acc_ema * 0.99) and (epoch % 50 == 0):
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_state': ema.shadow,
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_acc_ema': best_acc_ema,
                'config': config
            }
            save_dir = os.path.join('/mnt/storage2/zyc/CIM比赛/赛道5/赛题2', 'results/sota_v3')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{config.get('name', 'unknown')}_ep{epoch}.pt"))

    return {
        'experiment_type': 'p2_sota_v3',
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'best_acc_ema': best_acc_ema,
        'best_epoch_ema': best_epoch_ema,
        'final_acc': history['test_acc'][-1],
        'final_acc_ema': history['test_acc_ema'][-1],
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='赛题2 SOTA V3冲刺实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/sota_v3')
    args = parser.parse_args()

    configs = [
        # 配置1：ResNet34 + 时空噪声(0.01) + Mixup(0.2) + Label Smoothing + EMA + 300epochs
        {
            'name': 'resnet34_noise0.01_mixup_ls_ema_300ep',
            'device': args.device,
            'model_name': 'resnet34',
            'noise_strength': 0.01,
            'rho_temporal': 0.7,
            'beta': 1.0,
            'noise_schedule': 'cosine',
            'epochs': 300,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'lr': 0.1,
            'weight_decay': 1e-3,
            'warmup_epochs': 15,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'label_smoothing': 0.1,
            'ema_decay': 0.997,
            'use_cutout': False,
        },
        # 配置2：ResNet34 + 时空噪声(0.01) + Mixup(0.2) + Label Smoothing + EMA + Cutout + 300epochs
        {
            'name': 'resnet34_noise0.01_mixup_ls_ema_cutout_300ep',
            'device': args.device,
            'model_name': 'resnet34',
            'noise_strength': 0.01,
            'rho_temporal': 0.7,
            'beta': 1.0,
            'noise_schedule': 'cosine',
            'epochs': 300,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed + 1,
            'lr': 0.1,
            'weight_decay': 1e-3,
            'warmup_epochs': 15,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'label_smoothing': 0.1,
            'ema_decay': 0.997,
            'use_cutout': True,
            'cutout_length': 8,
        },
        # 配置3：ResNet34 + 时空噪声(0.008) + Mixup(0.15) + Label Smoothing + EMA + 300epochs
        {
            'name': 'resnet34_noise0.008_mixup0.15_ls_ema_300ep',
            'device': args.device,
            'model_name': 'resnet34',
            'noise_strength': 0.008,
            'rho_temporal': 0.7,
            'beta': 1.0,
            'noise_schedule': 'cosine',
            'epochs': 300,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed + 2,
            'lr': 0.1,
            'weight_decay': 1e-3,
            'warmup_epochs': 15,
            'use_mixup': True,
            'mixup_alpha': 0.15,
            'label_smoothing': 0.1,
            'ema_decay': 0.997,
            'use_cutout': False,
        },
    ]

    save_dir = os.path.join('/mnt/storage2/zyc/CIM比赛/赛道5/赛题2', args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"运行配置: {config['name']}")
        print(f"{'='*70}")

        result = run_experiment(config)
        result['config_name'] = config['name']
        all_results.append(result)

        save_path = os.path.join(save_dir, f"sota_v3_{config['name']}.json")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")

    print("\n" + "=" * 70)
    print("所有配置结果汇总:")
    print("=" * 70)

    best_result = None
    for r in all_results:
        print(f"  {r['config_name']}: "
              f"原始最佳={r['best_acc']:.2f}%(E{r['best_epoch']}), "
              f"EMA最佳={r['best_acc_ema']:.2f}%(E{r['best_epoch_ema']}), "
              f"最终={r['final_acc']:.2f}%")
        if best_result is None or r['best_acc_ema'] > best_result['best_acc_ema']:
            best_result = r

    print(f"\n最佳配置: {best_result['config_name']}")
    print(f"最佳精度(EMA): {best_result['best_acc_ema']:.2f}%")

    summary_path = os.path.join(save_dir, "sota_v3_summary.json")
    summary = {
        'best_config': best_result['config_name'],
        'best_acc': best_result['best_acc'],
        'best_epoch': best_result['best_epoch'],
        'best_acc_ema': best_result['best_acc_ema'],
        'best_epoch_ema': best_result['best_epoch_ema'],
        'all_results': [{
            'name': r['config_name'],
            'best_acc': r['best_acc'],
            'best_epoch': r['best_epoch'],
            'best_acc_ema': r['best_acc_ema'],
            'best_epoch_ema': r['best_epoch_ema'],
            'final_acc': r['final_acc']
        } for r in all_results]
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总已保存到: {summary_path}")


if __name__ == '__main__':
    main()
