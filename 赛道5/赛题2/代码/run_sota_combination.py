"""
SOTA组合优化实验 - 冲击90%+

组合策略：
1. P2-4 moderate_fixed时空噪声（87.35%）
2. Warmup + 梯度裁剪（已验证有效）
3. 更长训练轮次（100 epochs）
4. Mixup数据增强
5. Label Smoothing
6. SAM优化器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet import get_model


class SpatiotemporalNoiseLayer(nn.Module):
    """时空相关噪声层 - SOTA版本"""

    def __init__(self, layer, layer_idx, total_layers, noise_config):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.noise_strength = noise_config['noise_strength']
        self.rho_temporal = noise_config.get('rho_temporal', 0.5)
        self.beta = noise_config.get('beta', 0.5)

        # 时间维度噪声状态
        self.register_buffer('temporal_state', None)

        # 计算层次化噪声强度
        self.gamma_l = self.noise_strength * (1.0 + self.beta * layer_idx / max(total_layers - 1, 1))

    def forward(self, x):
        # 评估时不注入噪声
        if not self.training:
            if isinstance(self.layer, nn.Conv2d):
                return F.conv2d(x, self.layer.weight, self.layer.bias,
                              self.layer.stride, self.layer.padding,
                              self.layer.dilation, self.layer.groups)
            elif isinstance(self.layer, nn.Linear):
                return F.linear(x, self.layer.weight, self.layer.bias)

        # 训练时注入噪声
        weight_shape = self.layer.weight.shape
        fresh_noise = torch.randn(weight_shape, device=self.layer.weight.device)

        # 时间维度：AR(1)模型
        if self.temporal_state is None or self.temporal_state.shape != weight_shape:
            self.temporal_state = torch.zeros(weight_shape, device=self.layer.weight.device)

        temporal_noise = (self.rho_temporal * self.temporal_state +
                         np.sqrt(1 - self.rho_temporal**2) * fresh_noise)
        self.temporal_state = temporal_noise.detach().clone()

        # 应用层次化噪声强度
        weight_noise = temporal_noise * self.gamma_l
        noisy_weight = self.layer.weight + weight_noise

        if isinstance(self.layer, nn.Conv2d):
            return F.conv2d(x, noisy_weight, self.layer.bias,
                          self.layer.stride, self.layer.padding,
                          self.layer.dilation, self.layer.groups)
        elif isinstance(self.layer, nn.Linear):
            return F.linear(x, noisy_weight, self.layer.bias)


def inject_spatiotemporal_noise(model, noise_config):
    """注入时空噪声层"""
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


class SAMOptimizer:
    """SAM优化器 - 锐度感知最小化"""

    def __init__(self, optimizer, rho=0.05):
        self.optimizer = optimizer
        self.rho = rho

    def first_step(self, zero_grad=False):
        """计算扰动并应用"""
        grad_norm = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                perturbation = p.grad.data * self.rho / (grad_norm + 1e-12)
                p.data.add_(perturbation)

        if zero_grad:
            self.optimizer.zero_grad()

    def second_step(self):
        """恢复参数并更新"""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                perturbation = p.grad.data * self.rho / (p.grad.data.norm(2) + 1e-12)
                p.data.sub_(perturbation)

        self.optimizer.step()


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
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
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_noise_strength(epoch, total_epochs, base_strength, schedule='cosine'):
    """噪声退火策略"""
    if schedule == 'constant':
        return base_strength
    elif schedule == 'linear':
        return base_strength * (1 - epoch / total_epochs)
    elif schedule == 'cosine':
        return base_strength * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    else:
        return base_strength


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, total_epochs, config):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # 噪声退火
    if config.get('use_noise', True):
        current_noise = get_noise_strength(
            epoch, total_epochs,
            config['noise_strength'],
            config.get('noise_schedule', 'cosine')
        )

        # 更新所有噪声层的强度
        for module in model.modules():
            if isinstance(module, SpatiotemporalNoiseLayer):
                module.gamma_l = current_noise * (1.0 + module.beta * module.layer_idx / max(module.total_layers - 1, 1))

    # Mixup参数
    use_mixup = config.get('use_mixup', False)
    mixup_alpha = config.get('mixup_alpha', 0.2)

    # Label Smoothing
    label_smoothing = config.get('label_smoothing', 0.0)

    pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixup数据增强
        if use_mixup and np.random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            use_mixup_batch = True
        else:
            use_mixup_batch = False

        # SAM优化器
        use_sam = config.get('use_sam', False)
        if use_sam:
            # SAM第一步：计算扰动
            optimizer.first_step(zero_grad=True)

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        if use_mixup_batch:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        if use_sam:
            # SAM第二步：恢复参数并更新
            optimizer.second_step()
        else:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()

        # 统计精度
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
    """评估模型"""
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


def run_experiment(config):
    """运行单次实验"""
    import torchvision
    import torchvision.transforms as transforms

    device = config['device']
    print("=" * 70)
    print(f"实验类型: sota_combination")
    print(f"配置: {json.dumps(config, indent=2)}")
    print("=" * 70)

    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=True,
        download=False, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=False,
        download=False, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )

    # 模型初始化
    torch.manual_seed(config['seed'])
    model = get_model('resnet18', num_classes=10)

    # 噪声配置
    if config.get('use_noise', True):
        noise_config = {
            'noise_strength': config['noise_strength'],
            'rho_temporal': config.get('rho_temporal', 0.5),
            'beta': config.get('beta', 0.5)
        }
        model = inject_spatiotemporal_noise(model, noise_config)

    model = model.to(device)

    # 损失函数（支持Label Smoothing）
    label_smoothing = config.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 优化器
    base_optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )

    # SAM优化器
    use_sam = config.get('use_sam', False)
    if use_sam:
        optimizer = SAMOptimizer(base_optimizer, rho=config.get('sam_rho', 0.05))
    else:
        optimizer = base_optimizer

    # 学习率调度
    warmup_epochs = config['warmup_epochs']
    total_epochs = config['epochs']

    warmup_scheduler = LinearLR(
        base_optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        base_optimizer, T_max=total_epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        base_optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch, total_epochs, config
        )
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch}/{total_epochs}: '
              f'Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print(f'  -> 新最佳精度: {best_acc:.2f}% (Epoch {best_epoch})')

    return {
        'experiment_type': 'sota_combination',
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='SOTA组合优化实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/sota_combination')
    args = parser.parse_args()

    # SOTA组合配置
    configs = [
        # 配置1：基础组合（时空噪声 + warmup + 梯度裁剪 + 更长训练）
        {
            'name': 'base_combination',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'use_noise': True,
            'noise_strength': 0.01,      # moderate_fixed最佳配置
            'rho_temporal': 0.5,
            'beta': 0.5,
            'noise_schedule': 'cosine',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'use_mixup': False,
            'use_sam': False,
            'label_smoothing': 0.0
        },
        # 配置2：Mixup增强
        {
            'name': 'mixup_enhanced',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'use_noise': True,
            'noise_strength': 0.01,
            'rho_temporal': 0.5,
            'beta': 0.5,
            'noise_schedule': 'cosine',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'use_sam': False,
            'label_smoothing': 0.0
        },
        # 配置3：Label Smoothing
        {
            'name': 'label_smoothing',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'use_noise': True,
            'noise_strength': 0.01,
            'rho_temporal': 0.5,
            'beta': 0.5,
            'noise_schedule': 'cosine',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'use_mixup': False,
            'use_sam': False,
            'label_smoothing': 0.1
        },
        # 配置4：SAM优化器
        {
            'name': 'sam_optimizer',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'use_noise': True,
            'noise_strength': 0.01,
            'rho_temporal': 0.5,
            'beta': 0.5,
            'noise_schedule': 'cosine',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'use_mixup': False,
            'use_sam': True,
            'sam_rho': 0.05,
            'label_smoothing': 0.0
        },
        # 配置5：全组合（Mixup + Label Smoothing + SAM）
        {
            'name': 'full_combination',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'use_noise': True,
            'noise_strength': 0.01,
            'rho_temporal': 0.5,
            'beta': 0.5,
            'noise_schedule': 'cosine',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'use_sam': True,
            'sam_rho': 0.05,
            'label_smoothing': 0.1
        }
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

        # 保存单个结果
        save_path = os.path.join(save_dir, f"sota_{config['name']}.json")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")

    # 汇总结果
    print("\n" + "=" * 70)
    print("所有配置结果汇总:")
    print("=" * 70)

    best_result = None
    for r in all_results:
        print(f"  {r['config_name']}: Best Acc = {r['best_acc']:.2f}% (Epoch {r['best_epoch']}), Final Acc = {r['final_acc']:.2f}%")
        if best_result is None or r['best_acc'] > best_result['best_acc']:
            best_result = r

    print(f"\n最佳配置: {best_result['config_name']}")
    print(f"最佳精度: {best_result['best_acc']:.2f}%")

    # 保存汇总
    summary_path = os.path.join(save_dir, "sota_summary.json")
    summary = {
        'best_config': best_result['config_name'],
        'best_acc': best_result['best_acc'],
        'best_epoch': best_result['best_epoch'],
        'all_results': [{
            'name': r['config_name'],
            'best_acc': r['best_acc'],
            'best_epoch': r['best_epoch'],
            'final_acc': r['final_acc']
        } for r in all_results]
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总已保存到: {summary_path}")


if __name__ == '__main__':
    main()
