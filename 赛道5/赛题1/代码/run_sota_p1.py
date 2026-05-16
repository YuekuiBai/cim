"""
赛题1 SOTA冲刺实验 - 冲击90%+

基于当前最佳87.33%的优化策略：
1. ResNet34（更大模型容量，已有实验证明最鲁棒）
2. 混合Alpha训练 U(0, 0.3)（更温和的非线性范围）
3. Mixup数据增强 + RandomAugment
4. EMA指数移动平均
5. 100 epochs + Warmup + Cosine LR
6. 梯度裁剪
"""

import torch
import torch.nn as nn
import torch.optim as optim
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

from models.resnet import get_model, NonLinearWrapper


class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_to(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


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


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, total_epochs, 
                    alpha_max, ema=None, use_mixup=False, mixup_alpha=0.2):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 随机采样alpha
        alpha_t = np.random.uniform(0, alpha_max)
        if hasattr(model, 'set_alpha'):
            model.set_alpha(alpha_t)

        # Mixup
        if use_mixup and np.random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            use_mixup_batch = True
        else:
            use_mixup_batch = False

        optimizer.zero_grad()
        outputs = model(inputs)

        if use_mixup_batch:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # EMA更新
        if ema is not None:
            # 恢复alpha=0来更新EMA（评估时用的状态）
            if hasattr(model, 'set_alpha'):
                model.set_alpha(0.0)
            ema.update(model)
            # 恢复随机alpha用于下一轮
            if hasattr(model, 'set_alpha'):
                model.set_alpha(alpha_t)

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
def evaluate(model, testloader, criterion, device, ema=None):
    model.eval()
    
    # 使用EMA权重评估
    if ema is not None:
        ema.apply_to(model)
    
    # 确保alpha=0（clean评估）
    if hasattr(model, 'set_alpha'):
        model.set_alpha(0.0)
    
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

    # 恢复原权重
    if ema is not None:
        for name, param in model.named_parameters():
            if name in ema.shadow:
                # 这里不恢复，因为后续训练会继续更新
                pass
    
    return test_loss / len(testloader), 100.*correct/total


@torch.no_grad()
def evaluate_with_ema(model, ema_model, testloader, criterion, device):
    """用EMA副本评估，不影响原模型"""
    ema_model.eval()
    if hasattr(ema_model, 'set_alpha'):
        ema_model.set_alpha(0.0)
    
    test_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = ema_model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return test_loss / len(testloader), 100.*correct/total


def run_experiment(config):
    device = config['device']
    model_name = config.get('model_name', 'resnet34')
    alpha_max = config.get('alpha_max', 0.3)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    seed = config.get('seed', 42)
    use_mixup = config.get('use_mixup', True)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    ema_decay = config.get('ema_decay', 0.995)
    label_smoothing = config.get('label_smoothing', 0.1)

    print("=" * 70)
    print(f"实验: {config.get('name', 'unknown')}")
    print(f"模型: {model_name}, alpha_max: {alpha_max}, epochs: {epochs}")
    print(f"Mixup: {use_mixup} (alpha={mixup_alpha}), EMA: {ema_decay}")
    print("=" * 70)

    # 数据增强
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

    # 加载模型
    model = get_model(name=model_name, num_classes=10, pretrained=True, alpha=0.0)
    model = NonLinearWrapper(model, alpha=0.0)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=0.9,
        weight_decay=config.get('weight_decay', 1e-4),
        nesterov=True
    )

    # Warmup + Cosine LR
    warmup_epochs = config.get('warmup_epochs', 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # EMA
    ema = EMA(model, decay=ema_decay)
    ema_model = get_model(name=model_name, num_classes=10, pretrained=False, alpha=0.0)
    ema_model = NonLinearWrapper(ema_model, alpha=0.0)
    ema_model = ema_model.to(device)

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
            alpha_max, ema=ema, use_mixup=use_mixup, mixup_alpha=mixup_alpha
        )
        scheduler.step()

        # 原始模型评估
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # EMA模型评估（用深拷贝避免影响训练）
        with torch.no_grad():
            for name, param in ema_model.named_parameters():
                if name in ema.shadow:
                    param.data.copy_(ema.shadow[name])
        test_loss_ema, test_acc_ema = evaluate_with_ema(
            ema_model, ema_model, testloader, criterion, device
        )

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

    return {
        'experiment_type': 'p1_sota',
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
    parser = argparse.ArgumentParser(description='赛题1 SOTA冲刺实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/sota_p1')
    args = parser.parse_args()

    configs = [
        # 配置1：ResNet34 + 混合Alpha(0.3) + Mixup + EMA（主力配置）
        {
            'name': 'resnet34_alpha0.3_mixup_ema',
            'device': args.device,
            'model_name': 'resnet34',
            'alpha_max': 0.3,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 10,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'ema_decay': 0.995,
            'label_smoothing': 0.1
        },
        # 配置2：ResNet34 + 更温和Alpha(0.2) + 更长训练
        {
            'name': 'resnet34_alpha0.2_long_mixup_ema',
            'device': args.device,
            'model_name': 'resnet34',
            'alpha_max': 0.2,
            'epochs': 150,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 10,
            'use_mixup': True,
            'mixup_alpha': 0.15,
            'ema_decay': 0.997,
            'label_smoothing': 0.05
        },
        # 配置3：ResNet50 + 混合Alpha(0.2) + Mixup + EMA
        {
            'name': 'resnet50_alpha0.2_mixup_ema',
            'device': args.device,
            'model_name': 'resnet50',
            'alpha_max': 0.2,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'lr': 0.05,
            'weight_decay': 1e-4,
            'warmup_epochs': 10,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'ema_decay': 0.995,
            'label_smoothing': 0.1
        },
    ]

    save_dir = os.path.join('/mnt/storage2/zyc/CIM比赛/赛道5/赛题1', args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"运行配置: {config['name']}")
        print(f"{'='*70}")

        result = run_experiment(config)
        result['config_name'] = config['name']
        all_results.append(result)

        save_path = os.path.join(save_dir, f"sota_p1_{config['name']}.json")
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

    summary_path = os.path.join(save_dir, "sota_p1_summary.json")
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
