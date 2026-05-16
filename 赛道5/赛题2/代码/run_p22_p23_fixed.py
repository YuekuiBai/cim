"""
P2-2 和 P2-3 修复版实验脚本

修复内容：
P2-2 梯度方差自适应：
1. 增加训练轮次到50 epochs（匹配基线）
2. 添加warmup和梯度裁剪
3. 改进梯度方差追踪机制（使用per-element方差）
4. 增强自适应缩放的敏感度

P2-3 零阶校正：
1. 使用SPSA方法（2次前向传播代替220次）
2. 添加torch.no_grad()
3. 降低num_samples从5到1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import get_model
from ste.core import NoisyLinear, NoisyConv2d


def inject_ste_to_model(model, noise_config):
    """注入STE噪声层"""
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(model, name, NoisyConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                bias=child.bias is not None,
                noise_config=noise_config
            ))
        elif isinstance(child, nn.Linear):
            setattr(model, name, NoisyLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                noise_config=noise_config
            ))
        else:
            inject_ste_to_model(child, noise_config)
    return model


class ImprovedGradientVarianceTracker:
    """改进的梯度方差跟踪器 - 使用per-element方差"""

    def __init__(self, window_size=50, momentum=0.9):
        self.window_size = window_size
        self.momentum = momentum
        self.grad_second_moment = None  # E[g^2]
        self.grad_first_moment = None   # E[g]
        self.step_count = 0

    def update(self, model):
        """更新梯度方差估计（使用Welford在线算法）"""
        self.step_count += 1

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data.clone()

            if self.grad_second_moment is None:
                self.grad_second_moment = {}
                self.grad_first_moment = {}

            if name not in self.grad_second_moment:
                self.grad_second_moment[name] = grad.pow(2)
                self.grad_first_moment[name] = grad.clone()
            else:
                # 指数移动平均
                self.grad_second_moment[name] = (
                    self.momentum * self.grad_second_moment[name] +
                    (1 - self.momentum) * grad.pow(2)
                )
                self.grad_first_moment[name] = (
                    self.momentum * self.grad_first_moment[name] +
                    (1 - self.momentum) * grad
                )

    def get_adaptive_scale(self, name, base_scale=1.0):
        """获取自适应缩放因子（基于方差）"""
        if self.grad_second_moment is None or name not in self.grad_second_moment:
            return base_scale

        # Var[g] = E[g^2] - (E[g])^2
        variance = self.grad_second_moment[name] - self.grad_first_moment[name].pow(2)
        variance = variance.clamp(min=0)  # 确保非负

        # 使用更强的敏感度：scale = 1 / (1 + sqrt(var))
        scale = base_scale / (1.0 + torch.sqrt(variance + 1e-8))

        return scale


class SPSAZeroOrderCorrector:
    """SPSA零阶校正器 - 高效版本（2次前向传播）"""

    def __init__(self, alpha=0.2, perturbation_std=0.01):
        self.alpha = alpha
        self.perturbation_std = perturbation_std

    def compute_spsa_gradient(self, model, loss_fn, inputs, targets):
        """使用SPSA方法计算零阶梯度（只需2次前向传播）"""
        # 保存原始参数
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()

        # 生成共享的随机方向
        perturbation = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 使用Rademacher分布（+1/-1）而不是高斯分布
                perturbation[name] = torch.sign(torch.randn_like(param)) * self.perturbation_std

        # 正向扰动
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.add_(perturbation[name])

        with torch.no_grad():
            loss_plus = loss_fn(model(inputs), targets).item()

        # 恢复原始参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_params[name])

        # 负向扰动
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.sub_(perturbation[name])

        with torch.no_grad():
            loss_minus = loss_fn(model(inputs), targets).item()

        # 恢复原始参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_params[name])

        # 计算SPSA梯度估计
        zo_grad = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # SPSA梯度：(f(θ+Δ) - f(θ-Δ)) / (2Δ)
                zo_grad[name] = (loss_plus - loss_minus) / (2 * self.perturbation_std) * perturbation[name]

        return zo_grad

    def correct_gradient(self, ste_grad, zo_grad):
        """结合STE梯度和零阶梯度"""
        corrected = {}
        for name in ste_grad:
            if name in zo_grad:
                corrected[name] = ste_grad[name] + self.alpha * (zo_grad[name] - ste_grad[name])
            else:
                corrected[name] = ste_grad[name]
        return corrected


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


def run_gradient_variance_experiment(config):
    """运行P2-2梯度方差自适应实验（修复版）"""
    import torchvision
    import torchvision.transforms as transforms

    device = config['device']
    print("=" * 70)
    print(f"实验类型: gradient_variance_adaptive_fixed")
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
    noise_config = {
        'prog_noise_std': 0.01,
        'drift_factor': 0.005,
        'nonlinear_alpha': 0.1,
        'nonlinear_beta': 0.05,
        'output_noise_std': 0.01,
        'crosstalk_factor': 0.002,
    }

    model = inject_ste_to_model(model, noise_config)
    model = model.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )

    # 学习率调度：warmup + cosine
    warmup_epochs = config['warmup_epochs']
    total_epochs = config['epochs']

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # 梯度方差跟踪器
    var_tracker = ImprovedGradientVarianceTracker(window_size=50, momentum=0.9)

    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            # 更新梯度方差估计
            var_tracker.update(model)

            # 应用自适应缩放
            for name, param in model.named_parameters():
                if param.grad is not None:
                    scale = var_tracker.get_adaptive_scale(name, base_scale=1.0)
                    param.grad.data *= scale

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(100.*correct/total)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch}/{total_epochs}: '
              f'Train Acc={100.*correct/total:.2f}%, Test Acc={test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print(f'  -> 新最佳精度: {best_acc:.2f}% (Epoch {best_epoch})')

    return {
        'experiment_type': 'gradient_variance_adaptive_fixed',
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'history': history,
        'config': config
    }


def run_zero_order_experiment(config):
    """运行P2-3零阶校正实验（SPSA修复版）"""
    import torchvision
    import torchvision.transforms as transforms

    device = config['device']
    print("=" * 70)
    print(f"实验类型: zero_order_correction_spsa_fixed")
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
    noise_config = {
        'prog_noise_std': 0.01,
        'drift_factor': 0.005,
        'nonlinear_alpha': 0.1,
        'nonlinear_beta': 0.05,
        'output_noise_std': 0.01,
        'crosstalk_factor': 0.002,
    }

    model = inject_ste_to_model(model, noise_config)
    model = model.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )

    # 学习率调度
    warmup_epochs = config['warmup_epochs']
    total_epochs = config['epochs']

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # SPSA零阶校正器
    zo_corrector = SPSAZeroOrderCorrector(
        alpha=config.get('zo_alpha', 0.2),
        perturbation_std=config.get('perturbation_std', 0.01)
    )

    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            # 保存STE梯度
            ste_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    ste_grads[name] = param.grad.clone()

            # 使用SPSA计算零阶梯度（只需2次前向传播）
            zo_grads = zo_corrector.compute_spsa_gradient(model, criterion, inputs, targets)

            # 结合STE和零阶梯度
            corrected_grads = zo_corrector.correct_gradient(ste_grads, zo_grads)

            # 应用校正后的梯度
            for name, param in model.named_parameters():
                if name in corrected_grads:
                    param.grad.data.copy_(corrected_grads[name])

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(100.*correct/total)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch}/{total_epochs}: '
              f'Train Acc={100.*correct/total:.2f}%, Test Acc={test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print(f'  -> 新最佳精度: {best_acc:.2f}% (Epoch {best_epoch})')

    return {
        'experiment_type': 'zero_order_correction_spsa_fixed',
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='P2-2和P2-3修复版实验')
    parser.add_argument('--experiment', type=str, choices=['p22', 'p23', 'all'], default='all',
                       help='运行哪个实验: p22, p23, all')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/enhanced_experiments')
    args = parser.parse_args()

    # 处理设备：当CUDA_VISIBLE_DEVICES设置时，使用cuda:0
    device = args.device
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        # 当设置CUDA_VISIBLE_DEVICES时，PyTorch只能看到一个设备，使用cuda:0
        if device.startswith('cuda'):
            device = 'cuda:0'
    elif device == 'cuda' and torch.cuda.is_available():
        device = 'cuda:0'

    # P2-2配置（修复版）
    p22_config = {
        'name': 'gradient_variance_fixed',
        'device': device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'noise_strength': 0.5,  # 降低噪声强度
        'lr': 0.1,
        'weight_decay': 5e-4,
        'warmup_epochs': 5
    }

    # P2-3配置（SPSA修复版）
    p23_config = {
        'name': 'zero_order_spsa_fixed',
        'device': device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'noise_strength': 0.5,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'warmup_epochs': 5,
        'zo_alpha': 0.2,
        'perturbation_std': 0.01
    }

    save_dir = os.path.join('/mnt/storage2/zyc/CIM比赛/赛道5/赛题2', args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.experiment in ['p22', 'all']:
        print("\n" + "=" * 70)
        print("运行P2-2梯度方差自适应实验（修复版）")
        print("=" * 70)
        result = run_gradient_variance_experiment(p22_config)
        save_path = os.path.join(save_dir, 'gradient_variance_fixed.json')
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")

    if args.experiment in ['p23', 'all']:
        print("\n" + "=" * 70)
        print("运行P2-3零阶校正实验（SPSA修复版）")
        print("=" * 70)
        result = run_zero_order_experiment(p23_config)
        save_path = os.path.join(save_dir, 'zero_order_spsa_fixed.json')
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")


if __name__ == '__main__':
    main()
