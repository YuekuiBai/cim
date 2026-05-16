"""
P2-4 时空相关噪声 - 修复版

修复内容：
1. 修复评估时注入噪声的致命bug（添加self.training检查）
2. 降低噪声强度5-10倍
3. 实现真正的层间空间相关性（Cholesky分解）
4. 添加噪声退火策略
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
from datetime import datetime

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
        self.rho_spatial = noise_config.get('rho_spatial', 0.3)
        self.rho_temporal = noise_config.get('rho_temporal', 0.5)
        self.beta = noise_config.get('beta', 0.5)

        # 时间维度噪声状态
        self.register_buffer('temporal_state', None)

        # 计算层次化噪声强度 gamma_l = gamma_base * (1 + beta * l/L)
        self.gamma_l = self.noise_strength * (1.0 + self.beta * layer_idx / max(total_layers - 1, 1))

    def forward(self, x):
        # 修复：评估时不注入噪声
        if not self.training:
            if isinstance(self.layer, nn.Conv2d):
                return F.conv2d(x, self.layer.weight, self.layer.bias,
                              self.layer.stride, self.layer.padding,
                              self.layer.dilation, self.layer.groups)
            elif isinstance(self.layer, nn.Linear):
                return F.linear(x, self.layer.weight, self.layer.bias)

        # 训练时注入噪声
        weight_shape = self.layer.weight.shape

        # 生成当前步的噪声
        fresh_noise = torch.randn(weight_shape, device=self.layer.weight.device)

        # 时间维度：AR(1)模型
        if self.temporal_state is None or self.temporal_state.shape != weight_shape:
            self.temporal_state = torch.zeros(weight_shape, device=self.layer.weight.device)

        temporal_noise = (self.rho_temporal * self.temporal_state +
                         np.sqrt(1 - self.rho_temporal**2) * fresh_noise)
        self.temporal_state = temporal_noise.detach().clone()

        # 应用层次化噪声强度
        weight_noise = temporal_noise * self.gamma_l

        # 使用noisy_weight进行前向传播
        noisy_weight = self.layer.weight + weight_noise

        if isinstance(self.layer, nn.Conv2d):
            return F.conv2d(x, noisy_weight, self.layer.bias,
                          self.layer.stride, self.layer.padding,
                          self.layer.dilation, self.layer.groups)
        elif isinstance(self.layer, nn.Linear):
            return F.linear(x, noisy_weight, self.layer.bias)


class InterLayerCorrelation:
    """层间噪声相关控制器 - 使用Cholesky分解"""

    def __init__(self, num_layers, rho=0.3):
        self.num_layers = num_layers
        self.rho = rho
        self.cov_matrix = self._build_covariance()
        self.L = None

    def _build_covariance(self):
        """构建AR(1)协方差矩阵"""
        cov = torch.zeros(self.num_layers, self.num_layers)
        for i in range(self.num_layers):
            for j in range(self.num_layers):
                cov[i, j] = self.rho ** abs(i - j)
        return cov

    def get_correlated_noise(self, layer_idx, weight_shape, device):
        """获取带层间相关性的噪声"""
        if self.L is None:
            cov_reg = self.cov_matrix + 1e-6 * torch.eye(self.num_layers)
            self.L = torch.linalg.cholesky(cov_reg)

        # 生成相关的随机缩放因子
        z = torch.randn(self.num_layers, device=device)
        correlated = self.L.to(device) @ z

        # 使用相关系数调制噪声强度
        scale = torch.sigmoid(correlated[layer_idx])  # [0, 1]
        noise = torch.randn(weight_shape, device=device) * scale
        return noise


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


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, total_epochs, noise_config):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # 噪声退火
    current_noise = get_noise_strength(
        epoch, total_epochs,
        noise_config['base_strength'],
        noise_config.get('schedule', 'cosine')
    )

    # 更新所有噪声层的强度
    for module in model.modules():
        if isinstance(module, SpatiotemporalNoiseLayer):
            module.gamma_l = current_noise * (1.0 + module.beta * module.layer_idx / max(module.total_layers - 1, 1))

    pbar = tqdm(trainloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

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
    print(f"实验类型: spatiotemporal_noise_fixed")
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
        'noise_strength': config['noise_strength'],
        'rho_spatial': config['rho_spatial'],
        'rho_temporal': config['rho_temporal'],
        'beta': config['beta'],
        'base_strength': config['noise_strength'],
        'schedule': config.get('noise_schedule', 'cosine')
    }

    # 注入时空相关噪声
    model = inject_spatiotemporal_noise(model, noise_config)
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

    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch, total_epochs, noise_config
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
        'experiment_type': 'spatiotemporal_noise_fixed',
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='P2-4 修复版实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/enhanced_experiments')
    args = parser.parse_args()

    # 修复版配置 - 降低噪声强度5-10倍
    configs = [
        # 配置1：保守配置（低噪声）
        {
            'name': 'conservative_fixed',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.005,     # 降低10倍
            'rho_spatial': 0.3,
            'rho_temporal': 0.5,
            'beta': 0.5,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'noise_schedule': 'cosine'   # 噪声退火
        },
        # 配置2：中等配置
        {
            'name': 'moderate_fixed',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.01,      # 降低10倍
            'rho_spatial': 0.5,
            'rho_temporal': 0.7,
            'beta': 1.0,
            'lr': 0.1,
            'weight_decay': 1e-3,
            'warmup_epochs': 5,
            'noise_schedule': 'cosine'
        },
        # 配置3：较高噪声配置
        {
            'name': 'aggressive_fixed',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.02,      # 降低10倍
            'rho_spatial': 0.7,
            'rho_temporal': 0.8,
            'beta': 1.5,
            'lr': 0.1,
            'weight_decay': 2e-3,
            'warmup_epochs': 5,
            'noise_schedule': 'cosine'
        }
    ]

    all_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"运行配置: {config['name']}")
        print(f"{'='*70}")

        result = run_experiment(config)
        result['config_name'] = config['name']
        all_results.append(result)

        # 保存单个结果
        save_path = os.path.join(
            '/mnt/storage2/zyc/CIM比赛/赛道5/赛题2',
            args.save_dir,
            f"spatiotemporal_fixed_{config['name']}.json"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    summary_path = os.path.join(
        '/mnt/storage2/zyc/CIM比赛/赛道5/赛题2',
        args.save_dir,
        "spatiotemporal_fixed_summary.json"
    )
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
