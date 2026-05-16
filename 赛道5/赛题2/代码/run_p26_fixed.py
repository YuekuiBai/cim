"""
P2-6 正则化2.0 消融实验 - 修复版

对应专利三：基于偏差校正与正则化的STE梯度估计优化方法

核心改进：
1. 正则化权重从0.01降到0.001-0.0001
2. 添加warmup阶段（前5个epoch不加正则化）
3. 添加梯度裁剪
4. 使用官方噪声模型
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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet import get_model
from ste.core import NoisyLinear, NoisyConv2d


class NoiseAwareRegularizerV2Fixed:
    """
    噪声感知正则化2.0 - 修复版

    专利三的三项正则化：
    1. L2正则化：约束权重范数
    2. 梯度平滑正则化：约束权重变化幅度
    3. KL正则化：约束权重分布接近标准正态
    """

    def __init__(self, lambda_l2=0.001, lambda_grad_smooth=0.0005, lambda_kl=0.0001):
        self.lambda_l2 = lambda_l2
        self.lambda_grad_smooth = lambda_grad_smooth
        self.lambda_kl = lambda_kl
        self.previous_weights = None

    def compute_penalty(self, model):
        """计算三项正则化损失"""
        l2_penalty = 0.0
        grad_smooth_penalty = 0.0
        kl_penalty = 0.0
        num_params = 0

        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                num_params += param.numel()

                # L2正则化
                l2_penalty += torch.sum(param ** 2)

                # 梯度平滑正则化
                if self.previous_weights is not None and name in self.previous_weights:
                    grad_smooth_penalty += torch.sum((param - self.previous_weights[name]) ** 2)

                # KL正则化（约束分布接近标准正态）
                mean = param.mean()
                std = param.std() + 1e-8
                kl_penalty += 0.5 * (mean ** 2 + std ** 2 - torch.log(std ** 2) - 1)

        # 更新历史权重
        self.previous_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if 'weight' in name and param.requires_grad
        }

        # 归一化
        l2_penalty = l2_penalty / (num_params + 1e-8)
        grad_smooth_penalty = grad_smooth_penalty / (num_params + 1e-8)

        total_penalty = (self.lambda_l2 * l2_penalty +
                        self.lambda_grad_smooth * grad_smooth_penalty +
                        self.lambda_kl * kl_penalty)

        return total_penalty


class BiasCorrector:
    """
    偏差校正器 - 专利三

    使用EMA估计梯度偏差并校正
    """

    def __init__(self, ema_decay=0.9):
        self.ema_decay = ema_decay
        self.ema_gradients = {}

    def correct(self, gradient, param_id):
        if param_id not in self.ema_gradients:
            self.ema_gradients[param_id] = gradient.detach().clone()
        else:
            ema = self.ema_gradients[param_id]
            if ema.shape != gradient.shape:
                self.ema_gradients[param_id] = gradient.detach().clone()
            else:
                self.ema_gradients[param_id] = self.ema_decay * ema + (1 - self.ema_decay) * gradient.detach()

        # 校正
        corrected = gradient - (self.ema_gradients[param_id] - gradient.detach())
        return corrected


def inject_ste_to_model(model, noise_config):
    """将模型的Conv2d/Linear层替换为带噪声的版本"""
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


def run_experiment(config):
    """运行单次实验"""
    import torchvision
    import torchvision.transforms as transforms

    device = config['device']
    print("=" * 70)
    print(f"实验类型: regularizer_v2_fixed ({config['name']})")
    print(f"配置: lambda_l2={config['lambda_l2']}, lambda_grad_smooth={config['lambda_grad_smooth']}, lambda_kl={config['lambda_kl']}")
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
        'schedule': 'inverse'
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

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # 正则化器
    regularizer = NoiseAwareRegularizerV2Fixed(
        lambda_l2=config['lambda_l2'],
        lambda_grad_smooth=config['lambda_grad_smooth'],
        lambda_kl=config['lambda_kl']
    )

    # 偏差校正器
    bias_corrector = BiasCorrector(ema_decay=0.9)

    # 训练历史
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, total_epochs + 1):
        # 训练
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

            # Warmup后添加正则化
            if epoch > warmup_epochs:
                reg_loss = regularizer.compute_penalty(model)
                loss = loss + reg_loss

            loss.backward()

            # 偏差校正
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = bias_corrector.correct(param.grad.data, id(param))

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        train_acc = 100.*correct/total
        history['train_loss'].append(train_loss/len(trainloader))
        history['train_acc'].append(train_acc)

        scheduler.step()

        # 测试
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.*test_correct/test_total
        history['test_loss'].append(test_loss/len(testloader))
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print(f'  -> 新最佳精度: {best_acc:.2f}% (Epoch {best_epoch})')

    return {
        'experiment_type': f'regularizer_v2_fixed_{config["name"]}',
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='P2-6 正则化2.0修复版')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 多组配置进行消融实验
    configs = [
        # 配置1：轻度正则化
        {
            'name': 'light',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.5,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'lambda_l2': 0.0001,
            'lambda_grad_smooth': 0.00005,
            'lambda_kl': 0.00001
        },
        # 配置2：中度正则化
        {
            'name': 'moderate',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.5,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'lambda_l2': 0.001,
            'lambda_grad_smooth': 0.0005,
            'lambda_kl': 0.0001
        },
        # 配置3：仅L2正则化（消融）
        {
            'name': 'l2_only',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.5,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'lambda_l2': 0.001,
            'lambda_grad_smooth': 0.0,
            'lambda_kl': 0.0
        },
        # 配置4：仅梯度平滑（消融）
        {
            'name': 'smooth_only',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.5,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'lambda_l2': 0.0,
            'lambda_grad_smooth': 0.001,
            'lambda_kl': 0.0
        },
        # 配置5：无正则化基线
        {
            'name': 'no_reg',
            'device': args.device,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'noise_strength': 0.5,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'lambda_l2': 0.0,
            'lambda_grad_smooth': 0.0,
            'lambda_kl': 0.0
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
        save_path = f'/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/结果/enhanced_experiments/regularizer_v2_fixed_{config["name"]}.json'
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")

    # 汇总结果
    print("\n" + "=" * 70)
    print("消融实验结果汇总:")
    print("=" * 70)

    best_result = None
    for r in all_results:
        print(f"  {r['config_name']:15s}: Best Acc = {r['best_acc']:.2f}% (Epoch {r['best_epoch']}), Final Acc = {r['final_acc']:.2f}%")
        if best_result is None or r['best_acc'] > best_result['best_acc']:
            best_result = r

    print(f"\n最佳配置: {best_result['config_name']}")
    print(f"最佳精度: {best_result['best_acc']:.2f}%")

    # 保存汇总
    summary_path = '/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/结果/enhanced_experiments/regularizer_v2_fixed_summary.json'
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
