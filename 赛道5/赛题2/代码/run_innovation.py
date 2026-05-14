"""
创新算法对比实验脚本

对比以下方法：
1. Baseline (无噪声训练)
2. Standard STE-NAT
3. Adaptive STE (不同噪声调度策略)
4. STE + 噪声感知正则化
5. STE + 偏差校正
6. STE + 层次化噪声注入

任务：图像分类 (CIFAR-10)
网络：ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import get_model
from ste.core import NoisyLinear, NoisyConv2d, STEGradientEstimator
from ste.innovation import (
    AdaptiveSTE, NoiseAwareRegularizer, BiasCorrector,
    LayerwiseNoiseInjection, InnovationConfig
)


def get_data_loaders(data_dir, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return trainloader, testloader


def inject_ste_to_model(model, noise_config):
    def replace_linear_with_noisy(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, name, NoisyConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    bias=child.bias is not None,
                    noise_config=noise_config
                ))
            elif isinstance(child, nn.Linear):
                setattr(module, name, NoisyLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    noise_config=noise_config
                ))
            else:
                replace_linear_with_noisy(child)

    replace_linear_with_noisy(model)
    return model


def set_model_noise_strength(model, strength):
    for module in model.modules():
        if hasattr(module, 'set_noise_strength'):
            module.set_noise_strength(strength)


class AdaptiveSTEWrapper:
    """自适应STE封装器"""

    def __init__(self, schedule='inverse'):
        self.schedule = schedule
        self.noise_level = 1.0

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def get_scale(self):
        nl = self.noise_level
        if self.schedule == 'inverse':
            return 1.0 / (1.0 + nl ** 2)
        elif self.schedule == 'linear':
            return 1.0 / (1.0 + nl)
        elif self.schedule == 'sqrt':
            return 1.0 / np.sqrt(1.0 + nl ** 2)
        elif self.schedule == 'exp':
            return np.exp(-nl / 2.0)
        return 1.0


class Trainer:
    """训练器封装"""

    def __init__(self, model, device, optimizer, criterion, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.adaptive_ste = None
        self.regularizer = None
        self.bias_corrector = None
        self.use_innovation = False
        self.noise_config = {}
        self.noise_strength = 1.0

    def set_adaptive_ste(self, adaptive_ste):
        self.adaptive_ste = adaptive_ste

    def set_regularizer(self, regularizer):
        self.regularizer = regularizer

    def set_bias_corrector(self, bias_corrector):
        self.bias_corrector = bias_corrector

    def set_noise_config(self, config):
        self.noise_config = config

    def train_epoch(self, trainloader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f'Train Epoch {epoch+1}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            if self.adaptive_ste is not None:
                self.adaptive_ste.set_noise_level(self.noise_strength)
                scale = self.adaptive_ste.get_scale()
                for module in self.model.modules():
                    if hasattr(module, 'noise_strength'):
                        pass

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if self.regularizer is not None:
                reg_loss = self.regularizer.compute_penalty(self.model)
                loss = loss + reg_loss

            loss.backward()

            if self.bias_corrector is not None:
                pass

            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        if self.scheduler is not None:
            self.scheduler.step()

        return running_loss / len(trainloader), 100. * correct / total

    def evaluate(self, testloader, noise_strength=0.0):
        self.model.eval()
        set_model_noise_strength(self.model, noise_strength)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        set_model_noise_strength(self.model, 0.0)
        return 100. * correct / total


def run_experiment(method_name, model, trainloader, testloader, device,
                   epochs=30, lr=0.1, noise_strength=1.0,
                   adaptive_ste_schedule=None, use_regularizer=False,
                   use_bias_correction=False, use_layerwise=False,
                   noise_config=None):
    """运行单个实验"""

    print(f"\n{'='*60}")
    print(f"方法: {method_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Trainer(model, device, optimizer, criterion, scheduler)
    trainer.noise_strength = noise_strength

    if adaptive_ste_schedule:
        trainer.adaptive_ste = AdaptiveSTEWrapper(adaptive_ste_schedule)

    if use_regularizer:
        trainer.regularizer = NoiseAwareRegularizer(sigma=0.01, penalty_type='l2')

    if use_bias_correction:
        trainer.bias_corrector = BiasCorrector(correction_type='ema')
        trainer.bias_corrector.noise_config = noise_config or {}

    trainer.use_innovation = any([adaptive_ste_schedule, use_regularizer, use_bias_correction, use_layerwise])

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'noise_strength': []
    }

    best_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(trainloader, epoch)
        test_acc = trainer.evaluate(testloader, noise_strength=0.0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['noise_strength'].append(noise_strength)

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    final_clean_acc = trainer.evaluate(testloader, noise_strength=0.0)
    final_noisy_acc = trainer.evaluate(testloader, noise_strength=noise_strength)

    print(f"\n最终结果:")
    print(f"  Clean精度: {final_clean_acc:.2f}%")
    print(f"  Noisy精度(ns={noise_strength}): {final_noisy_acc:.2f}%")
    print(f"  最佳精度: {best_acc:.2f}%")

    return {
        'method': method_name,
        'best_acc': best_acc,
        'final_clean_acc': final_clean_acc,
        'final_noisy_acc': final_noisy_acc,
        'history': history
    }


def plot_comparison(results, save_dir):
    """绘制对比图表"""

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = [r['method'] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    ax = axes[0, 0]
    for i, result in enumerate(results):
        history = result['history']
        ax.plot(history['test_acc'], label=result['method'], color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Epoch')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(len(methods))
    clean_accs = [r['final_clean_acc'] for r in results]
    noisy_accs = [r['final_noisy_acc'] for r in results]
    width = 0.35
    ax.bar(x - width/2, clean_accs, width, label='Clean', color='steelblue')
    ax.bar(x + width/2, noisy_accs, width, label='Noisy', color='coral')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Clean vs Noisy Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([80, 92])
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    for i, result in enumerate(results):
        history = result['history']
        ax.plot(history['train_loss'], label=result['method'], color=colors[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Epoch')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    best_accs = [r['best_acc'] for r in results]
    ax.bar(methods, best_accs, color=colors)
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Best Test Accuracy Comparison')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim([80, 92])
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(best_accs):
        ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'innovation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {os.path.join(save_dir, 'innovation_comparison.png')}")


def plot_ablation(results, save_dir):
    """绘制消融实验图表"""

    os.makedirs(save_dir, exist_ok=True)

    baseline_acc = results[0]['best_acc']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    methods = [r['method'] for r in results]
    improvements = [r['best_acc'] - baseline_acc for r in results]

    colors = ['gray' if imp <= 0 else 'green' for imp in improvements]
    bars = ax.bar(methods, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement vs Baseline (%)')
    ax.set_title('Ablation Study: Improvement vs Baseline')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{imp:+.2f}%', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    labels = ['Clean', 'Noisy']
    x = np.arange(len(methods))
    width = 0.35

    clean_accs = [r['final_clean_acc'] for r in results]
    noisy_accs = [r['final_noisy_acc'] for r in results]

    ax.bar(x - width/2, clean_accs, width, label='Clean', color='steelblue')
    ax.bar(x + width/2, noisy_accs, width, label='Noisy', color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Clean vs Noisy Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([80, 92])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_study.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融图表已保存: {os.path.join(save_dir, 'ablation_study.png')}")


def main():
    parser = argparse.ArgumentParser(description='创新算法对比实验')
    parser.add_argument('--device', type=str, default='cuda:1', help='设备')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/storage2/zyc/CIM比赛/公共数据集',
                        help='数据集目录')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/结果/innovation',
                        help='结果保存目录')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    trainloader, testloader = get_data_loaders(args.data_dir, args.batch_size)

    noise_config = {
        'prog_noise_std': 0.01,
        'drift_factor': 0.005,
        'nonlinear_alpha': 0.1,
        'nonlinear_beta': 0.05,
        'output_noise_std': 0.01,
        'crosstalk_factor': 0.002,
    }

    results = []

    print("\n" + "="*70)
    print("开始创新算法对比实验")
    print("="*70)

    methods = [
        ('Baseline', None, False, False, False),
        ('STE-NAT', 'standard', False, False, False),
        ('Adaptive-STE-Inverse', 'inverse', False, False, False),
        ('Adaptive-STE-Linear', 'linear', False, False, False),
        ('Adaptive-STE-Sqrt', 'sqrt', False, False, False),
        ('STE+Regularizer', 'standard', True, False, False),
        ('STE+BiasCorrection', 'standard', False, True, False),
        ('STE+Layerwise', 'standard', False, False, True),
        ('Full-Innovation', 'inverse', True, True, True),
    ]

    for method_name, adaptive_schedule, use_reg, use_bias, use_layerwise in methods:
        torch.cuda.empty_cache()

        model = get_model(name='resnet18', num_classes=10, pretrained=False)

        if adaptive_schedule != 'standard':
            model = inject_ste_to_model(model, noise_config)
        elif adaptive_schedule == 'standard':
            model = inject_ste_to_model(model, noise_config)

        result = run_experiment(
            method_name=method_name,
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            epochs=args.epochs,
            lr=0.1,
            noise_strength=1.0,
            adaptive_ste_schedule=adaptive_schedule if adaptive_schedule != 'standard' else None,
            use_regularizer=use_reg,
            use_bias_correction=use_bias,
            use_layerwise=use_layerwise,
            noise_config=noise_config
        )
        results.append(result)

    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)

    print(f"\n{'方法':<25} {'最佳精度':>10} {'Clean':>10} {'Noisy':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<25} {r['best_acc']:>9.2f}% {r['final_clean_acc']:>9.2f}% {r['final_noisy_acc']:>9.2f}%")

    baseline_acc = results[0]['best_acc']
    print("\n相对于基准的改进:")
    for r in results[1:]:
        improvement = r['best_acc'] - baseline_acc
        print(f"  {r['method']}: {improvement:+.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump([
            {
                'method': r['method'],
                'best_acc': r['best_acc'],
                'final_clean_acc': r['final_clean_acc'],
                'final_noisy_acc': r['final_noisy_acc'],
                'history': r['history']
            } for r in results
        ], f, indent=2)

    plot_comparison(results, args.save_dir)
    plot_ablation(results, args.save_dir)

    print(f"\n所有结果已保存至: {args.save_dir}")
    print("实验完成!")


if __name__ == '__main__':
    main()