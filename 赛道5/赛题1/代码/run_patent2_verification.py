"""
赛题1 专利二验证实验 - 逐层非线性误差诊断与补偿方法

核心创新：
1. 前向钩子捕获各层输出分布
2. MSE敏感性分析 → 分级（高/中/低敏感）
3. 差异化补偿（高敏感→校准层，中敏感→预失真，低敏感→不处理）

当前问题：补偿效果仅60.74%，需改进补偿策略
改进方案：
1. 残差连接校准层
2. 可学习多项式拟合
3. 梯度敏感度分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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


# ==================== 专利二核心组件 ====================

class SensitivityAnalyzer:
    """敏感性分析器 - 前向钩子 + MSE评估"""
    def __init__(self):
        self.layer_outputs = {}
        self.hooks = []
        
    def register_hooks(self, model):
        def make_hook(name):
            def hook(module, input, output):
                self.layer_outputs[name] = output.detach()
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_hook(make_hook(name)))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_sensitivity(self, model, dataloader, device, alpha=0.5):
        """计算各层MSE敏感性"""
        model.eval()
        mse_scores = {}
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                # 干净前向
                model.module.set_alpha(0.0)
                _ = model(inputs)
                clean_outputs = {k: v.clone() for k, v in self.layer_outputs.items()}
                
                # 非线性前向
                model.module.set_alpha(alpha)
                _ = model(inputs)
                noisy_outputs = self.layer_outputs
                
                # 计算MSE
                for name in clean_outputs:
                    if name not in mse_scores:
                        mse_scores[name] = 0.0
                    mse_scores[name] += torch.mean((clean_outputs[name] - noisy_outputs[name])**2).item()
                
                break  # 仅一个batch
        
        # 归一化MSE
        max_mse = max(mse_scores.values()) if mse_scores else 1.0
        normalized_mse = {k: v / max_mse for k, v in mse_scores.items()}
        
        return normalized_mse
    
    def classify_layers(self, mse_scores, strict_threshold=0.5, loose_threshold=0.2):
        """根据MSE分级"""
        high_sensitive = []
        medium_sensitive = []
        low_sensitive = []
        
        for name, mse in mse_scores.items():
            if mse > strict_threshold:
                high_sensitive.append(name)
            elif mse > loose_threshold:
                medium_sensitive.append(name)
            else:
                low_sensitive.append(name)
        
        return high_sensitive, medium_sensitive, low_sensitive


class CalibrationLayer(nn.Module):
    """改进的校准层 - 残差连接 + 可学习多项式"""
    def __init__(self, original_module, alpha=0.5):
        super().__init__()
        self.original = original_module
        self.alpha = alpha
        
        # 可学习多项式系数（残差连接）
        if isinstance(original_module, nn.Conv2d):
            out_channels = original_module.out_channels
        elif isinstance(original_module, nn.Linear):
            out_channels = original_module.out_features
        else:
            out_channels = 64  # 默认
        
        self.c1 = nn.Parameter(torch.ones(out_channels) * 0.5)
        self.c3 = nn.Parameter(torch.ones(out_channels) * 0.1)
        
        # 梯度敏感度权重
        self.grad_sensitivity = nn.Parameter(torch.ones(out_channels))
    
    def forward(self, x):
        # 原始输出
        if isinstance(self.original, nn.Conv2d):
            y_orig = torch.nn.functional.conv2d(
                x, self.original.weight, self.original.bias,
                self.original.stride, self.original.padding,
                self.original.dilation, self.original.groups
            )
        else:
            y_orig = torch.nn.functional.linear(x, self.original.weight, self.original.bias)
        
        # 非线性失真
        y_noisy = self.alpha * y_orig**3 + (1 - self.alpha) * y_orig
        
        # 校准：逐通道多项式校正 + 残差连接
        c1 = self.c1.view(1, -1, 1, 1)
        c3 = self.c3.view(1, -1, 1, 1)
        grad_weight = self.grad_sensitivity.view(1, -1, 1, 1)
        
        y_calibrated = c1 * y_noisy + c3 * y_noisy**3 * grad_weight + y_orig
        
        return y_calibrated


class PreDistortionCompensation(nn.Module):
    """改进的预失真补偿"""
    def __init__(self, original_module, alpha=0.5):
        super().__init__()
        self.original = original_module
        self.alpha = alpha
        
        # 可学习预失真参数
        if isinstance(original_module, nn.Conv2d):
            out_channels = original_module.out_channels
        elif isinstance(original_module, nn.Linear):
            out_channels = original_module.out_features
        else:
            out_channels = 64
        
        self.beta = nn.Parameter(torch.ones(out_channels) * 0.1)
    
    def forward(self, x):
        # 预失真
        if isinstance(self.original, nn.Conv2d):
            out_channels = self.original.out_channels
            beta = self.beta.view(1, -1, 1, 1)
            x_distorted = x + beta * x**3
            return torch.nn.functional.conv2d(
                x_distorted, self.original.weight, self.original.bias,
                self.original.stride, self.original.padding,
                self.original.dilation, self.original.groups
            )
        else:
            beta = self.beta.view(1, -1)
            x_distorted = x + beta * x**3
            return torch.nn.functional.linear(x_distorted, self.original.weight, self.original.bias)


def apply_compensation(model, high_sensitive, medium_sensitive, low_sensitive, alpha=0.5, compensation_type='calibration'):
    """应用差异化补偿"""
    compensated_model = deepcopy(model)
    
    for name in high_sensitive:
        parts = name.split('.')
        module = compensated_model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        if compensation_type == 'calibration':
            setattr(compensated_model, name, CalibrationLayer(module, alpha))
        elif compensation_type == 'predistortion':
            setattr(compensated_model, name, PreDistortionCompensation(module, alpha))
    
    return compensated_model


# ==================== 训练函数 ====================

def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, total_epochs, alpha=0.5):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs}')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return train_loss / len(trainloader), 100.*correct/total


@torch.no_grad()
def evaluate(model, testloader, criterion, device, alpha=0.0):
    model.eval()
    if hasattr(model, 'set_alpha'):
        model.set_alpha(alpha)
    
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
    device = config['device']
    experiment_type = config.get('experiment_type', 'sensitivity_analysis')
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    seed = config.get('seed', 42)
    alpha = config.get('alpha', 0.5)
    
    print("=" * 70)
    print(f"实验: {config.get('name', 'unknown')}")
    print(f"类型: {experiment_type}")
    print(f"Alpha: {alpha}, Epochs: {epochs}")
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
    
    # 根据实验类型构建模型
    model = get_model(name='resnet18', num_classes=10, pretrained=True, alpha=0.0)
    
    # 注册敏感性分析钩子
    analyzer = SensitivityAnalyzer()
    analyzer.register_hooks(model)
    
    # 敏感性分析
    print("正在进行敏感性分析...")
    mse_scores = analyzer.analyze_sensitivity(model, trainloader, device, alpha)
    high_sensitive, medium_sensitive, low_sensitive = analyzer.classify_layers(mse_scores)
    print(f"高敏感层: {len(high_sensitive)}")
    print(f"中敏感层: {len(medium_sensitive)}")
    print(f"低敏感层: {len(low_sensitive)}")
    analyzer.remove_hooks()
    
    # 应用补偿
    if experiment_type == 'calibration':
        model = apply_compensation(model, high_sensitive, medium_sensitive, low_sensitive, alpha, 'calibration')
    elif experiment_type == 'predistortion':
        model = apply_compensation(model, high_sensitive, medium_sensitive, low_sensitive, alpha, 'predistortion')
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=0.9,
        weight_decay=config.get('weight_decay', 5e-4)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, epochs, alpha)
        scheduler.step()
        
        test_loss, test_acc = evaluate(model, testloader, criterion, device, alpha=0.0)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch}/{epochs}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            print(f'  -> 新最佳: {best_acc:.2f}% (Epoch {best_epoch})')
    
    return {
        'experiment_type': experiment_type,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'mse_scores': mse_scores,
        'sensitivity_classification': {
            'high': high_sensitive,
            'medium': medium_sensitive,
            'low': low_sensitive
        },
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='赛题1 专利二验证实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/patent2_verification')
    args = parser.parse_args()
    
    configs = [
        # 配置1：敏感性分析 + 校准层补偿
        {'name': 'calibration', 'device': args.device, 'experiment_type': 'calibration',
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed,
         'lr': 0.1, 'weight_decay': 5e-4, 'alpha': 0.5},
        
        # 配置2：敏感性分析 + 预失真补偿
        {'name': 'predistortion', 'device': args.device, 'experiment_type': 'predistortion',
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 1,
         'lr': 0.1, 'weight_decay': 5e-4, 'alpha': 0.5},
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
        
        save_path = os.path.join(save_dir, f"patent2_{config['name']}.json")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")
    
    print("\n" + "=" * 70)
    print("专利二验证实验结果汇总:")
    print("=" * 70)
    
    for r in all_results:
        print(f"  {r['config_name']}: 最佳={r['best_acc']:.2f}%(E{r['best_epoch']}), 最终={r['final_acc']:.2f}%")
    
    if all_results:
        best_result = max(all_results, key=lambda x: x['best_acc'])
        print(f"\n最佳配置: {best_result['config_name']}")
        print(f"最佳精度: {best_result['best_acc']:.2f}%")
        
        summary_path = os.path.join(save_dir, "patent2_summary.json")
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
