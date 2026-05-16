"""
专利二+专利一组合实验 - 分层Alpha训练方法

核心思想：
1. 专利二诊断各层非线性敏感性（MSE分析）
2. 根据敏感性分级，为不同层设置不同的alpha采样范围
   - 高敏感层：α ∈ U(0, 0.15) - 更温和，减少失真
   - 中敏感层：α ∈ U(0, 0.3) - 标准范围
   - 低敏感层：α ∈ U(0, 0.5) - 更大范围，充分利用模型容量
3. 配合专利一的逐通道归一化+余弦LR+Mixup+EMA

预期：87.33% → 89~90%+
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
                if hasattr(model, 'set_alpha'):
                    model.set_alpha(0.0)
                _ = model(inputs)
                clean_outputs = {k: v.clone() for k, v in self.layer_outputs.items()}
                
                # 非线性前向
                if hasattr(model, 'set_alpha'):
                    model.set_alpha(alpha)
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
        if max_mse < 1e-8:
            max_mse = 1.0
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


# ==================== 专利一核心组件 ====================

class NonLinearWrapper(nn.Module):
    """非线性包装器 - 支持分层Alpha"""
    def __init__(self, model, alpha=0.0):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.layerwise_alpha = {}  # 分层alpha设置
        self.use_layerwise = False  # 是否启用分层alpha
        
    def set_alpha(self, alpha):
        self.alpha = alpha
        
    def set_layerwise_alpha(self, layerwise_alpha_dict):
        """设置分层alpha范围"""
        self.layerwise_alpha = layerwise_alpha_dict
        self.use_layerwise = True
        
    def get_alpha_for_layer(self, layer_name):
        """获取特定层的alpha采样范围"""
        if not self.use_layerwise:
            return 0.0, self.alpha
        
        sensitivity = self.layerwise_alpha.get(layer_name, 'medium')
        if sensitivity == 'high':
            return 0.0, 0.15  # 高敏感层：U(0, 0.15)
        elif sensitivity == 'medium':
            return 0.0, 0.3   # 中敏感层：U(0, 0.3)
        else:
            return 0.0, 0.5   # 低敏感层：U(0, 0.5)
    
    def forward(self, x):
        return self.model(x)


def apply_nonlinearity_with_layerwise_alpha(x, alpha_min, alpha_max, use_channel_norm=True):
    """应用非线性变换，支持逐通道归一化"""
    alpha_t = np.random.uniform(alpha_min, alpha_max)
    
    if use_channel_norm:
        # 逐通道归一化防止数值溢出
        x_norm = x / (torch.max(torch.abs(x)) + 1e-8)
        return alpha_t * x_norm**3 + (1 - alpha_t) * x_norm
    else:
        return alpha_t * x**3 + (1 - alpha_t) * x


class LayerwiseNonLinearityHook:
    """为每个Conv2d/Linear层添加分层非线性钩子"""
    def __init__(self):
        self.hooks = []
        self.alpha_ranges = {}  # {layer_name: (alpha_min, alpha_max)}
        
    def set_alpha_range(self, layer_name, alpha_min, alpha_max):
        self.alpha_ranges[layer_name] = (alpha_min, alpha_max)
    
    def register_hooks(self, model):
        def make_forward_hook(name, alpha_range):
            def forward_hook(module, input, output):
                alpha_min, alpha_max = alpha_range
                alpha_t = np.random.uniform(alpha_min, alpha_max)
                
                # 逐通道归一化
                output_norm = output / (torch.max(torch.abs(output)) + 1e-8)
                return alpha_t * output_norm**3 + (1 - alpha_t) * output_norm
            return forward_hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                alpha_range = self.alpha_ranges.get(name, (0.0, 0.3))
                self.hooks.append(module.register_forward_hook(make_forward_hook(name, alpha_range)))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ==================== Mixup ====================

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


# ==================== EMA ====================

class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data


# ==================== 训练函数 ====================

def train_one_epoch_layerwise(model, trainloader, criterion, optimizer, device, epoch, total_epochs,
                              layerwise_hook, ema=None, use_mixup=False, mixup_alpha=0.2):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{total_epochs} (LayerwiseAlpha)')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return train_loss / len(trainloader), 100.*correct/total


@torch.no_grad()
def evaluate(model, testloader, criterion, device, layerwise_hook=None):
    model.eval()
    
    # 评估时移除非线性钩子
    if layerwise_hook is not None:
        layerwise_hook.remove_hooks()
    
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
    
    # 恢复钩子用于后续训练
    if layerwise_hook is not None:
        layerwise_hook.register_hooks(model)
    
    return test_loss / len(testloader), 100.*correct/total


@torch.no_grad()
def evaluate_with_ema(model, ema, testloader, criterion, device, layerwise_hook=None):
    """用EMA权重评估（临时应用EMA权重到原模型）"""
    model.eval()
    
    if layerwise_hook is not None:
        layerwise_hook.remove_hooks()
    
    # 保存原始权重并应用EMA权重
    restored = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in ema.shadow:
            restored[name] = param.data.clone()
            param.data.copy_(ema.shadow[name])
    
    # 评估
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
    
    # 恢复原始权重
    for name, data in restored.items():
        model.state_dict()[name].copy_(data)
    
    # 恢复钩子
    if layerwise_hook is not None:
        layerwise_hook.register_hooks(model)
    
    return test_loss / len(testloader), 100.*correct/total


def run_experiment(config):
    device = config['device']
    experiment_type = config.get('experiment_type', 'layerwise_alpha_combined')
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    seed = config.get('seed', 42)
    use_mixup = config.get('use_mixup', True)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    ema_decay = config.get('ema_decay', 0.995)
    label_smoothing = config.get('label_smoothing', 0.1)
    sensitivity_threshold_high = config.get('sensitivity_threshold_high', 0.5)
    sensitivity_threshold_low = config.get('sensitivity_threshold_low', 0.2)
    
    print("=" * 70)
    print(f"实验: {config.get('name', 'unknown')}")
    print(f"类型: {experiment_type}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"Mixup: {use_mixup} (α={mixup_alpha}), EMA: {ema_decay}")
    print(f"敏感性阈值: 高>{sensitivity_threshold_high}, 低<{sensitivity_threshold_low}")
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
    
    # 步骤1：加载模型并进行敏感性分析（专利二）
    print("步骤1：专利二 - 敏感性分析...")
    model_for_analysis = get_model(name='resnet34', num_classes=10, pretrained=True, alpha=0.0)
    model_for_analysis = model_for_analysis.to(device)
    
    analyzer = SensitivityAnalyzer()
    analyzer.register_hooks(model_for_analysis)
    mse_scores = analyzer.analyze_sensitivity(model_for_analysis, trainloader, device, alpha=0.5)
    high_sensitive, medium_sensitive, low_sensitive = analyzer.classify_layers(
        mse_scores, sensitivity_threshold_high, sensitivity_threshold_low
    )
    analyzer.remove_hooks()
    
    print(f"高敏感层: {len(high_sensitive)} 层")
    print(f"中敏感层: {len(medium_sensitive)} 层")
    print(f"低敏感层: {len(low_sensitive)} 层")
    
    # 保存敏感性分析结果
    sensitivity_result = {
        'high_sensitive': high_sensitive,
        'medium_sensitive': medium_sensitive,
        'low_sensitive': low_sensitive,
        'mse_scores': mse_scores
    }
    print(f"高敏感层列表: {high_sensitive[:5]}...")
    print(f"低敏感层列表: {low_sensitive[:5]}...")
    
    # 步骤2：应用分层Alpha训练（专利二+专利一组合）
    print("\n步骤2：专利二+专利一 - 分层Alpha训练...")
    model = get_model(name='resnet34', num_classes=10, pretrained=True, alpha=0.0)
    model = model.to(device)
    
    # 创建分层非线性钩子
    layerwise_hook = LayerwiseNonLinearityHook()
    for name in high_sensitive:
        layerwise_hook.set_alpha_range(name, 0.0, 0.15)  # 高敏感：U(0, 0.15)
    for name in medium_sensitive:
        layerwise_hook.set_alpha_range(name, 0.0, 0.3)   # 中敏感：U(0, 0.3)
    for name in low_sensitive:
        layerwise_hook.set_alpha_range(name, 0.0, 0.5)   # 低敏感：U(0, 0.5)
    
    layerwise_hook.register_hooks(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=0.9,
        weight_decay=config.get('weight_decay', 5e-4),
        nesterov=True
    )
    
    warmup_epochs = config.get('warmup_epochs', 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    ema = EMA(model, decay=ema_decay)
    ema_model = get_model(name='resnet34', num_classes=10, pretrained=False, alpha=0.0)
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
        train_loss, train_acc = train_one_epoch_layerwise(
            model, trainloader, criterion, optimizer, device, epoch, epochs,
            layerwise_hook, ema, use_mixup, mixup_alpha
        )
        scheduler.step()
        
        # 原始模型评估
        test_loss, test_acc = evaluate(model, testloader, criterion, device, layerwise_hook)
        
        # EMA模型评估
        test_loss_ema, test_acc_ema = evaluate_with_ema(
            model, ema, testloader, criterion, device, layerwise_hook
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
        'experiment_type': experiment_type,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'best_acc_ema': best_acc_ema,
        'best_epoch_ema': best_epoch_ema,
        'final_acc': history['test_acc'][-1],
        'final_acc_ema': history['test_acc_ema'][-1],
        'sensitivity_analysis': sensitivity_result,
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='专利二+专利一组合实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/patent1_2_combined')
    args = parser.parse_args()
    
    configs = [
        # 配置1：专利二+专利一组合（标准阈值）
        {
            'name': 'patent1_2_combined_standard',
            'device': args.device,
            'experiment_type': 'layerwise_alpha_combined',
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
            'label_smoothing': 0.1,
            'sensitivity_threshold_high': 0.5,
            'sensitivity_threshold_low': 0.2,
        },
        # 配置2：专利二+专利一组合（更严格阈值）
        {
            'name': 'patent1_2_combined_strict',
            'device': args.device,
            'experiment_type': 'layerwise_alpha_combined',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed + 1,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 10,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'ema_decay': 0.995,
            'label_smoothing': 0.1,
            'sensitivity_threshold_high': 0.6,
            'sensitivity_threshold_low': 0.1,
        },
        # 配置3：专利二+专利一组合（宽松阈值）
        {
            'name': 'patent1_2_combined_loose',
            'device': args.device,
            'experiment_type': 'layerwise_alpha_combined',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed + 2,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 10,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'ema_decay': 0.995,
            'label_smoothing': 0.1,
            'sensitivity_threshold_high': 0.4,
            'sensitivity_threshold_low': 0.3,
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
        
        save_path = os.path.join(save_dir, f"combined_{config['name']}.json")
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
    
    summary_path = os.path.join(save_dir, "combined_summary.json")
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
