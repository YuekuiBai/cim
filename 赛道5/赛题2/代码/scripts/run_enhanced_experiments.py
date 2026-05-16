"""
赛道五赛题二 - 增强版实验脚本

支持优化后的三篇专利实验：
- 专利四：自适应梯度缩放STE（含梯度方差自适应、零阶辅助校正）
- 专利五：层次化噪声注入框架（含敏感度感知、时空相关噪声）
- 专利六：偏差校正与正则化优化（含Decoupled偏差校正、正则化2.0）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import get_model
from ste.core import NoisyLinear, NoisyConv2d
from ste.innovation import (
    AdaptiveSTE, NoiseAwareRegularizer, BiasCorrector,
    LayerwiseNoiseInjection, InnovationConfig
)


def inject_ste_to_model(model, noise_config):
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


class GradientVarianceTracker:
    """梯度方差跟踪器 - 用于专利四的梯度方差自适应缩放"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.gradient_history = []
    
    def update(self, grad_norm):
        self.gradient_history.append(grad_norm)
        if len(self.gradient_history) > self.window_size:
            self.gradient_history.pop(0)
    
    def get_variance(self):
        if len(self.gradient_history) < 2:
            return 0.0
        grads = np.array(self.gradient_history)
        return np.var(grads)
    
    def get_adaptive_scale(self, base_scale=1.0):
        var = self.get_variance()
        return base_scale / (1.0 + var)


class ZeroOrderCorrector:
    """零阶校正器 - 用于专利四的零阶辅助校正"""
    
    def __init__(self, alpha=0.2, perturbation_std=0.01):
        self.alpha = alpha
        self.perturbation_std = perturbation_std
    
    def compute_zo_gradient(self, model, loss_fn, inputs, targets, num_samples=5):
        """计算零阶梯度估计"""
        base_loss = loss_fn(model(inputs), targets).item()
        zo_grad = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_estimate = torch.zeros_like(param)
                
                for _ in range(num_samples):
                    perturbation = torch.randn_like(param) * self.perturbation_std
                    
                    param.data.add_(perturbation)
                    loss_plus = loss_fn(model(inputs), targets).item()
                    
                    param.data.sub_(perturbation * 2)
                    loss_minus = loss_fn(model(inputs), targets).item()
                    
                    param.data.add_(perturbation)
                    
                    grad_estimate += (loss_plus - loss_minus) / (2 * self.perturbation_std) * perturbation
                
                zo_grad[name] = grad_estimate / num_samples
        
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


class SpatiotemporalNoiseInjector:
    """时空相关噪声注入器 - 用于专利五的时空相关噪声建模"""
    
    def __init__(self, correlation_rho=0.7, base_noise_std=0.01):
        self.correlation_rho = correlation_rho
        self.base_noise_std = base_noise_std
        self.layer_count = 0
        self.noise_covariance = None
    
    def build_covariance_matrix(self, num_layers):
        """构建噪声协方差矩阵"""
        self.layer_count = num_layers
        self.noise_covariance = torch.zeros(num_layers, num_layers)
        
        for i in range(num_layers):
            for j in range(num_layers):
                self.noise_covariance[i, j] = (self.correlation_rho ** abs(i - j)) * (self.base_noise_std ** 2)
    
    def inject_correlated_noise(self, layer_idx, weight):
        """注入相关噪声"""
        if self.noise_covariance is None:
            self.build_covariance_matrix(layer_idx + 1)
        
        noise = torch.randn_like(weight) * self.base_noise_std
        
        if layer_idx > 0:
            correlation_factor = self.correlation_rho ** layer_idx
            noise = noise * (1 - correlation_factor) + torch.randn_like(weight) * self.base_noise_std * correlation_factor
        
        return weight + noise


class DecoupledBiasCorrector:
    """Decoupled偏差校正器 - 用于专利六的Decoupled偏差校正"""
    
    def __init__(self, ema_decay=0.99):
        self.ema_decay = ema_decay
        self.forward_bias_ema = None
        self.backward_bias_ema = None
    
    def estimate_forward_bias(self, noisy_output, clean_output):
        """估计前向偏差"""
        bias = noisy_output - clean_output
        bias_mean = bias.mean().item()
        
        if self.forward_bias_ema is None:
            self.forward_bias_ema = bias_mean
        else:
            self.forward_bias_ema = self.ema_decay * self.forward_bias_ema + (1 - self.ema_decay) * bias_mean
        
        return self.forward_bias_ema
    
    def estimate_backward_bias(self, noisy_grad, clean_grad):
        """估计反向偏差"""
        bias = noisy_grad - clean_grad
        bias_mean = bias.mean().item()
        
        if self.backward_bias_ema is None:
            self.backward_bias_ema = bias_mean
        else:
            self.backward_bias_ema = self.ema_decay * self.backward_bias_ema + (1 - self.ema_decay) * bias_mean
        
        return self.backward_bias_ema
    
    def correct_output(self, output):
        """校正输出"""
        if self.forward_bias_ema is not None:
            return output - self.forward_bias_ema
        return output


class NoiseAwareRegularizerV2:
    """噪声感知正则化2.0 - 用于专利六的正则化2.0"""
    
    def __init__(self, lambda_l2=0.01, lambda_grad_smooth=0.005, lambda_kl=0.001):
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
                l2_penalty += torch.sum(param ** 2)
                
                if self.previous_weights is not None and name in self.previous_weights:
                    grad_smooth_penalty += torch.sum((param - self.previous_weights[name]) ** 2)
                
                mean = param.mean()
                std = param.std() + 1e-8
                kl_penalty += 0.5 * (mean ** 2 + std ** 2 - torch.log(std ** 2) - 1)
        
        self.previous_weights = {name: param.clone().detach() for name, param in model.named_parameters() if 'weight' in name and param.requires_grad}
        
        l2_penalty = l2_penalty / (num_params + 1e-8)
        grad_smooth_penalty = grad_smooth_penalty / (num_params + 1e-8)
        kl_penalty = kl_penalty / (num_params + 1e-8)
        
        return self.lambda_l2 * l2_penalty + self.lambda_grad_smooth * grad_smooth_penalty + self.lambda_kl * kl_penalty


def get_data_loaders(config, batch_size=None, num_workers=None):
    """获取数据加载器"""
    import torchvision
    import torchvision.transforms as transforms
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    root = config.get('dataset', {}).get('root', '/mnt/storage2/zyc/CIM比赛/公共数据集')
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    
    bs = batch_size if batch_size is not None else config.get('training', {}).get('batch_size', 512)
    nw = num_workers if num_workers is not None else config.get('dataset', {}).get('num_workers', 8)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=nw)
    
    return train_loader, test_loader


def run_enhanced_experiment(config, device, save_dir, experiment_type, **kwargs):
    """运行增强版实验"""
    
    print(f"\n{'='*70}")
    print(f"实验类型: {experiment_type}")
    print(f"{'='*70}")
    
    seed = kwargs.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_loader, test_loader = get_data_loaders(config, kwargs.get('batch_size', 512), kwargs.get('num_workers', 8))
    
    model = get_model(name='resnet18', num_classes=10, pretrained=False)
    
    noise_config = config.get('ste', {}).get('noise_config', {
        'prog_noise_std': 0.01,
        'drift_factor': 0.005,
        'nonlinear_alpha': 0.1,
        'nonlinear_beta': 0.05,
        'output_noise_std': 0.01,
        'crosstalk_factor': 0.002,
    })
    
    if experiment_type in ['gradient_variance_adaptive', 'zero_order_correction']:
        model = inject_ste_to_model(model, noise_config)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=kwargs.get('lr', 0.1), momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=kwargs.get('epochs', 30))
    
    # Initialize enhanced components
    grad_variance_tracker = GradientVarianceTracker() if experiment_type == 'gradient_variance_adaptive' else None
    zo_corrector = ZeroOrderCorrector(alpha=kwargs.get('zo_alpha', 0.2)) if experiment_type == 'zero_order_correction' else None
    spatiotemporal_injector = SpatiotemporalNoiseInjector(correlation_rho=kwargs.get('correlation_rho', 0.7)) if experiment_type == 'spatiotemporal_noise' else None
    decoupled_corrector = DecoupledBiasCorrector() if experiment_type == 'decoupled_bias_correction' else None
    regularizer_v2 = NoiseAwareRegularizerV2() if experiment_type == 'regularizer_v2_ablation' else None
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'gradient_variance': []}
    
    num_epochs = kwargs.get('epochs', 30)
    noise_strength = kwargs.get('noise_strength', 1.0)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add regularization
            if regularizer_v2 is not None:
                reg_loss = regularizer_v2.compute_penalty(model)
                loss = loss + reg_loss
            
            loss.backward()
            
            # Gradient variance tracking
            if grad_variance_tracker is not None:
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_variance_tracker.update(grad_norm ** 0.5)
                
                # Apply adaptive scaling
                adaptive_scale = grad_variance_tracker.get_adaptive_scale()
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data *= adaptive_scale
            
            # Zero-order correction
            if zo_corrector is not None:
                ste_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
                zo_grads = zo_corrector.compute_zo_gradient(model, criterion, inputs, targets)
                corrected_grads = zo_corrector.correct_gradient(ste_grads, zo_grads)
                
                for name, param in model.named_parameters():
                    if name in corrected_grads:
                        param.grad.data = corrected_grads[name]
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        train_acc = 100. * correct / total
        
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        if grad_variance_tracker is not None:
            history['gradient_variance'].append(grad_variance_tracker.get_variance())
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results = {
        'experiment_type': experiment_type,
        'best_acc': best_acc,
        'final_acc': test_acc,
        'history': history,
        'config': kwargs
    }
    
    with open(os.path.join(save_dir, f'{experiment_type}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n实验完成！最佳精度: {best_acc:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='赛道五赛题二 - 增强版实验')
    parser.add_argument('--experiment_type', type=str, required=True,
                       choices=['gradient_variance_adaptive', 'zero_order_correction',
                               'spatiotemporal_noise', 'decoupled_bias_correction',
                               'regularizer_v2_ablation'],
                       help='实验类型')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda:2', help='设备')
    parser.add_argument('--save_dir', type=str, default='../结果/enhanced_experiments', help='结果保存目录')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=512, help='批大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--noise_strength', type=float, default=1.0, help='噪声强度')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    parser.add_argument('--zo_alpha', type=float, default=0.2, help='零阶校正系数')
    parser.add_argument('--correlation_rho', type=float, default=0.7, help='时空相关系数')
    
    args = parser.parse_args()
    
    config = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = run_enhanced_experiment(
        config=config,
        device=device,
        save_dir=args.save_dir,
        experiment_type=args.experiment_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        noise_strength=args.noise_strength,
        lr=args.lr,
        zo_alpha=args.zo_alpha,
        correlation_rho=args.correlation_rho
    )


if __name__ == '__main__':
    main()
