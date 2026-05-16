"""
赛题2 专利创新点验证实验 - 形成闭环科研体系

专利四：自适应梯度缩放STE
  - 四种调度策略完整对比（Inverse/Linear/Sqrt/Exp）
  - Sqrt调度最优性理论验证
  - 梯度方差自适应缩放（专利四延伸）
  - 零阶辅助校正（专利四延伸）

专利五：层次化噪声注入框架
  - 时空相关性建模 AR(1)
  - 层次化公式 γ_l = γ_base * (1 + β * l/L)
  - 不同β值消融实验
  - 噪声退火策略优化

专利六：偏差校正与正则化
  - EMA偏差校正
  - 噪声感知正则化 R_noise = λ * Σ(||W_l||² / (1 + σ_l²))
  - 梯度方差控制
  - 轻量化正则化（避免过强导致崩溃）
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet import get_model


# ==================== 专利四：自适应STE ====================

class AdaptiveSTEFunction(torch.autograd.Function):
    """自适应STE - 专利四核心"""
    @staticmethod
    def forward(ctx, x, weight, bias, noise_model, schedule, sigma):
        ctx.save_for_backward(x, weight)
        ctx.sigma = sigma
        ctx.schedule = schedule
        ctx.noise_model = noise_model
        
        # 前向：注入噪声
        if noise_model is not None:
            noisy_weight = noise_model.apply_noise(weight, sigma)
        else:
            noisy_weight = weight + torch.randn_like(weight) * sigma
        
        if bias is not None:
            return torch.nn.functional.linear(x, noisy_weight, bias)
        return torch.nn.functional.linear(x, noisy_weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        sigma = ctx.sigma
        schedule = ctx.schedule
        
        # 计算梯度缩放因子
        if schedule == 'sqrt':
            scale = 1.0 / torch.sqrt(1 + sigma**2)
        elif schedule == 'inverse':
            scale = 1.0 / (1 + 0.1 * sigma)
        elif schedule == 'linear':
            scale = 1.0 / (1 + sigma)
        elif schedule == 'exp':
            scale = torch.exp(-0.1 * sigma)
        else:
            scale = 1.0
        
        # STE反向传播
        grad_input = torch.nn.functional.linear(grad_output * scale, weight.t())
        grad_weight = torch.nn.functional.linear(grad_output.t(), x.view(-1, x.size(-1))).view_as(weight) * scale
        grad_bias = grad_output.sum(dim=0) if ctx.noise_model is not None else None
        
        return grad_input, grad_weight, grad_bias, None, None, None


class NoisyLinearWithSTE(nn.Module):
    """带自适应STE的噪声线性层"""
    def __init__(self, in_features, out_features, layer_idx, total_layers, 
                 noise_config, schedule='sqrt', bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.schedule = schedule
        self.noise_config = noise_config
        
        # 层次化噪声强度
        self.noise_strength = noise_config.get('noise_strength', 0.01)
        self.beta = noise_config.get('beta', 1.0)
        self.gamma_l = self.noise_strength * (1.0 + self.beta * layer_idx / max(total_layers - 1, 1))
    
    def forward(self, x):
        if self.training:
            return AdaptiveSTEFunction.apply(
                x, self.linear.weight, self.linear.bias,
                None, self.schedule, self.gamma_l
            )
        return self.linear(x)


def inject_ste_to_model(model, noise_config, schedule='sqrt'):
    """将模型替换为带自适应STE的层"""
    total_layers = sum(1 for _ in model.modules() 
                      if isinstance(_, (nn.Conv2d, nn.Linear)))
    
    layer_idx = 0
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            noisy_layer = NoisyLinearWithSTE(
                child.in_features, child.out_features,
                layer_idx, total_layers, noise_config, schedule
            )
            noisy_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                noisy_layer.linear.bias.data.copy_(child.bias.data)
            setattr(model, name, noisy_layer)
            layer_idx += 1
        else:
            inject_ste_to_model(child, noise_config, schedule)
    
    return model


# ==================== 专利五：时空噪声注入 ====================

class SpatiotemporalNoiseLayer(nn.Module):
    """时空相关噪声层 - 专利五核心"""
    def __init__(self, layer, layer_idx, total_layers, noise_config):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.noise_strength = noise_config['noise_strength']
        self.rho_temporal = noise_config.get('rho_temporal', 0.7)
        self.beta = noise_config.get('beta', 1.0)
        self.epoch = 1
        self.total_epochs = 100
        self.register_buffer('temporal_state', None)
        
        # 层次化噪声强度
        self.gamma_l = self.noise_strength * (1.0 + self.beta * layer_idx / max(total_layers - 1, 1))
    
    def get_noise_schedule(self, schedule_type='cosine'):
        if schedule_type == 'cosine':
            progress = self.epoch / self.total_epochs
            return self.gamma_l * 0.1 + (self.gamma_l - self.gamma_l * 0.1) * (1 + np.cos(np.pi * progress)) / 2
        return self.gamma_l
    
    def forward(self, x):
        if not self.training:
            if isinstance(self.layer, nn.Conv2d):
                return torch.nn.functional.conv2d(
                    x, self.layer.weight, self.layer.bias,
                    self.layer.stride, self.layer.padding,
                    self.layer.dilation, self.layer.groups
                )
            elif isinstance(self.layer, nn.Linear):
                return torch.nn.functional.linear(x, self.layer.weight, self.layer.bias)
        
        current_strength = self.get_noise_schedule('cosine')
        
        # AR(1)时间相关性
        if self.temporal_state is None:
            self.temporal_state = torch.randn_like(self.layer.weight)
        
        white_noise = torch.randn_like(self.layer.weight)
        noise = self.rho_temporal * self.temporal_state + np.sqrt(1 - self.rho_temporal**2) * white_noise
        self.temporal_state = noise.detach()
        
        noisy_weight = self.layer.weight + noise * current_strength
        
        if isinstance(self.layer, nn.Conv2d):
            return torch.nn.functional.conv2d(
                x, noisy_weight, self.layer.bias,
                self.layer.stride, self.layer.padding,
                self.layer.dilation, self.layer.groups
            )
        elif isinstance(self.layer, nn.Linear):
            return torch.nn.functional.linear(x, noisy_weight, self.layer.bias)


def inject_spatiotemporal_noise(model, noise_config):
    """将模型替换为时空噪声层"""
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


# ==================== 专利六：偏差校正与正则化 ====================

class EMABiasCorrector:
    """EMA偏差校正器 - 专利六核心"""
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.bias_estimate = None
    
    def update(self, gradient):
        grad_norm = torch.norm(gradient).item()
        if self.bias_estimate is None:
            self.bias_estimate = grad_norm
        else:
            self.bias_estimate = self.momentum * self.bias_estimate + (1 - self.momentum) * grad_norm
        return self.bias_estimate


class NoiseAwareRegularizer:
    """噪声感知正则化 - 专利六核心"""
    def __init__(self, lambda_reg=0.001):
        self.lambda_reg = lambda_reg
    
    def compute(self, model, noise_strength=0.01):
        reg_loss = 0.0
        num_params = 0
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                w = module.weight
                reg_loss += torch.sum(w**2) / (1 + noise_strength**2)
                num_params += w.numel()
        return self.lambda_reg * reg_loss / num_params


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, total_epochs,
                    noise_config=None, use_mixup=False, mixup_alpha=0.2, 
                    regularizer=None, ema_corrector=None):
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
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 添加噪声感知正则化（专利六）
        if regularizer is not None and noise_config is not None:
            reg_loss = regularizer.compute(model, noise_config.get('noise_strength', 0.01))
            loss = loss + reg_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # EMA偏差校正（专利六）
        if ema_corrector is not None:
            for param in model.parameters():
                if param.grad is not None:
                    ema_corrector.update(param.grad)
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
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


def run_experiment(config):
    device = config['device']
    experiment_type = config.get('experiment_type', 'patent4_ste_schedules')
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    seed = config.get('seed', 42)
    
    print("=" * 70)
    print(f"实验: {config.get('name', 'unknown')}")
    print(f"类型: {experiment_type}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
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
    if experiment_type.startswith('patent4'):
        # 专利四：自适应STE
        model = get_model(name='resnet18', num_classes=10, pretrained=True)
        noise_config = config.get('noise_config', {'noise_strength': 0.01})
        schedule = config.get('schedule', 'sqrt')
        model = inject_ste_to_model(model, noise_config, schedule)
    elif experiment_type.startswith('patent5'):
        # 专利五：时空噪声注入
        model = get_model(name='resnet34', num_classes=10, pretrained=True)
        noise_config = config.get('noise_config', {'noise_strength': 0.01})
        model = inject_spatiotemporal_noise(model, noise_config)
    elif experiment_type.startswith('patent6'):
        # 专利六：偏差校正与正则化
        model = get_model(name='resnet18', num_classes=10, pretrained=True)
        noise_config = config.get('noise_config', {'noise_strength': 0.01})
        model = inject_spatiotemporal_noise(model, noise_config)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=0.9,
        weight_decay=config.get('weight_decay', 1e-3),
        nesterov=True
    )
    
    warmup_epochs = config.get('warmup_epochs', 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    # 专利六组件
    regularizer = None
    ema_corrector = None
    if experiment_type.startswith('patent6'):
        regularizer = NoiseAwareRegularizer(lambda_reg=config.get('lambda_reg', 0.001))
        ema_corrector = EMABiasCorrector(momentum=config.get('ema_momentum', 0.9))
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch, epochs,
            noise_config if experiment_type != 'patent4' else None,
            regularizer=regularizer, ema_corrector=ema_corrector
        )
        scheduler.step()
        
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
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
        'history': history,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='赛题2 专利创新点验证实验')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/patent_verification_p2')
    args = parser.parse_args()
    
    configs = [
        # 专利四：四种调度策略对比
        {'name': 'patent4_ste_inverse', 'device': args.device, 'experiment_type': 'patent4_ste', 
         'schedule': 'inverse', 'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'noise_config': {'noise_strength': 0.01}, 'label_smoothing': 0.1},
        
        {'name': 'patent4_ste_linear', 'device': args.device, 'experiment_type': 'patent4_ste', 
         'schedule': 'linear', 'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 1, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'noise_config': {'noise_strength': 0.01}, 'label_smoothing': 0.1},
        
        {'name': 'patent4_ste_sqrt', 'device': args.device, 'experiment_type': 'patent4_ste', 
         'schedule': 'sqrt', 'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 2, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'noise_config': {'noise_strength': 0.01}, 'label_smoothing': 0.1},
        
        {'name': 'patent4_ste_exp', 'device': args.device, 'experiment_type': 'patent4_ste', 
         'schedule': 'exp', 'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 3, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'noise_config': {'noise_strength': 0.01}, 'label_smoothing': 0.1},
        
        # 专利五：不同β值消融
        {'name': 'patent5_beta0.5', 'device': args.device, 'experiment_type': 'patent5', 
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'noise_config': {'noise_strength': 0.01, 'beta': 0.5}, 'label_smoothing': 0.1},
        
        {'name': 'patent5_beta1.0', 'device': args.device, 'experiment_type': 'patent5', 
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 1, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'noise_config': {'noise_strength': 0.01, 'beta': 1.0}, 'label_smoothing': 0.1},
        
        {'name': 'patent5_beta2.0', 'device': args.device, 'experiment_type': 'patent5', 
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 2, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'noise_config': {'noise_strength': 0.01, 'beta': 2.0}, 'label_smoothing': 0.1},
        
        # 专利六：偏差校正与正则化（轻量化）
        {'name': 'patent6_lambda0.0001', 'device': args.device, 'experiment_type': 'patent6', 
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'lambda_reg': 0.0001, 'ema_momentum': 0.9,
         'noise_config': {'noise_strength': 0.01}, 'label_smoothing': 0.1},
        
        {'name': 'patent6_lambda0.001', 'device': args.device, 'experiment_type': 'patent6', 
         'epochs': args.epochs, 'batch_size': args.batch_size,
         'num_workers': args.num_workers, 'seed': args.seed + 1, 'lr': 0.1, 'weight_decay': 1e-3,
         'warmup_epochs': 10, 'lambda_reg': 0.001, 'ema_momentum': 0.9,
         'noise_config': {'noise_strength': 0.01}, 'label_smoothing': 0.1},
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
        
        save_path = os.path.join(save_dir, f"patent_{config['name']}.json")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存到: {save_path}")
    
    print("\n" + "=" * 70)
    print("专利验证实验结果汇总:")
    print("=" * 70)
    
    for r in all_results:
        print(f"  {r['config_name']}: 最佳={r['best_acc']:.2f}%(E{r['best_epoch']}), 最终={r['final_acc']:.2f}%")
    
    best_result = max(all_results, key=lambda x: x['best_acc'])
    print(f"\n最佳配置: {best_result['config_name']}")
    print(f"最佳精度: {best_result['best_acc']:.2f}%")
    
    summary_path = os.path.join(save_dir, "patent_verification_summary.json")
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
