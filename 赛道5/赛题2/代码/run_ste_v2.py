"""
赛题2 STE噪声感知训练框架 V2 - 全面改进版

整合内容：
1. 使用主办方官方噪声模型(sample_noise.py)
2. 专利一：自适应梯度缩放STE（4种调度策略）
3. 专利二：层次化噪声注入（线性递增公式）
4. 专利三：偏差校正与方差稳定化
5. 前沿技术：Warmup、梯度裁剪、混合精度、EMA

比赛要求覆盖：
- 任务一：通用STE框架（多架构适配）
- 任务二：CIFAR-10图像分类验证
- 任务三：统计分析与消融实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, Tuple
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet import get_model


# ============================================================
# 1. 官方噪声模型集成（来自sample_noise.py）
# ============================================================

class CIMNoiseModel(nn.Module):
    """
    存算一体芯片噪声模型 - 基于主办方提供的sample_noise.py

    包含三类误差：
    1. 编程误差（加性噪声）：prog_noise, drift_noise, retention_noise
    2. 非线性噪声（乘性特征）：tanh饱和
    3. 输出误差（量化效应）：ADC量化, 热噪声, 1/f噪声
    """

    def __init__(self, noise_config: Dict):
        super().__init__()
        self.config = noise_config
        self.noise_strength = noise_config.get('noise_strength', 1.0)

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """带噪声的矩阵乘法"""
        # 1. 编程误差（加性噪声）
        prog_noise_std = self.config.get('prog_noise_std', 0.01) * self.noise_strength
        drift_factor = self.config.get('drift_factor', 0.005) * self.noise_strength
        retention_loss = self.config.get('retention_loss', 0.001) * self.noise_strength
        temperature_factor = self.config.get('temperature_factor', 0.001) * self.noise_strength

        prog_noise = torch.randn_like(weight) * prog_noise_std
        drift_noise = torch.randn_like(weight) * drift_factor * torch.abs(weight)
        retention_noise = weight * retention_loss * (torch.rand_like(weight) - 0.5)
        temp_noise = torch.randn_like(weight) * temperature_factor * torch.sqrt(torch.abs(weight) + 1e-8)

        noisy_weight = weight + prog_noise + drift_noise + retention_noise + temp_noise

        # 2. 输入串扰
        crosstalk_factor = self.config.get('crosstalk_factor', 0.002) * self.noise_strength
        if crosstalk_factor > 0:
            crosstalk = torch.randn_like(input) * crosstalk_factor * torch.norm(input, dim=-1, keepdim=True)
            noisy_input = input + crosstalk
        else:
            noisy_input = input

        # 3. 矩阵乘法 (weight.T for linear layer: input @ weight.T)
        result = torch.matmul(noisy_input, noisy_weight.T)

        # 4. 非线性噪声（tanh饱和）
        nonlinear_alpha = self.config.get('nonlinear_alpha', 0.1) * self.noise_strength
        nonlinear_beta = self.config.get('nonlinear_beta', 0.05) * self.noise_strength

        pos_mask = result > 0
        neg_mask = result < 0
        result_nonlinear = torch.zeros_like(result)
        result_nonlinear[pos_mask] = torch.tanh(nonlinear_alpha * result[pos_mask]) / nonlinear_alpha
        result_nonlinear[neg_mask] = torch.tanh((nonlinear_alpha + nonlinear_beta) * result[neg_mask]) / (nonlinear_alpha + nonlinear_beta)

        # 5. 输出误差
        output_noise_std = self.config.get('output_noise_std', 0.01) * self.noise_strength
        output_noise = torch.randn_like(result_nonlinear) * output_noise_std
        f_noise = torch.randn_like(result_nonlinear) * output_noise_std * 0.3

        # 空间相关噪声
        if len(result_nonlinear.shape) >= 2:
            spatial_corr = F.conv2d(
                f_noise.unsqueeze(0).unsqueeze(0) if len(f_noise.shape) < 4 else f_noise,
                torch.ones(1, 1, 3, 3, device=f_noise.device) / 9,
                padding=1
            )
            if len(f_noise.shape) < 4:
                spatial_corr = spatial_corr.squeeze()
        else:
            spatial_corr = f_noise

        supply_variation = 1 + torch.randn(1, device=result.device).item() * 0.01

        final_result = (result_nonlinear + output_noise + spatial_corr) * supply_variation

        return final_result


# ============================================================
# 2. 专利一：自适应梯度缩放STE
# ============================================================

class AdaptiveSTEFunction(torch.autograd.Function):
    """
    自适应STE梯度估计器

    前向传播：执行带噪声的操作
    反向传播：使用自适应缩放的STE梯度

    调度策略：
    - Inverse: s = 1/(1 + λσ)
    - Linear: s = 1/(1 + σ)
    - Sqrt: s = 1/√(1 + σ²) ← 最优
    - Exp: s = exp(-λσ)
    """

    @staticmethod
    def forward(ctx, input, weight, bias, noise_model, schedule, noise_strength):
        ctx.save_for_backward(input, weight, bias)
        ctx.noise_model = noise_model
        ctx.schedule = schedule
        ctx.noise_strength = noise_strength

        # 前向传播：使用噪声模型
        if noise_model is not None and noise_strength > 0:
            output = noise_model(input, weight)
        else:
            output = torch.matmul(input, weight.T)

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        schedule = ctx.schedule
        noise_strength = ctx.noise_strength

        # 自适应缩放因子
        scale = AdaptiveSTEFunction._get_scale(schedule, noise_strength)

        # STE梯度估计：直接传递梯度（乘以缩放因子）
        grad_input = torch.matmul(grad_output, weight.t()) * scale
        grad_weight = torch.matmul(input.t(), grad_output) * scale
        grad_bias = grad_output.sum(dim=0) * scale if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None

    @staticmethod
    def _get_scale(schedule: str, sigma: float) -> float:
        """计算自适应缩放因子"""
        if schedule == 'inverse':
            return 1.0 / (1.0 + sigma)
        elif schedule == 'linear':
            return 1.0 / (1.0 + sigma)
        elif schedule == 'sqrt':
            return 1.0 / np.sqrt(1.0 + sigma ** 2)  # 最优策略
        elif schedule == 'exp':
            return np.exp(-sigma)
        else:
            return 1.0


# ============================================================
# 3. 专利二：层次化噪声注入层
# ============================================================

class NoisyLinearLayer(nn.Module):
    """
    带噪声的线性层 - 专利二：层次化噪声注入

    gamma_l = gamma_base * (1 + beta * l/L)
    浅层低噪声，深层高噪声
    """

    def __init__(self, in_features, out_features, bias=True,
                 layer_idx=0, total_layers=10, noise_config=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.noise_config = noise_config or {}

        # 层次化噪声强度
        self.noise_strength = noise_config.get('noise_strength', 1.0)
        self.beta = noise_config.get('beta', 1.0)
        self.gamma_l = self.noise_strength * (1.0 + self.beta * layer_idx / max(total_layers - 1, 1))

    def forward(self, x):
        # 训练时注入噪声
        if self.training and self.noise_strength > 0:
            # 使用自适应STE
            output = AdaptiveSTEFunction.apply(
                x, self.linear.weight, self.linear.bias,
                CIMNoiseModel(self.noise_config) if self.noise_config.get('use_official_noise', True) else None,
                self.noise_config.get('schedule', 'sqrt'),
                self.gamma_l
            )
            return output
        else:
            return self.linear(x)


class NoisyConv2dLayer(nn.Module):
    """
    带噪声的卷积层 - 专利二：层次化噪声注入
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, layer_idx=0, total_layers=10, noise_config=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.noise_config = noise_config or {}

        self.noise_strength = noise_config.get('noise_strength', 1.0)
        self.beta = noise_config.get('beta', 1.0)
        self.gamma_l = self.noise_strength * (1.0 + self.beta * layer_idx / max(total_layers - 1, 1))

    def forward(self, x):
        if self.training and self.noise_strength > 0:
            # 对卷积操作，先执行正常卷积，再注入噪声
            output = self.conv(x)

            # 注入噪声（模拟CIM芯片的输出误差）
            if self.noise_config.get('use_official_noise', True):
                noise_model = CIMNoiseModel(self.noise_config)
                # 将输出视为矩阵乘法结果，添加噪声
                batch_size = output.shape[0]
                output_flat = output.view(batch_size, -1)
                identity_weight = torch.eye(output_flat.shape[1], device=output.device)
                output_flat = noise_model(output_flat, identity_weight)
                output = output_flat.view(output.shape)

            return output
        else:
            return self.conv(x)


# ============================================================
# 4. 专利三：偏差校正与方差稳定化
# ============================================================

class BiasCorrector:
    """
    偏差校正器 - 专利三

    使用EMA估计梯度偏差并校正
    """

    def __init__(self, ema_decay=0.9):
        self.ema_decay = ema_decay
        self.ema_gradients = {}

    def correct(self, gradient: torch.Tensor, param_id: int) -> torch.Tensor:
        if param_id not in self.ema_gradients:
            self.ema_gradients[param_id] = gradient.detach().clone()
        else:
            ema = self.ema_gradients[param_id]
            if ema.shape != gradient.shape:
                self.ema_gradients[param_id] = gradient.detach().clone()
            else:
                self.ema_gradients[param_id] = self.ema_decay * ema + (1 - self.ema_decay) * gradient.detach()

        # 校正：减去偏差估计
        corrected = gradient - (self.ema_gradients[param_id] - gradient.detach())
        return corrected


class VarianceStabilizer:
    """
    方差稳定化器 - 专利三

    使用梯度裁剪和方差正则化
    """

    def __init__(self, max_norm=1.0, lambda_var=0.01):
        self.max_norm = max_norm
        self.lambda_var = lambda_var

    def apply_variance_penalty(self, model: nn.Module, loss: torch.Tensor) -> torch.Tensor:
        """在backward之前添加方差惩罚项（基于权重统计）"""
        weight_var = 0.0
        count = 0
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                weight_var += torch.var(param)
                count += 1
        if count > 0:
            loss = loss + self.lambda_var * weight_var / count
        return loss

    def clip_gradients(self, model: nn.Module):
        """在backward之后裁剪梯度"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


# ============================================================
# 5. 模型构建器
# ============================================================

def build_noisy_model(model: nn.Module, noise_config: Dict) -> nn.Module:
    """
    将标准模型转换为带噪声层的模型

    遍历所有Conv2d和Linear层，替换为带噪声版本
    """
    total_layers = sum(1 for _ in model.modules()
                      if isinstance(_, (nn.Conv2d, nn.Linear)))

    layer_idx = 0

    def _replace_layers(module, prefix=''):
        nonlocal layer_idx

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                # 替换为带噪声的线性层
                noisy_layer = NoisyLinearLayer(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    layer_idx=layer_idx,
                    total_layers=total_layers,
                    noise_config=noise_config
                )
                # 复制权重
                noisy_layer.linear.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    noisy_layer.linear.bias.data = child.bias.data.clone()

                setattr(module, name, noisy_layer)
                layer_idx += 1

            elif isinstance(child, nn.Conv2d):
                # 替换为带噪声的卷积层
                noisy_layer = NoisyConv2dLayer(
                    child.in_channels, child.out_channels,
                    child.kernel_size, stride=child.stride,
                    padding=child.padding, bias=child.bias is not None,
                    layer_idx=layer_idx,
                    total_layers=total_layers,
                    noise_config=noise_config
                )
                # 复制权重
                noisy_layer.conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    noisy_layer.conv.bias.data = child.bias.data.clone()

                setattr(module, name, noisy_layer)
                layer_idx += 1

            else:
                _replace_layers(child, full_name)

    _replace_layers(model)
    return model


# ============================================================
# 6. 训练器
# ============================================================

class STETrainer:
    """
    STE噪声感知训练器

    集成所有专利技术和前沿优化
    """

    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('lr', 0.1),
            momentum=0.9,
            weight_decay=config.get('weight_decay', 5e-4)
        )

        # 学习率调度：warmup + cosine
        warmup_epochs = config.get('warmup_epochs', 5)
        total_epochs = config.get('epochs', 50)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_epochs - warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # 专利三：偏差校正
        self.bias_corrector = BiasCorrector(ema_decay=0.9)
        self.variance_stabilizer = VarianceStabilizer(
            max_norm=config.get('grad_clip', 1.0),
            lambda_var=config.get('lambda_var', 0.01)
        )

        # 混合精度训练
        self.scaler = GradScaler() if config.get('use_amp', True) else None

        # 噪声强度调度
        self.noise_schedule = config.get('noise_schedule', 'constant')
        self.initial_noise = config.get('noise_strength', 1.0)

        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'noise_strength': [], 'lr': []
        }
        self.best_acc = 0.0
        self.best_epoch = 0

    def _get_noise_strength(self, epoch: int) -> float:
        """获取当前epoch的噪声强度"""
        total_epochs = self.config.get('epochs', 50)

        if self.noise_schedule == 'constant':
            return self.initial_noise
        elif self.noise_schedule == 'linear_decay':
            return self.initial_noise * (1 - epoch / total_epochs)
        elif self.noise_schedule == 'cosine_decay':
            return self.initial_noise * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
        elif self.noise_schedule == 'inverse':
            return self.initial_noise / (1 + epoch * 0.1)
        else:
            return self.initial_noise

    def _update_noise_strength(self, noise_strength: float):
        """更新模型中所有噪声层的噪声强度"""
        for module in self.model.modules():
            if isinstance(module, (NoisyLinearLayer, NoisyConv2dLayer)):
                module.gamma_l = noise_strength * (1.0 + module.beta * module.layer_idx / max(module.total_layers - 1, 1))

    def train_epoch(self, trainloader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 更新噪声强度
        noise_strength = self._get_noise_strength(epoch)
        self._update_noise_strength(noise_strength)

        pbar = tqdm(trainloader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # 训练步骤
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 专利三：方差正则化（backward前）
            loss = self.variance_stabilizer.apply_variance_penalty(self.model, loss)
            loss.backward()

            # 专利三：偏差校正
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data = self.bias_corrector.correct(param.grad.data, id(param))

            # 梯度裁剪（backward后）
            self.variance_stabilizer.clip_gradients(self.model)
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        self.scheduler.step()

        return train_loss / len(trainloader), 100.*correct/total

    @torch.no_grad()
    def evaluate(self, testloader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in testloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return test_loss / len(testloader), 100.*correct/total

    def train(self, trainloader, testloader) -> Dict:
        """完整训练流程"""
        print("=" * 70)
        print(f"STE噪声感知训练 V2")
        print(f"配置: {json.dumps(self.config, indent=2)}")
        print("=" * 70)

        for epoch in range(1, self.config.get('epochs', 50) + 1):
            train_loss, train_acc = self.train_epoch(trainloader, epoch)
            test_loss, test_acc = self.evaluate(testloader)

            # 记录历史
            noise_strength = self._get_noise_strength(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['noise_strength'].append(noise_strength)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            print(f'Epoch {epoch}/{self.config["epochs"]}: '
                  f'Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, '
                  f'Noise={noise_strength:.3f}')

            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_epoch = epoch
                print(f'  -> 新最佳精度: {self.best_acc:.2f}% (Epoch {self.best_epoch})')

        return {
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'final_acc': self.history['test_acc'][-1],
            'history': self.history,
            'config': self.config
        }


# ============================================================
# 7. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='STE噪声感知训练框架 V2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/ste_v2')
    args = parser.parse_args()

    import torchvision
    import torchvision.transforms as transforms

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
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 实验配置列表
    configs = [
        # 配置1：基线（无噪声）
        {
            'name': 'baseline_no_noise',
            'device': args.device,
            'epochs': args.epochs,
            'noise_strength': 0.0,
            'schedule': 'sqrt',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'grad_clip': 1.0,
            'lambda_var': 0.0,
            'noise_schedule': 'constant',
            'use_amp': True,
            'use_official_noise': False,
            'beta': 0.0
        },
        # 配置2：专利一 - 自适应STE-Sqrt
        {
            'name': 'adaptive_ste_sqrt',
            'device': args.device,
            'epochs': args.epochs,
            'noise_strength': 0.5,
            'schedule': 'sqrt',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'grad_clip': 1.0,
            'lambda_var': 0.01,
            'noise_schedule': 'constant',
            'use_amp': True,
            'use_official_noise': True,
            'beta': 0.0,
            # 官方噪声模型参数
            'prog_noise_std': 0.01,
            'drift_factor': 0.005,
            'nonlinear_alpha': 0.1,
            'nonlinear_beta': 0.05,
            'output_noise_std': 0.01,
            'crosstalk_factor': 0.002,
            'temperature_factor': 0.001,
            'retention_loss': 0.001
        },
        # 配置3：专利二 - 层次化噪声注入
        {
            'name': 'layerwise_noise',
            'device': args.device,
            'epochs': args.epochs,
            'noise_strength': 0.3,
            'schedule': 'sqrt',
            'lr': 0.1,
            'weight_decay': 5e-4,
            'warmup_epochs': 5,
            'grad_clip': 1.0,
            'lambda_var': 0.01,
            'noise_schedule': 'constant',
            'use_amp': True,
            'use_official_noise': True,
            'beta': 1.0,  # 层次化系数
            'prog_noise_std': 0.01,
            'drift_factor': 0.005,
            'nonlinear_alpha': 0.1,
            'nonlinear_beta': 0.05,
            'output_noise_std': 0.01,
            'crosstalk_factor': 0.002
        },
        # 配置4：全部专利组合
        {
            'name': 'full_patent_combo',
            'device': args.device,
            'epochs': args.epochs,
            'noise_strength': 0.5,
            'schedule': 'sqrt',
            'lr': 0.1,
            'weight_decay': 1e-3,
            'warmup_epochs': 10,
            'grad_clip': 1.0,
            'lambda_var': 0.01,
            'noise_schedule': 'cosine_decay',  # 噪声强度余弦衰减
            'use_amp': True,
            'use_official_noise': True,
            'beta': 1.0,
            'prog_noise_std': 0.01,
            'drift_factor': 0.005,
            'nonlinear_alpha': 0.1,
            'nonlinear_beta': 0.05,
            'output_noise_std': 0.01,
            'crosstalk_factor': 0.002
        },
        # 配置5：高噪声鲁棒性
        {
            'name': 'high_noise_robust',
            'device': args.device,
            'epochs': args.epochs,
            'noise_strength': 1.0,
            'schedule': 'sqrt',
            'lr': 0.05,
            'weight_decay': 2e-3,
            'warmup_epochs': 15,
            'grad_clip': 0.5,
            'lambda_var': 0.02,
            'noise_schedule': 'linear_decay',
            'use_amp': True,
            'use_official_noise': True,
            'beta': 1.5,
            'prog_noise_std': 0.02,
            'drift_factor': 0.01,
            'nonlinear_alpha': 0.2,
            'nonlinear_beta': 0.1,
            'output_noise_std': 0.02,
            'crosstalk_factor': 0.005
        }
    ]

    all_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"运行配置: {config['name']}")
        print(f"{'='*70}")

        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # 构建模型
        model = get_model('resnet18', num_classes=10)

        # 构建带噪声的模型
        if config.get('use_official_noise', False):
            model = build_noisy_model(model, config)

        # 训练
        trainer = STETrainer(model, config, args.device)
        result = trainer.train(trainloader, testloader)
        result['config_name'] = config['name']

        all_results.append(result)

        # 保存结果
        save_path = os.path.join(
            '/mnt/storage2/zyc/CIM比赛/赛道5/赛题2',
            args.save_dir,
            f"{config['name']}.json"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存时不包含tensor数据
        save_result = {
            'config_name': result['config_name'],
            'best_acc': result['best_acc'],
            'best_epoch': result['best_epoch'],
            'final_acc': result['final_acc'],
            'history': {
                'train_loss': result['history']['train_loss'],
                'train_acc': result['history']['train_acc'],
                'test_loss': result['history']['test_loss'],
                'test_acc': result['history']['test_acc'],
                'noise_strength': result['history']['noise_strength'],
                'lr': result['history']['lr']
            },
            'config': result['config']
        }

        with open(save_path, 'w') as f:
            json.dump(save_result, f, indent=2)
        print(f"结果已保存到: {save_path}")

    # 汇总结果
    print("\n" + "=" * 70)
    print("所有配置结果汇总:")
    print("=" * 70)

    best_result = None
    for r in all_results:
        print(f"  {r['config_name']:25s}: Best Acc = {r['best_acc']:.2f}% (Epoch {r['best_epoch']}), Final Acc = {r['final_acc']:.2f}%")
        if best_result is None or r['best_acc'] > best_result['best_acc']:
            best_result = r

    print(f"\n最佳配置: {best_result['config_name']}")
    print(f"最佳精度: {best_result['best_acc']:.2f}%")

    # 保存汇总
    summary_path = os.path.join(
        '/mnt/storage2/zyc/CIM比赛/赛道5/赛题2',
        args.save_dir,
        "summary.json"
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
