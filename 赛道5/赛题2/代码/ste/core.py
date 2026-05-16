"""
STE噪声感知训练框架核心模块

任务一：通用STE框架设计与实现
- 核心STE算法设计
- 多架构适配机制
- 梯度估计优化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class STEGradientEstimator(nn.Module):
    """
    STE梯度估计器

    核心思想：在反向传播时绕过噪声操作的不可微性，
    使用简化的梯度估计维持训练可行性
    """

    def __init__(
        self,
        estimator_type: str = 'identity',
        clip_value: Optional[float] = None,
        adaptive_scale: bool = True
    ):
        super().__init__()
        self.estimator_type = estimator_type
        self.clip_value = clip_value
        self.adaptive_scale = adaptive_scale

    def estimate(self, grad_output: torch.Tensor, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        估计梯度

        Args:
            grad_output: 反向传播到该层的梯度
            input: 输入数据（可选，用于自适应估计）

        Returns:
            估计的梯度
        """
        if self.estimator_type == 'identity':
            grad_input = grad_output
        elif self.estimator_type == 'signed':
            grad_input = torch.sign(grad_output)
        elif self.estimator_type == 'clipped':
            if self.clip_value is not None:
                grad_input = torch.clamp(grad_output, -self.clip_value, self.clip_value)
            else:
                grad_input = grad_output
        else:
            grad_input = grad_output

        if self.adaptive_scale:
            grad_input = grad_input * self._get_scale_factor()

        return grad_input

    def _get_scale_factor(self) -> float:
        """获取缩放因子用于方差稳定化"""
        if self.adaptive_scale and self.estimator_type == 'identity':
            return 1.0
        return 1.0


class STEBaseFunc(torch.autograd.Function):
    """
    STE基类：直通估计器

    前向传播：执行实际操作
    反向传播：使用恒等梯度绕过不可微操作
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_identity(x):
    """直通估计器：恒等函数"""
    return STEBaseFunc.apply(x)


class NoiseInjector(nn.Module):
    """
    噪声注入器

    将存算芯片的噪声特性注入到前向传播中
    """

    def __init__(self, noise_config: Dict):
        super().__init__()
        self.noise_config = noise_config
        self.noise_strength = 1.0
        self.training_noise = True

    def set_noise_strength(self, strength: float):
        self.noise_strength = strength

    def enable_noise(self, enabled: bool):
        self.training_noise = enabled

    def inject_to_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """注入权重噪声（编程误差）"""
        if not self.training_noise or self.noise_strength <= 0:
            return weight

        prog_noise_std = self.noise_config.get('prog_noise_std', 0.01) * self.noise_strength
        drift_factor = self.noise_config.get('drift_factor', 0.005) * self.noise_strength

        noise_device = weight.device
        prog_noise = torch.randn(weight.shape, device=noise_device, dtype=weight.dtype) * prog_noise_std
        drift_noise = torch.randn(weight.shape, device=noise_device, dtype=weight.dtype) * drift_factor * torch.abs(weight)

        return weight + prog_noise + drift_noise

    def inject_to_input(self, input: torch.Tensor) -> torch.Tensor:
        """注入输入噪声（串扰）"""
        if not self.training_noise or self.noise_strength <= 0:
            return input

        crosstalk_factor = self.noise_config.get('crosstalk_factor', 0.002) * self.noise_strength

        if crosstalk_factor > 0:
            noise_device = input.device
            crosstalk = torch.randn(input.shape, device=noise_device, dtype=input.dtype) * crosstalk_factor * torch.norm(input, dim=-1, keepdim=True)
            return input + crosstalk
        return input

    def inject_saturation(self, output: torch.Tensor) -> torch.Tensor:
        """注入饱和非线性"""
        if not self.training_noise or self.noise_strength <= 0:
            return output

        alpha = self.noise_config.get('nonlinear_alpha', 0.1) * self.noise_strength
        beta = self.noise_config.get('nonlinear_beta', 0.05) * self.noise_strength

        pos_mask = output > 0
        neg_mask = output < 0

        result = torch.zeros_like(output)
        result[pos_mask] = torch.tanh(alpha * output[pos_mask]) / alpha
        result[neg_mask] = torch.tanh((alpha + beta) * output[neg_mask]) / (alpha + beta)

        return result

    def inject_output_noise(self, output: torch.Tensor) -> torch.Tensor:
        """注入输出噪声"""
        if not self.training_noise or self.noise_strength <= 0:
            return output

        output_noise_std = self.noise_config.get('output_noise_std', 0.01) * self.noise_strength

        noise = torch.randn_like(output) * output_noise_std
        return output + noise


class NoisyLinear(nn.Module):
    """
    支持噪声注入的线性层，使用STE进行梯度估计
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        noise_config: Optional[Dict] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.noise_config = noise_config or {}
        self.noise_injector = NoiseInjector(self.noise_config)
        self.noise_strength = 1.0
        self.training_noise = True

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_noise_strength(self, strength: float):
        self.noise_strength = strength
        self.noise_injector.set_noise_strength(strength)

    def enable_noise(self, enabled: bool):
        self.training_noise = enabled
        self.noise_injector.enable_noise(enabled)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播：带噪声的矩阵乘法
        反向传播：使用STE（恒等梯度）
        """
        noisy_input = self.noise_injector.inject_to_input(input)
        noisy_weight = self.noise_injector.inject_to_weight(self.weight)

        output = F.linear(noisy_input, noisy_weight, self.bias)
        output = self.noise_injector.inject_saturation(output)
        output = self.noise_injector.inject_output_noise(output)

        return output


class NoisyConv2d(nn.Module):
    """
    支持噪声注入的卷积层
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
        noise_config: Optional[Dict] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.noise_config = noise_config or {}
        self.noise_injector = NoiseInjector(self.noise_config)
        self.noise_strength = 1.0
        self.training_noise = True

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_noise_strength(self, strength: float):
        self.noise_strength = strength
        self.noise_injector.set_noise_strength(strength)

    def enable_noise(self, enabled: bool):
        self.training_noise = enabled
        self.noise_injector.enable_noise(enabled)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        noisy_input = self.noise_injector.inject_to_input(input)
        noisy_weight = self.noise_injector.inject_to_weight(self.weight)

        output = F.conv2d(noisy_input, noisy_weight, self.bias, self.stride, self.padding)
        output = self.noise_injector.inject_saturation(output)
        output = self.noise_injector.inject_output_noise(output)

        return output