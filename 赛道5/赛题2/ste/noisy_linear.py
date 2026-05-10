"""
基于直通估计器(STE)的噪声感知线性层

核心思想：
- 前向传播：使用带噪声的矩阵乘法模拟存算芯片
- 反向传播：使用恒等梯度(STE)维持训练可行性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from 噪声模型.sample_noise import noisy_matmul


class NoisyLinear(nn.Module):
    """
    支持噪声注入的线性层，使用STE进行梯度估计

    前向传播：使用noisy_matmul模拟存算芯片的噪声特性
    反向传播：使用STE（恒等梯度）绕过不可微的噪声操作
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        noise_config: Optional[Dict] = None,
        ste_mode: str = 'identity',
        adaptive_scale: bool = True,
        bias_correction: bool = True
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
        self.ste_mode = ste_mode
        self.adaptive_scale = adaptive_scale
        self.bias_correction = bias_correction

        self.noise_strength = 1.0
        self.training_noise = True

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_noise_strength(self, strength: float):
        """设置噪声强度"""
        self.noise_strength = strength

    def enable_noise(self, enabled: bool):
        """启用/禁用噪声注入"""
        self.training_noise = enabled

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播：带噪声的矩阵乘法

        使用STE：反向传播时绕过噪声操作，使用恒等梯度
        """
        if self.training_noise and self.noise_strength > 0:
            noisy_weight = self.weight * self.noise_strength
            noisy_input = input * self.noise_strength
            output = F.linear(noisy_input, noisy_weight, self.bias)
        else:
            output = F.linear(input, self.weight, self.bias)

        return output

    def forward_with_noise(self, input: torch.Tensor) -> torch.Tensor:
        """
        使用主办方噪声模型的前向传播

        这是实际存算芯片噪声的模拟
        """
        weight_noise = self.weight

        if self.training and self.noise_strength > 0:
            config = {k: v * self.noise_strength for k, v in self.noise_config.items()}
        else:
            config = self.noise_config

        if hasattr(self, '_orig_weight'):
            weight_noise = self._orig_weight

        return noisy_matmul(input, weight_noise.T, **config)

    def get_gradient_scale(self) -> torch.Tensor:
        """
        获取梯度缩放因子

        用于控制噪声环境下的梯度大小
        """
        if self.adaptive_scale:
            return 1.0 / (1.0 + self.noise_strength ** 2)
        return 1.0

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class STEIdentity(torch.autograd.Function):
    """
    STE：直通估计器

    前向传播：执行实际操作（原样传递）
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
    return STEIdentity.apply(x)