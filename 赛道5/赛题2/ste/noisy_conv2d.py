"""
基于直通估计器(STE)的噪声感知卷积层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NoisyConv2d(nn.Module):
    """
    支持噪声注入的卷积层，使用STE进行梯度估计
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
        noise_config: Optional[Dict] = None,
        ste_mode: str = 'identity',
        adaptive_scale: bool = True
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
        self.ste_mode = ste_mode
        self.adaptive_scale = adaptive_scale
        self.noise_strength = 1.0
        self.training_noise = True

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_noise_strength(self, strength: float):
        self.noise_strength = strength

    def enable_noise(self, enabled: bool):
        self.training_noise = enabled

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播：带噪声的卷积
        """
        if self.training_noise and self.noise_strength > 0:
            noisy_weight = self.weight * self.noise_strength
            noisy_input = input * self.noise_strength
            output = F.conv2d(noisy_input, noisy_weight, self.bias, self.stride, self.padding)
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)

        return output

    def get_gradient_scale(self) -> float:
        if self.adaptive_scale:
            return 1.0 / (1.0 + self.noise_strength ** 2)
        return 1.0

    def extra_repr(self) -> str:
        s = f'in_channels={self.in_channels}, out_channels={self.out_channels}'
        s += f', kernel_size={self.kernel_size}'
        s += f', stride={self.stride}'
        s += f', padding={self.padding}'
        s += f', bias={self.bias is not None}'
        return s