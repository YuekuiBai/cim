"""
STE噪声感知训练框架核心模块 - 增强版

创新算法实现：
1. 自适应STE (Adaptive STE)
2. 噪声感知正则化
3. 偏差校正机制
4. 层次化噪声注入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdaptiveSTE(nn.Module):
    """
    自适应STE梯度估计器

    核心创新：根据噪声水平动态调整梯度缩放因子

    参考CoRR 2025论文的自适应机制设计
    """

    def __init__(
        self,
        estimator_type: str = 'identity',
        clip_value: Optional[float] = None,
        adaptive_scale: bool = True,
        noise_schedule: str = 'inverse'
    ):
        super().__init__()
        self.estimator_type = estimator_type
        self.clip_value = clip_value
        self.adaptive_scale = adaptive_scale
        self.noise_schedule = noise_schedule
        self.noise_level = 1.0

    def set_noise_level(self, noise_level: float):
        self.noise_level = noise_level

    def estimate(self, grad_output: torch.Tensor, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.estimator_type == 'identity':
            grad_input = grad_output
        elif self.estimator_type == 'signed':
            grad_input = torch.sign(grad_output)
        elif self.estimator_type == 'clipped':
            grad_input = torch.clamp(grad_output, -self.clip_value, self.clip_value) if self.clip_value else grad_output
        else:
            grad_input = grad_output

        if self.adaptive_scale:
            grad_input = grad_input * self._get_adaptive_scale()

        return grad_input

    def _get_adaptive_scale(self) -> float:
        """
        根据噪声水平计算自适应缩放因子

        噪声调度策略：
        - 'inverse': scale = 1 / (1 + noise^2)
        - 'linear': scale = 1 / (1 + noise)
        - 'sqrt': scale = 1 / sqrt(1 + noise^2)
        - 'exp': scale = exp(-noise / 2)
        """
        nl = self.noise_level
        if self.noise_schedule == 'inverse':
            return 1.0 / (1.0 + nl ** 2)
        elif self.noise_schedule == 'linear':
            return 1.0 / (1.0 + nl)
        elif self.noise_schedule == 'sqrt':
            return 1.0 / torch.sqrt(1.0 + nl ** 2 + 1e-8)
        elif self.noise_schedule == 'exp':
            return torch.exp(-nl / 2.0 * torch.ones(1)).item()
        return 1.0


class NoiseAwareRegularizer(nn.Module):
    """
    噪声感知正则化器

    核心思想：鼓励权重分布更加紧凑，减少对噪声的敏感性
    """
    def __init__(self, sigma: float = 0.01, penalty_type: str = 'l2'):
        super().__init__()
        self.sigma = sigma
        self.penalty_type = penalty_type

    def compute_penalty(self, model: nn.Module) -> torch.Tensor:
        penalty = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name:
                if self.penalty_type == 'l2':
                    penalty = penalty + torch.sum(param ** 2)
                elif self.penalty_type == 'l1':
                    penalty = penalty + torch.sum(torch.abs(param))
                elif self.penalty_type == 'kl':
                    mean = param.mean()
                    std = param.std() + 1e-8
                    penalty = penalty + torch.sum((param - mean) ** 2 / (2 * std ** 2))
        return self.sigma * penalty


class BiasCorrector(nn.Module):
    """
    偏差校正机制

    估计并校正噪声引入的梯度偏差
    """
    def __init__(self, correction_type: str = 'ema'):
        super().__init__()
        self.correction_type = correction_type
        self.bias_history = []
        self.ema_bias = None
        self.ema_decay = 0.9

    def estimate_bias(self, grad: torch.Tensor, noise_config: Dict, noise_strength: float) -> torch.Tensor:
        prog_noise_std = noise_config.get('prog_noise_std', 0.01) * noise_strength
        drift_factor = noise_config.get('drift_factor', 0.005) * noise_strength

        bias_estimate = torch.zeros_like(grad)

        if self.correction_type == 'analytic':
            bias_estimate = -grad * (prog_noise_std ** 2 + drift_factor ** 2)
        elif self.correction_type == 'ema':
            batch_bias = -grad * (prog_noise_std ** 2 + drift_factor ** 2)
            if self.ema_bias is None:
                self.ema_bias = batch_bias
            else:
                self.ema_bias = self.ema_decay * self.ema_bias + (1 - self.ema_decay) * batch_bias
            bias_estimate = self.ema_bias

        return bias_estimate

    def correct(self, grad: torch.Tensor, noise_config: Dict, noise_strength: float) -> torch.Tensor:
        bias_estimate = self.estimate_bias(grad, noise_config, noise_strength)
        return grad - bias_estimate


class LayerwiseNoiseInjection(nn.Module):
    """
    层次化噪声注入

    根据层深度和类型动态调整噪声强度
    """
    def __init__(self, noise_config: Dict, depth_scaling: float = 0.1):
        super().__init__()
        self.noise_config = noise_config
        self.depth_scaling = depth_scaling
        self.layer_scales = {
            'conv1': 0.8,
            'conv2': 1.0,
            'fc': 1.2,
        }

    def get_layer_noise(self, layer_name: str, layer_depth: int, layer_type: str) -> Tuple[float, float, float]:
        base_scale = self.layer_scales.get(layer_type, 1.0)
        depth_factor = 1.0 / (1.0 + self.depth_scaling * layer_depth)

        scale = base_scale * depth_factor

        prog_noise = self.noise_config.get('prog_noise_std', 0.01) * scale
        output_noise = self.noise_config.get('output_noise_std', 0.01) * scale
        crosstalk = self.noise_config.get('crosstalk_factor', 0.002) * scale

        return prog_noise, output_noise, crosstalk


class STEGradientEstimator(nn.Module):
    """
    STE梯度估计器（增强版）

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
        if self.adaptive_scale and self.estimator_type == 'identity':
            return 1.0
        return 1.0


class STEBaseFunc(torch.autograd.Function):
    """
    STE基类：直通估计器
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
    噪声注入器（增强版）
    """

    def __init__(self, noise_config: Dict):
        super().__init__()
        self.noise_config = noise_config
        self.noise_strength = 1.0
        self.training_noise = True
        self.use_layerwise = False
        self.layer_info = {'depth': 0, 'type': 'conv'}

    def set_noise_strength(self, strength: float):
        self.noise_strength = strength

    def enable_noise(self, enabled: bool):
        self.training_noise = enabled

    def set_layer_info(self, depth: int, layer_type: str):
        self.layer_info = {'depth': depth, 'type': layer_type}
        self.use_layerwise = True

    def inject_to_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.training_noise or self.noise_strength <= 0:
            return weight

        prog_noise_std = self.noise_config.get('prog_noise_std', 0.01) * self.noise_strength
        drift_factor = self.noise_config.get('drift_factor', 0.005) * self.noise_strength

        prog_noise = torch.randn_like(weight) * prog_noise_std
        drift_noise = torch.randn_like(weight) * drift_factor * torch.abs(weight)

        return weight + prog_noise + drift_noise

    def inject_to_input(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training_noise or self.noise_strength <= 0:
            return input

        crosstalk_factor = self.noise_config.get('crosstalk_factor', 0.002) * self.noise_strength

        if crosstalk_factor > 0:
            crosstalk = torch.randn_like(input) * crosstalk_factor * torch.norm(input, dim=-1, keepdim=True)
            return input + crosstalk
        return input

    def inject_saturation(self, output: torch.Tensor) -> torch.Tensor:
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
        if not self.training_noise or self.noise_strength <= 0:
            return output

        output_noise_std = self.noise_config.get('output_noise_std', 0.01) * self.noise_strength

        noise = torch.randn_like(output) * output_noise_std
        return output + noise


class NoisyLinear(nn.Module):
    """支持噪声注入的线性层"""

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
        noisy_input = self.noise_injector.inject_to_input(input)
        noisy_weight = self.noise_injector.inject_to_weight(self.weight)

        output = F.linear(noisy_input, noisy_weight, self.bias)
        output = self.noise_injector.inject_saturation(output)
        output = self.noise_injector.inject_output_noise(output)

        return output


class NoisyConv2d(nn.Module):
    """支持噪声注入的卷积层"""

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


class InnovationConfig:
    """创新算法配置"""

    def __init__(self, config_dict=None):
        cfg = config_dict or {}

        self.adaptive_ste = cfg.get('adaptive_ste', {
            'enabled': True,
            'schedule': 'inverse',
            'adaptive_scale': True
        })

        self.regularizer = cfg.get('regularizer', {
            'enabled': True,
            'sigma': 0.01,
            'penalty_type': 'l2'
        })

        self.bias_correction = cfg.get('bias_correction', {
            'enabled': True,
            'correction_type': 'ema'
        })

        self.layerwise_noise = cfg.get('layerwise_noise', {
            'enabled': False,
            'depth_scaling': 0.1
        })

        self.noise_schedules = ['inverse', 'linear', 'sqrt', 'exp']