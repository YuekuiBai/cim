"""
非线性误差注入模块

赛题要求：将非线性映射模型嵌入到可进行矩阵计算的算子
（linear, conv2d, convtranspose2d 等）的输入（激活值）中

非线性模型：x' = α · x³ + (1 - α) · x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class NonLinearInjection(nn.Module):
    """
    非线性注入模块
    
    在激活值上应用非线性失真：x' = α · x³ + (1 - α) · x
    """
    def __init__(self, alpha: float = 0.0, per_channel: bool = False):
        """
        Args:
            alpha: 非线性强度参数
            per_channel: 是否按通道独立归一化（用于卷积层）
        """
        super().__init__()
        self.alpha = alpha
        self.per_channel = per_channel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0 or not self.training and self.alpha == 0:
            return x
        
        if self.per_channel and x.dim() == 4:
            # 卷积层：按通道归一化 (N, C, H, W)
            max_val = x.abs().amax(dim=(2, 3), keepdim=True)
            max_val = max_val.clamp(min=1e-8)
        else:
            # 全连接层或全局归一化
            max_val = x.abs().max()
            if max_val == 0:
                return x
        
        x_norm = x / max_val
        y = self.alpha * (x_norm ** 3) + (1 - self.alpha) * x_norm
        return y * max_val


class NonLinearLinear(nn.Module):
    """
    带非线性注入的Linear层
    
    在输入激活值上注入非线性失真后再进行线性变换
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.0, 
                 bias: bool = True, per_channel: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.nonlinearity = NonLinearInjection(alpha=alpha, per_channel=per_channel)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在输入激活值上注入非线性
        x_noisy = self.nonlinearity(x)
        # 执行线性变换
        return self.linear(x_noisy)
    
    def set_alpha(self, alpha: float):
        """动态设置非线性强度"""
        self.alpha = alpha
        self.nonlinearity.alpha = alpha


class NonLinearConv2d(nn.Module):
    """
    带非线性注入的Conv2d层
    
    在输入激活值上注入非线性失真后再进行卷积运算
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 alpha: float = 0.0, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True,
                 per_channel: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, 
                              dilation=dilation, groups=groups, bias=bias)
        self.nonlinearity = NonLinearInjection(alpha=alpha, per_channel=per_channel)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在输入激活值上注入非线性
        x_noisy = self.nonlinearity(x)
        # 执行卷积运算
        return self.conv(x_noisy)
    
    def set_alpha(self, alpha: float):
        """动态设置非线性强度"""
        self.alpha = alpha
        self.nonlinearity.alpha = alpha


class NonLinearConvTranspose2d(nn.Module):
    """
    带非线性注入的ConvTranspose2d层
    
    在输入激活值上注入非线性失真后再进行转置卷积运算
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 alpha: float = 0.0, stride: int = 1, padding: int = 0,
                 output_padding: int = 0, groups: int = 1, bias: bool = True,
                 dilation: int = 1, per_channel: bool = True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation
        )
        self.nonlinearity = NonLinearInjection(alpha=alpha, per_channel=per_channel)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在输入激活值上注入非线性
        x_noisy = self.nonlinearity(x)
        # 执行转置卷积运算
        return self.conv_transpose(x_noisy)
    
    def set_alpha(self, alpha: float):
        """动态设置非线性强度"""
        self.alpha = alpha
        self.nonlinearity.alpha = alpha


class InverseNonLinearity(nn.Module):
    """
    逆非线性变换（用于预失真补偿）
    
    近似逆：x' ≈ x - α · x³
    """
    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0:
            return x
        
        max_val = x.abs().max()
        if max_val == 0:
            return x
        
        x_norm = x / max_val
        # 近似逆变换
        y = x_norm - self.alpha * (x_norm ** 3)
        return y * max_val


class CalibrationLayer(nn.Module):
    """
    可学习的校准层
    
    通过学习多项式系数来补偿非线性失真：
    y = c1 * x + c3 * x³
    """
    def __init__(self, num_features: int, learn_cubic: bool = True):
        super().__init__()
        self.coeff_1 = nn.Parameter(torch.ones(num_features))
        self.coeff_3 = nn.Parameter(torch.zeros(num_features)) if learn_cubic else None
        self.learn_cubic = learn_cubic
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # 卷积层输出：(N, C, H, W)
            coeff_1 = self.coeff_1.view(1, -1, 1, 1)
            if self.learn_cubic:
                coeff_3 = self.coeff_3.view(1, -1, 1, 1)
                return coeff_1 * x + coeff_3 * (x ** 3)
            return coeff_1 * x
        else:
            # 全连接层输出：(N, C)
            if self.learn_cubic:
                return self.coeff_1 * x + self.coeff_3 * (x ** 3)
            return self.coeff_1 * x


def inject_nonlinearity_to_model(model: nn.Module, alpha: float = 0.0, 
                                  target_layers: tuple = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
    """
    将非线性注入应用到模型的所有目标层
    
    Args:
        model: 要处理的模型
        alpha: 非线性强度
        target_layers: 需要注入非线性的层类型
    
    Returns:
        处理后的模型
    """
    for name, module in model.named_modules():
        if isinstance(module, target_layers):
            # 获取父模块和属性名
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            
            # 创建带非线性注入的替换层
            if isinstance(module, nn.Linear):
                new_module = NonLinearLinear(
                    module.in_features, module.out_features,
                    alpha=alpha, bias=module.bias is not None
                )
                new_module.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.linear.bias.data = module.bias.data.clone()
            elif isinstance(module, nn.Conv2d):
                new_module = NonLinearConv2d(
                    module.in_channels, module.out_channels,
                    module.kernel_size[0], alpha=alpha,
                    stride=module.stride[0], padding=module.padding[0],
                    dilation=module.dilation[0], groups=module.groups,
                    bias=module.bias is not None
                )
                new_module.conv.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.conv.bias.data = module.bias.data.clone()
            elif isinstance(module, nn.ConvTranspose2d):
                new_module = NonLinearConvTranspose2d(
                    module.in_channels, module.out_channels,
                    module.kernel_size[0], alpha=alpha,
                    stride=module.stride[0], padding=module.padding[0],
                    output_padding=module.output_padding[0],
                    groups=module.groups, bias=module.bias is not None,
                    dilation=module.dilation[0]
                )
                new_module.conv_transpose.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.conv_transpose.bias.data = module.bias.data.clone()
            else:
                continue
            
            setattr(parent, attr_name, new_module)
    
    return model


def set_model_alpha(model: nn.Module, alpha: float):
    """
    设置模型中所有非线性注入层的alpha值
    
    Args:
        model: 模型
        alpha: 非线性强度
    """
    for module in model.modules():
        if isinstance(module, (NonLinearLinear, NonLinearConv2d, NonLinearConvTranspose2d)):
            module.set_alpha(alpha)


class NonLinearWrapper(nn.Module):
    """
    模型包装器：为现有模型添加非线性注入
    
    使用方法：
        model = resnet18(pretrained=True)
        model = NonLinearWrapper(model, alpha=0.2)
    """
    def __init__(self, model: nn.Module, alpha: float = 0.0):
        super().__init__()
        self.model = inject_nonlinearity_to_model(model, alpha=alpha)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def set_alpha(self, alpha: float):
        """动态设置非线性强度"""
        self.alpha = alpha
        set_model_alpha(self.model, alpha)
    
    def get_original_model(self) -> nn.Module:
        """获取原始模型（移除非线性注入）"""
        # 这里返回的是带注入的模型，如需原始模型需要重新加载
        return self.model
