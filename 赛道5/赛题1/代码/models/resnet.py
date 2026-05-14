"""
神经网络模型定义

支持带非线性注入的模型构建
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List, Type, Union
from noise.nonlinearity import (
    NonLinearWrapper, 
    NonLinearLinear, 
    NonLinearConv2d,
    NonLinearConvTranspose2d,
    inject_nonlinearity_to_model,
    set_model_alpha
)


def get_model(name: str = 'resnet18', num_classes: int = 10, 
              pretrained: bool = False, alpha: float = 0.0) -> nn.Module:
    """
    获取模型
    
    Args:
        name: 模型名称 (resnet18, resnet34, resnet50, mobilenet_v2)
        num_classes: 分类数
        pretrained: 是否使用预训练权重
        alpha: 非线性强度，0表示无非线性
    
    Returns:
        模型实例
    """
    # 获取基础模型（预训练时先加载1000类，再替换最后一层）
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model: {name}")
    
    # 替换最后一层以适配目标类别数
    if name in ('resnet18', 'resnet34', 'resnet50'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # 如果需要非线性注入，使用包装器
    if alpha > 0:
        model = NonLinearWrapper(model, alpha=alpha)
    
    return model


class LayerOutputHook:
    """
    用于获取网络各层输出的钩子
    
    用于敏感性分析中的单层输出分布偏移分析
    """
    def __init__(self):
        self.outputs = {}
        self.handles = []
    
    def register_hook(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        注册前向钩子
        
        Args:
            model: 模型
            layer_names: 要监控的层名列表，None表示监控所有卷积层和全连接层
        """
        for name, module in model.named_modules():
            if layer_names is None:
                # 默认监控所有卷积层和全连接层
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    handle = module.register_forward_hook(self._make_hook(name))
                    self.handles.append(handle)
            elif name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
    
    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.outputs[name] = output.detach()
        return hook
    
    def get_outputs(self) -> dict:
        """获取所有捕获的输出"""
        return self.outputs
    
    def clear(self):
        """清除所有输出"""
        self.outputs = {}
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []


class NonLinearResNet(nn.Module):
    """
    自定义ResNet，在每个卷积层前注入非线性
    
    用于精细控制非线性注入位置
    """
    def __init__(self, block, layers, num_classes: int = 10, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.inplanes = 64
        
        # 初始卷积层
        self.conv1 = NonLinearConv2d(3, 64, kernel_size=7, stride=2, padding=3, 
                                      bias=False, alpha=alpha)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = NonLinearLinear(512 * block.expansion, num_classes, alpha=alpha)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                NonLinearConv2d(self.inplanes, planes * block.expansion, 
                               kernel_size=1, stride=stride, bias=False, alpha=self.alpha),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, alpha=self.alpha))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, alpha=self.alpha))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def set_alpha(self, alpha: float):
        """动态设置非线性强度"""
        self.alpha = alpha
        for module in self.modules():
            if isinstance(module, (NonLinearLinear, NonLinearConv2d)):
                module.set_alpha(alpha)


class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, alpha=0.0):
        super().__init__()
        self.conv1 = NonLinearConv2d(inplanes, planes, kernel_size=3, stride=stride,
                                      padding=1, bias=False, alpha=alpha)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = NonLinearConv2d(planes, planes, kernel_size=3, stride=1,
                                      padding=1, bias=False, alpha=alpha)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, alpha=0.0):
        super().__init__()
        self.conv1 = NonLinearConv2d(inplanes, planes, kernel_size=1, bias=False, alpha=alpha)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = NonLinearConv2d(planes, planes, kernel_size=3, stride=stride,
                                      padding=1, bias=False, alpha=alpha)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = NonLinearConv2d(planes, planes * self.expansion, kernel_size=1,
                                      bias=False, alpha=alpha)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


def non_linear_resnet18(num_classes: int = 10, alpha: float = 0.0) -> NonLinearResNet:
    """构建带非线性注入的ResNet18"""
    return NonLinearResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, alpha=alpha)


def non_linear_resnet34(num_classes: int = 10, alpha: float = 0.0) -> NonLinearResNet:
    """构建带非线性注入的ResNet34"""
    return NonLinearResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, alpha=alpha)


def non_linear_resnet50(num_classes: int = 10, alpha: float = 0.0) -> NonLinearResNet:
    """构建带非线性注入的ResNet50"""
    return NonLinearResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, alpha=alpha)
