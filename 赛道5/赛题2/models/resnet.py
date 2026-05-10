"""
神经网络模型定义

支持基于STE的噪声感知训练框架
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_model(name: str = 'resnet18', num_classes: int = 10,
              pretrained: bool = False, ste_config: dict = None) -> nn.Module:
    """
    获取模型

    Args:
        name: 模型名称 (resnet18, resnet34, resnet50)
        num_classes: 分类数
        pretrained: 是否使用预训练权重
        ste_config: STE配置字典

    Returns:
        模型实例
    """
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if name in ('resnet18', 'resnet34', 'resnet50'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


class LayerOutputHook:
    """获取网络各层输出的钩子"""
    def __init__(self):
        self.outputs = {}
        self.handles = []

    def register_hook(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        for name, module in model.named_modules():
            if layer_names is None:
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
        return self.outputs

    def clear(self):
        self.outputs = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()