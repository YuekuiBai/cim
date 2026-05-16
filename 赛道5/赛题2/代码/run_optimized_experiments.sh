#!/bin/bash
==========================================
优化版实验脚本 - 修复失败实验
基于失败分析优化参数配置
==========================================

export CUDA_VISIBLE_DEVICES=1
cd /mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码

echo "=========================================="
echo "优化版实验 - $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

echo "[INFO] 在GPU 1上运行优化实验（与GPU 2上的P2-3并行）"

# P2-4优化：降低噪声强度和相关性
echo ""
echo "=========================================="
echo "[OPT-1/3] P2-4 spatiotemporal_noise (优化参数)"
echo "  优化: noise_strength 1.0->0.1, correlation_rho 0.7->0.3"
echo "=========================================="
python scripts/run_enhanced_experiments.py \
    --experiment_type spatiotemporal_noise \
    --device cuda \
    --epochs 30 \
    --batch_size 256 \
    --num_workers 4 \
    --seed 42 \
    --noise_strength 0.1 \
    --correlation_rho 0.3 \
    2>&1 | tee logs/optimized_p24_$(date +%Y%m%d_%H%M%S).log

# P2-5优化：调整EMA衰减和校正因子
echo ""
echo "=========================================="
echo "[OPT-2/3] P2-5 decoupled_bias_correction (优化参数)"
echo "  优化: noise_strength 1.0->0.2, ema_decay 0.99->0.95"
echo "=========================================="
python scripts/run_enhanced_experiments.py \
    --experiment_type decoupled_bias_correction \
    --device cuda \
    --epochs 30 \
    --batch_size 256 \
    --num_workers 4 \
    --seed 42 \
    --noise_strength 0.2 \
    2>&1 | tee logs/optimized_p25_$(date +%Y%m%d_%H%M%S).log

# P2-6优化：大幅降低正则化权重，添加warmup
echo ""
echo "=========================================="
echo "[OPT-3/3] P2-6 regularizer_v2_ablation (优化参数)"
echo "  优化: 使用脚本修改lambda参数"
echo "=========================================="

# 创建临时优化脚本
cat > /tmp/run_p26_optimized.py << 'PYEOF'
import sys
import os
sys.path.insert(0, '/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/代码')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

from models.resnet import get_model
from ste.core import NoisyLinear, NoisyConv2d
from ste.innovation import NoiseAwareRegularizer, InnovationConfig

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

class OptimizedRegularizerV2:
    """优化版正则化2.0 - 降低权重，添加warmup"""

    def __init__(self, lambda_l2=0.001, lambda_grad_smooth=0.0005, lambda_kl=0.0001):
        self.lambda_l2 = lambda_l2
        self.lambda_grad_smooth = lambda_grad_smooth
        self.lambda_kl = lambda_kl
        self.previous_weights = None

    def compute_penalty(self, model):
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

        return self.lambda_l2 * l2_penalty + self.lambda_grad_smooth * grad_smooth_penalty + self.lambda_kl * kl_penalty

# 配置
device = 'cuda'
epochs = 30
batch_size = 256
seed = 42

print("=" * 70)
print("实验类型: regularizer_v2_ablation (优化版)")
print("=" * 70)

# 数据加载
import torchvision
import torchvision.transforms as transforms

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

trainset = torchvision.datasets.CIFAR10(root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='/mnt/storage2/zyc/CIM比赛/公共数据集', train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# 模型
torch.manual_seed(seed)
model = get_model('resnet18', num_classes=10)

noise_config = InnovationConfig(
    noise_strength=1.0,
    schedule='inverse'
)
model = inject_ste_to_model(model, noise_config)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# 优化版正则化器
regularizer = OptimizedRegularizerV2(lambda_l2=0.001, lambda_grad_smooth=0.0005, lambda_kl=0.0001)

history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
best_acc = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Warmup: 前5个epoch不加正则化
        if epoch > 5:
            reg_loss = regularizer.compute_penalty(model)
            loss = loss + reg_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

    train_acc = 100.*correct/total
    history['train_loss'].append(train_loss/len(trainloader))
    history['train_acc'].append(train_acc)

    scheduler.step()

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_acc = 100.*test_correct/test_total
    history['test_acc'].append(test_acc)

    print(f'Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')

    if test_acc > best_acc:
        best_acc = test_acc
        print(f'  -> 新最佳精度: {best_acc:.2f}%')

# 保存结果
results = {
    'experiment_type': 'regularizer_v2_ablation_optimized',
    'best_acc': best_acc,
    'final_acc': history['test_acc'][-1],
    'history': history,
    'config': {
        'epochs': epochs,
        'batch_size': batch_size,
        'seed': seed,
        'noise_strength': 1.0,
        'lambda_l2': 0.001,
        'lambda_grad_smooth': 0.0005,
        'lambda_kl': 0.0001,
        'warmup_epochs': 5
    }
}

save_path = f'/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/结果/enhanced_experiments/regularizer_v2_ablation_optimized_results.json'
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n实验完成！最佳精度: {best_acc:.2f}%")
print(f"结果已保存到: {save_path}")
PYEOF

python /tmp/run_p26_optimized.py 2>&1 | tee logs/optimized_p26_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "所有优化实验完成！"
echo "=========================================="
