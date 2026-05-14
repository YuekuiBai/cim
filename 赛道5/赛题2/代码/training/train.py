"""
STE噪声感知训练模块

任务二：STE噪声感知训练（STE-Aware Training）
- 在训练阶段注入存算芯片噪声
- 对比STE-NAT与标准训练的性能差异
- 分析收敛性和泛化性
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class STETrainingConfig:
    """STE训练配置"""
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_scheduler: str = 'cosine',
        warmup_epochs: int = 5,
        noise_strength: float = 1.0,
        noise_schedule: str = 'fixed',
        save_dir: str = 'results/ste_training',
        grad_clip: Optional[float] = None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.noise_strength = noise_strength
        self.noise_schedule = noise_schedule
        self.save_dir = save_dir
        self.grad_clip = grad_clip


class STEAwareTrainer:
    """STE噪声感知训练器"""

    def __init__(self, model: nn.Module, config: STETrainingConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

        if config.lr_scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        else:
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'noise_strength': []
        }

        os.makedirs(config.save_dir, exist_ok=True)

    def get_noise_strength(self, epoch: int) -> float:
        """根据调度策略获取当前噪声强度"""
        if self.config.noise_schedule == 'fixed':
            return self.config.noise_strength
        elif self.config.noise_schedule == 'gradual':
            progress = min(epoch / (self.config.epochs * 0.5), 1.0)
            return self.config.noise_strength * progress
        elif self.config.noise_schedule == 'decay':
            return self.config.noise_strength * (0.95 ** epoch)
        elif self.config.noise_schedule == 'cyclic':
            cycle = epoch % 10
            return self.config.noise_strength * (1.0 - 0.5 * cycle / 10)
        return self.config.noise_strength

    def set_model_noise(self, strength: float):
        """设置模型所有层的噪声强度"""
        for module in self.model.modules():
            if hasattr(module, 'set_noise_strength'):
                module.set_noise_strength(strength)

    def train_epoch(self, train_loader, noise_strength: float) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        self.set_model_noise(noise_strength)

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Noise={noise_strength:.2f}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        return running_loss / len(train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self, test_loader, noise_strength: float = 0.0) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        self.set_model_noise(noise_strength)

        test_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return test_loss / len(test_loader), 100. * correct / total

    def train(self, train_loader, test_loader) -> Dict:
        """完整训练流程"""
        best_acc = 0.0

        for epoch in range(self.config.epochs):
            noise_strength = self.get_noise_strength(epoch)

            train_loss, train_acc = self.train_epoch(train_loader, noise_strength)
            test_loss, test_acc = self.evaluate(test_loader, noise_strength=0.0)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['noise_strength'].append(noise_strength)

            self.scheduler.step()

            print(f'Epoch {epoch+1}/{self.config.epochs}: '
                  f'Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                  f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%')

            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model('best_model.pth')

        self.save_history()

        return self.history

    def save_model(self, filename: str):
        """保存模型"""
        path = os.path.join(self.config.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)

    def save_history(self):
        """保存训练历史"""
        path = os.path.join(self.config.save_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def compare_training_strategies(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    strategies: Dict[str, Dict],
    save_dir: str
) -> Dict:
    """
    对比多种训练策略

    Args:
        model: 基础模型
        train_loader: 训练数据
        test_loader: 测试数据
        device: 设备
        strategies: 策略配置字典
        save_dir: 保存目录

    Returns:
        各策略的结果
    """
    results = {}

    for strategy_name, strategy_config in strategies.items():
        print(f"\n{'='*60}")
        print(f"训练策略: {strategy_name}")
        print(f"{'='*60}")

        model_copy = type(model)(**model.__dict__.get('_config', {}))

        config = STETrainingConfig(
            epochs=strategy_config.get('epochs', 30),
            lr=strategy_config.get('lr', 0.1),
            noise_strength=strategy_config.get('noise_strength', 1.0),
            noise_schedule=strategy_config.get('noise_schedule', 'fixed'),
            save_dir=os.path.join(save_dir, strategy_name)
        )

        trainer = STEAwareTrainer(model_copy, config, device)
        history = trainer.train(train_loader, test_loader)

        results[strategy_name] = {
            'best_acc': max(history['test_acc']),
            'final_acc': history['test_acc'][-1],
            'history': history
        }

    return results