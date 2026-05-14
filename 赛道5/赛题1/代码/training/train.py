"""
非线性感知训练模块

任务二：非线性感知训练（Nonlinearity-Aware Training）
- 在训练阶段加入非线性映射模型
- 探究对网络收敛性、泛化性的影响
- 对比微调（fine-tuning）与从头训练（training from scratch）策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Callable
from noise.nonlinearity import NonLinearWrapper, set_model_alpha


class TrainingConfig:
    """训练配置"""
    def __init__(self,
                 epochs: int = 100,
                 batch_size: int = 128,
                 lr: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 5e-4,
                 lr_scheduler: str = 'cosine',
                 warmup_epochs: int = 5,
                 alpha: float = 0.0,
                 alpha_schedule: str = 'fixed',  # 'fixed', 'gradual', 'random'
                 save_dir: str = '../结果/training'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.save_dir = save_dir


class NonlinearityAwareTrainer:
    """非线性感知训练器"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        if config.lr_scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        else:
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'alpha_values': []
        }
    
    def get_alpha(self, epoch: int) -> float:
        """根据调度策略获取当前alpha值"""
        if self.config.alpha_schedule == 'fixed':
            return self.config.alpha
        elif self.config.alpha_schedule == 'gradual':
            # 渐进式增加alpha
            progress = min(epoch / (self.config.epochs * 0.5), 1.0)
            return self.config.alpha * progress
        elif self.config.alpha_schedule == 'random':
            # 随机采样alpha
            return np.random.uniform(0, self.config.alpha)
        return self.config.alpha
    
    def train_epoch(self, train_loader, alpha: float) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 设置当前alpha
        if isinstance(self.model, NonLinearWrapper):
            self.model.set_alpha(alpha)
        else:
            set_model_alpha(self.model, alpha)
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Training (α={alpha:.3f})')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def evaluate(self, test_loader, alpha: float = 0.0) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        
        # 设置评估时的alpha
        if isinstance(self.model, NonLinearWrapper):
            self.model.set_alpha(alpha)
        else:
            set_model_alpha(self.model, alpha)
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(test_loader), 100. * correct / total
    
    def train(self, train_loader, test_loader, 
              eval_alphas: List[float] = None) -> Dict:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            eval_alphas: 评估时要测试的alpha值列表
        
        Returns:
            训练历史
        """
        if eval_alphas is None:
            eval_alphas = [0.0, self.config.alpha]
        
        print(f"\n开始非线性感知训练 (训练alpha={self.config.alpha}, 策略={self.config.alpha_schedule})")
        print("=" * 60)
        
        best_acc = 0
        best_epoch = 0
        
        for epoch in range(self.config.epochs):
            # 获取当前alpha
            current_alpha = self.get_alpha(epoch)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, current_alpha)
            
            # 评估（在清洁条件下）
            test_loss, test_acc = self.evaluate(test_loader, alpha=0.0)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['alpha_values'].append(current_alpha)
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, test_acc)
            
            print(f'Epoch {epoch+1}/{self.config.epochs}: '
                  f'Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                  f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}% '
                  f'(Best: {best_acc:.2f}% @ epoch {best_epoch+1})')
        
        # 最终评估：在不同alpha下的表现
        print("\n最终评估（不同非线性强度下）：")
        final_results = {}
        for alpha in eval_alphas:
            _, acc = self.evaluate(test_loader, alpha=alpha)
            final_results[f'alpha_{alpha}'] = acc
            print(f'  α={alpha:.2f}: Accuracy={acc:.2f}%')
        
        self.history['final_evaluation'] = final_results
        self.history['best_accuracy'] = best_acc
        self.history['best_epoch'] = best_epoch
        
        # 保存训练历史
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, accuracy: float):
        """保存模型检查点"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, os.path.join(self.config.save_dir, filename))
    
    def save_history(self):
        """保存训练历史"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        with open(os.path.join(self.config.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)


def finetune_with_nonlinearity(model: nn.Module, train_loader, test_loader,
                                alpha: float, device: torch.device,
                                epochs: int = 20, lr: float = 0.01) -> Dict:
    """
    微调训练：从预训练模型开始，加入非线性感知训练
    
    Args:
        model: 预训练模型
        train_loader: 训练数据
        test_loader: 测试数据
        alpha: 非线性强度
        device: 设备
        epochs: 微调轮数
        lr: 学习率（通常比从头训练小）
    
    Returns:
        训练结果
    """
    config = TrainingConfig(
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        alpha_schedule='fixed',
        save_dir=f'../结果/training/finetune_alpha_{alpha}'
    )
    
    trainer = NonlinearityAwareTrainer(model, config, device)
    return trainer.train(train_loader, test_loader)


def train_from_scratch(model_fn: Callable, train_loader, test_loader,
                       alpha: float, device: torch.device,
                       epochs: int = 100, lr: float = 0.1) -> Dict:
    """
    从头训练：使用非线性感知训练从头训练模型
    
    Args:
        model_fn: 模型创建函数（无参数）
        train_loader: 训练数据
        test_loader: 测试数据
        alpha: 非线性强度
        device: 设备
        epochs: 训练轮数
        lr: 学习率
    
    Returns:
        训练结果
    """
    # 创建新模型
    model = model_fn()
    
    config = TrainingConfig(
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        alpha_schedule='fixed',
        save_dir=f'../结果/training/scratch_alpha_{alpha}'
    )
    
    trainer = NonlinearityAwareTrainer(model, config, device)
    return trainer.train(train_loader, test_loader)


def compare_training_strategies(model_fn: Callable, pretrained_model: nn.Module,
                                 train_loader, test_loader, alpha_values: List[float],
                                 device: torch.device,
                                 finetune_epochs: int = 20,
                                 scratch_epochs: int = 100) -> Dict:
    """
    对比微调与从头训练策略
    
    Args:
        model_fn: 模型创建函数
        pretrained_model: 预训练模型
        train_loader: 训练数据
        test_loader: 测试数据
        alpha_values: 要测试的非线性强度列表
        device: 设备
        finetune_epochs: 微调轮数
        scratch_epochs: 从头训练轮数
    
    Returns:
        对比结果
    """
    results = {
        'finetune': {},
        'scratch': {},
        'comparison': {}
    }
    
    print("\n" + "=" * 70)
    print("开始对比实验：微调 vs 从头训练")
    print("=" * 70)
    
    for alpha in alpha_values:
        print(f"\n{'='*50}")
        print(f"Alpha = {alpha}")
        print(f"{'='*50}")
        
        # 1. 微调训练
        print("\n[1/2] 微调训练...")
        finetune_model = pretrained_model
        finetune_results = finetune_with_nonlinearity(
            finetune_model, train_loader, test_loader,
            alpha, device, epochs=finetune_epochs
        )
        results['finetune'][alpha] = finetune_results
        
        # 2. 从头训练
        print("\n[2/2] 从头训练...")
        scratch_results = train_from_scratch(
            model_fn, train_loader, test_loader,
            alpha, device, epochs=scratch_epochs
        )
        results['scratch'][alpha] = scratch_results
        
        # 对比
        results['comparison'][alpha] = {
            'finetune_best_acc': finetune_results['best_accuracy'],
            'scratch_best_acc': scratch_results['best_accuracy'],
            'finetune_final_clean_acc': finetune_results['final_evaluation'].get(f'alpha_0.0', 0),
            'scratch_final_clean_acc': scratch_results['final_evaluation'].get(f'alpha_0.0', 0),
            'finetune_final_noisy_acc': finetune_results['final_evaluation'].get(f'alpha_{alpha}', 0),
            'scratch_final_noisy_acc': scratch_results['final_evaluation'].get(f'alpha_{alpha}', 0),
        }
        
        print(f"\n对比结果 (α={alpha}):")
        print(f"  微调 - 最佳精度: {finetune_results['best_accuracy']:.2f}%")
        print(f"  从头训练 - 最佳精度: {scratch_results['best_accuracy']:.2f}%")
    
    # 保存对比结果
    os.makedirs('../结果/training/comparison', exist_ok=True)
    with open('../结果/training/comparison/comparison_results.json', 'w') as f:
        json.dump(results['comparison'], f, indent=2)
    
    return results


def analyze_convergence(history: Dict) -> Dict:
    """
    分析训练收敛性
    
    Args:
        history: 训练历史
    
    Returns:
        收敛性分析结果
    """
    train_acc = np.array(history['train_acc'])
    test_acc = np.array(history['test_acc'])
    train_loss = np.array(history['train_loss'])
    
    # 找到收敛点（测试精度不再显著提升）
    best_idx = np.argmax(test_acc)
    
    # 计算收敛速度（达到最佳精度90%所需的epoch数）
    target_acc = test_acc[best_idx] * 0.9
    convergence_epoch = np.argmax(test_acc >= target_acc) + 1 if np.any(test_acc >= target_acc) else -1
    
    # 计算稳定性（最后10个epoch的测试精度标准差）
    stability = np.std(test_acc[-10:]) if len(test_acc) >= 10 else np.std(test_acc)
    
    # 计算泛化差距
    generalization_gap = train_acc[-1] - test_acc[-1]
    
    return {
        'best_accuracy': float(test_acc[best_idx]),
        'best_epoch': int(best_idx + 1),
        'convergence_epoch': int(convergence_epoch),
        'stability': float(stability),
        'generalization_gap': float(generalization_gap),
        'final_train_acc': float(train_acc[-1]),
        'final_test_acc': float(test_acc[-1]),
        'final_train_loss': float(train_loss[-1])
    }


def plot_training_comparison(results: Dict, save_dir: str):
    """绘制训练对比图"""
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    
    # 绘制不同alpha下的精度对比
    alphas = sorted(results['comparison'].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：清洁精度对比
    ax1 = axes[0]
    finetune_clean = [results['comparison'][a]['finetune_final_clean_acc'] for a in alphas]
    scratch_clean = [results['comparison'][a]['scratch_final_clean_acc'] for a in alphas]
    
    x = np.arange(len(alphas))
    width = 0.35
    
    ax1.bar(x - width/2, finetune_clean, width, label='Fine-tuning', color='steelblue')
    ax1.bar(x + width/2, scratch_clean, width, label='From Scratch', color='coral')
    
    ax1.set_xlabel('Alpha', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Clean Accuracy Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：噪声精度对比
    ax2 = axes[1]
    finetune_noisy = [results['comparison'][a]['finetune_final_noisy_acc'] for a in alphas]
    scratch_noisy = [results['comparison'][a]['scratch_final_noisy_acc'] for a in alphas]
    
    ax2.bar(x - width/2, finetune_noisy, width, label='Fine-tuning', color='steelblue')
    ax2.bar(x + width/2, scratch_noisy, width, label='From Scratch', color='coral')
    
    ax2.set_xlabel('Alpha', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Noisy Accuracy Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存至: {save_dir}/training_comparison.png")
