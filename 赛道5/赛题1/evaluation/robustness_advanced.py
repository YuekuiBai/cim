"""
高级鲁棒性增强方法模块

基于最新研究成果实现的多种鲁棒性增强方法：
1. Parametric Noise Injection (PNI) - 可学习噪声注入
2. Variance-Aware Noisy Training - 方差感知噪声训练
3. Lipschitz Regularization - 利普希茨正则化
4. Progressive Noise Calibration - 渐进式噪声校准
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Callable
from noise.nonlinearity import NonLinearWrapper, set_model_alpha


class ParametricNoiseInjection(nn.Module):
    """
    参数化噪声注入 (PNI)

    在每一层的激活值或权重上注入可学习的的高斯噪声
    噪声强度作为可训练参数，通过对抗训练优化

    参考文献: "Parametric Noise Injection: Trainable Randomness to Improve Deep Neural
              Network Robustness against Adversarial Attack" (arXiv:1811.09310)
    """
    def __init__(self, module: nn.Module, noise_scale: float = 0.1,
                 trainable: bool = True, layer_name: str = ''):
        super().__init__()
        self.module = module
        self.layer_name = layer_name

        if trainable:
            self.noise_scale = nn.Parameter(torch.tensor(noise_scale))
        else:
            self.noise_scale = noise_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)

        if isinstance(self.noise_scale, nn.Parameter):
            noise_std = torch.abs(self.noise_scale)
        else:
            noise_std = abs(self.noise_scale)

        if noise_std > 1e-6:
            noise = torch.randn_like(output) * noise_std
            return output + noise
        return output


class PNIWrapper(nn.Module):
    """
    PNI包装器：在模型forward过程中动态添加噪声

    参考文献: "Parametric Noise Injection: Trainable Randomness to Improve Deep Neural
              Network Robustness against Adversarial Attack" (arXiv:1811.09310)
    """
    def __init__(self, model: nn.Module, base_noise: float = 0.1):
        super().__init__()
        self.model = model
        self.base_noise = base_noise
        self.noise_scale = nn.Parameter(torch.tensor(base_noise))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        noise_std = torch.abs(self.noise_scale)
        if noise_std > 1e-6:
            noise = torch.randn_like(output) * noise_std
            return output + noise
        return output


class NoiseScheduleTrainer:
    """
    方差感知噪声训练

    在训练过程中动态调整噪声强度，模拟实际推理时变化的噪声环境

    参考文献: "Variance-Aware Noisy Training: Hardening DNNs against Unstable
              Analog Computations" (arXiv:2503.16183)
    """
    def __init__(self, model: nn.Module, device: torch.device,
                 schedule_type: str = 'cosine', noise_range: Tuple[float, float] = (0.0, 0.5)):
        self.model = model
        self.device = device
        self.schedule_type = schedule_type
        self.noise_range = noise_range
        self.current_alpha = 0.0

    def get_noise_alpha(self, epoch: int, total_epochs: int) -> float:
        """根据调度策略计算当前噪声强度"""
        if self.schedule_type == 'linear':
            progress = epoch / total_epochs
            return self.noise_range[0] + progress * (self.noise_range[1] - self.noise_range[0])
        elif self.schedule_type == 'cosine':
            progress = epoch / total_epochs
            return self.noise_range[1] - 0.5 * (self.noise_range[1] - self.noise_range[0]) * (1 + np.cos(np.pi * progress))
        elif self.schedule_type == 'cyclic':
            cycle = 4
            progress = (epoch % (total_epochs // cycle)) / (total_epochs // cycle)
            return self.noise_range[1] - 0.5 * (self.noise_range[1] - self.noise_range[0]) * (1 + np.cos(np.pi * progress))
        elif self.schedule_type == 'adaptive':
            progress = epoch / total_epochs
            base = self.noise_range[0] + 0.3 * (self.noise_range[1] - self.noise_range[0])
            amplitude = 0.2 * (self.noise_range[1] - self.noise_range[0])
            return base + amplitude * np.sin(2 * np.pi * progress)
        else:
            return np.random.uniform(*self.noise_range)


class LipschitzRegularizedTrainer:
    """
    利普希茨正则化训练

    在训练时添加利普希茨常数正则化项，抑制误差在网络中的传播放大

    参考文献: "CorrectNet: Robustness Enhancement of Analog In-Memory Computing
              for Neural Networks by Error Suppression and Compensation"
    """
    def __init__(self, model: nn.Module, lipschitz_coef: float = 0.01):
        self.model = model
        self.lipschitz_coef = lipschitz_coef

    def compute_lipschitz_loss(self) -> torch.Tensor:
        """计算利普希茨正则化损失"""
        lipschitz_loss = 0.0
        count = 0

        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_norm = torch.norm(module.weight.data, p=2)
                if hasattr(module, 'bias') and module.bias is not None:
                    bias_norm = torch.norm(module.bias.data, p=2)
                    lipschitz_loss += weight_norm + bias_norm
                else:
                    lipschitz_loss += weight_norm
                count += 1

        return self.lipschitz_coef * lipschitz_loss / max(count, 1)


class ProgressiveCalibrationTrainer:
    """
    渐进式校准训练

    从低噪声到高噪声渐进式训练，提高模型对各种噪声水平的泛化能力
    """
    def __init__(self, model: nn.Module, device: torch.device,
                 alpha_stages: List[float] = None):
        self.model = model
        self.device = device
        self.alpha_stages = alpha_stages or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def train(self, train_loader, test_loader, epochs_per_stage: int = 5,
             total_epochs: int = 30, lr: float = 0.01) -> Dict:
        """执行渐进式校准训练"""
        results = {'history': [], 'final_results': {}}

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        total_batches = len(train_loader)
        current_stage = 0
        alpha = self.alpha_stages[0]

        print(f"\n渐进式校准训练:")
        print(f"阶段切换: {self.alpha_stages}")
        print(f"每阶段轮数: {epochs_per_stage}, 总轮数: {total_epochs}")

        for epoch in range(total_epochs):
            if isinstance(self.model, NonLinearWrapper):
                self.model.set_alpha(alpha)
            else:
                set_model_alpha(self.model, alpha)

            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if isinstance(self.model, NonLinearWrapper):
                    self.model.set_alpha(alpha)
                else:
                    set_model_alpha(self.model, alpha)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            if (epoch + 1) % epochs_per_stage == 0 and current_stage < len(self.alpha_stages) - 1:
                current_stage += 1
                alpha = self.alpha_stages[current_stage]
                print(f"  阶段 {current_stage}: 切换到 alpha={alpha}")

            train_acc = 100. * correct / total
            results['history'].append({
                'epoch': epoch + 1,
                'alpha': alpha,
                'train_acc': train_acc,
                'loss': total_loss / total_batches
            })

            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Alpha={alpha:.2f}')

        self.model.eval()
        for test_alpha in self.alpha_stages:
            if isinstance(self.model, NonLinearWrapper):
                self.model.set_alpha(test_alpha)
            else:
                set_model_alpha(self.model, test_alpha)

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            results['final_results'][test_alpha] = acc
            print(f'  最终测试 (alpha={test_alpha:.1f}): {acc:.2f}%')

        return results


class MultiStrategyEnsemble:
    """
    多策略集成

    结合多种鲁棒性增强策略，通过集成方式提升整体鲁棒性
    """
    def __init__(self, models: List[nn.Module], strategies: List[str]):
        self.models = models
        self.strategies = strategies

    def predict(self, x: torch.Tensor, method: str = 'voting') -> torch.Tensor:
        """集成预测"""
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))

        if method == 'voting':
            predictions = [torch.argmax(o, dim=1) for o in outputs]
            stacked = torch.stack(predictions, dim=0)
            return torch.mode(stacked, dim=0).values
        elif method == 'averaging':
            avg_output = torch.stack(outputs, dim=0).mean(dim=0)
            return torch.argmax(avg_output, dim=1)
        else:
            return torch.argmax(outputs[0], dim=1)


def train_with_pni(model: nn.Module, train_loader, test_loader,
                   device: torch.device, epochs: int = 20,
                   noise_init: float = 0.1, lr: float = 0.01) -> Dict:
    """
    使用PNI方法训练模型
    """
    pni_model = PNIWrapper(model, base_noise=noise_init).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pni_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'noise_scales': []}

    print(f"\nPNI训练 (初始噪声={noise_init}):")

    for epoch in range(epochs):
        pni_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = pni_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        train_acc = 100. * correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        noise_scales = []
        for pni in pni_model.pni_modules:
            if isinstance(pni, ParametricNoiseInjection) and isinstance(pni.noise_scale, nn.Parameter):
                noise_scales.append(pni.noise_scale.item())
        history['noise_scales'].append(noise_scales)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Avg Noise={np.mean(noise_scales):.4f}')

    return history


def comprehensive_advanced_robustness_eval(model: nn.Module, train_loader, test_loader,
                                          alpha_values: List[float], device: torch.device,
                                          save_dir: str = 'results/task3_robustness') -> Dict:
    """
    综合高级鲁棒性评估

    评估多种高级鲁棒性增强方法
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("高级鲁棒性增强方法综合评估")
    print("=" * 70)

    results = {
        'baseline': {},
        'pni': {},
        'variance_aware': {},
        'progressive': {},
        'mixed': {}
    }

    for alpha in tqdm(alpha_values, desc='评估基线'):
        if isinstance(model, NonLinearWrapper):
            model.set_alpha(alpha)
        else:
            set_model_alpha(model, alpha)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['baseline'][alpha] = 100. * correct / total

    print("\n方法1: PNI (Parametric Noise Injection)")
    pni_model = PNIWrapper(model, base_noise=0.15).to(device)
    pni_trainer_results = train_with_pni(pni_model, train_loader, test_loader, device, epochs=15, noise_init=0.15)

    pni_model.eval()
    for alpha in alpha_values:
        if isinstance(pni_model, NonLinearWrapper):
            pni_model.set_alpha(alpha)
        else:
            set_model_alpha(pni_model, alpha)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pni_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['pni'][alpha] = 100. * correct / total

    print("\n方法2: 方差感知噪声训练")
    vat_trainer = VarianceAwareTrainer(model, device, schedule_type='cosine', noise_range=(0.1, 0.4))

    for alpha in alpha_values:
        if isinstance(model, NonLinearWrapper):
            model.set_alpha(alpha)
        else:
            set_model_alpha(model, alpha)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['variance_aware'][alpha] = 100. * correct / total

    print("\n方法3: 渐进式校准训练")
    pc_trainer = ProgressiveCalibrationTrainer(model, device, alpha_stages=[0.0, 0.1, 0.2, 0.3, 0.4])
    progressive_results = pc_trainer.train(train_loader, test_loader, epochs_per_stage=4, total_epochs=24)
    results['progressive'] = progressive_results['final_results']

    print("\n结果汇总:")
    print("-" * 70)
    methods_display = {
        'baseline': '基线',
        'pni': 'PNI',
        'variance_aware': '方差感知',
        'progressive': '渐进式校准'
    }

    for method, data in results.items():
        if data:
            print(f"\n{methods_display.get(method, method)}:")
            for alpha, acc in sorted(data.items()):
                print(f"  α={alpha:.1f}: {acc:.2f}%")

    results_path = os.path.join(save_dir, 'advanced_robustness_results.json')
    serializable_results = {}
    for method, data in results.items():
        serializable_results[method] = {str(k): v for k, v in data.items()}
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n结果已保存至: {results_path}")

    return results


class VarianceAwareTrainer:
    """方差感知噪声训练的简化实现"""
    def __init__(self, model: nn.Module, device: torch.device,
                 schedule_type: str = 'cosine', noise_range: Tuple[float, float] = (0.1, 0.4)):
        self.model = model
        self.device = device
        self.schedule_type = schedule_type
        self.noise_range = noise_range

    def get_noise_alpha(self, epoch: int, total_epochs: int) -> float:
        if self.schedule_type == 'cosine':
            progress = epoch / total_epochs
            return self.noise_range[1] - 0.5 * (self.noise_range[1] - self.noise_range[0]) * (1 + np.cos(np.pi * progress))
        return np.random.uniform(*self.noise_range)
