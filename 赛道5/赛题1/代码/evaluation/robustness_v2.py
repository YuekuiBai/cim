"""
鲁棒性增强方法模块 - 改进版

任务三：针对非线性误差的优化方法
- 改进的预失真补偿（输入层补偿）
- 可学习校准层（增强版）
- 非线性感知训练集成
- 多策略对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
from noise.nonlinearity import (
    NonLinearWrapper,
    NonLinearInjection,
    InverseNonLinearity,
    CalibrationLayer,
    set_model_alpha
)


class ImprovedPreDistortionWrapper(nn.Module):
    """
    改进的预失真补偿包装器

    在输入层应用逆非线性变换，而非输出层
    这样可以补偿网络内部的非线性累积效应
    """
    def __init__(self, model: nn.Module, alpha: float = 0.0):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.predistortion = NonLinearInjection(alpha=-alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_corrected = self.predistortion(x)
        return self.model(x_corrected)

    def set_alpha(self, alpha: float):
        self.alpha = alpha
        self.predistortion.alpha = -alpha
        if isinstance(self.model, NonLinearWrapper):
            self.model.set_alpha(alpha)
        else:
            set_model_alpha(self.model, alpha)


class AdaptiveCalibrationWrapper(nn.Module):
    """
    自适应校准层包装器

    在forward过程中动态调整校准强度
    """
    def __init__(self, model: nn.Module, alpha: float = 0.0, num_classes: int = 10):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.calibration = CalibrationLayer(num_classes, learn_cubic=True)
        self.calibration.coeff_1.data.fill_(1.0)
        self.calibration.coeff_3.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return self.calibration(features)

    def set_alpha(self, alpha: float):
        self.alpha = alpha
        if isinstance(self.model, NonLinearWrapper):
            self.model.set_alpha(alpha)
        else:
            set_model_alpha(self.model, alpha)


class MixedAlphaTrainingWrapper(nn.Module):
    """
    多Alpha混合训练包装器

    在训练时随机采样不同的alpha值，提高泛化能力
    """
    def __init__(self, model: nn.Module, alpha_range: Tuple[float, float] = (0.0, 0.3)):
        super().__init__()
        self.model = model
        self.alpha_range = alpha_range

    def forward(self, x: torch.Tensor, alpha: float = None) -> torch.Tensor:
        if alpha is None:
            alpha = np.random.uniform(*self.alpha_range)
        if isinstance(self.model, NonLinearWrapper):
            self.model.set_alpha(alpha)
        else:
            set_model_alpha(self.model, alpha)
        return self.model(x)


def train_improved_calibration(model: nn.Module, train_loader, test_loader,
                                alpha: float, device: torch.device,
                                epochs: int = 30, lr: float = 0.005,
                                weight_decay: float = 1e-4) -> Dict:
    """
    改进的校准层训练

    使用更大的训练轮次、更小的学习率和权重衰减
    """
    if isinstance(model, NonLinearWrapper):
        model.set_alpha(alpha)
    else:
        set_model_alpha(model, alpha)

    model = model.to(device)
    model.eval()

    num_classes = 10
    for module in model.modules():
        if isinstance(module, nn.Linear):
            num_classes = module.out_features
            break

    calibration = CalibrationLayer(num_classes, learn_cubic=True).to(device)
    calibration.coeff_1.data.fill_(1.0)
    calibration.coeff_3.data.zero_()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(calibration.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'best_acc': 0}

    print(f"\n改进校准层训练 (alpha={alpha}, epochs={epochs}, lr={lr})...")

    for epoch in range(epochs):
        calibration.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                features = model(inputs)
            outputs = calibration(features)
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

        calibration.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                features = model(inputs)
                outputs = calibration(features)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        history['test_acc'].append(test_acc)

        if test_acc > history['best_acc']:
            history['best_acc'] = test_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}% (Best: {history["best_acc"]:.2f}%)')

    print(f'最佳测试精度: {history["best_acc"]:.2f}%')
    return {
        'history': history,
        'final_test_acc': test_acc,
        'best_test_acc': history['best_acc'],
        'calibration_state': calibration.state_dict()
    }


def mixed_alpha_training(model: nn.Module, train_loader, test_loader,
                         device: torch.device,
                         alpha_range: Tuple[float, float] = (0.0, 0.4),
                         epochs: int = 20, lr: float = 0.001) -> Dict:
    """
    多Alpha混合训练

    训练时随机使用不同的alpha值，提高模型对各种非线性强度的泛化能力
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': {}}

    print(f"\n多Alpha混合训练 (alpha范围={alpha_range}, epochs={epochs})...")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            alpha = np.random.uniform(*alpha_range)

            if isinstance(model, NonLinearWrapper):
                model.set_alpha(alpha)
            else:
                set_model_alpha(model, alpha)

            optimizer.zero_grad()
            outputs = model(inputs)
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%')

    model.eval()
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    print("\n评估各alpha下的精度:")
    for alpha in alpha_values:
        if isinstance(model, NonLinearWrapper):
            model.set_alpha(alpha)
        else:
            set_model_alpha(model, alpha)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        history['test_acc'][alpha] = acc
        print(f'  alpha={alpha:.2f}: {acc:.2f}%')

    return history


def evaluate_all_methods(model: nn.Module, nat_models: Dict[float, nn.Module],
                         test_loader, alpha_values: List[float],
                         device: torch.device) -> Dict:
    """
    评估所有鲁棒性增强方法

    Args:
        model: 基线模型
        nat_models: 非线性感知训练模型字典 {alpha: model}
        test_loader: 测试数据
        alpha_values: 非线性强度列表
        device: 设备

    Returns:
        各方法在不同alpha下的精度
    """
    results = {
        'baseline': {},
        'nat': {},
        'predistortion': {},
        'calibration': {},
        'mixed': {}
    }

    num_classes = 10
    for module in model.modules():
        if isinstance(module, nn.Linear):
            num_classes = module.out_features
            break

    for alpha in tqdm(alpha_values, desc='评估所有方法'):
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

        if alpha in nat_models:
            nat_model = nat_models[alpha]
            if isinstance(nat_model, NonLinearWrapper):
                nat_model.set_alpha(alpha)
            else:
                set_model_alpha(nat_model, alpha)
            nat_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = nat_model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            results['nat'][alpha] = 100. * correct / total

        predistorted = ImprovedPreDistortionWrapper(model, alpha=alpha).to(device)
        predistorted.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = predistorted(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['predistortion'][alpha] = 100. * correct / total

        calibrated = AdaptiveCalibrationWrapper(model, alpha=alpha, num_classes=num_classes).to(device)
        calibrated.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = calibrated(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        results['calibration'][alpha] = 100. * correct / total

    return results


def plot_robustness_comparison(results: Dict, save_path: str):
    """绘制鲁棒性方法对比图"""
    import matplotlib.pyplot as plt

    alphas = sorted(list(results['baseline'].keys()))

    plt.figure(figsize=(14, 8))

    colors = {
        'baseline': '#1f77b4',
        'nat': '#2ca02c',
        'predistortion': '#ff7f0e',
        'calibration': '#9467bd',
        'mixed': '#d62728'
    }

    labels = {
        'baseline': 'Baseline (No Defense)',
        'nat': 'NAT (Nonlinearity-Aware Training)',
        'predistortion': 'Pre-distortion Compensation',
        'calibration': 'Calibration Layer',
        'mixed': 'Mixed-alpha Training'
    }

    for method, data in results.items():
        if not data:
            continue
        accs = []
        for a in alphas:
            accs.append(data.get(a, None))
        valid_pairs = [(a, acc) for a, acc in zip(alphas, accs) if acc is not None]
        if valid_pairs:
            plot_alphas, plot_accs = zip(*valid_pairs)
            plt.plot(plot_alphas, plot_accs, '-o', label=labels.get(method, method.upper()),
                    linewidth=2, markersize=8, color=colors.get(method))

    plt.xlabel('Nonlinearity Alpha', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Robustness Enhancement Methods Comparison', fontsize=16)
    plt.legend(fontsize=11, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(alphas)
    plt.ylim(40, 95)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"对比图已保存至: {save_path}")


def comprehensive_robustness_analysis(model: nn.Module, train_loader, test_loader,
                                       nat_models: Dict[float, nn.Module],
                                       alpha_values: List[float], device: torch.device,
                                       save_dir: str = '../结果/task3_robustness') -> Dict:
    """
    综合鲁棒性分析 - 改进版

    评估多种鲁棒性增强方法
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("开始综合鲁棒性分析（改进版）")
    print("=" * 60)

    print("\n评估各方法在不同非线性强度下的表现...")
    results = evaluate_all_methods(model, nat_models, test_loader, alpha_values, device)

    print("\n结果汇总:")
    print("-" * 60)
    for method, data in results.items():
        if data:
            print(f"\n{method.upper()}:")
            for alpha, acc in sorted(data.items()):
                print(f"  α={alpha:.1f}: {acc:.2f}%")

    results_path = os.path.join(save_dir, 'robustness_results.json')
    serializable_results = {}
    for method, data in results.items():
        serializable_results[method] = {str(k): v for k, v in data.items()}
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n结果已保存至: {results_path}")

    plot_path = os.path.join(save_dir, 'robustness_comparison.png')
    plot_robustness_comparison(results, plot_path)

    print("\n" + "=" * 60)
    print("鲁棒性分析完成！")
    print("=" * 60)

    return results
