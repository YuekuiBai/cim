"""
鲁棒性分析模块

任务三：鲁棒性增强方法
- STE-NAT: STE噪声感知训练
- STE-SAM: STE + Sharpness-Aware Minimization
- STE-OVF: STE + Optimal Vertex Fisher
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_model_noise_strength(model: nn.Module, strength: float):
    """设置模型所有层的噪声强度"""
    for module in model.modules():
        if hasattr(module, 'set_noise_strength'):
            module.set_noise_strength(strength)


def evaluate_robustness(
    model: nn.Module,
    test_loader,
    noise_strengths: List[float],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    评估模型在不同噪声强度下的鲁棒性
    """
    results = {f'noise_{ns}': [] for ns in noise_strengths}
    results['clean'] = []

    model = model.to(device)

    for noise_strength in noise_strengths:
        set_model_noise_strength(model, noise_strength)
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

        acc = 100. * correct / total
        results[f'noise_{noise_strength}'] = acc

    set_model_noise_strength(model, 0.0)
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

    results['clean'] = 100. * correct / total

    return results


def ste_sam_training(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epsilon: float = 0.1,
    rho: float = 0.5,
    epochs: int = 30
) -> Dict:
    """
    STE + SAM (Sharpness-Aware Minimization) 训练

    SAM通过寻找均匀平坦的损失 minima 来提高泛化能力
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'sam_radius': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data = p.data - epsilon * torch.sign(p.grad)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        _, test_acc = evaluate_single(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['sam_radius'].append(epsilon)

        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, '
              f'Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')

    return history


def ste_ovf_training(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    delta: float = 0.1,
    epochs: int = 30
) -> Dict:
    """
    STE + OVF (Optimal Vertex Fisher) 训练

    OVF通过在权重的顶点方向上进行扰动来提高鲁棒性
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        perturbation = delta * torch.sign(p.grad)
                        p.data = p.data + perturbation

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        _, test_acc = evaluate_single(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, '
              f'Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')

    return history


@torch.no_grad()
def evaluate_single(model: nn.Module, test_loader, device: torch.device) -> Tuple[float, float]:
    """评估单个模型"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss_sum += criterion(outputs, targets).item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return loss_sum / len(test_loader), 100. * correct / total


def comprehensive_robustness_analysis(
    baseline_model: nn.Module,
    train_loader,
    test_loader,
    nat_models: Dict,
    noise_strength_values: List[float],
    device: torch.device,
    save_dir: str = 'results/task3_robustness'
) -> Dict:
    """
    综合鲁棒性分析
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("任务三：鲁棒性增强方法分析")
    print("=" * 60)

    results = {}

    print("\n评估基准模型...")
    results['baseline'] = evaluate_robustness(baseline_model, test_loader, noise_strength_values, device)
    print(f"基准模型 Clean Accuracy: {results['baseline']['clean']:.2f}%")

    for name, model in nat_models.items():
        print(f"\n评估{name}模型...")
        results[name] = evaluate_robustness(model, test_loader, noise_strength_values, device)
        print(f"{name} Clean Accuracy: {results[name]['clean']:.2f}%")

    plot_robustness_comparison(results, noise_strength_values, save_dir)

    save_path = os.path.join(save_dir, 'robustness_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n结果已保存至: {save_path}")

    return results


def plot_robustness_comparison(results: Dict, noise_strengths: List[float], save_dir: str):
    """绘制鲁棒性对比图"""
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    for idx, (name, values) in enumerate(results.items()):
        clean_acc = values['clean']
        noise_accs = [values.get(f'noise_{ns}', clean_acc) for ns in noise_strengths]

        x_vals = [0] + noise_strengths
        y_vals = [clean_acc] + noise_accs

        label = name.replace('_', ' ').title()
        plt.plot(x_vals, y_vals, color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)], linewidth=2, markersize=8, label=label)

    plt.xlabel('Noise Strength', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Robustness Comparison', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks([0] + noise_strengths)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_comparison.png'), dpi=150)
    plt.close()
    print(f"鲁棒性对比图已保存至: {os.path.join(save_dir, 'robustness_comparison.png')}")