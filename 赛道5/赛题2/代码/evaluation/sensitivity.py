"""
敏感性分析模块

任务一：噪声敏感性分析
- 整网精度衰减趋势
- 单层输出分布偏移
- 误差逐层累积行为
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

from models.resnet import LayerOutputHook


def set_model_noise_strength(model: nn.Module, strength: float):
    """设置模型所有层的噪声强度"""
    for module in model.modules():
        if hasattr(module, 'set_noise_strength'):
            module.set_noise_strength(strength)


def sensitivity_analysis(
    model: nn.Module,
    test_loader,
    noise_strength_values: List[float],
    device: torch.device
) -> List[Dict]:
    """
    敏感性分析：整网精度衰减趋势
    """
    results = []
    model = model.to(device)

    for noise_strength in tqdm(noise_strength_values, desc='噪声敏感性分析'):
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
        results.append({'noise_strength': noise_strength, 'accuracy': acc})
        print(f'noise_strength={noise_strength:.2f}, accuracy={acc:.2f}%')

    return results


def layer_output_distribution_analysis(
    model: nn.Module,
    test_loader,
    noise_strength_values: List[float],
    device: torch.device,
    num_samples: int = 100
) -> Dict[str, Dict]:
    """
    单层输出分布偏移分析
    """
    model = model.to(device)
    hook = LayerOutputHook()
    hook.register_hook(model)

    results = {}

    print("获取基准输出分布...")
    set_model_noise_strength(model, 0.0)
    model.eval()

    baseline_outputs = {}
    sample_count = 0

    with torch.no_grad():
        for inputs, _ in test_loader:
            if sample_count >= num_samples:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            for name, output in hook.get_outputs().items():
                if name not in baseline_outputs:
                    baseline_outputs[name] = []
                baseline_outputs[name].append(output.cpu().numpy())
            sample_count += inputs.size(0)

    for name in baseline_outputs:
        baseline_outputs[name] = np.concatenate(baseline_outputs[name], axis=0)
        results[name] = {
            'baseline_mean': float(np.mean(baseline_outputs[name])),
            'baseline_std': float(np.std(baseline_outputs[name])),
            'noise_outputs': {}
        }

    hook.remove()

    for noise_strength in noise_strength_values:
        print(f"分析 noise_strength={noise_strength}...")
        set_model_noise_strength(model, noise_strength)

        hook = LayerOutputHook()
        hook.register_hook(model)
        model.eval()

        noise_outputs = {}
        sample_count = 0

        with torch.no_grad():
            for inputs, _ in test_loader:
                if sample_count >= num_samples:
                    break
                inputs = inputs.to(device)
                _ = model(inputs)
                for name, output in hook.get_outputs().items():
                    if name not in noise_outputs:
                        noise_outputs[name] = []
                    noise_outputs[name].append(output.cpu().numpy())
                sample_count += inputs.size(0)

        for name in noise_outputs:
            noise_outputs[name] = np.concatenate(noise_outputs[name], axis=0)
            results[name]['noise_outputs'][noise_strength] = {
                'mean': float(np.mean(noise_outputs[name])),
                'std': float(np.std(noise_outputs[name])),
                'kl_div': float(np.mean(
                    np.log(noise_outputs[name].std() + 1e-8) -
                    np.log(baseline_outputs[name].std() + 1e-8) +
                    (baseline_outputs[name].std()**2 + (baseline_outputs[name].mean() - noise_outputs[name].mean())**2) /
                    (2 * noise_outputs[name].std()**2 + 1e-8) - 0.5
                ))
            }

        hook.remove()

    return results


def comprehensive_sensitivity_analysis(
    model: nn.Module,
    test_loader,
    noise_strength_values: List[float],
    device: torch.device,
    save_dir: str = 'results/task1_sensitivity'
) -> Dict:
    """
    综合敏感性分析
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("任务一：噪声敏感性分析")
    print("=" * 60)

    accuracy_results = sensitivity_analysis(model, test_loader, noise_strength_values, device)

    save_path = os.path.join(save_dir, 'sensitivity_results.json')
    with open(save_path, 'w') as f:
        json.dump(accuracy_results, f, indent=2)
    print(f"结果已保存至: {save_path}")

    plot_sensitivity_curve(accuracy_results, save_dir)

    return {
        'accuracy_results': accuracy_results
    }


def plot_sensitivity_curve(results: List[Dict], save_dir: str):
    """绘制敏感性曲线"""
    noise_strengths = [r['noise_strength'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(noise_strengths, accuracies, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Noise Strength', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Noise Sensitivity Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(noise_strengths)
    plt.ylim(0, 100)

    for i, (x, y) in enumerate(zip(noise_strengths, accuracies)):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sensitivity_curve.png'), dpi=150)
    plt.close()
    print(f"敏感性曲线已保存至: {os.path.join(save_dir, 'sensitivity_curve.png')}")