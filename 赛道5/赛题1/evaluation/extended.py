"""
拓展研究模块

拓展研究方向：
- 网络结构与参数量对非线性误差影响
- 对比分析随机高斯噪声注入与非线性失真的不同影响机制
- 结合量化误差与非线性误差对模型推理精度的影响分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import os
import json
from noise.nonlinearity import NonLinearWrapper, set_model_alpha, NonLinearInjection


class GaussianNoiseInjection(nn.Module):
    """高斯噪声注入模块"""
    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.std == 0 or not self.training:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise


class QuantizationNoise(nn.Module):
    """量化噪声模块"""
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.scale = 2 ** (bits - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 模拟量化-反量化过程
        x_clipped = torch.clamp(x, -self.scale, self.scale - 1)
        x_quantized = torch.round(x_clipped)
        return x_quantized


class CombinedNoise(nn.Module):
    """组合噪声模块：量化 + 非线性"""
    def __init__(self, alpha: float = 0.0, bits: int = 8):
        super().__init__()
        self.nonlinearity = NonLinearInjection(alpha=alpha)
        self.quantization = QuantizationNoise(bits=bits)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlinearity(x)
        x = self.quantization(x)
        return x


def compare_noise_types(model: nn.Module, test_loader, 
                        alpha_values: List[float], 
                        noise_std_values: List[float],
                        device: torch.device) -> Dict:
    """
    对比非线性失真与高斯噪声的影响
    
    Args:
        model: 模型
        test_loader: 测试数据
        alpha_values: 非线性强度列表
        noise_std_values: 高斯噪声标准差列表
        device: 设备
    
    Returns:
        对比结果
    """
    model = model.to(device)
    model.eval()
    
    results = {
        'nonlinearity': {},
        'gaussian': {},
        'comparison': {}
    }
    
    # 1. 评估非线性失真
    print("\n评估非线性失真影响...")
    for alpha in tqdm(alpha_values, desc='非线性评估'):
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
        
        results['nonlinearity'][alpha] = 100. * correct / total
    
    # 2. 评估高斯噪声
    print("\n评估高斯噪声影响...")
    # 重置模型非线性
    if isinstance(model, NonLinearWrapper):
        model.set_alpha(0.0)
    else:
        set_model_alpha(model, 0.0)
    
    for std in tqdm(noise_std_values, desc='高斯噪声评估'):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # 注入高斯噪声
                noisy_inputs = inputs + torch.randn_like(inputs) * std
                outputs = model(noisy_inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        results['gaussian'][std] = 100. * correct / total
    
    # 3. 分析对比
    # 找到造成相似精度下降的噪声强度对
    for alpha in alpha_values:
        nonlin_acc = results['nonlinearity'][alpha]
        # 找到最接近的高斯噪声强度
        closest_std = min(noise_std_values, 
                         key=lambda s: abs(results['gaussian'][s] - nonlin_acc))
        results['comparison'][alpha] = {
            'nonlinearity_acc': nonlin_acc,
            'equivalent_gaussian_std': closest_std,
            'gaussian_acc': results['gaussian'][closest_std]
        }
    
    return results


def analyze_quantization_nonlinearity(model: nn.Module, test_loader,
                                       alpha_values: List[float],
                                       quant_bits: List[int],
                                       device: torch.device) -> Dict:
    """
    分析量化误差与非线性误差的组合影响
    
    Args:
        model: 模型
        test_loader: 测试数据
        alpha_values: 非线性强度列表
        quant_bits: 量化位数列表
        device: 设备
    
    Returns:
        组合影响分析结果
    """
    model = model.to(device)
    model.eval()
    
    results = {
        'quantization_only': {},
        'combined': {}
    }
    
    # 1. 仅量化误差
    print("\n评估量化误差影响...")
    for bits in tqdm(quant_bits, desc='量化评估'):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # 模拟量化
                scale = 2 ** (bits - 1)
                inputs_q = torch.clamp(inputs * scale, -scale, scale - 1)
                inputs_q = torch.round(inputs_q) / scale
                
                outputs = model(inputs_q)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        results['quantization_only'][bits] = 100. * correct / total
    
    # 2. 量化 + 非线性组合
    print("\n评估量化+非线性组合影响...")
    for bits in quant_bits:
        results['combined'][bits] = {}
        for alpha in tqdm(alpha_values, desc=f'{bits}bit量化'):
            if isinstance(model, NonLinearWrapper):
                model.set_alpha(alpha)
            else:
                set_model_alpha(model, alpha)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    # 量化
                    scale = 2 ** (bits - 1)
                    inputs_q = torch.clamp(inputs * scale, -scale, scale - 1)
                    inputs_q = torch.round(inputs_q) / scale
                    
                    outputs = model(inputs_q)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            results['combined'][bits][alpha] = 100. * correct / total
    
    return results


def analyze_model_architecture(model_fns: Dict[str, callable], 
                                test_loader,
                                alpha_values: List[float],
                                device: torch.device) -> Dict:
    """
    分析不同网络结构对非线性误差的敏感性
    
    Args:
        model_fns: 模型名称到创建函数的映射
        test_loader: 测试数据
        alpha_values: 非线性强度列表
        device: 设备
    
    Returns:
        各模型的敏感性分析结果
    """
    results = {}
    
    for model_name, model_fn in model_fns.items():
        print(f"\n分析模型: {model_name}")
        model = model_fn()
        model = model.to(device)
        model.eval()
        
        results[model_name] = {'alpha_results': {}}
        
        for alpha in tqdm(alpha_values, desc=model_name):
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
            
            results[model_name]['alpha_results'][alpha] = 100. * correct / total
        
        # 计算敏感性指标
        accs = list(results[model_name]['alpha_results'].values())
        results[model_name]['sensitivity'] = {
            'baseline_acc': accs[0] if accs else 0,
            'max_drop': accs[0] - min(accs) if accs else 0,
            'drop_rate': (accs[0] - min(accs)) / accs[0] * 100 if accs and accs[0] > 0 else 0
        }
    
    return results


def plot_noise_comparison(results: Dict, save_path: str):
    """绘制噪声类型对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：非线性 vs 高斯噪声精度曲线
    ax1 = axes[0]
    
    alphas = sorted(results['nonlinearity'].keys())
    nonlin_accs = [results['nonlinearity'][a] for a in alphas]
    ax1.plot(alphas, nonlin_accs, 'b-o', label='Nonlinearity', linewidth=2, markersize=8)
    
    stds = sorted(results['gaussian'].keys())
    gauss_accs = [results['gaussian'][s] for s in stds]
    ax1.plot(stds, gauss_accs, 'r-s', label='Gaussian Noise', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Noise Intensity (α or σ)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Nonlinearity vs Gaussian Noise Impact', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 右图：等效噪声强度映射
    ax2 = axes[1]
    
    alphas = sorted(results['comparison'].keys())
    equiv_stds = [results['comparison'][a]['equivalent_gaussian_std'] for a in alphas]
    
    ax2.bar(range(len(alphas)), equiv_stds, color='steelblue')
    ax2.set_xlabel('Nonlinearity Alpha', fontsize=12)
    ax2.set_ylabel('Equivalent Gaussian Std', fontsize=12)
    ax2.set_title('Equivalent Noise Intensity Mapping', fontsize=14)
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"噪声对比图已保存至: {save_path}")


def plot_quantization_analysis(results: Dict, save_path: str):
    """绘制量化分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：不同量化位数下的精度
    ax1 = axes[0]
    
    bits = sorted(results['quantization_only'].keys())
    quant_accs = [results['quantization_only'][b] for b in bits]
    
    ax1.bar(range(len(bits)), quant_accs, color='coral')
    ax1.set_xlabel('Quantization Bits', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Quantization Only Impact', fontsize=14)
    ax1.set_xticks(range(len(bits)))
    ax1.set_xticklabels([f'{b}bit' for b in bits])
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：量化+非线性组合影响
    ax2 = axes[1]
    
    bits = sorted(results['combined'].keys())
    alphas = sorted(results['combined'][bits[0]].keys()) if bits else []
    
    for bits_val in bits:
        accs = [results['combined'][bits_val].get(a, 0) for a in alphas]
        ax2.plot(alphas, accs, '-o', label=f'{bits_val}bit', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Nonlinearity Alpha', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Quantization + Nonlinearity Combined Impact', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"量化分析图已保存至: {save_path}")


def plot_architecture_comparison(results: Dict, save_path: str):
    """绘制模型架构对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    model_names = list(results.keys())
    
    # 左图：各模型精度曲线
    ax1 = axes[0]
    
    for name in model_names:
        alphas = sorted(results[name]['alpha_results'].keys())
        accs = [results[name]['alpha_results'][a] for a in alphas]
        ax1.plot(alphas, accs, '-o', label=name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Nonlinearity Alpha', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Architecture Sensitivity Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 右图：敏感性指标对比
    ax2 = axes[1]
    
    sensitivities = [results[name]['sensitivity']['drop_rate'] for name in model_names]
    
    ax2.bar(range(len(model_names)), sensitivities, color='steelblue')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy Drop Rate (%)', fontsize=12)
    ax2.set_title('Model Sensitivity to Nonlinearity', fontsize=14)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"架构对比图已保存至: {save_path}")


def comprehensive_extended_analysis(model: nn.Module, test_loader,
                                     alpha_values: List[float],
                                     device: torch.device,
                                     save_dir: str = 'results/extended') -> Dict:
    """
    综合拓展分析
    
    执行所有拓展研究实验
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始综合拓展研究分析")
    print("=" * 60)
    
    all_results = {}
    
    # 1. 噪声类型对比
    print("\n[1/3] 噪声类型对比分析...")
    noise_results = compare_noise_types(
        model, test_loader, alpha_values,
        noise_std_values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        device=device
    )
    plot_noise_comparison(noise_results, os.path.join(save_dir, 'noise_comparison.png'))
    all_results['noise_comparison'] = noise_results
    
    # 2. 量化+非线性组合分析
    print("\n[2/3] 量化+非线性组合分析...")
    quant_results = analyze_quantization_nonlinearity(
        model, test_loader, alpha_values,
        quant_bits=[8, 6, 4, 2],
        device=device
    )
    plot_quantization_analysis(quant_results, os.path.join(save_dir, 'quantization_analysis.png'))
    all_results['quantization_analysis'] = quant_results
    
    # 3. 保存结果
    with open(os.path.join(save_dir, 'extended_results.json'), 'w') as f:
        # 转换不可序列化的值
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)
    
    print("\n" + "=" * 60)
    print("拓展研究分析完成！")
    print(f"结果保存在: {save_dir}")
    print("=" * 60)
    
    return all_results
