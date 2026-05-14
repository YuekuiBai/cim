"""
敏感性分析模块

任务一：非线性误差的敏感性分析
- 整网精度衰减趋势
- 单层输出分布偏移
- 误差在网络中的逐层累积行为
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import os
from noise.nonlinearity import NonLinearWrapper, set_model_alpha
from models.resnet import LayerOutputHook


def sensitivity_analysis(model: nn.Module, test_loader, alpha_values: List[float], 
                         device: torch.device) -> List[Dict]:
    """
    敏感性分析：整网精度衰减趋势
    
    Args:
        model: 模型（已包装NonLinearWrapper）
        test_loader: 测试数据加载器
        alpha_values: 非线性强度列表
        device: 设备
    
    Returns:
        结果列表，每个元素包含alpha和accuracy
    """
    results = []
    model = model.to(device)
    
    for alpha in tqdm(alpha_values, desc='整网精度敏感性分析'):
        # 设置非线性强度
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
        
        acc = 100. * correct / total
        results.append({'alpha': alpha, 'accuracy': acc})
        print(f'alpha={alpha:.2f}, accuracy={acc:.2f}%')
    
    return results


def layer_output_distribution_analysis(model: nn.Module, test_loader, 
                                        alpha_values: List[float], device: torch.device,
                                        num_samples: int = 100) -> Dict[str, Dict]:
    """
    单层输出分布偏移分析
    
    分析不同非线性强度下各层输出的分布变化
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        alpha_values: 非线性强度列表
        device: 设备
        num_samples: 分析样本数
    
    Returns:
        各层在不同alpha下的输出统计信息
    """
    model = model.to(device)
    
    # 注册钩子获取各层输出
    hook = LayerOutputHook()
    hook.register_hook(model)
    
    results = {}
    
    # 先获取基准输出（alpha=0）
    print("获取基准输出分布...")
    if isinstance(model, NonLinearWrapper):
        model.set_alpha(0.0)
    else:
        set_model_alpha(model, 0.0)
    
    model.eval()
    baseline_outputs = {}
    sample_count = 0
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            if sample_count >= num_samples:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            
            for layer_name, output in hook.get_outputs().items():
                if layer_name not in baseline_outputs:
                    baseline_outputs[layer_name] = []
                baseline_outputs[layer_name].append(output.cpu().numpy().flatten())
            
            hook.clear()
            sample_count += inputs.size(0)
    
    # 计算基准统计量
    for layer_name in baseline_outputs:
        all_outputs = np.concatenate(baseline_outputs[layer_name])
        results[layer_name] = {
            'baseline_mean': float(np.mean(all_outputs)),
            'baseline_std': float(np.std(all_outputs)),
            'baseline_min': float(np.min(all_outputs)),
            'baseline_max': float(np.max(all_outputs)),
            'alpha_stats': {}
        }
    
    # 分析不同alpha下的分布偏移
    for alpha in tqdm(alpha_values, desc='单层分布偏移分析'):
        if alpha == 0:
            continue
            
        if isinstance(model, NonLinearWrapper):
            model.set_alpha(alpha)
        else:
            set_model_alpha(model, alpha)
        
        hook.clear()
        sample_count = 0
        noisy_outputs = {}
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                if sample_count >= num_samples:
                    break
                inputs = inputs.to(device)
                _ = model(inputs)
                
                for layer_name, output in hook.get_outputs().items():
                    if layer_name not in noisy_outputs:
                        noisy_outputs[layer_name] = []
                    noisy_outputs[layer_name].append(output.cpu().numpy().flatten())
                
                hook.clear()
                sample_count += inputs.size(0)
        
        # 计算分布偏移指标
        for layer_name in noisy_outputs:
            all_outputs = np.concatenate(noisy_outputs[layer_name])
            results[layer_name]['alpha_stats'][alpha] = {
                'mean': float(np.mean(all_outputs)),
                'std': float(np.std(all_outputs)),
                'min': float(np.min(all_outputs)),
                'max': float(np.max(all_outputs)),
                'mean_shift': float(np.mean(all_outputs) - results[layer_name]['baseline_mean']),
                'std_change': float(np.std(all_outputs) - results[layer_name]['baseline_std']),
            }
    
    hook.remove_hooks()
    return results


def layer_error_accumulation_analysis(model: nn.Module, test_loader, 
                                       alpha_values: List[float], device: torch.device,
                                       num_samples: int = 50) -> Dict[str, Dict]:
    """
    误差逐层累积行为分析
    
    分析非线性误差在网络中如何逐层累积
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        alpha_values: 非线性强度列表
        device: 设备
        num_samples: 分析样本数
    
    Returns:
        各层的误差累积信息
    """
    model = model.to(device)
    
    # 注册钩子
    hook = LayerOutputHook()
    hook.register_hook(model)
    
    results = {}
    
    # 获取基准输出
    print("获取基准输出...")
    if isinstance(model, NonLinearWrapper):
        model.set_alpha(0.0)
    else:
        set_model_alpha(model, 0.0)
    
    model.eval()
    baseline_layer_outputs = []
    sample_count = 0
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            if sample_count >= num_samples:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            
            # 按层顺序保存输出
            layer_outputs = {}
            for layer_name, output in hook.get_outputs().items():
                layer_outputs[layer_name] = output.cpu().clone()
            baseline_layer_outputs.append(layer_outputs)
            
            hook.clear()
            sample_count += inputs.size(0)
    
    # 分析不同alpha下的误差累积
    for alpha in tqdm(alpha_values, desc='逐层误差累积分析'):
        if alpha == 0:
            continue
        
        if isinstance(model, NonLinearWrapper):
            model.set_alpha(alpha)
        else:
            set_model_alpha(model, alpha)
        
        hook.clear()
        sample_count = 0
        noisy_layer_outputs = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                if sample_count >= num_samples:
                    break
                inputs = inputs.to(device)
                _ = model(inputs)
                
                layer_outputs = {}
                for layer_name, output in hook.get_outputs().items():
                    layer_outputs[layer_name] = output.cpu().clone()
                noisy_layer_outputs.append(layer_outputs)
                
                hook.clear()
                sample_count += inputs.size(0)
        
        # 计算每层的误差
        layer_names = list(baseline_layer_outputs[0].keys())
        for layer_name in layer_names:
            if layer_name not in results:
                results[layer_name] = {'alpha_errors': {}}
            
            # 计算该层在所有样本上的平均误差
            mse_errors = []
            for baseline, noisy in zip(baseline_layer_outputs, noisy_layer_outputs):
                if layer_name in baseline and layer_name in noisy:
                    mse = torch.mean((baseline[layer_name] - noisy[layer_name]) ** 2).item()
                    mse_errors.append(mse)
            
            results[layer_name]['alpha_errors'][alpha] = {
                'mse_mean': float(np.mean(mse_errors)),
                'mse_std': float(np.std(mse_errors)),
            }
    
    hook.remove_hooks()
    return results


def plot_sensitivity(results: List[Dict], save_path: Optional[str] = None):
    """绘制整网精度敏感性曲线"""
    alphas = [r['alpha'] for r in results]
    accs = [r['accuracy'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, accs, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Nonlinearity Alpha', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Sensitivity to Non-linear Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, max(alphas) + 0.1, 0.1))
    plt.ylim(0, 100)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    plt.close()


def plot_layer_distribution_shift(results: Dict, save_dir: str):
    """绘制各层输出分布偏移图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择关键层进行可视化
    key_layers = list(results.keys())[:10]  # 最多显示10层
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, layer_name in enumerate(key_layers):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        layer_data = results[layer_name]
        
        alphas = sorted(layer_data['alpha_stats'].keys())
        mean_shifts = [layer_data['alpha_stats'][a]['mean_shift'] for a in alphas]
        std_changes = [layer_data['alpha_stats'][a]['std_change'] for a in alphas]
        
        ax2 = ax.twinx()
        
        line1, = ax.plot(alphas, mean_shifts, 'b-o', label='Mean Shift', linewidth=2)
        line2, = ax2.plot(alphas, std_changes, 'r-s', label='Std Change', linewidth=2)
        
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Mean Shift', color='b')
        ax2.set_ylabel('Std Change', color='r')
        ax.set_title(layer_name[:20] + '...' if len(layer_name) > 20 else layer_name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'layer_distribution_shift.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    plt.close()


def plot_error_accumulation(results: Dict, save_dir: str):
    """绘制误差逐层累积图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有alpha值
    first_layer = list(results.keys())[0]
    alphas = sorted(results[first_layer]['alpha_errors'].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：各层MSE随alpha变化
    ax1 = axes[0]
    for layer_name, layer_data in results.items():
        mse_values = [layer_data['alpha_errors'][a]['mse_mean'] for a in alphas]
        # 简化层名显示
        short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        ax1.plot(alphas, mse_values, '-o', label=short_name, linewidth=1.5, markersize=4)
    
    ax1.set_xlabel('Alpha', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Layer-wise Error vs Alpha', fontsize=14)
    ax1.legend(fontsize=8, ncol=2, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 右图：固定alpha下误差随层深度累积
    ax2 = axes[1]
    layer_names = list(results.keys())
    layer_indices = range(len(layer_names))
    
    for alpha in alphas:
        mse_values = [results[name]['alpha_errors'][alpha]['mse_mean'] for name in layer_names]
        ax2.plot(layer_indices, mse_values, '-o', label=f'α={alpha}', linewidth=1.5, markersize=4)
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Error Accumulation Across Layers', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'error_accumulation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    plt.close()


def comprehensive_sensitivity_analysis(model: nn.Module, test_loader, 
                                        alpha_values: List[float], device: torch.device,
                                        save_dir: str = '../结果/sensitivity'):
    """
    综合敏感性分析
    
    执行完整的敏感性分析并生成所有图表
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始综合敏感性分析")
    print("=" * 60)
    
    # 1. 整网精度衰减趋势
    print("\n[1/3] 整网精度衰减趋势分析...")
    accuracy_results = sensitivity_analysis(model, test_loader, alpha_values, device)
    plot_sensitivity(accuracy_results, os.path.join(save_dir, 'accuracy_vs_alpha.png'))
    
    # 2. 单层输出分布偏移
    print("\n[2/3] 单层输出分布偏移分析...")
    distribution_results = layer_output_distribution_analysis(
        model, test_loader, alpha_values, device, num_samples=100
    )
    plot_layer_distribution_shift(distribution_results, save_dir)
    
    # 3. 误差逐层累积
    print("\n[3/3] 误差逐层累积分析...")
    accumulation_results = layer_error_accumulation_analysis(
        model, test_loader, alpha_values, device, num_samples=50
    )
    plot_error_accumulation(accumulation_results, save_dir)
    
    # 保存数值结果
    import json
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    all_results = {
        'accuracy': accuracy_results,
        'distribution': convert_to_serializable(distribution_results),
        'accumulation': convert_to_serializable(accumulation_results)
    }
    
    with open(os.path.join(save_dir, 'sensitivity_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("敏感性分析完成！")
    print(f"结果保存在: {save_dir}")
    print("=" * 60)
    
    return all_results
