#!/usr/bin/env python3
"""
生成赛题二实验结果图表 - 修复版
使用真实实验数据生成图表
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = "/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/图表"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results():
    """加载所有实验结果"""
    base = "/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/结果"
    results = {}
    
    # 加载创新算法结果 (使用 innovation_gpu1)
    with open(f"{base}/innovation_gpu1/results.json", "r") as f:
        innovation_data = json.load(f)
        results['innovation'] = innovation_data
    
    return results

def plot_innovation_comparison(results):
    """绘制创新算法对比图 - 基于 innovation_gpu1/results.json"""
    data = results['innovation']
    
    methods = [d['method'] for d in data]
    accuracies = [d['best_acc'] for d in data]
    
    # 按精度排序
    sorted_indices = np.argsort(accuracies)[::-1]
    methods = [methods[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    baseline_acc = 85.32
    colors = ['#2ecc71' if acc >= baseline_acc - 0.05 else '#3498db' for acc in accuracies]
    
    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(methods))
    
    bars = ax.barh(y_pos, accuracies, color=colors, height=0.6)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 0.03, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_title('Innovation Algorithm Comparison (ResNet18/CIFAR-10)', fontsize=13, fontweight='bold')
    ax.set_xlim(84, 86)
    ax.axvline(x=baseline_acc, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.5, 
               label=f'Baseline ({baseline_acc}%)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/图4_创新算法对比.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存：{OUTPUT_DIR}/图4_创新算法对比.png (数据：{len(data)} 个方法)")

def plot_ablation_study(results):
    """绘制消融实验图"""
    data = results['innovation']
    
    # 选择关键方法
    ablation_methods = ['Baseline', 'STE+Layerwise', 'STE+BiasCorrection', 'Adaptive-STE-Sqrt']
    ablation_accs = []
    
    for method in ablation_methods:
        found = False
        for item in data:
            if item['method'] == method:
                ablation_accs.append(item['best_acc'])
                found = True
                break
        if not found:
            ablation_accs.append(0)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    bars = ax.bar(ablation_methods, ablation_accs, color=colors, width=0.6)
    
    for bar, acc in zip(bars, ablation_accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Ablation Study: Component Analysis', fontsize=13, fontweight='bold')
    ax.set_ylim(83, 86)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/图10_消融实验.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存：{OUTPUT_DIR}/图10_消融实验.png")

def plot_training_curves(results):
    """绘制训练曲线"""
    data = results['innovation']
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 选择几个代表性方法
    plot_methods = ['Baseline', 'Adaptive-STE-Sqrt', 'STE+BiasCorrection']
    colors = {'Baseline': '#3498db', 'Adaptive-STE-Sqrt': '#2ecc71', 'STE+BiasCorrection': '#e74c3c'}
    
    for item in data:
        method = item['method']
        if method not in plot_methods:
            continue
            
        history = item['history']
        color = colors.get(method, '#999999')
        
        if 'train_loss' in history and len(history['train_loss']) > 0:
            axes[0].plot(range(len(history['train_loss'])), history['train_loss'], 
                        label=method, linewidth=2, color=color)
        if 'train_acc' in history and len(history['train_acc']) > 0:
            axes[1].plot(range(len(history['train_acc'])), history['train_acc'], 
                        label=method, linewidth=2, color=color)
    
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss Curves', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.2)
    
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Training Accuracy Curves', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/图2_训练曲线.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存：{OUTPUT_DIR}/图2_训练曲线.png")

def plot_schedule_comparison():
    """绘制调度策略对比图（理论曲线）"""
    noise_levels = np.linspace(0, 2.0, 100)
    
    schedules = {
        'Inverse': lambda x: 1 / (1 + 0.5 * x),
        'Linear': lambda x: 1 / (1 + x),
        'Sqrt (Best)': lambda x: 1 / np.sqrt(1 + x**2),
        'Exp': lambda x: np.exp(-0.3 * x),
    }
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for (name, func), color in zip(schedules.items(), colors):
        values = func(noise_levels)
        ax.plot(noise_levels, values, label=name, linewidth=2.5, color=color)
    
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Scale Factor', fontsize=11)
    ax.set_title('STE Gradient Scale: Schedule Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/图3_调度策略对比.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存：{OUTPUT_DIR}/图3_调度策略对比.png")

def plot_framework_architecture():
    """绘制框架架构图（示意图）"""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis('off')
    
    ax.text(0.5, 0.93, 'STE Noise-Aware Training Framework',
            fontsize=15, fontweight='bold', ha='center', transform=ax.transAxes)
    
    components = [
        (0.5, 0.82, 'NoiseInjector', '#3498db'),
        (0.25, 0.65, 'NoisyLinear', '#2ecc71'),
        (0.75, 0.65, 'STE Estimator', '#e74c3c'),
        (0.5, 0.48, 'Adaptive STE', '#9b59b6'),
        (0.5, 0.30, 'Regularizer', '#f39c12'),
        (0.5, 0.12, 'Backprop', '#1abc9c'),
    ]
    
    for x, y, text, color in components:
        rect = plt.Rectangle((x-0.12, y-0.05), 0.24, 0.1,
                            facecolor=color, edgecolor='black', linewidth=1.5,
                            transform=ax.transAxes, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=10, ha='center', va='center', 
               transform=ax.transAxes, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/图1_框架架构图.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"已保存：{OUTPUT_DIR}/图1_框架架构图.png")

if __name__ == "__main__":
    print("开始生成赛题二图表...")
    print(f"输出目录：{OUTPUT_DIR}")
    
    results = load_results()
    print(f"加载了 {len(results['innovation'])} 个创新算法结果")
    
    # 打印数据摘要
    print("\n创新算法数据摘要:")
    for item in results['innovation']:
        print(f"  {item['method']}: {item['best_acc']:.2f}%")
    
    print("\n生成图表...")
    plot_innovation_comparison(results)
    plot_ablation_study(results)
    plot_training_curves(results)
    plot_schedule_comparison()
    plot_framework_architecture()
    
    print("\n所有图表生成完成！")
