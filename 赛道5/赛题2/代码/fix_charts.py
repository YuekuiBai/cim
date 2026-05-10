#!/usr/bin/env python3
"""
修复图表 - 仅使用真实实验数据
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
plt.rcParams['legend.fontsize'] = 9

OUTPUT_DIR = "/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/results/figures"

def load_real_data():
    """加载所有真实实验数据"""
    base = "/mnt/storage2/zyc/CIM比赛/赛道5/赛题2/results"

    with open(f"{base}/innovation/results.json", "r") as f:
        innovation_data = json.load(f)

    with open(f"{base}/task1_sensitivity/sensitivity_results.json", "r") as f:
        sensitivity_data = json.load(f)

    with open(f"{base}/task3_evaluation/evaluation_results.json", "r") as f:
        eval_data = json.load(f)

    return innovation_data, sensitivity_data, eval_data

def plot_ablation_from_real_data(innovation_data):
    """从真实数据绘制消融实验图"""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    method_map = {
        'Baseline': 'Baseline',
        'STE-NAT': 'STE',
        'Adaptive-STE-Inverse': '+Inverse',
        'Adaptive-STE-Linear': '+Linear',
        'Adaptive-STE-Sqrt': '+Sqrt (Ours)',
        'STE+BiasCorrection': '+BiasCorr',
        'STE+Layerwise': '+Layerwise',
        'STE+Regularizer': '+Regularizer',
        'Full-Innovation': 'Full'
    }

    # 只绘制有真实数据的组件
    available_methods = [m for m in innovation_data if m['method'] in method_map]
    available_methods = sorted(available_methods, key=lambda x: x['best_acc'], reverse=True)

    components = [method_map[m['method']] for m in available_methods]
    accuracies = [m['best_acc'] for m in available_methods]

    colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]

    bars = ax.barh(components, accuracies, color=colors, height=0.6)

    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 0.05, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10)

    ax.set_xlabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Ablation Study: Component Contribution', fontsize=13, fontweight='bold')
    ax.set_xlim(50, 87)
    ax.axvline(x=85.15, color='red', linestyle='--', alpha=0.6, label='Baseline')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ablation_study.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已更新: {OUTPUT_DIR}/ablation_study.png")

def plot_statistical_from_real_data(innovation_data):
    """从真实数据绘制统计分析图（单次实验，无多次运行）"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：各方法最终精度对比
    ax = axes[0]
    methods = [m['method'] for m in innovation_data]
    final_accs = [m['final_clean_acc'] for m in innovation_data]

    colors = ['#2ecc71' if acc == max(final_accs) else '#3498db' for acc in final_accs]

    x = np.arange(len(methods))
    bars = ax.bar(x, final_accs, color=colors, width=0.6)

    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Method Comparison: Final Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(50, 90)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=8)

    # 右图：噪声鲁棒性（clean vs noisy accuracy差距）
    ax = axes[1]

    methods_for_bar = []
    gaps = []

    for m in innovation_data:
        clean = m['final_clean_acc']
        noisy = m['final_noisy_acc']
        gap = clean - noisy  # 精度损失
        methods_for_bar.append(m['method'])
        gaps.append(gap)

    colors_gap = ['#e74c3c' if g > 1.5 else '#f39c12' if g > 1.0 else '#27ae60' for g in gaps]

    x = np.arange(len(methods_for_bar))
    bars = ax.bar(x, gaps, color=colors_gap, width=0.6)

    ax.set_ylabel('Accuracy Gap (%)')
    ax.set_title('Noise Robustness: Clean vs Noisy Acc', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_for_bar, rotation=30, ha='right', fontsize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{gap:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/statistical_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已更新: {OUTPUT_DIR}/statistical_analysis.png")

def plot_sensitivity_from_real_data(sensitivity_data):
    """从真实数据绘制敏感性分析图"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # 从真实数据提取
    noise_levels = sorted(set(item['noise_strength'] for item in sensitivity_data))

    # 按方法分组
    baseline_by_noise = {}
    ste_nat_by_noise = {}

    for item in sensitivity_data:
        nl = item['noise_strength']
        acc = item['accuracy']
        method = item.get('method', 'Unknown')

        if 'baseline' in method.lower() or nl not in baseline_by_noise:
            if nl not in baseline_by_noise:
                baseline_by_noise[nl] = acc
        else:
            if nl not in ste_nat_by_noise:
                ste_nat_by_noise[nl] = acc

    # 绘制
    nl_sorted = sorted(baseline_by_noise.keys())
    baseline_accs = [baseline_by_noise[nl] for nl in nl_sorted]

    ax.plot(nl_sorted, baseline_accs, 'o-', label='Baseline',
            color='#3498db', linewidth=2, markersize=8)

    if ste_nat_by_noise:
        nl_sorted_ste = sorted(ste_nat_by_noise.keys())
        ste_nat_accs = [ste_nat_by_noise[nl] for nl in nl_sorted_ste]
        ax.plot(nl_sorted_ste, ste_nat_accs, 's-', label='STE-NAT',
                color='#e74c3c', linewidth=2, markersize=8)

    ax.set_xlabel('Noise Level (σ)', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Noise Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(80, 90)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sensitivity_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已更新: {OUTPUT_DIR}/sensitivity_analysis.png")

def main():
    print("=" * 60)
    print("使用真实数据重新生成图表...")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/3] 从真实数据生成消融实验图...")
    innovation_data, sensitivity_data, eval_data = load_real_data()
    plot_ablation_from_real_data(innovation_data)

    print("\n[2/3] 从真实数据生成统计分析图...")
    plot_statistical_from_real_data(innovation_data)

    print("\n[3/3] 从真实数据生成敏感性分析图...")
    plot_sensitivity_from_real_data(sensitivity_data)

    print("\n" + "=" * 60)
    print("图表已更新！所有数据均来自真实实验结果")
    print("=" * 60)

    files = sorted(os.listdir(OUTPUT_DIR))
    print("\n当前图表文件:")
    for f in files:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  • {f} ({size:.1f} KB)")

if __name__ == "__main__":
    main()