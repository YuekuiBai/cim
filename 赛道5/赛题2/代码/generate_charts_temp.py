#!/usr/bin/env python3
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = "/mnt/storage2/zyc/CIM 比赛/赛道 5/赛题 2/results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base = "/mnt/storage2/zyc/CIM 比赛/赛道 5/赛题 2/results"
with open(f"{base}/innovation_gpu1/results.json", "r") as f:
    innovation_data = json.load(f)

print(f"加载了 {len(innovation_data)} 个创新算法结果")
print("\n创新算法数据摘要:")
for item in innovation_data:
    print(f"  {item['method']}: {item['best_acc']:.2f}%")

# 1. innovation_comparison
methods = [d['method'] for d in innovation_data]
accuracies = [d['best_acc'] for d in innovation_data]
sorted_indices = np.argsort(accuracies)[::-1]
methods = [methods[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]

baseline_acc = 85.32
colors = ['#2ecc71' if acc >= baseline_acc - 0.05 else '#3498db' for acc in accuracies]

fig, ax = plt.subplots(figsize=(11, 7))
y_pos = np.arange(len(methods))
bars = ax.barh(y_pos, accuracies, color=colors, height=0.6)

for bar, acc in zip(bars, accuracies):
    ax.text(acc + 0.03, bar.get_y() + bar.get_height()/2, f'{acc:.2f}%', va='center', fontsize=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=9)
ax.set_xlabel('Accuracy (%)')
ax.set_title('Innovation Algorithm Comparison (ResNet18/CIFAR-10)', fontweight='bold')
ax.set_xlim(84, 86)
ax.axvline(x=baseline_acc, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Baseline ({baseline_acc}%)')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/innovation_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"\n已保存：{OUTPUT_DIR}/innovation_comparison.png")

# 2. ablation_study
ablation_methods = ['Baseline', 'STE+Layerwise', 'STE+BiasCorrection', 'Adaptive-STE-Sqrt']
ablation_accs = []
for method in ablation_methods:
    found = False
    for item in innovation_data:
        if item['method'] == method:
            ablation_accs.append(item['best_acc'])
            found = True
            break
    if not found:
        ablation_accs.append(0)

fig, ax = plt.subplots(figsize=(9, 6))
bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = ax.bar(ablation_methods, ablation_accs, color=bar_colors, width=0.6)

for bar, acc in zip(bars, ablation_accs):
    if acc > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{acc:.2f}%', ha='center', fontsize=10)

ax.set_ylabel('Accuracy (%)')
ax.set_title('Ablation Study: Component Analysis', fontweight='bold')
ax.set_ylim(83, 86)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ablation_study.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"已保存：{OUTPUT_DIR}/ablation_study.png")

# 3. training_curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_methods = ['Baseline', 'Adaptive-STE-Sqrt', 'STE+BiasCorrection']
method_colors = {'Baseline': '#3498db', 'Adaptive-STE-Sqrt': '#2ecc71', 'STE+BiasCorrection': '#e74c3c'}

for item in innovation_data:
    method = item['method']
    if method not in plot_methods:
        continue
    history = item['history']
    color = method_colors.get(method, '#999999')
    
    if 'train_loss' in history and len(history['train_loss']) > 0:
        axes[0].plot(range(len(history['train_loss'])), history['train_loss'], label=method, linewidth=2, color=color)
    if 'train_acc' in history and len(history['train_acc']) > 0:
        axes[1].plot(range(len(history['train_acc'])), history['train_acc'], label=method, linewidth=2, color=color)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Curves', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.2)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training Accuracy Curves', fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"已保存：{OUTPUT_DIR}/training_curves.png")

# 4. schedule_comparison
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

ax.set_xlabel('Noise Level')
ax.set_ylabel('Scale Factor')
ax.set_title('STE Gradient Scale: Schedule Comparison', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/schedule_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"已保存：{OUTPUT_DIR}/schedule_comparison.png")

print("\n所有图表生成完成！")
