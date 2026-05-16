import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import os

os.makedirs('../图表', exist_ok=True)

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

print("="*60)
print("生成可视化图表...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]
baseline_acc = [81.95, 80.99, 77.63, 70.95, 60.40, 48.49]
ax1.plot(alphas, baseline_acc, 'b-o', linewidth=2, markersize=8, label='Baseline')
ax1.fill_between(alphas, baseline_acc, alpha=0.3)
ax1.set_xlabel('Alpha (Nonlinear Strength)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Task 1: Sensitivity Analysis\nAccuracy vs Nonlinear Strength', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 0.5)
ax1.set_ylim(40, 90)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
for i, (a, acc) in enumerate(zip(alphas, baseline_acc)):
    ax1.annotate(f'{acc:.1f}%', (a, acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

ax2 = axes[0, 1]
nat_02 = [84.34, 86.12, 87.30, 83.97, 71.45, 56.32]
mixed_alpha = [78.13, 81.46, 82.93, 83.12, 81.83, 79.13]
nat_mixed = [79.03, 82.59, 83.74, 83.92, 82.42, 78.56]
curriculum_nat = [69.30, 76.60, 81.22, 83.30, 83.94, 82.48]
ax2.plot(alphas, nat_02, 'g-s', linewidth=2, markersize=7, label='NAT (α=0.2)')
ax2.plot(alphas, mixed_alpha, 'r-^', linewidth=2, markersize=7, label='Mixed Alpha')
ax2.plot(alphas, nat_mixed, 'm-D', linewidth=2, markersize=7, label='NAT+Mixed Alpha')
ax2.plot(alphas, curriculum_nat, 'c-p', linewidth=2, markersize=7, label='Curriculum NAT')
ax2.plot(alphas, baseline_acc, 'b--o', linewidth=1.5, markersize=5, alpha=0.5, label='Baseline')
ax2.set_xlabel('Alpha (Nonlinear Strength)', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Task 3: Robustness Enhancement Methods\nComparison of Advanced Methods', fontsize=14, fontweight='bold')
ax2.legend(loc='lower left', fontsize=9)
ax2.set_xlim(0, 0.5)
ax2.set_ylim(50, 95)
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
resnet18 = [81.41, 81.27, 80.84, 79.85, 78.70, 76.93]
resnet34 = [80.86, 80.90, 80.57, 80.11, 78.83, 77.07]
mobilenet = [73.40, 73.36, 72.94, 71.98, 70.54, 68.03]
ax3.plot(alphas, resnet34, 'g-s', linewidth=2, markersize=8, label='ResNet34 (21.3M)')
ax3.plot(alphas, resnet18, 'b-^', linewidth=2, markersize=8, label='ResNet18 (11.2M)')
ax3.plot(alphas, mobilenet, 'r-o', linewidth=2, markersize=8, label='MobileNetV2 (2.2M)')
ax3.set_xlabel('Alpha (Nonlinear Strength)', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_title('Extended Research 1: Network Structure\nParameter Count vs Robustness', fontsize=14, fontweight='bold')
ax3.legend(loc='lower left', fontsize=10)
ax3.set_xlim(0, 0.5)
ax3.set_ylim(65, 85)
ax3.grid(True, alpha=0.3)
decay_r18 = resnet18[0] - resnet18[5]
decay_r34 = resnet34[0] - resnet34[5]
decay_mob = mobilenet[0] - mobilenet[5]
ax3.text(0.02, 0.02, f'α=0→0.5 Decay:\nResNet34: {decay_r34:.1f}%\nResNet18: {decay_r18:.1f}%\nMobileNetV2: {decay_mob:.1f}%',
         transform=ax3.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4 = axes[1, 1]
sigma = [0.0, 0.05, 0.10, 0.15, 0.20]
gaussian_acc = [57.11, 56.33, 52.37, 46.48, 39.87]
nl_acc = [57.11, 54.13, 51.36, 48.60, 45.81, 42.76]
ax4_twin = ax4.twinx()
ax4.plot([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], nl_acc, 'b-o', linewidth=2, markersize=8, label='Nonlinear失真 (α)')
ax4_twin.plot(sigma, gaussian_acc, 'r-s', linewidth=2, markersize=8, label='Gaussian噪声 (σ)')
ax4.set_xlabel('Alpha / Sigma', fontsize=12)
ax4.set_ylabel('Nonlinear Accuracy (%)', color='b', fontsize=12)
ax4_twin.set_ylabel('Gaussian Accuracy (%)', color='r', fontsize=12)
ax4.set_title('Extended Research 2: Noise Comparison\nGaussian vs Nonlinear', fontsize=14, fontweight='bold')
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)
ax4.set_xlim(0, 0.5)
ax4.set_ylim(35, 60)
ax4_twin.set_ylim(35, 60)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../图表/图1_实验总览.png', dpi=150, bbox_inches='tight')
print("保存: ../图表/图1_实验总览.png")

quant_acc = {
    'FP32': [57.11, 54.13, 51.36, 48.60, 45.81, 42.76],
    'INT8': [57.11, 54.15, 51.39, 48.63, 45.86, 42.87],
    '4-bit': [56.66, 53.76, 50.84, 48.22, 45.30, 42.42],
    '2-bit': [55.22, 52.72, 50.21, 47.72, 45.18, 42.54]
}

fig2, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
for (name, acc), color, marker in zip(quant_acc.items(), colors, markers):
    ax.plot(alphas, acc, color=color, marker=marker, linewidth=2, markersize=8, label=name)

ax.set_xlabel('Alpha (Nonlinear Strength)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Extended Research 3: Quantization + Nonlinear\nImpact of Different Bit-widths', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.set_xlim(0, 0.5)
ax.set_ylim(35, 62)
ax.grid(True, alpha=0.3)

fp32_decay = quant_acc['FP32'][0] - quant_acc['FP32'][5]
int8_decay = quant_acc['INT8'][0] - quant_acc['INT8'][5]
bit4_decay = quant_acc['4-bit'][0] - quant_acc['4-bit'][5]
bit2_decay = quant_acc['2-bit'][0] - quant_acc['2-bit'][5]

summary = f'α=0→0.5 Decay:\nFP32: {fp32_decay:.1f}%\nINT8: {int8_decay:.1f}%\n4-bit: {bit4_decay:.1f}%\n2-bit: {bit2_decay:.1f}%'
ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('../图表/图2_量化与非线性分析.png', dpi=150, bbox_inches='tight')
print("保存: ../图表/图2_量化与非线性分析.png")

fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

methods = ['Baseline', 'Pre-dist', 'NAT(α=0.2)', 'Mixed α', 'NAT+Mix', 'Curriculum']
avg_acc = [70.07, 74.68, 78.25, 81.10, 81.71, 79.47]
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']

bars = axes[0].bar(methods, avg_acc, color=colors, edgecolor='black', linewidth=1.2)
axes[0].set_ylabel('Average Accuracy (%)', fontsize=12)
axes[0].set_title('Task 3: Method Comparison\nAverage Accuracy Across All α', fontsize=13, fontweight='bold')
axes[0].set_ylim(65, 85)
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, avg_acc):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{acc:.1f}%', ha='center', fontsize=9)

networks = ['ResNet34', 'ResNet18', 'MobileNetV2']
decays = [3.79, 4.48, 5.37]
colors = ['#2ca02c', '#1f77b4', '#d62728']
bars = axes[1].bar(networks, decays, color=colors, edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('Accuracy Decay (%)', fontsize=12)
axes[1].set_title('Extended 1: Network Robustness\nα=0→0.5 Decay Comparison', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 7)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, decays):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{d:.1f}%', ha='center', fontsize=10)

noise_types = ['Nonlinear\n(α=0.2)', 'Gaussian\n(σ=0.1)', 'Nonlinear\n(α=0.4)', 'Gaussian\n(σ=0.15)']
losses = [5.75, 4.74, 11.30, 10.63]
colors = ['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e']
patterns = ['', '', '//', '//']
bars = axes[2].bar(noise_types, losses, color=colors, edgecolor='black', linewidth=1.2, hatch='')
axes[2].set_ylabel('Accuracy Loss (%)', fontsize=12)
axes[2].set_title('Extended 2: Noise Equivalence\nEquivalent Decay Conditions', fontsize=13, fontweight='bold')
axes[2].set_ylim(0, 14)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].legend([plt.Rectangle((0,0),1,1,fc='#1f77b4'), plt.Rectangle((0,0),1,1,fc='#ff7f0e')],
               ['Nonlinear', 'Gaussian'], loc='upper left')

plt.tight_layout()
plt.savefig('../图表/图3_方法对比汇总.png', dpi=150, bbox_inches='tight')
print("保存: ../图表/图3_方法对比汇总.png")

print("\n" + "="*60)
print("所有图表生成完成！")
print("="*60)