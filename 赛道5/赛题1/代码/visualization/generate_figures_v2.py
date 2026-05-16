#!/usr/bin/env python3
"""
生成赛题一补充图表 (图4-图10)
基于已有实验数据，覆盖评分要点：敏感性分析、NAT训练、鲁棒性增强、拓展研究、机理分析
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = '../结果'
OUTPUT_DIR = '../图表'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 数据加载
# ============================================================

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

sensitivity = load_json(f'{RESULT_DIR}/task1_sensitivity/sensitivity_results.json')
training_01 = load_json(f'{RESULT_DIR}/task2_training/alpha_0.1/training_history.json')
training_02 = load_json(f'{RESULT_DIR}/task2_training/alpha_0.2/training_history.json')
training_03 = load_json(f'{RESULT_DIR}/task2_training/alpha_0.3/training_history.json')
finetune_rig = load_json(f'{RESULT_DIR}/task2_training/finetune_rigorous/finetune_results.json')
scratch_rig = load_json(f'{RESULT_DIR}/task2_training/scratch_rigorous/scratch_results.json')
comparison = load_json(f'{RESULT_DIR}/task2_training/finetune_vs_scratch/comparison_results.json')
robustness = load_json(f'{RESULT_DIR}/task3_robustness/robustness_results.json')

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

print("=" * 60)
print("生成补充图表 (图4-图10)")
print("=" * 60)

# ============================================================
# 图4: 敏感性分析 (2x2)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) 精度衰减曲线
ax = axes[0, 0]
acc_data = sensitivity['accuracy']
alpha_vals = [d['alpha'] for d in acc_data]
acc_vals = [d['accuracy'] for d in acc_data]
ax.plot(alpha_vals, acc_vals, 'b-o', linewidth=2.5, markersize=9, label='Baseline')
ax.fill_between(alpha_vals, acc_vals, alpha=0.15, color='b')
ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='70% threshold')
ax.axvline(x=0.2, color='g', linestyle=':', alpha=0.5, label=r'$\alpha$=0.2 safe')
ax.axvline(x=0.3, color='orange', linestyle=':', alpha=0.5, label=r'$\alpha$=0.3 danger')
for a, acc in zip(alpha_vals, acc_vals):
    ax.annotate(f'{acc:.1f}%', (a, acc), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel(r'Nonlinearity Parameter $\alpha$', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('(a) Accuracy Degradation\nSensitivity to Nonlinear Strength', fontsize=13, fontweight='bold')
ax.set_xlim(0, 0.5)
ax.set_ylim(40, 90)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)

# (b) 逐层MSE误差热力图
ax = axes[0, 1]
accum = sensitivity['accumulation']
layer_names = []
short_names = []
for key in accum.keys():
    layer_names.append(key)
    # 简化层名
    short = key.replace('model.', '').replace('.conv', '').replace('.0.', '.0.').replace('.1.', '.1.')
    if 'downsample' in short:
        short = short.replace('.0.downsample.0', '.ds')
    short_names.append(short)

mse_matrix = []
for layer in layer_names:
    row = []
    for a in ['0.1', '0.2', '0.3', '0.4', '0.5']:
        row.append(accum[layer]['alpha_errors'][a]['mse_mean'])
    mse_matrix.append(row)

mse_matrix = np.array(mse_matrix)
# 排除fc层以便热力图更清晰（fc层MSE远大于conv层）
conv_mask = [i for i, n in enumerate(layer_names) if 'fc' not in n]
mse_conv = mse_matrix[conv_mask]
short_conv = [short_names[i] for i in conv_mask]

im = ax.imshow(mse_conv, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels([r'$\alpha$=0.1', r'$\alpha$=0.2', r'$\alpha$=0.3', r'$\alpha$=0.4', r'$\alpha$=0.5'], fontsize=9)
ax.set_yticks(range(len(short_conv)))
ax.set_yticklabels(short_conv, fontsize=7)
ax.set_title('(b) Per-Layer MSE Error Heatmap\n(Conv Layers Only)', fontsize=13, fontweight='bold')
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('MSE', fontsize=10)

# (c) 各alpha下标准差变化
ax = axes[1, 0]
dist = sensitivity['distribution']
fc_std_changes = []
conv_std_changes_avg = []
for a in ['0.1', '0.2', '0.3', '0.4', '0.5']:
    fc_sc = dist['model.fc.linear']['alpha_stats'][a]['std_change']
    fc_std_changes.append(abs(fc_sc))
    conv_scs = []
    for layer in dist:
        if 'fc' not in layer:
            conv_scs.append(abs(dist[layer]['alpha_stats'][a]['std_change']))
    conv_std_changes_avg.append(np.mean(conv_scs))

x = np.arange(5)
width = 0.35
bars1 = ax.bar(x - width/2, conv_std_changes_avg, width, label='Conv Layers (avg)', color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, fc_std_changes, width, label='FC Layer', color='#e74c3c', edgecolor='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([r'$\alpha$=0.1', r'$\alpha$=0.2', r'$\alpha$=0.3', r'$\alpha$=0.4', r'$\alpha$=0.5'], fontsize=9)
ax.set_xlabel(r'Nonlinearity Parameter $\alpha$', fontsize=12)
ax.set_ylabel('|Std Change|', fontsize=12)
ax.set_title('(c) Weight Distribution Shift\nStd Change by Layer Type', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# (d) 误差逐层累积趋势（选几个alpha）
ax = axes[1, 1]
conv_indices = [i for i, n in enumerate(layer_names) if 'fc' not in n]
conv_layers_plot = [layer_names[i] for i in conv_indices]
layer_depths = list(range(len(conv_layers_plot)))

for a_idx, a in enumerate(['0.1', '0.2', '0.3', '0.5']):
    mse_vals = [accum[l]['alpha_errors'][a]['mse_mean'] for l in conv_layers_plot]
    cumulative = np.cumsum(mse_vals)
    color = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][a_idx]
    ax.plot(layer_depths, cumulative, '-o', linewidth=2, markersize=5,
            color=color, label=r'$\alpha$=' + a)

ax.set_xlabel('Layer Depth (Conv Layers)', fontsize=12)
ax.set_ylabel('Cumulative MSE', fontsize=12)
ax.set_title('(d) Error Accumulation Across Layers\nCumulative MSE vs Layer Depth', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/图4_敏感性分析.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: 图4_敏感性分析.png")

# ============================================================
# 图5: NAT训练曲线 (2x2)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

datasets = [
    (training_01, r'NAT $\alpha$=0.1', '#3498db'),
    (training_02, r'NAT $\alpha$=0.2', '#2ecc71'),
    (training_03, r'NAT $\alpha$=0.3', '#e74c3c'),
]

# (a) Training Loss
ax = axes[0, 0]
for data, label, color in datasets:
    epochs = range(len(data['train_loss']))
    ax.plot(epochs, data['train_loss'], linewidth=2, color=color, label=label)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('(a) Training Loss\nConvergence Under Different Alpha', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) Test Loss
ax = axes[0, 1]
for data, label, color in datasets:
    epochs = range(len(data['test_loss']))
    ax.plot(epochs, data['test_loss'], linewidth=2, color=color, label=label)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Loss', fontsize=12)
ax.set_title('(b) Test Loss\nGeneralization Performance', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (c) Training Accuracy
ax = axes[1, 0]
for data, label, color in datasets:
    epochs = range(len(data['train_acc']))
    ax.plot(epochs, data['train_acc'], linewidth=2, color=color, label=label)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Accuracy (%)', fontsize=12)
ax.set_title('(c) Training Accuracy\nLearning Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d) Test Accuracy
ax = axes[1, 1]
for data, label, color in datasets:
    epochs = range(len(data['test_acc']))
    ax.plot(epochs, data['test_acc'], linewidth=2, color=color, label=label)
    best_ep = data['best_epoch']
    best_acc = data['best_accuracy']
    ax.annotate(f'{best_acc:.1f}%', (best_ep, best_acc),
                textcoords="offset points", xytext=(10, 8), fontsize=9, color=color, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('(d) Test Accuracy\nBest Performance per Alpha', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/图5_NAT训练曲线.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: 图5_NAT训练曲线.png")

# ============================================================
# 图6: 微调 vs 从头训练 (1x2)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) 3-seed rigorous comparison with error bars
ax = axes[0]
alpha_labels = [r'$\alpha$=0.1', r'$\alpha$=0.2', r'$\alpha$=0.3']
x = np.arange(3)
width = 0.35

# finetune: keys are "0.1", "0.2", "0.3"
ft_means = [np.mean(finetune_rig[k]) for k in ['0.1', '0.2', '0.3']]
ft_stds = [np.std(finetune_rig[k]) for k in ['0.1', '0.2', '0.3']]
sc_means = [np.mean(scratch_rig['scratch_results'][k]) for k in ['0.1', '0.2', '0.3']]
sc_stds = [np.std(scratch_rig['scratch_results'][k]) for k in ['0.1', '0.2', '0.3']]

bars1 = ax.bar(x - width/2, ft_means, width, yerr=ft_stds, label='Finetune',
               color='#2ecc71', edgecolor='black', linewidth=1, capsize=5)
bars2 = ax.bar(x + width/2, sc_means, width, yerr=sc_stds, label='From Scratch',
               color='#e74c3c', edgecolor='black', linewidth=1, capsize=5)

for bar, mean, std in zip(bars1, ft_means, ft_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
            f'{mean:.2f}%', ha='center', fontsize=9, fontweight='bold')
for bar, mean, std in zip(bars2, sc_means, sc_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
            f'{mean:.2f}%', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(alpha_labels, fontsize=11)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('(a) Rigorous Comparison (3 Seeds)\nFinetune vs From Scratch', fontsize=13, fontweight='bold')
ax.set_ylim(78, 88)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# (b) Accuracy across all alpha values
ax = axes[1]
ft_accs = [comparison['finetune'][str(a)] for a in alphas]
sc_accs = [comparison['scratch'][str(a)] for a in alphas]
ax.plot(alphas, ft_accs, 'g-s', linewidth=2, markersize=8, label='Finetune')
ax.plot(alphas, sc_accs, 'r-^', linewidth=2, markersize=8, label='From Scratch')
ax.fill_between(alphas, ft_accs, sc_accs, alpha=0.1, color='green')
for a, ft, sc in zip(alphas, ft_accs, sc_accs):
    ax.annotate(f'{ft:.1f}', (a, ft), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color='green')
    ax.annotate(f'{sc:.1f}', (a, sc), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8, color='red')
ax.axhline(y=comparison['finetune_avg'], color='green', linestyle='--', alpha=0.4)
ax.axhline(y=comparison['scratch_avg'], color='red', linestyle='--', alpha=0.4)
ax.text(0.48, comparison['finetune_avg'] + 0.5, f"Avg: {comparison['finetune_avg']:.1f}%", fontsize=9, color='green')
ax.text(0.48, comparison['scratch_avg'] + 0.5, f"Avg: {comparison['scratch_avg']:.1f}%", fontsize=9, color='red')
ax.set_xlabel(r'Nonlinearity Parameter $\alpha$', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('(b) Robustness Across Alpha\nFinetune Maintains Stability', fontsize=13, fontweight='bold')
ax.set_xlim(0, 0.5)
ax.set_ylim(50, 90)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/图6_微调vs从头训练.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: 图6_微调vs从头训练.png")

# ============================================================
# 图7: 鲁棒性方法对比
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

# 从robustness_results和图表.md中汇总所有方法
methods_data = {
    'Baseline': {0.0: 81.95, 0.1: 80.99, 0.2: 77.63, 0.3: 70.95, 0.4: 60.40, 0.5: 48.49},
    'Predistortion': {0.0: 81.95, 0.1: 81.39, 0.2: 79.71, 0.3: 75.31, 0.4: 68.95, 0.5: 60.74},
    r'NAT Fixed ($\alpha$=0.2)': {0.0: 84.34, 0.1: 86.12, 0.2: 87.30, 0.3: 83.97, 0.4: 71.45, 0.5: 56.32},
    'Mixed Alpha': {0.0: 78.13, 0.1: 81.46, 0.2: 82.93, 0.3: 83.12, 0.4: 81.83, 0.5: 79.13},
    'NAT+Mixed Alpha': {0.0: 79.03, 0.1: 82.59, 0.2: 83.74, 0.3: 83.92, 0.4: 82.42, 0.5: 78.56},
    'Curriculum NAT': {0.0: 69.30, 0.1: 76.60, 0.2: 81.22, 0.3: 83.30, 0.4: 83.94, 0.5: 82.48},
    'OVF Training': {0.0: 77.40, 0.1: 81.58, 0.2: 83.47, 0.3: 83.64, 0.4: 82.16, 0.5: 78.11},
    'SAM Training': {0.0: 76.45, 0.1: 80.30, 0.2: 82.29, 0.3: 82.75, 0.4: 81.15, 0.5: 77.44},
}

colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
markers = ['o', 's', '^', 'D', 'p', 'v', '<', '>']

for (name, data), color, marker in zip(methods_data.items(), colors, markers):
    a_vals = sorted(data.keys())
    acc_vals = [data[a] for a in a_vals]
    lw = 2.5 if 'NAT+Mixed' in name else 1.8
    ax.plot(a_vals, acc_vals, color=color, marker=marker, linewidth=lw, markersize=7, label=name)

ax.axhline(y=70, color='gray', linestyle='--', alpha=0.3)
ax.text(0.01, 70.5, '70% threshold', fontsize=8, color='gray')
ax.set_xlabel(r'Nonlinearity Parameter $\alpha$', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Robustness Enhancement Methods Comparison\nNAT+Mixed Alpha Achieves Best Stability', fontsize=14, fontweight='bold')
ax.set_xlim(0, 0.5)
ax.set_ylim(45, 92)
ax.legend(fontsize=8.5, loc='lower left', ncol=2)
ax.grid(True, alpha=0.3)

# 添加平均精度标注
avg_accs = {name: np.mean(list(data.values())) for name, data in methods_data.items()}
best_method = max(avg_accs, key=avg_accs.get)
ax.text(0.97, 0.97, f'Best Avg: {best_method}\n{avg_accs[best_method]:.2f}%',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/图7_鲁棒性方法对比.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: 图7_鲁棒性方法对比.png")

# ============================================================
# 图8: 网络结构敏感性 (1x2)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

resnet18 = [81.41, 81.27, 80.84, 79.85, 78.70, 76.93]
resnet34 = [80.86, 80.90, 80.57, 80.11, 78.83, 77.07]
mobilenet = [73.40, 73.36, 72.94, 71.98, 70.54, 68.03]

# (a) Accuracy curves
ax = axes[0]
ax.plot(alphas, resnet34, 'g-s', linewidth=2, markersize=8, label='ResNet34 (21.3M)')
ax.plot(alphas, resnet18, 'b-^', linewidth=2, markersize=8, label='ResNet18 (11.2M)')
ax.plot(alphas, mobilenet, 'r-o', linewidth=2, markersize=8, label='MobileNetV2 (2.2M)')
for a, r34, r18, mob in zip(alphas, resnet34, resnet18, mobilenet):
    ax.annotate(f'{r34:.1f}', (a, r34), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color='green')
    ax.annotate(f'{mob:.1f}', (a, mob), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8, color='red')
ax.set_xlabel(r'Nonlinearity Parameter $\alpha$', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('(a) Network Robustness Comparison\nDeeper Networks Are More Robust', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.set_xlim(0, 0.5)
ax.set_ylim(65, 85)
ax.grid(True, alpha=0.3)

# (b) Decay bar chart
ax = axes[1]
decay_r18 = resnet18[0] - resnet18[5]
decay_r34 = resnet34[0] - resnet34[5]
decay_mob = mobilenet[0] - mobilenet[5]
networks = ['ResNet34\n(21.3M)', 'ResNet18\n(11.2M)', 'MobileNetV2\n(2.2M)']
decays = [decay_r34, decay_r18, decay_mob]
colors = ['#2ca02c', '#1f77b4', '#d62728']
bars = ax.bar(networks, decays, color=colors, edgecolor='black', linewidth=1.2)
for bar, d in zip(bars, decays):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{d:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy Decay (%)', fontsize=12)
ax.set_title(r'(b) Decay Comparison ($\alpha$=0$\rightarrow$0.5)' + '\nResNet34 Most Robust', fontsize=13, fontweight='bold')
ax.set_ylim(0, 7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/图8_网络结构敏感性.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: 图8_网络结构敏感性.png")

# ============================================================
# 图9: 非线性失真可视化 (1x2)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

x = np.linspace(-2, 2, 500)
alpha_vals_plot = [0.0, 0.1, 0.2, 0.3, 0.5]
colors_nl = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

# (a) Transfer function
ax = axes[0]
for a, color in zip(alpha_vals_plot, colors_nl):
    y = a * x**3 + (1 - a) * x
    label = r'$\alpha$=' + f'{a}' + (' (linear)' if a == 0 else '')
    lw = 2.5 if a == 0 else 1.8
    ls = '--' if a == 0 else '-'
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, label=label)
ax.plot(x, x, 'k:', linewidth=1, alpha=0.3)
ax.set_xlabel('Input x', fontsize=12)
ax.set_ylabel("Output x'", fontsize=12)
ax.set_title(r"(a) Nonlinear Transfer Function" + "\n" + r"$x' = \alpha x^3 + (1-\alpha)x$", fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# (b) Derivative (gain)
ax = axes[1]
for a, color in zip(alpha_vals_plot, colors_nl):
    dydx = 3 * a * x**2 + (1 - a)
    lw = 2.5 if a == 0 else 1.8
    ls = '--' if a == 0 else '-'
    ax.plot(x, dydx, color=color, linewidth=lw, linestyle=ls, label=r'$\alpha$=' + f'{a}')
ax.axhline(y=1, color='k', linestyle=':', alpha=0.3)
ax.set_xlabel('Input x', fontsize=12)
ax.set_ylabel("Gain dx'/dx", fontsize=12)
ax.set_title('(b) Gain Compression Effect\nDerivative Shows Non-Uniform Gain', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(-2, 2)
ax.set_ylim(0, 3.5)
ax.grid(True, alpha=0.3)

# 标注增益压缩区域
ax.annotate('Gain compression\nat large |x|', xy=(1.5, 3 * 0.5 * 1.5**2 + 0.5),
            xytext=(0.5, 3.2), fontsize=9, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/图9_非线性失真可视化.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: 图9_非线性失真可视化.png")

# ============================================================
# 图10: 误差累积机理
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

# 选择conv层（排除fc）按block分组
layer_order = []
for key in accum.keys():
    if 'fc' not in key:
        layer_order.append(key)

# 简化名称
def short_layer_name(name):
    n = name.replace('model.', '').replace('.conv', '')
    n = n.replace('.0.downsample.0', '.ds')
    n = n.replace('.0.', '.0.').replace('.1.', '.1.')
    return n

short_names = [short_layer_name(l) for l in layer_order]
x = np.arange(len(layer_order))
width = 0.15

alpha_show = ['0.1', '0.2', '0.3', '0.4', '0.5']
bar_colors = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

for i, (a, color) in enumerate(zip(alpha_show, bar_colors)):
    mse_vals = [accum[l]['alpha_errors'][a]['mse_mean'] for l in layer_order]
    offset = (i - 2) * width
    ax.bar(x + offset, mse_vals, width, label=r'$\alpha$=' + a,
           color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

# 按ResNet block标注背景色
block_colors = ['#e8f5e9', '#e3f2fd', '#fff3e0', '#fce4ec']
block_names = ['Layer1', 'Layer2', 'Layer3', 'Layer4']
block_starts = [0, 4, 8, 13]
block_ends = [4, 8, 13, 17]

for bc, bn, bs, be in zip(block_colors, block_names, block_starts, block_ends):
    ax.axvspan(bs - 0.5, be - 0.5, alpha=0.2, color=bc)
    ax.text((bs + be) / 2 - 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
            bn, ha='center', fontsize=9, fontweight='bold', color='gray')

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
ax.set_xlabel('Network Layer', fontsize=12)
ax.set_ylabel('MSE Error', fontsize=12)
ax.set_title('Per-Layer MSE Error at Different Nonlinearity Levels\nError Concentrated in First Conv of Each Block', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, title=r'$\alpha$', title_fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/图10_误差累积机理.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: 图10_误差累积机理.png")

# ============================================================
# 完成
# ============================================================

print("\n" + "=" * 60)
print("所有补充图表生成完成！")
print("=" * 60)
print(f"\n共生成 7 张新图表，保存至: {OUTPUT_DIR}/")
print("  图4_敏感性分析.png")
print("  图5_NAT训练曲线.png")
print("  图6_微调vs从头训练.png")
print("  图7_鲁棒性方法对比.png")
print("  图8_网络结构敏感性.png")
print("  图9_非线性失真可视化.png")
print("  图10_误差累积机理.png")
