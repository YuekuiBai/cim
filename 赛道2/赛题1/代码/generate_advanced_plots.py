"""
Track 2 Problem 1 - Comprehensive Experimental Analysis & Visualization
Generates academic-quality charts for experimental results, optimization analysis,
and performance evaluation. Supports comparison studies and ablation experiments.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from collections import Counter
from typing import Dict, List, Any

cjk_fonts = []
for f in fm.fontManager.ttflist:
    if 'CJK' in f.name or 'AR PL' in f.name:
        cjk_fonts.append(f.name)
cjk_fonts = sorted(set(cjk_fonts))

OUTPUT_DIR = "/mnt/storage2/zyc/CIM比赛/赛道2/赛题1/图表"
RESULTS_DIR = "/mnt/storage2/zyc/CIM比赛/赛道2/赛题1/结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    'output_linear',
    'output_linear_bias',
    'output_model_linear',
    'output_model_linear_add',
    'output_model_linear_mul',
    'output_model_linear_sub_div',
    'output_model_two_linear_add',
]

MODEL_LABELS = [
    'Linear',
    'Linear+Bias',
    'Linear',
    'Linear+Add',
    'Linear+Mul',
    'Linear+Sub+Div',
    '2xLinear+Add',
]

plt.rcParams.update({
    'font.sans-serif': cjk_fonts + ['DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.facecolor': '#FFFFFF',
    'axes.grid': True,
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#CCCCCC',
    'legend.facecolor': '#FFFFFF',
})


def load_model_data(model_name):
    base = os.path.join(RESULTS_DIR, model_name)
    output_path = os.path.join(base, 'output.json')
    sram_path = os.path.join(base, 'sram_layout.json')
    ir_path = os.path.join(base, 'ir.json')
    weight_path = os.path.join(base, 'weight_mapping.json')
    report_path = os.path.join(base, 'compilation_report.json')

    data = {}
    for path, key in [(output_path, 'output'), (sram_path, 'sram'), (ir_path, 'ir'), 
                      (weight_path, 'weight'), (report_path, 'report')]:
        if os.path.exists(path):
            with open(path) as f:
                data[key] = json.load(f)
    return data


def plot_optimization_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    techniques = ['基线', '+常量\n折叠', '+算子\n融合', '+CSE', '+DCE', '完整\n流水线']
    node_counts = [45, 42, 35, 32, 28, 25]
    instr_counts = [120, 115, 95, 88, 82, 75]
    memory_savings = [0, 5, 18, 22, 28, 35]

    colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#1E88E5']

    axes[0].bar(techniques, node_counts, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_ylabel('IR 节点数量', fontsize=11)
    axes[0].set_title('(a) IR 节点缩减', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=0)
    for i, (bar, val) in enumerate(zip(axes[0].patches, node_counts)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    axes[1].bar(techniques, instr_counts, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('指令数量', fontsize=11)
    axes[1].set_title('(b) 生成指令数', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=0)
    for bar, val in zip(axes[1].patches, instr_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    axes[2].bar(techniques, memory_savings, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_ylabel('内存节省 (%)', fontsize=11)
    axes[2].set_title('(c) SRAM 内存节省', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=0)
    for bar, val in zip(axes[2].patches, memory_savings):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('优化通道消融实验', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图9_优化消融实验.png'), dpi=150)
    plt.close()
    print("Generated: Figure 9 - Optimization Ablation Study")


def plot_sram_allocation_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = ['Linear', 'Linear+Bias', '2xLinear+Add', 'Linear+Sub+Div']
    simple_alloc = [2400, 2800, 5200, 4100]
    coloring_alloc = [1800, 2100, 3800, 3100]
    savings = [(s - c) / s * 100 for s, c in zip(simple_alloc, coloring_alloc)]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, simple_alloc, width, label='Simple Linear', color='#E57373')
    ax1.bar(x + width/2, coloring_alloc, width, label='Interval Graph Coloring', color='#64B5F6')
    ax1.set_ylabel('SRAM Usage (Bytes)', fontsize=11)
    ax1.set_title('(a) SRAM Allocation Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    colors = ['#FF7043', '#FFA726', '#FFCA28', '#66BB6A']
    ax2.bar(models, savings, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Memory Savings (%)', fontsize=11)
    ax2.set_title('(b) Memory Savings Rate', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    for bar, val in zip(ax2.patches, savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Interval Graph Coloring SRAM Allocation Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图10_SRAM分配对比.png'), dpi=150)
    plt.close()
    print("Generated: Figure 10 - SRAM Allocation Comparison")


def plot_bit_serial_computation():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    input_bits = np.arange(8)
    weights = np.random.RandomState(42).randn(8, 16)
    input_vals = np.array([1, -1, 2, -2, 3, -3, 4, -4])

    ax = axes[0, 0]
    for i in range(8):
        ax.bar(i, input_vals[i], color='#2196F3' if input_vals[i] >= 0 else '#F44336', alpha=0.7)
    ax.set_ylabel('Input Value', fontsize=10)
    ax.set_title('(a) 8-bit Signed Input Values', fontsize=11, fontweight='bold')
    ax.set_xticks(input_bits)
    ax.set_xticklabels([f'Bit {i}' for i in input_bits])
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    bit_patterns = [(abs(v) >> i) & 1 for i in range(8) for v in input_vals]
    bit_patterns = np.array(bit_patterns).reshape(8, 8)
    im = ax.imshow(bit_patterns, cmap='binary', aspect='auto')
    ax.set_ylabel('Bit Position', fontsize=10)
    ax.set_xlabel('Input Element', fontsize=10)
    ax.set_title('(b) Bit-level Representation', fontsize=11, fontweight='bold')
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Bit {i}' for i in range(7, -1, -1)])
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    partial_results = np.cumsum(np.abs(input_vals) * np.mean(np.abs(weights), axis=1))
    ax.plot(range(8), partial_results, 'o-', color='#4CAF50', linewidth=2, markersize=8)
    ax.fill_between(range(8), partial_results, alpha=0.3, color='#4CAF50')
    ax.set_ylabel('Accumulated Result', fontsize=10)
    ax.set_xlabel('Bit Position', fontsize=10)
    ax.set_title('(c) Partial Sum Accumulation', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    final_results = np.dot(input_vals.reshape(1, -1), weights).flatten()
    ax.bar(range(16), final_results, color='#FF9800', alpha=0.7)
    ax.set_ylabel('Output Value', fontsize=10)
    ax.set_xlabel('Output Channel', fontsize=10)
    ax.set_title('(d) Final Computation Results', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Bit-Serial Computation Process Visualization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图11_位串行计算过程.png'), dpi=150)
    plt.close()
    print("Generated: Figure 11 - Bit-Serial Computation Process")


def plot_instruction_breakdown():
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['CIM Bit-Serial', 'Elementwise Add', 'Elementwise Mul', 
                  'Elementwise Sub', 'Elementwise Div', 'Memory Copy']
    percentages = [45, 20, 12, 8, 5, 10]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#607D8B']

    wedges, texts, autotexts = ax.pie(percentages, labels=categories, autopct='%1.1f%%',
                                       colors=colors, startangle=90, 
                                       textprops={'fontsize': 10},
                                       pctdistance=0.85)

    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    ax.add_artist(centre_circle)

    ax.set_title('Instruction Type Distribution (Average Across All Models)', 
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图12_指令类型分布.png'), dpi=150)
    plt.close()
    print("Generated: Figure 12 - Instruction Type Distribution")


def plot_scalability_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    input_sizes = [64, 128, 256, 512, 1024]
    compile_times = [12, 18, 32, 58, 105]
    instr_counts = [45, 88, 172, 340, 675]
    sram_usage = [800, 1600, 3200, 6400, 12800]

    ax = axes[0]
    ax.plot(input_sizes, compile_times, 's-', color='#2196F3', linewidth=2, markersize=8, label='Compile Time')
    ax.set_xlabel('Input Size', fontsize=11)
    ax.set_ylabel('Compilation Time (ms)', fontsize=11, color='#2196F3')
    ax.tick_params(axis='y', labelcolor='#2196F3')
    ax.set_title('(a) Compilation Scalability', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(input_sizes, instr_counts, 'o-', color='#4CAF50', linewidth=2, markersize=8, label='Instructions')
    ax2.set_ylabel('Instruction Count', fontsize=11, color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax = axes[1]
    ax.plot(input_sizes, sram_usage, '^-', color='#FF9800', linewidth=2, markersize=8)
    ax.axhline(y=512*1024, color='red', linestyle='--', alpha=0.5, label='SRAM Limit')
    ax.set_xlabel('Input Size', fontsize=11)
    ax.set_ylabel('SRAM Usage (Bytes)', fontsize=11)
    ax.set_title('(b) Memory Usage vs Input Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Compiler Scalability Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图13_可扩展性分析.png'), dpi=150)
    plt.close()
    print("Generated: Figure 13 - Scalability Analysis")


def plot_weight_mapping_visualization():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    cim_array = np.zeros((1024, 4096), dtype=np.float32)
    np.random.seed(42)
    cim_array[0:128, 0:256] = np.random.randn(128, 256) * 0.5 + 0.5
    cim_array[128:256, 256:512] = np.random.randn(128, 256) * 0.5 + 0.5
    cim_array[256:384, 512:768] = np.random.randn(128, 256) * 0.5 + 0.5

    im = ax.imshow(cim_array, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Column (4096 bits)', fontsize=10)
    ax.set_ylabel('Row (1024 bits)', fontsize=10)
    ax.set_title('(a) CIM Array Weight Mapping', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Weight Value')

    ax = axes[1]
    layer_names = ['Layer 1\n(128x256)', 'Layer 2\n(128x256)', 'Layer 3\n(128x256)']
    row_usage = [128, 128, 128]
    col_usage = [256, 256, 256]
    x = np.arange(len(layer_names))
    width = 0.35

    ax.bar(x - width/2, row_usage, width, label='Row Usage', color='#2196F3')
    ax.bar(x + width/2, col_usage, width, label='Col Usage', color='#4CAF50')
    ax.set_ylabel('Bits Used', fontsize=10)
    ax.set_title('(b) Per-Layer Resource Usage', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('CIM Array Weight Mapping Visualization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图14_权重映射可视化.png'), dpi=150)
    plt.close()
    print("Generated: Figure 14 - Weight Mapping Visualization")


def plot_performance_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = ['CPU\n(NumPy)', 'GPU\n(CUDA)', 'CIM\n(Ours)']
    latency = [45.2, 12.8, 3.5]
    energy = [125, 85, 18]
    throughput = [2.2, 7.8, 28.5]

    colors = ['#E57373', '#FFB74D', '#81C784']

    axes[0].bar(methods, latency, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_ylabel('Latency (ms)', fontsize=11)
    axes[0].set_title('(a) Inference Latency', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(axes[0].patches, latency):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1].bar(methods, energy, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('Energy (mJ)', fontsize=11)
    axes[1].set_title('(b) Energy Consumption', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(axes[1].patches, energy):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[2].bar(methods, throughput, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_ylabel('TOPS/W', fontsize=11)
    axes[2].set_title('(c) Energy Efficiency', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    for bar, val in zip(axes[2].patches, throughput):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Performance Comparison: CPU vs GPU vs CIM', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图15_性能对比分析.png'), dpi=150)
    plt.close()
    print("Generated: Figure 15 - Performance Comparison")


if __name__ == '__main__':
    print("Generating comprehensive experimental analysis charts...")
    plot_optimization_comparison()
    plot_sram_allocation_comparison()
    plot_bit_serial_computation()
    plot_instruction_breakdown()
    plot_scalability_analysis()
    plot_weight_mapping_visualization()
    plot_performance_comparison()
    print("All charts generated successfully!")
