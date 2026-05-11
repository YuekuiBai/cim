"""
Track 2 Problem 2 - Experimental Results Plotting Script
Generates all experiment-related visualization charts
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from collections import Counter

cjk_fonts = []
for f in fm.fontManager.ttflist:
    if 'CJK' in f.name or 'AR PL' in f.name:
        cjk_fonts.append(f.name)
cjk_fonts = sorted(set(cjk_fonts))

OUTPUT_DIR = "/mnt/storage2/zyc/CIM比赛/赛道2/赛题2/图表"
RESULTS_DIR = "/mnt/storage2/zyc/CIM比赛/赛道2/赛题2/结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def load_result(model_name, filename):
    path = os.path.join(RESULTS_DIR, model_name, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def plot_subcube_usage():
    """图1: Sub-Cube 使用分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, title) in enumerate([('output_simple', '简单模型'), ('output_moe', 'MoE 模型')]):
        ax = axes[idx]
        solution = load_result(model, 'solution.json')
        placements = solution.get('placements', [])

        subcube_counts = Counter()
        subcube_volumes = {}
        for p in placements:
            sc_id = p['subcube_id']
            subcube_counts[sc_id] += 1
            subcube_volumes[sc_id] = subcube_volumes.get(sc_id, 0) + p.get('volume', 0)

        sc_ids = sorted(subcube_counts.keys())
        counts = [subcube_counts[s] for s in sc_ids]

        colors = plt.cm.Set3(np.linspace(0, 1, len(sc_ids)))
        ax.bar([str(s) for s in sc_ids], counts, color=colors, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Sub-Cube 编号', fontsize=11)
        ax.set_ylabel('放置次数', fontsize=11)
        ax.set_title(f'{title}: Sub-Cube 使用分布', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('赛题二：Sub-Cube 使用分布', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图1_SubCube使用分布.png'), dpi=150)
    plt.close()
    print("Generated: Figure 1 - Sub-Cube Usage Distribution")


def plot_volume_distribution():
    """图2: 各 Sub-Cube 体积分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, title) in enumerate([('output_simple', '简单模型'), ('output_moe', 'MoE 模型')]):
        ax = axes[idx]
        solution = load_result(model, 'solution.json')
        placements = solution.get('placements', [])

        subcube_volumes = {}
        for p in placements:
            sc_id = p['subcube_id']
            subcube_volumes[sc_id] = subcube_volumes.get(sc_id, 0) + p.get('volume', 0)

        sc_ids = sorted(subcube_volumes.keys())
        volumes = [subcube_volumes[s] for s in sc_ids]

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sc_ids)))
        ax.bar([str(s) for s in sc_ids], volumes, color=colors, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Sub-Cube 编号', fontsize=11)
        ax.set_ylabel('总体积', fontsize=11)
        ax.set_title(f'{title}: 各 Sub-Cube 体积分布', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('赛题二：各 Sub-Cube 体积分布', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图2_体积分布.png'), dpi=150)
    plt.close()
    print("Generated: Figure 2 - Volume Distribution")


def plot_z_depth_usage():
    """图3: Z 轴深度使用"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, title) in enumerate([('output_simple', '简单模型'), ('output_moe', 'MoE 模型')]):
        ax = axes[idx]
        solution = load_result(model, 'solution.json')
        placements = solution.get('placements', [])

        subcube_z_ranges = {}
        for p in placements:
            sc_id = p['subcube_id']
            z_range = p.get('z', [0, 1])
            if sc_id not in subcube_z_ranges:
                subcube_z_ranges[sc_id] = []
            subcube_z_ranges[sc_id].append(z_range)

        sc_ids = sorted(subcube_z_ranges.keys())
        max_z = [max(zr[1] for zr in subcube_z_ranges[s]) for s in sc_ids]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sc_ids)))
        ax.bar([str(s) for s in sc_ids], max_z, color=colors, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Sub-Cube 编号', fontsize=11)
        ax.set_ylabel('最大 Z 深度', fontsize=11)
        ax.set_title(f'{title}: Z 轴深度使用', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('赛题二：Z 轴深度使用', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_Z轴深度使用.png'), dpi=150)
    plt.close()
    print("Generated: Figure 3 - Z-Axis Depth Usage")


def plot_model_comparison():
    """图4: 模型对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['算子数', 'Section 数', '使用 Sub-Cube 数', '放置次数']
    simple_vals = []
    moe_vals = []

    for model in ['output_simple', 'output_moe']:
        solution = load_result(model, 'solution.json')
        placements = solution.get('placements', [])
        subcubes_used = len(set(p['subcube_id'] for p in placements))

        sections_data = load_result(model, 'weight_sections.json')
        if isinstance(sections_data, list):
            num_sections = len(sections_data)
        else:
            num_sections = len(sections_data.get('sections', []))

        operators_data = load_result(model, 'parsed_operators.json')
        if isinstance(operators_data, list):
            num_ops = len(operators_data)
        else:
            num_ops = len(operators_data.get('operators', []))

        if model == 'output_simple':
            simple_vals = [num_ops, num_sections, subcubes_used, len(placements)]
        else:
            moe_vals = [num_ops, num_sections, subcubes_used, len(placements)]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, simple_vals, width, label='简单模型', color='#2196F3')
    ax.bar(x + width/2, moe_vals, width, label='MoE 模型', color='#FF9800')

    for i, (sv, mv) in enumerate(zip(simple_vals, moe_vals)):
        ax.text(i - width/2, sv + max(simple_vals)*0.02, str(sv), ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, mv + max(moe_vals)*0.02, str(mv), ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('数量', fontsize=11)
    ax.set_title('赛题二：简单模型 vs MoE 模型对比', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图4_模型对比.png'), dpi=150)
    plt.close()
    print("Generated: Figure 4 - Model Comparison")


def plot_3d_placement_visualization():
    """图5: 3D 映射可视化"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    solution = load_result('output_simple', 'solution.json')
    placements = solution.get('placements', [])

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for i, p in enumerate(placements):
        z_range = p.get('z', [0, 1])
        y_range = p.get('y', [0, 1])
        x_range = p.get('x', [0, 1])

        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]
        dz = z_range[1] - z_range[0]

        ax.bar3d(x_range[0], y_range[0], z_range[0], dx, dy, dz,
                 color=colors[i % len(colors)], alpha=0.8, edgecolor='white')

        ax.text(x_range[0] + dx/2, y_range[0] + dy/2, z_range[0] + dz/2,
                p['section'][:12], ha='center', va='center', fontsize=7)

    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    ax.set_title('赛题二：3D 映射可视化 (简单模型)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图5_3D映射可视化.png'), dpi=150)
    plt.close()
    print("Generated: Figure 5 - 3D Placement Visualization")


def plot_subcube_load_balance():
    """图6: Sub-Cube 负载均衡分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, title) in enumerate([('output_simple', '简单模型'), ('output_moe', 'MoE 模型')]):
        ax = axes[idx]
        solution = load_result(model, 'solution.json')
        placements = solution.get('placements', [])

        subcube_volumes = {}
        for p in placements:
            sc_id = p['subcube_id']
            subcube_volumes[sc_id] = subcube_volumes.get(sc_id, 0) + p.get('volume', 0)

        sc_ids = sorted(subcube_volumes.keys())
        volumes = [subcube_volumes[s] for s in sc_ids]

        if len(volumes) > 1:
            mean_vol = np.mean(volumes)
            std_vol = np.std(volumes)
            cv = std_vol / mean_vol * 100 if mean_vol > 0 else 0
        else:
            cv = 0

        ax.bar([str(s) for s in sc_ids], volumes, color=plt.cm.Set2(np.linspace(0, 1, len(sc_ids))),
               edgecolor='white', linewidth=0.5)
        ax.axhline(y=np.mean(volumes), color='red', linestyle='--', alpha=0.7, label=f'均值: {np.mean(volumes):.0f}')

        ax.text(0.95, 0.95, f'CV: {cv:.1f}%', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Sub-Cube 编号', fontsize=11)
        ax.set_ylabel('体积', fontsize=11)
        ax.set_title(f'{title}: 负载均衡分析', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('赛题二：Sub-Cube 负载均衡分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图6_负载均衡分析.png'), dpi=150)
    plt.close()
    print("Generated: Figure 6 - Load Balance Analysis")


def plot_placement_heatmap():
    """图7: Sub-Cube 映射热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    solution = load_result('output_moe', 'solution.json')
    placements = solution.get('placements', [])

    max_sc = max(p['subcube_id'] for p in placements) + 1
    grid_size = int(np.ceil(np.sqrt(max_sc)))

    heatmap = np.zeros((grid_size, grid_size))
    for p in placements:
        sc_id = p['subcube_id']
        row = sc_id // grid_size
        col = sc_id % grid_size
        heatmap[row][col] += p.get('volume', 0)

    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')
    ax.set_title('赛题二：Sub-Cube 映射热力图 (MoE 模型)', fontsize=13, fontweight='bold')
    ax.set_xlabel('列索引', fontsize=11)
    ax.set_ylabel('行索引', fontsize=11)

    for i in range(grid_size):
        for j in range(grid_size):
            val = heatmap[i][j]
            if val > 0:
                label = f'{val/1e6:.1f}M' if val > 1e6 else f'{val/1e3:.0f}K'
                ax.text(j, i, label, ha='center', va='center', fontsize=8, fontweight='bold')

    plt.colorbar(im, ax=ax, label='总体积')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图7_映射热力图.png'), dpi=150)
    plt.close()
    print("Generated: Figure 7 - Placement Heatmap")


def plot_section_size_distribution():
    """图8: Section 大小分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, title) in enumerate([('output_simple', '简单模型'), ('output_moe', 'MoE 模型')]):
        ax = axes[idx]
        solution = load_result(model, 'solution.json')
        placements = solution.get('placements', [])

        volumes = [p.get('volume', 0) for p in placements]

        # Use box plot for better distribution visualization
        bp = ax.boxplot(volumes, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='#2196F3', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='#2196F3'),
                        capprops=dict(color='#2196F3'))

        # Add statistics annotation
        stats_text = f'均值: {np.mean(volumes):.0f}\n中位数: {np.median(volumes):.0f}\n标准差: {np.std(volumes):.0f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel('Section 体积', fontsize=11)
        ax.set_title(f'{title}: Section 大小分布', fontsize=12, fontweight='bold')
        ax.set_xticks([1])
        ax.set_xticklabels(['所有 Section'], fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('赛题二：Section 大小分布 (箱线图)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图8_Section大小分布.png'), dpi=150)
    plt.close()
    print("Generated: Figure 8 - Section Size Distribution")


if __name__ == '__main__':
    print("Generating Problem 2 experimental result charts...")
    plot_subcube_usage()
    plot_volume_distribution()
    plot_z_depth_usage()
    plot_model_comparison()
    plot_3d_placement_visualization()
    plot_subcube_load_balance()
    plot_placement_heatmap()
    plot_section_size_distribution()
    print("Problem 2 chart generation complete!")
