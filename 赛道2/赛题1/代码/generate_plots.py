"""
Track 2 Problem 1 - Experimental Results Plotting Script
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

# Detect available CJK fonts
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
    'figure.facecolor': '#FFFFFF',
})


def load_model_data(model_name):
    base = os.path.join(RESULTS_DIR, model_name)
    output_path = os.path.join(base, 'output.json')
    sram_path = os.path.join(base, 'sram_layout.json')
    ir_path = os.path.join(base, 'ir.json')
    weight_path = os.path.join(base, 'weight_mapping.json')

    data = {}
    for path, key in [(output_path, 'output'), (sram_path, 'sram'), (ir_path, 'ir'), (weight_path, 'weight')]:
        if os.path.exists(path):
            with open(path) as f:
                data[key] = json.load(f)
    return data


def plot_instruction_counts():
    """Figure 1: Instruction count statistics per model (stacked bar chart)"""
    fig, ax = plt.subplots(figsize=(14, 7))

    opcode_labels = {
        'cim.bit.i8': 'CIM Bit-Serial (i8)',
        'cim.bit.i32': 'CIM Bit-Serial (i32)',
        'elt.mul.i32.vi': 'Elementwise Mul (vi)',
        'elt.mul.i32.vv': 'Elementwise Mul (vv)',
        'elt.add.i32.vv': 'Elementwise Add (vv)',
        'elt.add.i32.vi': 'Elementwise Add (vi)',
        'elt.sub.i32.vv': 'Elementwise Sub (vv)',
        'elt.sub.i32.vi': 'Elementwise Sub (vi)',
        'elt.div.i32.vi': 'Elementwise Div (vi)',
        'mem.copy.i32.i32': 'Memory Copy',
    }

    opcode_colors = {
        'cim.bit.i8': '#1565C0',
        'cim.bit.i32': '#1976D2',
        'elt.mul.i32.vi': '#2E7D32',
        'elt.mul.i32.vv': '#388E3C',
        'elt.add.i32.vv': '#E65100',
        'elt.add.i32.vi': '#F57C00',
        'elt.sub.i32.vv': '#C62828',
        'elt.sub.i32.vi': '#D32F2F',
        'elt.div.i32.vi': '#6A1B9A',
        'mem.copy.i32.i32': '#455A64',
    }

    all_opcodes = sorted(opcode_colors.keys())
    bottom = np.zeros(len(MODELS))

    for opcode in all_opcodes:
        counts = []
        for m in MODELS:
            data = load_model_data(m)
            insts = data.get('output', {}).get('instructions', [])
            opcodes = [i['opcode'] for i in insts]
            counts.append(opcodes.count(opcode))
        if sum(counts) > 0:
            ax.bar(MODEL_LABELS, counts, bottom=bottom, label=opcode_labels[opcode],
                   color=opcode_colors[opcode], edgecolor='white', linewidth=0.8, width=0.7)
            bottom += np.array(counts)

    ax.set_ylabel('指令数量', fontsize=13, fontweight='bold')
    ax.set_title('赛题一：各模型指令数量统计', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9.5, frameon=True,
              shadow=False, edgecolor='#CCCCCC', title='指令类型', title_fontsize=10)
    ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图1_指令数量统计.png'), dpi=150, facecolor='white')
    plt.close()
    print("Generated: Figure 1 - Instruction Count Statistics")


def plot_sram_usage():
    """Figure 2: SRAM usage comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    sram_usage = []
    for m in MODELS:
        data = load_model_data(m)
        sram = data.get('sram', {})
        total = sram.get('total_used', 0)
        sram_usage.append(total)

    colors = ['#1565C0', '#2E7D32', '#E65100', '#C62828', '#6A1B9A', '#00838F', '#4E342E']
    bars = ax.bar(MODEL_LABELS, sram_usage, color=colors, edgecolor='white', linewidth=0.8, width=0.7)

    for bar, val in zip(bars, sram_usage):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{val} B', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=512 * 1024, color='#C62828', linestyle='--', alpha=0.6, linewidth=1.5, label='SRAM 上限 (512KB)')
    ax.set_ylabel('SRAM 使用量 (Bytes)', fontsize=13, fontweight='bold')
    ax.set_title('赛题一：各模型 SRAM 使用量对比', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=10, frameon=True, edgecolor='#CCCCCC')
    ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图2_SRAM使用量.png'), dpi=150, facecolor='white')
    plt.close()
    print("Generated: Figure 2 - SRAM Usage")


def plot_sram_region_breakdown():
    """Figure 3: SRAM region allocation breakdown"""
    fig, ax = plt.subplots(figsize=(14, 7))

    region_types = ['input', 'output', 'acc', 'tmp', 'bias']
    region_labels = {'input': 'Input Buffer', 'output': 'Output Buffer', 'acc': 'Accumulator',
                     'tmp': 'Temporary', 'bias': 'Bias'}
    region_colors = {'input': '#1565C0', 'output': '#2E7D32', 'acc': '#E65100',
                     'tmp': '#C62828', 'bias': '#6A1B9A'}

    bottom = np.zeros(len(MODELS))

    for rtype in region_types:
        sizes = []
        for m in MODELS:
            data = load_model_data(m)
            sram = data.get('sram', {})
            regions = sram.get('regions', [])
            total = sum(r['size'] for r in regions if rtype in r.get('name', '').lower())
            sizes.append(total)
        if sum(sizes) > 0:
            ax.bar(MODEL_LABELS, sizes, bottom=bottom, label=region_labels[rtype],
                   color=region_colors[rtype], edgecolor='white', linewidth=0.8, width=0.7)
            bottom += np.array(sizes)

    ax.set_ylabel('SRAM 大小 (Bytes)', fontsize=13, fontweight='bold')
    ax.set_title('赛题一：SRAM 区域分配详情', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=10, frameon=True, edgecolor='#CCCCCC')
    ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_SRAM区域分配.png'), dpi=150, facecolor='white')
    plt.close()
    print("Generated: Figure 3 - SRAM Region Allocation")


def plot_opcode_distribution():
    """Figure 4: Instruction type distribution per model (pie charts)"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        data = load_model_data(m)
        insts = data.get('output', {}).get('instructions', [])
        opcodes = [i['opcode'] for i in insts]
        counts = Counter(opcodes)

        ax = axes[idx]
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 8},
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.0})
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title(MODEL_LABELS[idx], fontsize=11, fontweight='bold', pad=8)

        legend_labels = []
        for l in labels:
            if 'cim' in l:
                legend_labels.append(f'CIM: {l}')
            elif 'elt' in l:
                legend_labels.append(f'ELT: {l}')
            elif 'mem' in l:
                legend_labels.append(f'MEM: {l}')
            else:
                legend_labels.append(l)
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=True,
                  edgecolor='#CCCCCC')

    axes[-1].axis('off')

    plt.suptitle('赛题一：各模型指令类型分布', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, '图4_指令类型占比.png'), dpi=150)
    plt.close()
    print("Generated: Figure 4 - Instruction Type Distribution")


def plot_compilation_flow():
    """Figure 5: Instruction category statistics"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(MODELS))
    width = 0.25

    cim_counts = []
    elt_counts = []
    mem_counts = []

    for m in MODELS:
        data = load_model_data(m)
        insts = data.get('output', {}).get('instructions', [])
        opcodes = [i['opcode'] for i in insts]
        cim = sum(1 for o in opcodes if 'cim' in o)
        elt = sum(1 for o in opcodes if 'elt' in o)
        mem = sum(1 for o in opcodes if 'mem' in o)
        cim_counts.append(cim)
        elt_counts.append(elt)
        mem_counts.append(mem)

    ax.bar(x - width, cim_counts, width, label='CIM 指令', color='#1565C0', edgecolor='white', linewidth=0.8)
    ax.bar(x, elt_counts, width, label='Elementwise 指令', color='#2E7D32', edgecolor='white', linewidth=0.8)
    ax.bar(x + width, mem_counts, width, label='Memory Copy 指令', color='#E65100', edgecolor='white', linewidth=0.8)

    ax.set_ylabel('指令数量', fontsize=13, fontweight='bold')
    ax.set_title('赛题一：指令分类统计', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10, frameon=True, edgecolor='#CCCCCC')
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图5_指令分类统计.png'), dpi=150, facecolor='white')
    plt.close()
    print("Generated: Figure 5 - Instruction Category Statistics")


def plot_sram_utilization_rate():
    """Figure 6: SRAM utilization rate (percentage)"""
    fig, ax = plt.subplots(figsize=(12, 6))

    SRAM_CAPACITY = 512 * 1024
    rates = []
    for m in MODELS:
        data = load_model_data(m)
        sram = data.get('sram', {})
        total = sram.get('total_used', 0)
        rates.append(total / SRAM_CAPACITY * 100)

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(rates)))
    bars = ax.bar(MODEL_LABELS, rates, color=colors, edgecolor='white', linewidth=0.8, width=0.7)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('SRAM 利用率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('赛题一：SRAM 利用率（总容量 512KB）', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图6_SRAM利用率.png'), dpi=150, facecolor='white')
    plt.close()
    print("Generated: Figure 6 - SRAM Utilization Rate")


def plot_ir_structure():
    """Figure 7: IR computation graph with data flow (model_two_linear_add)"""
    import networkx as nx

    data = load_model_data('output_model_two_linear_add')
    ir = data.get('ir', {})
    nodes = ir.get('nodes', [])
    tensors = ir.get('tensors', {})

    node_colors = {'linear': '#1565C0', 'elementwise': '#2E7D32'}
    tensor_color = '#78909C'
    node_labels_map = {'linear': 'Linear', 'elementwise': 'Elementwise'}

    G = nx.DiGraph()
    op_node_ids = []

    for node in nodes:
        node_id = node['id']
        node_type = node.get('type', 'unknown')
        inputs = node.get('inputs', [])
        outputs = node.get('outputs', [])
        attrs = node.get('attrs', {})

        label = node_labels_map.get(node_type, node_type)
        if attrs.get('op'):
            label += f'\n({attrs["op"]})'

        G.add_node(node_id, label=label, type='op', node_type=node_type)
        op_node_ids.append(node_id)

        for inp in inputs:
            G.add_edge(inp, node_id, edge_type='tensor_in')
        for out in outputs:
            G.add_edge(node_id, out, edge_type='tensor_out')

    op_node_ids.sort()

    tensor_names = []
    for nid in op_node_ids:
        node = [n for n in nodes if n['id'] == nid][0]
        for inp in node.get('inputs', []):
            if inp not in tensor_names:
                tensor_names.append(inp)
        for out in node.get('outputs', []):
            if out not in tensor_names:
                tensor_names.append(out)

    pos = {}
    n_ops = len(op_node_ids)
    n_tensors = len(tensor_names)

    for i, nid in enumerate(op_node_ids):
        pos[nid] = (0.72, 0.88 - i * 0.22)

    for i, tname in enumerate(tensor_names):
        pos[tname] = (0.28, 0.77 - (i + 0.5) * 0.22)

    fig, ax = plt.subplots(figsize=(12, 6))

    op_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'op']
    tensor_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != 'op']

    for ntype, color in node_colors.items():
        nodelist = [n for n in op_nodes if G.nodes[n].get('node_type') == ntype]
        if nodelist:
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=color,
                                   node_size=4500, alpha=0.92, edgecolors='white',
                                   linewidths=2.5, ax=ax, node_shape='s')

    if tensor_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=tensor_nodes, node_color=tensor_color,
                               node_size=1800, alpha=0.85, edgecolors='white',
                               linewidths=1.5, ax=ax, node_shape='o')

    node_labels = {n: G.nodes[n]['label'] for n in op_nodes}
    tensor_labels = {n: n for n in tensor_nodes}

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold',
                            font_color='white', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=tensor_labels, font_size=8, font_weight='normal',
                            font_color='#333333', ax=ax)

    edges = G.edges()
    for u, v in edges:
        edge_type = G.edges[u, v].get('edge_type', 'tensor_out')
        if edge_type == 'tensor_in':
            color = '#90A4AE'
            style = 'dashed'
            width = 1.5
        else:
            color = '#37474F'
            style = 'solid'
            width = 2.0
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], style=style,
                               edge_color=color, width=width, arrows=True,
                               arrowsize=15, alpha=0.7, ax=ax,
                               arrowstyle='-|>', connectionstyle='arc3,rad=0')

    legend_ops = [plt.Line2D([0], [0], marker='s', color='white', markerfacecolor=node_colors[t],
                              markersize=12, label=node_labels_map[t], markeredgecolor='white',
                              markeredgewidth=2, linestyle='None') for t in node_colors]
    legend_tensor = [plt.Line2D([0], [0], marker='o', color='white', markerfacecolor=tensor_color,
                                 markersize=10, label='Tensor', markeredgecolor='white',
                                 markeredgewidth=1.5, linestyle='None')]
    ax.legend(handles=legend_ops + legend_tensor, loc='upper right', frameon=True,
              edgecolor='#CCCCCC', fontsize=10, title='图例', title_fontsize=11)

    ax.set_title('赛题一：IR 计算图数据流 (model_two_linear_add)', fontsize=15, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图7_IR计算图结构.png'), dpi=150, facecolor='white')
    plt.close()
    print("Generated: Figure 7 - IR Graph Structure")


def plot_total_instructions_comparison():
    """Figure 8: Total instruction count comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    totals = []
    for m in MODELS:
        data = load_model_data(m)
        insts = data.get('output', {}).get('instructions', [])
        totals.append(len(insts))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(totals)))
    bars = ax.bar(MODEL_LABELS, totals, color=colors, edgecolor='white', linewidth=0.8, width=0.7)

    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('总指令数', fontsize=13, fontweight='bold')
    ax.set_title('赛题一：总指令数对比', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticklabels(MODEL_LABELS, rotation=25, ha='right', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图8_总指令数对比.png'), dpi=150, facecolor='white')
    plt.close()
    print("Generated: Figure 8 - Total Instruction Count Comparison")


if __name__ == '__main__':
    print("Generating Problem 1 experimental result charts...")
    plot_instruction_counts()
    plot_sram_usage()
    plot_sram_region_breakdown()
    plot_opcode_distribution()
    plot_compilation_flow()
    plot_sram_utilization_rate()
    plot_ir_structure()
    plot_total_instructions_comparison()
    print("Problem 1 chart generation complete!")
