import numpy as np
import matplotlib.pyplot as plt
import json
import os
import yaml

def load_results(output_dir: str):
    results = {}
    result_files = [
        'array_simulation_results.json',
        'nonlinearity_sweep_results.json',
        'network_mapping_results.json',
        'performance_evaluation_results.json'
    ]

    for fname in result_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                key = fname.replace('.json', '')
                results[key] = json.load(f)

    config_path = os.path.join(os.path.dirname(output_dir), '代码', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            results['config'] = yaml.safe_load(f)

    return results

def plot_nonlinearity_sweep(results, output_dir: str):
    if 'nonlinearity_sweep_results' not in results:
        return

    nl_data = results['nonlinearity_sweep_results']
    alpha_values = nl_data['alpha_values']
    snr_values = nl_data['snr_values']
    power_values = nl_data['power_values']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(alpha_values, snr_values, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Nonlinearity Factor (alpha)', fontsize=12)
    axes[0].set_ylabel('SNR (dB)', fontsize=12)
    axes[0].set_title('SNR vs Nonlinearity', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, max(alpha_values)])

    axes[1].plot(alpha_values, power_values, 'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel('Nonlinearity Factor (alpha)', fontsize=12)
    axes[1].set_ylabel('Power (W)', fontsize=12)
    axes[1].set_title('Power vs Nonlinearity', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, max(alpha_values)])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nonlinearity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'nonlinearity_analysis.png')}")

def plot_ppa_metrics(results, output_dir: str):
    if 'performance_evaluation_results' not in results:
        return

    ppa = results['performance_evaluation_results']['ppa']
    accuracy = results['performance_evaluation_results']['accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ppa_labels = ['Performance\n(TOPS)', 'Power\n(W)', 'Area\n(mm²)']
    ppa_values = [ppa['performance_tops'], ppa['power_w'], ppa['area_mm2']]
    colors = ['steelblue', 'coral', 'seagreen']

    bars = axes[0].bar(ppa_labels, ppa_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_title('PPA Metrics', fontsize=14)
    axes[0].set_ylabel('Value')
    for bar, val in zip(bars, ppa_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10)

    efficiency_labels = ['Energy Efficiency\n(TOPS/W)', 'Area Efficiency\n(TOPS/mm²)']
    efficiency_values = [ppa['energy_efficiency_tops_w'], ppa['area_efficiency_tops_mm2']]
    colors2 = ['coral', 'seagreen']

    bars2 = axes[1].bar(efficiency_labels, efficiency_values, color=colors2, alpha=0.7, edgecolor='black')
    axes[1].set_title('Efficiency Metrics', fontsize=14)
    axes[1].set_ylabel('Value')
    for bar, val in zip(bars2, efficiency_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10)

    acc_labels = ['Baseline\n(FP32)', 'Expected\n(CIM)']
    acc_values = [accuracy['baseline_accuracy'], accuracy['expected_accuracy']]
    colors3 = ['steelblue', 'tomato']

    bars3 = axes[2].bar(acc_labels, acc_values, color=colors3, alpha=0.7, edgecolor='black')
    axes[2].set_title('Accuracy Comparison', fontsize=14)
    axes[2].set_ylabel('Top-1 Accuracy (%)')
    axes[2].set_ylim([60, 75])
    for bar, val in zip(bars3, acc_values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppa_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'ppa_metrics.png')}")

def plot_layer_mapping(results, output_dir: str):
    if 'network_mapping_results' not in results:
        return

    mapping = results['network_mapping_results']
    layer_names = [f"L{i+1}" for i in range(len(mapping['layer_mappings']))]
    arrays_needed = [lm['arrays_needed'] for lm in mapping['layer_mappings']]
    utilizations = [lm['utilization'] * 100 for lm in mapping['layer_mappings']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    bars = axes[0].bar(layer_names, arrays_needed, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Number of Arrays', fontsize=12)
    axes[0].set_title('Array Usage per Layer', fontsize=14)
    for bar, val in zip(bars, arrays_needed):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{val}',
                    ha='center', va='bottom', fontsize=9)

    bars2 = axes[1].bar(layer_names, utilizations, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Utilization (%)', fontsize=12)
    axes[1].set_title('Array Utilization per Layer', fontsize=14)
    axes[1].set_ylim([0, 100])
    for bar, val in zip(bars2, utilizations):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_mapping.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'layer_mapping.png')}")

def plot_latency_breakdown(results, output_dir: str):
    if 'performance_evaluation_results' not in results:
        return

    latency = results['performance_evaluation_results']['latency']

    labels = ['DAC', 'Sense Amp', 'ADC']
    values = [latency['dac_latency_ns'], latency['sense_latency_ns'], latency['adc_latency_ns']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 12})
    ax.set_title('Latency Breakdown', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'latency_breakdown.png')}")

def plot_array_simulation(results, output_dir: str):
    if 'array_simulation_results' not in results:
        return

    arr_result = results['array_simulation_results']

    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ['Max Current', 'Min Current', 'Avg Power', 'Array Efficiency']
    values = [arr_result['max_current'], arr_result['min_current'],
              arr_result['avg_power'] * 1e6, arr_result['array_efficiency'] * 1e9]
    units = ['A', 'A', 'μW', 'nW/cell']

    colors = ['steelblue', 'coral', 'seagreen', 'gold']
    bars = ax.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Array Simulation Results', fontsize=14)

    for bar, val, unit in zip(bars, values, units):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2., f' {val:.2e} {unit}',
                ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'array_simulation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'array_simulation.png')}")

def generate_all_charts(output_dir: str):
    print("=" * 60)
    print("生成图表")
    print("=" * 60)

    results = load_results(output_dir)

    plot_nonlinearity_sweep(results, output_dir)
    plot_ppa_metrics(results, output_dir)
    plot_layer_mapping(results, output_dir)
    plot_latency_breakdown(results, output_dir)
    plot_array_simulation(results, output_dir)

    print("\n所有图表生成完成!")

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "../结果"
    generate_all_charts(output_dir)
