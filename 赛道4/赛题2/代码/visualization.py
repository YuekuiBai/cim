import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict

def plot_iv_characteristics(results: Dict, output_dir: str):
    fet_data = results['fet_iv_characteristics']
    vgs = np.array(fet_data['vgs'])
    ids = np.array(fet_data['ids'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(vgs, ids, 'b-', linewidth=2)
    ax.set_xlabel('V_GS (V)', fontsize=12)
    ax.set_ylabel('I_DS (A)', fontsize=12)
    ax.set_title('FeFET I_DS vs V_GS Characteristics', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.8])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fet_iv_curve.png'), dpi=150)
    plt.close()

def plot_linearity_analysis(results: Dict, output_dir: str):
    linearity = results['linearity_analysis']
    ideal = np.array(linearity['ideal_outputs'])
    actual = np.array(linearity['actual_outputs'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(ideal, actual, 'b.', alpha=0.5, markersize=4, label='Actual')
    ax1.plot([min(ideal), max(ideal)], [min(ideal), max(ideal)], 'r--', linewidth=2, label='Ideal')
    ax1.set_xlabel('Ideal Output', fontsize=12)
    ax1.set_ylabel('Actual Output', fontsize=12)
    ax1.set_title(f'Linearity Analysis\nCorrelation: {linearity["correlation"]:.4f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    error = np.abs(ideal - actual)
    ax2.plot(error, 'r-', linewidth=1.5)
    ax2.fill_between(range(len(error)), error, alpha=0.3)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title(f'Linear Error: {linearity["linearity_error_percent"]:.2f}%', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linearity_analysis.png'), dpi=150)
    plt.close()

def plot_temperature_analysis(results: Dict, output_dir: str):
    temp_data = results['temperature_analysis']['temperature_analysis']
    temps = [r['temperature'] for r in temp_data]
    snrs = [r['snr'] for r in temp_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    means = [r['mean_output'] for r in temp_data]
    stds = [r['std_output'] for r in temp_data]
    ax1.errorbar(temps, means, yerr=stds, fmt='bo-', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Output Current (A)', fontsize=12)
    ax1.set_title('Temperature Effect on Output', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(range(len(temps)), snrs, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(temps)))
    ax2.set_xticklabels([f'{t}K' for t in temps])
    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('SNR', fontsize=12)
    ax2.set_title('SNR vs Temperature', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_analysis.png'), dpi=150)
    plt.close()

def plot_mismatch_analysis(results: Dict, output_dir: str):
    mismatch_data = results['mismatch_analysis']['mismatch_analysis']
    mfs = [r['mismatch_factor'] for r in mismatch_data]
    cvs = [r['cv'] for r in mismatch_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mfs, cvs, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Mismatch Factor (σ/μ)', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('Effect of Device Mismatch on Output Variation', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mismatch_analysis.png'), dpi=150)
    plt.close()

def plot_summary_dashboard(results: Dict, output_dir: str):
    linearity = results['linearity_analysis']
    temp_data = results['temperature_analysis']['temperature_analysis']
    mismatch_data = results['mismatch_analysis']['mismatch_analysis']

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    fet_data = results['fet_iv_characteristics']
    ax1.semilogy(fet_data['vgs'], fet_data['ids'], 'b-', linewidth=1.5)
    ax1.set_xlabel('V_GS (V)')
    ax1.set_ylabel('I_DS (A)')
    ax1.set_title('FeFET I-V Curve')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ideal = np.array(linearity['ideal_outputs'])
    actual = np.array(linearity['actual_outputs'])
    ax2.plot(ideal, actual, 'b.', alpha=0.3, markersize=2)
    ax2.plot([min(ideal), max(ideal)], [min(ideal), max(ideal)], 'r--', linewidth=1.5)
    ax2.set_xlabel('Ideal')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Linearity (r={linearity["correlation"]:.3f})')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    temps = [r['temperature'] for r in temp_data]
    snrs = [r['snr'] for r in temp_data]
    ax3.bar(range(len(temps)), snrs, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(temps)))
    ax3.set_xticklabels([f'{t}K' for t in temps])
    ax3.set_title('SNR vs Temperature')
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = fig.add_subplot(2, 3, 4)
    mfs = [r['mismatch_factor'] for r in mismatch_data]
    cvs = [r['cv'] for r in mismatch_data]
    ax4.plot(mfs, cvs, 'ro-', linewidth=1.5, markersize=6)
    ax4.set_xlabel('Mismatch Factor')
    ax4.set_ylabel('CV')
    ax4.set_title('Mismatch Effect')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    config = results['config_summary']
    labels = ['Weight\nBits', 'Input\nBits', 'Array\nRows', 'Array\nCols']
    values = [config['weight_bits'], config['input_bits'],
             int(config['array_size'].split('x')[0]),
             int(config['array_size'].split('x')[1])]
    ax5.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax5.set_ylabel('Value')
    ax5.set_title('Device Configuration')
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    Device: {config['device']}
    Array Size: {config['array_size']}

    Linearity Metrics:
      Correlation: {linearity['correlation']:.4f}
      Max Error: {linearity['max_error']:.2e}
      Linear Error: {linearity['linearity_error_percent']:.2f}%

    Temperature Range: 250K - 400K
    Mismatch Range: 1% - 20%
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('FeFET Analog CIM System Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()

def visualize_results(results_file: str, output_dir: str = None):
    with open(results_file, 'r') as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(results_file)

    print("Generating visualizations...")

    plot_iv_characteristics(results, output_dir)
    print(f"  - I-V characteristics: fet_iv_curve.png")

    plot_linearity_analysis(results, output_dir)
    print(f"  - Linearity analysis: linearity_analysis.png")

    plot_temperature_analysis(results, output_dir)
    print(f"  - Temperature analysis: temperature_analysis.png")

    plot_mismatch_analysis(results, output_dir)
    print(f"  - Mismatch analysis: mismatch_analysis.png")

    plot_summary_dashboard(results, output_dir)
    print(f"  - Summary dashboard: summary_dashboard.png")

    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1] if len(sys.argv) > 1 else "../结果/device_simulation_results.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(results_file)
    visualize_results(results_file, output_dir)
