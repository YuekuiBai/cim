import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os
from typing import Dict

# 设置中文字体
for font in ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False


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
    ax1.set_xlabel('Ideal Output (A)', fontsize=12)
    ax1.set_ylabel('Actual Output (A)', fontsize=12)
    ax1.set_title(f'Linearity Analysis\nR={linearity["correlation"]:.4f}, R²={linearity["correlation"]**2:.4f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    error_pct = np.abs(ideal - actual) / (np.max(ideal) - np.min(ideal) + 1e-30) * 100
    ax2.plot(error_pct, 'r-', linewidth=1.5)
    ax2.fill_between(range(len(error_pct)), error_pct, alpha=0.3)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title(f'Max Error: {linearity["max_error"]:.2f}%', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linearity_analysis.png'), dpi=150)
    plt.close()


def plot_temperature_analysis(results: Dict, output_dir: str):
    temp_data = results['temperature_analysis']['temperature_analysis']
    temps = [r['temperature'] for r in temp_data]
    snrs = [r['snr'] for r in temp_data]
    vth_shifts = [r.get('vth_shift_mV', 0) for r in temp_data]
    mu_factors = [r.get('mobility_factor', 1) for r in temp_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SNR vs Temperature
    ax1 = axes[0, 0]
    ax1.bar(range(len(temps)), snrs, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(temps)))
    ax1.set_xticklabels([f'{t}K' for t in temps])
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('SNR', fontsize=12)
    ax1.set_title('SNR vs Temperature', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Vth Shift
    ax2 = axes[0, 1]
    ax2.plot(temps, vth_shifts, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('Vth Shift (mV)', fontsize=12)
    ax2.set_title('Threshold Voltage Shift vs Temperature', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Mobility Factor
    ax3 = axes[1, 0]
    ax3.plot(temps, mu_factors, 'gs-', linewidth=2, markersize=8)
    ax3.set_xlabel('Temperature (K)', fontsize=12)
    ax3.set_ylabel('Mobility Factor (mu/mu_300K)', fontsize=12)
    ax3.set_title('Carrier Mobility Degradation', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Output Current with Error Bars
    ax4 = axes[1, 1]
    means = [r['mean_output'] for r in temp_data]
    stds = [r['std_output'] for r in temp_data]
    ax4.errorbar(temps, means, yerr=stds, fmt='bo-', capsize=5, linewidth=2, markersize=8)
    ax4.set_xlabel('Temperature (K)', fontsize=12)
    ax4.set_ylabel('Mean Output Current (A)', fontsize=12)
    ax4.set_title('Output Current vs Temperature', fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Temperature Effects on FeFET CIM System', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mismatch_analysis(results: Dict, output_dir: str):
    mismatch_data = results['mismatch_analysis']['mismatch_analysis']
    n_trials = results['mismatch_analysis'].get('n_trials', 100)
    mfs = [r['mismatch_factor'] for r in mismatch_data]
    cvs = [r['cv_across_trials'] for r in mismatch_data]
    yields = [r.get('yield_5pct', 0) for r in mismatch_data]
    p5s = [r['p5'] for r in mismatch_data]
    p95s = [r['p95'] for r in mismatch_data]
    means = [r['mean_output'] for r in mismatch_data]
    stds = [r['std_across_trials'] for r in mismatch_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # CV vs Mismatch Factor
    ax1 = axes[0, 0]
    ax1.plot(mfs, cvs, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Mismatch Factor (σ/μ)', fontsize=12)
    ax1.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax1.set_title(f'Output Variation vs Mismatch (Monte Carlo n={n_trials})', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Yield vs Mismatch Factor
    ax2 = axes[0, 1]
    ax2.bar(range(len(mfs)), [y * 100 for y in yields], color='green', alpha=0.7)
    ax2.set_xticks(range(len(mfs)))
    ax2.set_xticklabels([f'{mf:.0%}' for mf in mfs])
    ax2.set_xlabel('Mismatch Factor (σ/μ)', fontsize=12)
    ax2.set_ylabel('Yield (%)', fontsize=12)
    ax2.set_title('Yield (within 5% of mean) vs Mismatch', fontsize=14)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')

    # P5-P95 Range
    ax3 = axes[1, 0]
    x_pos = range(len(mfs))
    ax3.errorbar(x_pos, means,
                 yerr=[np.array(means) - np.array(p5s), np.array(p95s) - np.array(means)],
                 fmt='bo-', capsize=8, linewidth=2, markersize=8, label='P5-P95 Range')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{mf:.0%}' for mf in mfs])
    ax3.set_xlabel('Mismatch Factor (σ/μ)', fontsize=12)
    ax3.set_ylabel('Output Current (A)', fontsize=12)
    ax3.set_title('Output Distribution (P5-P95) vs Mismatch', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Std vs Mismatch Factor
    ax4 = axes[1, 1]
    ax4.plot(mfs, stds, 'ms-', linewidth=2, markersize=8)
    ax4.set_xlabel('Mismatch Factor (σ/μ)', fontsize=12)
    ax4.set_ylabel('Std Across Trials (A)', fontsize=12)
    ax4.set_title('Standard Deviation vs Mismatch', fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Device Mismatch Analysis (Monte Carlo)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mismatch_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(results: Dict, output_dir: str):
    linearity = results['linearity_analysis']
    temp_data = results['temperature_analysis']['temperature_analysis']
    mismatch_data = results['mismatch_analysis']['mismatch_analysis']
    config = results['config_summary']

    fig = plt.figure(figsize=(16, 12))

    # I-V Curve
    ax1 = fig.add_subplot(3, 3, 1)
    fet_data = results['fet_iv_characteristics']
    ax1.semilogy(fet_data['vgs'], fet_data['ids'], 'b-', linewidth=1.5)
    ax1.set_xlabel('V_GS (V)')
    ax1.set_ylabel('I_DS (A)')
    ax1.set_title('FeFET I-V Curve')
    ax1.grid(True, alpha=0.3)

    # Linearity
    ax2 = fig.add_subplot(3, 3, 2)
    ideal = np.array(linearity['ideal_outputs'])
    actual = np.array(linearity['actual_outputs'])
    ax2.plot(ideal, actual, 'b.', alpha=0.3, markersize=2)
    ax2.plot([min(ideal), max(ideal)], [min(ideal), max(ideal)], 'r--', linewidth=1.5)
    ax2.set_xlabel('Ideal')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Linearity (R={linearity["correlation"]:.4f})')
    ax2.grid(True, alpha=0.3)

    # SNR vs Temperature
    ax3 = fig.add_subplot(3, 3, 3)
    temps = [r['temperature'] for r in temp_data]
    snrs = [r['snr'] for r in temp_data]
    ax3.bar(range(len(temps)), snrs, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(temps)))
    ax3.set_xticklabels([f'{t}K' for t in temps])
    ax3.set_title('SNR vs Temperature')
    ax3.grid(True, alpha=0.3, axis='y')

    # Vth Shift
    ax4 = fig.add_subplot(3, 3, 4)
    vth_shifts = [r.get('vth_shift_mV', 0) for r in temp_data]
    ax4.plot(temps, vth_shifts, 'ro-', linewidth=1.5, markersize=6)
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Vth Shift (mV)')
    ax4.set_title('Vth Temperature Drift')
    ax4.grid(True, alpha=0.3)

    # Mismatch CV
    ax5 = fig.add_subplot(3, 3, 5)
    mfs = [r['mismatch_factor'] for r in mismatch_data]
    cvs = [r['cv_across_trials'] for r in mismatch_data]
    ax5.plot(mfs, cvs, 'ro-', linewidth=1.5, markersize=6)
    ax5.set_xlabel('Mismatch Factor')
    ax5.set_ylabel('CV')
    ax5.set_title('Mismatch Effect')
    ax5.grid(True, alpha=0.3)

    # Yield
    ax6 = fig.add_subplot(3, 3, 6)
    yields = [r.get('yield_5pct', 0) * 100 for r in mismatch_data]
    ax6.bar(range(len(mfs)), yields, color='green', alpha=0.7)
    ax6.set_xticks(range(len(mfs)))
    ax6.set_xticklabels([f'{mf:.0%}' for mf in mfs])
    ax6.set_ylabel('Yield (%)')
    ax6.set_title('Yield (5% tolerance)')
    ax6.set_ylim([0, 105])
    ax6.grid(True, alpha=0.3, axis='y')

    # Device Config
    ax7 = fig.add_subplot(3, 3, 7)
    labels = ['Weight\nBits', 'Input\nBits', 'Array\nRows', 'Array\nCols']
    values = [config['weight_bits'], config['input_bits'],
             int(config['array_size'].split('x')[0]),
             int(config['array_size'].split('x')[1])]
    ax7.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    ax7.set_ylabel('Value')
    ax7.set_title('Device Configuration')
    ax7.grid(True, alpha=0.3, axis='y')

    # Mobility Factor
    ax8 = fig.add_subplot(3, 3, 8)
    mu_factors = [r.get('mobility_factor', 1) for r in temp_data]
    ax8.plot(temps, mu_factors, 'gs-', linewidth=1.5, markersize=6)
    ax8.set_xlabel('Temperature (K)')
    ax8.set_ylabel('mu/mu_300K')
    ax8.set_title('Mobility Degradation')
    ax8.grid(True, alpha=0.3)

    # Summary Text
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    k_val = config.get('k_value', 0)
    summary_text = f"""Device: {config['device']}
Array Size: {config['array_size']}
k (mu*Cox*W/L): {k_val:.2e}

Linearity:
  R = {linearity['correlation']:.4f}
  R² = {linearity['correlation']**2:.4f}
  Max Error: {linearity['max_error']:.2f}%

Temperature: 250K - 400K
  Vth Drift: {vth_shifts[0]:.1f} ~ {vth_shifts[-1]:.1f} mV

Mismatch: Monte Carlo
  Yield(5%): {yields[1]:.0f}% @ σ/μ=5%"""
    ax9.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax9.transAxes,
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
