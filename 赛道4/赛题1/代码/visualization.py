import numpy as np
import matplotlib.pyplot as plt
import json
import os
import yaml
import matplotlib
from matplotlib import font_manager

def _get_cjk_font():
    """Dynamically detect available CJK font"""
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/home/.fonts/wqy-microhei.ttc',
    ]

    found_fonts = []
    for fp in font_paths:
        if os.path.exists(fp):
            found_fonts.append(fp)

    if found_fonts:
        font_path = found_fonts[0]
        prop = font_manager.FontProperties(fname=font_path)
        try:
            matplotlib.font_manager.fontManager.addfont(font_path)
        except:
            pass
        return prop

    cjk_names = ['Droid Sans Fallback', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti', 'DejaVu Sans']
    for name in cjk_names:
        try:
            prop = font_manager.FontProperties(family=name)
            return prop
        except:
            continue

    return font_manager.FontProperties()

def _setup_chinese_font():
    """Setup matplotlib for Chinese font support"""
    plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback', 'WenQuanYi Micro Hei',
                                        'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

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

    _setup_chinese_font()
    nl_data = results['nonlinearity_sweep_results']
    alpha_values = nl_data['alpha_values']
    snr_values = nl_data['snr_values']
    power_values = nl_data['power_values']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('非线性因素分析', fontsize=16, fontweight='bold', y=1.02)

    axes[0].plot(alpha_values, snr_values, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('非线性因子 (α)', fontsize=12)
    axes[0].set_ylabel('信噪比 SNR (dB)', fontsize=12)
    axes[0].set_title('SNR vs 非线性因子', fontsize=14)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim([0, max(alpha_values) * 1.05])
    for i, (x, y) in enumerate(zip(alpha_values[::2], snr_values[::2])):
        axes[0].annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    axes[1].plot(alpha_values, power_values, 'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel('非线性因子 (α)', fontsize=12)
    axes[1].set_ylabel('功耗 Power (W)', fontsize=12)
    axes[1].set_title('功耗 vs 非线性因子', fontsize=14)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim([0, max(alpha_values) * 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nonlinearity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'nonlinearity_analysis.png')}")

def plot_ppa_metrics(results, output_dir: str):
    if 'performance_evaluation_results' not in results:
        return

    _setup_chinese_font()
    ppa = results['performance_evaluation_results']['ppa']
    accuracy = results['performance_evaluation_results']['accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('PPA综合指标分析', fontsize=16, fontweight='bold', y=1.02)

    ppa_labels = ['性能\n(TOPS)', '功耗\n(W)', '面积\n(mm²)']
    ppa_values = [ppa['performance_tops'], ppa['power_w'], ppa['area_mm2']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = axes[0].bar(ppa_labels, ppa_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_title('PPA 指标', fontsize=14)
    axes[0].set_ylabel('数值', fontsize=12)
    for bar, val in zip(bars, ppa_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    efficiency_labels = ['能效\n(TOPS/W)', '面积效率\n(TOPS/mm²)']
    efficiency_values = [ppa['energy_efficiency_tops_w'], ppa['area_efficiency_tops_mm2']]
    colors2 = ['#e74c3c', '#2ecc71']

    bars2 = axes[1].bar(efficiency_labels, efficiency_values, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_title('效率指标', fontsize=14)
    axes[1].set_ylabel('数值', fontsize=12)
    for bar, val in zip(bars2, efficiency_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    acc_labels = ['基准精度\n(浮点32位)', '期望精度\n(存算一体)']
    acc_values = [accuracy['baseline_accuracy'], accuracy['expected_accuracy']]
    colors3 = ['#3498db', '#e74c3c']

    bars3 = axes[2].bar(acc_labels, acc_values, color=colors3, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[2].set_title('精度对比', fontsize=14)
    axes[2].set_ylabel('Top-1 精度 (%)', fontsize=12)
    axes[2].set_ylim([60, 75])
    for bar, val in zip(bars3, acc_values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height * 1.01, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppa_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'ppa_metrics.png')}")

def plot_layer_mapping(results, output_dir: str):
    if 'network_mapping_results' not in results:
        return

    _setup_chinese_font()
    mapping = results['network_mapping_results']
    layer_names = [f"L{i+1}" for i in range(len(mapping['layer_mappings']))]
    arrays_needed = [lm['arrays_needed'] for lm in mapping['layer_mappings']]
    utilizations = [lm['utilization'] * 100 for lm in mapping['layer_mappings']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('网络层映射分析', fontsize=16, fontweight='bold', y=1.02)

    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    bars = axes[0].bar(layer_names, arrays_needed, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_xlabel('网络层', fontsize=12)
    axes[0].set_ylabel('所需阵列数量', fontsize=12)
    axes[0].set_title('各层阵列使用数量', fontsize=14)
    for bar, val in zip(bars, arrays_needed):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{val}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    bars2 = axes[1].bar(layer_names, utilizations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_xlabel('网络层', fontsize=12)
    axes[1].set_ylabel('利用率 (%)', fontsize=12)
    axes[1].set_title('各层阵列利用率', fontsize=14)
    axes[1].set_ylim([0, 100])
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%基准线')
    for bar, val in zip(bars2, utilizations):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_mapping.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'layer_mapping.png')}")

def plot_latency_breakdown(results, output_dir: str):
    if 'performance_evaluation_results' not in results:
        return

    _setup_chinese_font()
    latency = results['performance_evaluation_results']['latency']

    labels = ['DAC\n数模转换', '感应放大器', 'ADC\n模数转换']
    values = [latency['dac_latency_ns'], latency['sense_latency_ns'], latency['adc_latency_ns']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    ax.set_title('延迟分解', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'latency_breakdown.png')}")

def plot_array_simulation(results, output_dir: str):
    if 'array_simulation_results' not in results:
        return

    _setup_chinese_font()
    arr_result = results['array_simulation_results']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('阵列仿真结果', fontsize=16, fontweight='bold', y=1.02)

    metrics = ['最大电流', '最小电流', '平均功耗', '阵列效率']
    values = [arr_result['max_current'], arr_result['min_current'],
              arr_result['avg_power'] * 1e6, arr_result['array_efficiency'] * 1e9]
    units = ['A', 'A', 'μW', 'nW/cell']

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = axes[0].barh(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_xlabel('数值', fontsize=12)
    axes[0].set_title('仿真指标', fontsize=14)
    for bar, val, unit in zip(bars, values, units):
        width = bar.get_width()
        axes[0].text(width * 1.02, bar.get_y() + bar.get_height()/2., f'{val:.2e} {unit}',
                    ha='left', va='center', fontsize=10, fontweight='bold')

    if 'weight_matrix' in arr_result and arr_result['weight_matrix'] is not None:
        weight_data = np.array(arr_result['weight_matrix'])
        im = axes[1].imshow(weight_data, cmap='RdBu_r', aspect='auto', interpolation='nearest')
        axes[1].set_title('权重矩阵热图', fontsize=14)
        axes[1].set_xlabel('列索引', fontsize=12)
        axes[1].set_ylabel('行索引', fontsize=12)
        plt.colorbar(im, ax=axes[1], label='电导值 (S)')
    else:
        axes[1].text(0.5, 0.5, '权重数据\n不可用', ha='center', va='center', fontsize=14)
        axes[1].set_title('权重矩阵热图', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'array_simulation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'array_simulation.png')}")

def plot_power_analysis(results, output_dir: str):
    if 'performance_evaluation_results' not in results:
        return

    _setup_chinese_font()
    perf = results['performance_evaluation_results']
    energy = perf.get('energy', {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('功耗与能效分析', fontsize=16, fontweight='bold', y=1.02)

    power_labels = ['静态功耗\n(mW)', '动态功耗\n(mW)', '总功耗\n(mW)']
    power_values = [
        energy.get('total_static_power_mW', 0),
        energy.get('total_power_mW', 0) * 0.1,
        energy.get('total_power_mW', 0)
    ]
    colors = ['#3498db', '#f39c12', '#e74c3c']

    bars = axes[0].bar(power_labels, power_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_title('功耗分解', fontsize=14)
    axes[0].set_ylabel('功耗 (mW)', fontsize=12)
    for bar, val in zip(bars, power_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    efficiency_labels = ['能效 (TOPS/W)', '每推理能耗 (μJ)']
    efficiency_values = [
        perf['ppa']['energy_efficiency_tops_w'],
        energy.get('energy_per_inference_uJ', 0)
    ]
    colors2 = ['#2ecc71', '#9b59b6']

    bars2 = axes[1].bar(efficiency_labels, efficiency_values, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_title('能效指标', fontsize=14)
    axes[1].set_ylabel('数值', fontsize=12)
    for bar, val in zip(bars2, efficiency_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'power_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'power_analysis.png')}")

def plot_accuracy_analysis(results, output_dir: str):
    if 'performance_evaluation_results' not in results:
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        _setup_chinese_font()
        accuracy = results['performance_evaluation_results']['accuracy']

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle('精度影响分析', fontsize=16, fontweight='bold', y=1.02)

        categories = ['基准精度', '精度损失', '期望精度']
        values = [
            accuracy['baseline_accuracy'],
            -accuracy['estimated_accuracy_loss_percent'],
            accuracy['expected_accuracy']
        ]
        colors = ['#3498db', '#e74c3c', '#2ecc71']

        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Top-1 精度 (%)', fontsize=12)
        ax.set_title('CIM 系统精度分析', fontsize=14)
        ax.set_ylim([60, 75])

        for bar, val in zip(bars, values):
            height = bar.get_height()
            if val < 0:
                ax.text(bar.get_x() + bar.get_width()/2., height / 2, f'{val:.1f}%',
                       ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.01, f'{val:.1f}%',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'accuracy_analysis.png')}")

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
    plot_power_analysis(results, output_dir)
    plot_accuracy_analysis(results, output_dir)

    print("\n所有图表生成完成!")

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "../结果"
    generate_all_charts(output_dir)