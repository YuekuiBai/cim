import numpy as np
import yaml
from typing import Dict, List, Tuple
import json
import os

class PerformanceEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.device_cfg = config['device']
        self.peripherals = config['peripherals']
        self.sim_cfg = config['simulation']

    def calculate_latency(self, num_arrays: int, array_size: Tuple[int, int],
                          word_length: int) -> Dict:
        dac_time = self.peripherals['dac']['settling_time']
        adc_time = 1 / self.peripherals['adc']['sampling_rate']
        sense_time = self.peripherals['sense_amplifier']['latency']

        per_array_time = dac_time + sense_time + adc_time
        total_latency = per_array_time * num_arrays * word_length

        return {
            'dac_latency_ns': dac_time * 1e9,
            'adc_latency_ns': adc_time * 1e9,
            'sense_latency_ns': sense_time * 1e9,
            'per_array_latency_ns': per_array_time * 1e9,
            'total_latency_ns': total_latency * 1e9,
            'total_latency_ms': total_latency * 1e3,
            'latency_per_layer_ns': per_array_time * word_length * 1e9
        }

    def calculate_energy(self, num_arrays: int, array_size: Tuple[int, int],
                        power_per_cell: float = 1e-6) -> Dict:
        total_cells = num_arrays * array_size[0] * array_size[1]
        static_power = total_cells * power_per_cell
        dynamic_power_factor = 0.1
        total_power = static_power * (1 + dynamic_power_factor)

        latency_result = self.calculate_latency(num_arrays, array_size, 1024)
        energy_per_inference = total_power * latency_result['total_latency_ns'] * 1e-9

        return {
            'static_power_per_cell_uW': power_per_cell * 1e6,
            'total_static_power_mW': static_power * 1e3,
            'total_power_mW': total_power * 1e3,
            'energy_per_inference_uJ': energy_per_inference * 1e6
        }

    def calculate_area(self, num_arrays: int, array_size: Tuple[int, int],
                      cell_size: float = 10e-12) -> Dict:
        array_area = array_size[0] * array_size[1] * cell_size
        peripheral_factor = 2.0
        total_area = num_arrays * array_area * peripheral_factor

        return {
            'cell_area_um2': cell_size * 1e12,
            'array_area_mm2': array_area * 1e6,
            'total_area_mm2': total_area * 1e6,
            'area_efficiency': 1 / peripheral_factor
        }

    def evaluate_accuracy_impact(self) -> Dict:
        noise = self.sim_cfg.get('noise_std', 0.01)
        nonlinearity = self.sim_cfg.get('nonlinearity', 0.05)

        snr = 1 / (noise + nonlinearity + 1e-10)
        estimated_accuracy_loss = min((noise + nonlinearity) * 10, 20)

        return {
            'estimated_snr_db': 20 * np.log10(snr) if snr > 0 else 0,
            'noise_contribution': noise,
            'nonlinearity_contribution': nonlinearity,
            'estimated_accuracy_loss_percent': estimated_accuracy_loss,
            'baseline_accuracy': 69.8,
            'expected_accuracy': 69.8 - estimated_accuracy_loss,
            'requires_calibration': nonlinearity > 0.02
        }

    def identify_bottlenecks(self, num_arrays: int, total_bandwidth: float) -> List[Dict]:
        bottlenecks = []

        if num_arrays > 16:
            bottlenecks.append({
                'type': '阵列利用率',
                'severity': 'high',
                'description': f'大量小阵列导致硬件资源利用率低 ({num_arrays} arrays)',
                'suggestion': '考虑使用更大尺寸阵列或层间复用'
            })

        if total_bandwidth > 1e12:
            bottlenecks.append({
                'type': '带宽',
                'severity': 'medium',
                'description': '数据搬移成为瓶颈',
                'suggestion': '优化数据流，减少阵列间数据传递'
            })

        adc_res = 1 / (2 ** self.peripherals['adc']['bits'])
        if adc_res < 0.001:
            bottlenecks.append({
                'type': 'ADC精度',
                'severity': 'medium',
                'description': 'ADC精度限制整体计算精度',
                'suggestion': '提高ADC位数或采用过采样技术'
            })

        return bottlenecks

    def compute_ppa(self, num_arrays: int) -> Dict:
        array_size = tuple(map(int, self.device_cfg['array_size'].split('x')))
        latency = self.calculate_latency(num_arrays, array_size, 1024)
        energy = self.calculate_energy(num_arrays, array_size)
        area = self.calculate_area(num_arrays, array_size)

        peak_throughput = num_arrays * array_size[0] * array_size[1] * 100e6

        return {
            'performance_tops': peak_throughput * 1e-12,
            'power_w': energy['total_power_mW'] * 1e-3,
            'area_mm2': area['total_area_mm2'],
            'energy_efficiency_tops_w': (peak_throughput * 1e-12) / (energy['total_power_mW'] * 1e-3),
            'area_efficiency_tops_mm2': (peak_throughput * 1e-12) / area['total_area_mm2']
        }

def run_performance_evaluation(config_path: str = "config.yaml", output_dir: str = "../结果"):
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    evaluator = PerformanceEvaluator(config)
    array_size = tuple(map(int, config['device']['array_size'].split('x')))

    num_arrays = 32
    total_bandwidth = 1e11

    print("=" * 60)
    print("性能评估报告")
    print("=" * 60)

    latency = evaluator.calculate_latency(num_arrays, array_size, 1024)
    print("\n【延迟分析】")
    print("-" * 40)
    for k, v in latency.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    energy = evaluator.calculate_energy(num_arrays, array_size)
    print("\n【能效分析】")
    print("-" * 40)
    for k, v in energy.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    area = evaluator.calculate_area(num_arrays, array_size)
    print("\n【面积分析】")
    print("-" * 40)
    for k, v in area.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    accuracy = evaluator.evaluate_accuracy_impact()
    print("\n【精度影响】")
    print("-" * 40)
    for k, v in accuracy.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    bottlenecks = evaluator.identify_bottlenecks(num_arrays, total_bandwidth)
    print("\n【瓶颈分析】")
    print("-" * 40)
    for bp in bottlenecks:
        print(f"  [{bp['severity'].upper()}] {bp['type']}: {bp['description']}")
        print(f"    建议: {bp['suggestion']}")

    ppa = evaluator.compute_ppa(num_arrays)
    print("\n【PPA综合指标】")
    print("-" * 40)
    for k, v in ppa.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    results = {
        'latency': latency,
        'energy': energy,
        'area': area,
        'accuracy': accuracy,
        'bottlenecks': bottlenecks,
        'ppa': ppa,
        'config': {
            'num_arrays': num_arrays,
            'array_size': config['device']['array_size']
        }
    }

    output_file = os.path.join(output_dir, 'performance_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存至: {output_file}")

    return results

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../结果"
    run_performance_evaluation(config_path, output_dir)
