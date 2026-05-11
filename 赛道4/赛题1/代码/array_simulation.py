import numpy as np
import yaml
from typing import Tuple, Optional, Dict, List
import json
import os

class RRAMCell:
    def __init__(self, conductance: float, bit_precision: int = 8):
        self.g_max = 1e-3
        self.g_min = 10e-6
        self.conductance = conductance
        self.bit_precision = bit_precision

    def program(self, value: float):
        normalized = np.clip(value, 0, 1)
        self.conductance = self.g_min + normalized * (self.g_max - self.g_min)

    def read(self, voltage: float) -> float:
        return self.conductance * voltage

    def apply_nonlinearity(self, voltage: float, alpha: float = 0.05) -> float:
        return alpha * voltage**3 + (1 - alpha) * voltage

class CrossbarArray:
    def __init__(self, rows: int, cols: int, bit_precision: int = 8):
        self.rows = rows
        self.cols = cols
        self.bit_precision = bit_precision
        self.cells = [[RRAMCell(50e-6, bit_precision) for _ in range(cols)] for _ in range(rows)]
        self.weight_matrix = np.zeros((rows, cols))

    def set_weight_matrix(self, weights: np.ndarray):
        self.weight_matrix = weights
        rows, cols = weights.shape
        for i in range(min(rows, self.rows)):
            for j in range(min(cols, self.cols)):
                normalized = (weights[i, j] + 1) / 2
                self.cells[i][j].program(normalized)

    def matrix_vector_multiply(self, input_voltage: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        if len(input_voltage) != self.rows:
            raise ValueError(f"Input size {len(input_voltage)} must match array rows {self.rows}")
        output_current = np.zeros(self.cols)
        for j in range(self.cols):
            for i in range(self.rows):
                v = input_voltage[i]
                v_nl = self.cells[i][j].apply_nonlinearity(v)
                output_current[j] += self.cells[i][j].read(v_nl)
        if noise_std > 0:
            output_current += np.random.randn(self.cols) * noise_std
        return output_current

    def calculate_power(self, voltage: np.ndarray) -> float:
        power = 0.0
        for j in range(self.cols):
            for i in range(self.rows):
                power += self.cells[i][j].conductance * (voltage[i] ** 2)
        return power

class ADC:
    def __init__(self, bits: int, vref: float = 1.0):
        self.bits = bits
        self.vref = vref
        self.resolution = vref / (2 ** bits)

    def convert(self, analog_value: float) -> float:
        digital = int(analog_value / self.resolution)
        max_val = 2 ** self.bits - 1
        return np.clip(digital, 0, max_val) * self.resolution

class DAC:
    def __init__(self, bits: int, vref: float = 1.0):
        self.bits = bits
        self.vref = vref
        self.resolution = vref / (2 ** bits)

    def convert(self, digital_value: float) -> float:
        return digital_value * self.resolution

class PeripheralCircuit:
    def __init__(self, dac_bits: int = 8, adc_bits: int = 10, vref: float = 1.0):
        self.dac = DAC(dac_bits, vref)
        self.adc = ADC(adc_bits, vref)

    def input_conversion(self, digital_input: np.ndarray) -> np.ndarray:
        return np.array([self.dac.convert(x) for x in digital_input])

    def output_conversion(self, analog_output: np.ndarray) -> np.ndarray:
        return np.array([self.adc.convert(x) for x in analog_output])

class CIMSimulator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        array_cfg = self.config['device']
        rows, cols = map(int, array_cfg['array_size'].split('x'))
        self.array = CrossbarArray(rows, cols, array_cfg['bit_precision'])

        periph = self.config['peripherals']
        self.peripherals = PeripheralCircuit(
            periph['dac']['bits'],
            periph['adc']['bits'],
            periph['dac']['voltage_range'][1]
        )

        self.sim_cfg = self.config['simulation']

    def simulate_weight_mapping(self, weights: np.ndarray) -> Dict:
        self.array.set_weight_matrix(weights)

        test_input = np.random.rand(self.array.rows) * 0.8
        analog_output = self.array.matrix_vector_multiply(
            test_input,
            self.sim_cfg.get('noise_std', 0.01)
        )

        digital_output = self.peripherals.output_conversion(analog_output)

        power = self.array.calculate_power(test_input)

        return {
            'weight_shape': weights.shape,
            'input_shape': test_input.shape,
            'output_shape': analog_output.shape,
            'max_current': float(np.max(analog_output)),
            'min_current': float(np.min(analog_output)),
            'avg_power': float(power),
            'array_efficiency': float(power / (self.array.rows * self.array.cols)),
            'snr': float(20 * np.log10(np.mean(np.abs(analog_output)) / (self.sim_cfg.get('noise_std', 0.01) + 1e-10)))
        }

    def sweep_nonlinearity(self, weights: np.ndarray, alpha_values: List[float]) -> Dict:
        results = {'alpha_values': alpha_values, 'snr_values': [], 'power_values': []}
        original_alpha = 0.05

        test_input = np.random.rand(self.array.rows) * 0.8
        self.array.set_weight_matrix(weights)

        for alpha in alpha_values:
            for i in range(self.array.rows):
                for j in range(self.array.cols):
                    self.array.cells[i][j].apply_nonlinearity = lambda v, a=alpha: a * v**3 + (1 - a) * v

            output = self.array.matrix_vector_multiply(test_input, 0.0)
            signal = np.mean(np.abs(output))
            noise = np.std(output)
            snr = 20 * np.log10(signal / (noise + 1e-10))
            power = self.array.calculate_power(test_input)

            results['snr_values'].append(float(snr))
            results['power_values'].append(float(power))

        return results

def run_array_simulation(config_path: str = "config.yaml", output_dir: str = "../结果"):
    os.makedirs(output_dir, exist_ok=True)

    sim = CIMSimulator(config_path)

    test_weights = np.random.randn(128, 128) * 0.5
    result = sim.simulate_weight_mapping(test_weights)

    result_file = os.path.join(output_dir, 'array_simulation_results.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    print("存算阵列仿真结果")
    print("=" * 60)
    print(f"权重形状: {result['weight_shape']}")
    print(f"输入形状: {result['input_shape']}")
    print(f"输出形状: {result['output_shape']}")
    print(f"最大电流: {result['max_current']:.6f} A")
    print(f"最小电流: {result['min_current']:.6f} A")
    print(f"平均功耗: {result['avg_power']:.6f} W")
    print(f"阵列效率: {result['array_efficiency']:.9f} W/cell")
    print(f"信噪比: {result['snr']:.2f} dB")
    print(f"\n结果已保存至: {result_file}")

    alpha_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    nl_results = sim.sweep_nonlinearity(test_weights, alpha_values)

    nl_file = os.path.join(output_dir, 'nonlinearity_sweep_results.json')
    with open(nl_file, 'w') as f:
        json.dump(nl_results, f, indent=2)

    print("\n" + "=" * 60)
    print("非线性参数扫描结果")
    print("=" * 60)
    print(f"{'Alpha':<10} {'SNR (dB)':<15} {'Power (W)':<15}")
    print("-" * 40)
    for alpha, snr, power in zip(nl_results['alpha_values'], nl_results['snr_values'], nl_results['power_values']):
        print(f"{alpha:<10} {snr:<15.2f} {power:<15.6f}")
    print(f"\n结果已保存至: {nl_file}")

    return result, nl_results

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../结果"
    run_array_simulation(config_path, output_dir)
