import numpy as np
import yaml
from typing import Tuple, Dict, List
import json
import os

class FeFETPhysics:
    def __init__(self, config: Dict):
        self.config = config
        self.physics = config['physics']
        self.work_region = config['work_region']

    def drain_current(self, vgs: float, vds: float, vth: float) -> float:
        if vgs <= vth:
            return 1e-6

        vgt = vgs - vth
        k = 50e-6

        if vds < vgt:
            ids = k * ((vgt - vds / 2) * vds)
        else:
            ids = k * (vgt ** 2 / 2)

        return max(ids, 1e-6)

    def threshold_voltage(self, polarization: float) -> float:
        vt_base = self.work_region['vt_range'][0]
        vt_range = self.work_region['vt_range'][1] - self.work_region['vt_range'][0]
        return vt_base + polarization * vt_range

    def fet_sweep(self, vgs_range: Tuple[float, float],
                  vds: float, num_points: int = 100) -> Dict:
        vgs_values = np.linspace(vgs_range[0], vgs_range[1], num_points)
        ids_values = []

        for vgs in vgs_values:
            vth = self.threshold_voltage(0.5)
            ids = self.drain_current(vgs, vds, vth)
            ids_values.append(ids)

        return {
            'vgs': vgs_values.tolist(),
            'ids': ids_values,
            'vds': vds
        }

class FeFETCell:
    def __init__(self, weight: float, config: Dict):
        self.config = config
        self.weight = weight
        self.non_ideal = config['non_ideal']

        vt_min, vt_max = config['work_region']['vt_range']
        vt_range = vt_max - vt_min
        self.vth = vt_min + (weight + 1) / 2 * vt_range

        self.mismatch = np.random.normal(0, self.non_ideal['mismatch_factor'] * self.vth)
        self.vth += self.mismatch
        self.noise = self.non_ideal['noise_std']

    def read_current(self, input_voltage: float, vds: float) -> float:
        if input_voltage <= self.vth:
            return 1e-6 + np.random.normal(0, self.noise * 1e-6)

        vgt = input_voltage - self.vth
        k = 50e-6

        if vds < vgt:
            ids = k * ((vgt - vds / 2) * vds)
        else:
            ids = k * (vgt ** 2 / 2)

        noise = np.random.normal(0, self.noise * 1e-6)
        return max(ids, 1e-6) + noise

class DifferentialPairCircuit:
    def __init__(self, config: Dict):
        self.config = config
        self.ref_voltage = config['sign_handling']['reference_voltage']

    def weight_to_vth(self, weight: float) -> float:
        vt_min, vt_max = self.config['work_region']['vt_range']
        vt_range = vt_max - vt_min
        return vt_min + (weight + 1) / 2 * vt_range

class LinearCompensator:
    def __init__(self, config: Dict):
        self.config = config
        self.compensation = config['compensation']
        self.lut_size = self.compensation['look_up_table_size']
        self.pre_distortion_lut = self._generate_lut()

    def _generate_lut(self) -> np.ndarray:
        lut = np.zeros(self.lut_size)
        alpha = 0.1
        for i in range(self.lut_size):
            x = i / (self.lut_size - 1)
            lut[i] = x / (1 - alpha * (1 - x)) if abs(1 - alpha * (1 - x)) > 1e-10 else x
        return lut

    def compensate(self, value: float) -> float:
        idx = int(np.clip(value, 0, 1) * (self.lut_size - 1))
        return self.pre_distortion_lut[idx]

class MultiBitEncoder:
    def __init__(self, config: Dict):
        self.config = config
        self.weight_bits = config['multi_bit']['weight_bits']
        self.input_bits = config['multi_bit']['input_bits']
        self.levels = 2 ** self.weight_bits

    def encode_weight(self, weight: float) -> int:
        normalized = (weight + 1) / 2
        quantized = int(normalized * (self.levels - 1))
        return np.clip(quantized, 0, self.levels - 1)

    def decode_weight(self, code: int) -> float:
        normalized = code / (self.levels - 1)
        return normalized * 2 - 1

    def encode_input(self, input_val: float) -> int:
        normalized = np.clip(input_val, 0, 1)
        levels = 2 ** self.input_bits
        quantized = int(normalized * (levels - 1))
        return np.clip(quantized, 0, levels - 1)

class ArraySimulator:
    def __init__(self, config: Dict):
        self.config = config
        self.rows = int(config['device']['array_size'].split('x')[0])
        self.cols = int(config['device']['array_size'].split('x')[1])
        self.fet = FeFETPhysics(config)
        self.diff_pair = DifferentialPairCircuit(config)
        self.compensator = LinearCompensator(config)
        self.encoder = MultiBitEncoder(config)

        self.weight_matrix = np.zeros((self.rows, self.cols))
        self.cells = [[None for _ in range(self.cols)] for _ in range(self.rows)]

    def initialize_cells(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.weight_matrix[i, j] = np.random.uniform(-1, 1)
                self.cells[i][j] = FeFETCell(self.weight_matrix[i, j], self.config)

    def matrix_vector_multiply(self, input_vector: np.ndarray,
                              vds: float = 0.1) -> np.ndarray:
        if len(input_vector) != self.rows:
            raise ValueError(f"Input size {len(input_vector)} must match rows {self.rows}")

        output = np.zeros(self.cols)

        for j in range(self.cols):
            column_sum = 0.0
            for i in range(self.rows):
                vgs = input_vector[i]
                cell = self.cells[i][j]
                current = cell.read_current(vgs, vds)
                column_sum += current
            output[j] = column_sum

        return output

    def analyze_linearity(self, input_range: Tuple[float, float],
                        num_points: int = 100) -> Dict:
        inputs = np.linspace(input_range[0], input_range[1], num_points)

        mean_weight = np.mean(self.weight_matrix)
        ideal_outputs = inputs * mean_weight * 1e6

        actual_outputs = []
        for inp in inputs:
            input_vec = np.full(self.rows, inp)
            actual = self.matrix_vector_multiply(input_vec, vds=0.1)
            actual_outputs.append(np.mean(actual) * 1e6)

        actual_outputs = np.array(actual_outputs)

        ideal_arr = ideal_outputs - np.mean(ideal_outputs)
        actual_arr = actual_outputs - np.mean(actual_outputs)
        std_ideal = np.std(ideal_arr) + 1e-10
        std_actual = np.std(actual_arr) + 1e-10
        ideal_norm = ideal_arr / std_ideal
        actual_norm = actual_arr / std_actual
        correlation = np.corrcoef(ideal_norm, actual_norm)[0, 1]

        ideal_max = np.max(ideal_outputs)
        actual_max = np.max(actual_outputs)
        scale_factor = ideal_max / (actual_max + 1e-10) if actual_max > 1e-10 else 1.0
        scaled_actual = actual_outputs * scale_factor
        max_error = np.max(np.abs(ideal_outputs - scaled_actual)) / ideal_max * 100 if ideal_max > 1e-10 else 0
        mse = np.mean((ideal_outputs - scaled_actual) ** 2)

        return {
            'input_range': input_range,
            'ideal_outputs': ideal_outputs.tolist(),
            'actual_outputs': actual_outputs.tolist(),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'max_error': float(max_error),
            'mse': float(mse),
            'linearity_error_percent': float(max_error)
        }

    def analyze_temperature_effect(self, temperatures: List[float],
                                 input_val: float) -> Dict:
        results = []
        original_noise = self.config['non_ideal']['noise_std']

        for temp in temperatures:
            noise_factor = 1 + 0.01 * (temp - 300) / 100
            self.config['non_ideal']['noise_std'] = original_noise * noise_factor

            input_vec = np.full(self.rows, input_val)
            output = self.matrix_vector_multiply(input_vec, vds=0.1)
            mean_out = np.mean(output)
            std_out = np.std(output) + 1e-12

            results.append({
                'temperature': temp,
                'mean_output': float(mean_out),
                'std_output': float(std_out),
                'snr': float(np.abs(mean_out) / std_out)
            })

        self.config['non_ideal']['noise_std'] = original_noise
        return {'temperature_analysis': results}

    def analyze_mismatch_effect(self, mismatch_factors: List[float],
                               input_val: float) -> Dict:
        results = []
        original_mismatch = self.config['non_ideal']['mismatch_factor']

        for mf in mismatch_factors:
            self.config['non_ideal']['mismatch_factor'] = mf
            self.initialize_cells()

            input_vec = np.full(self.rows, input_val)
            output = self.matrix_vector_multiply(input_vec, vds=0.1)
            mean_out = np.mean(output)
            std_out = np.std(output) + 1e-12

            results.append({
                'mismatch_factor': mf,
                'mean_output': float(mean_out),
                'std_output': float(std_out),
                'cv': float(std_out / np.abs(mean_out))
            })

        self.config['non_ideal']['mismatch_factor'] = original_mismatch
        self.initialize_cells()
        return {'mismatch_analysis': results}

def run_device_simulation(config_path: str = "config.yaml",
                         output_dir: str = "../结果") -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(config['experiment']['seed'])

    sim = ArraySimulator(config)
    sim.initialize_cells()

    vgs_range = config['work_region']['vgs_range']
    linearity = sim.analyze_linearity(
        (vgs_range[0] + 0.4, vgs_range[1] * 0.9),
        num_points=50
    )

    temps = [250, 300, 350, 400]
    temp_analysis = sim.analyze_temperature_effect(temps, input_val=0.9)

    mismatch_factors = [0.01, 0.05, 0.1, 0.2]
    mismatch_analysis = sim.analyze_mismatch_effect(mismatch_factors, input_val=0.9)

    fet_sweep = sim.fet.fet_sweep(vgs_range, vds=0.1, num_points=100)

    results = {
        'linearity_analysis': linearity,
        'temperature_analysis': temp_analysis,
        'mismatch_analysis': mismatch_analysis,
        'fet_iv_characteristics': fet_sweep,
        'config_summary': {
            'device': config['device']['name'],
            'array_size': config['device']['array_size'],
            'weight_bits': config['multi_bit']['weight_bits'],
            'input_bits': config['multi_bit']['input_bits']
        }
    }

    output_file = os.path.join(output_dir, 'device_simulation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("FeFET器件仿真结果")
    print("=" * 60)
    print(f"\n器件: {config['device']['name']}")
    print(f"阵列规模: {config['device']['array_size']}")
    print(f"\n【线性度分析】")
    print(f"  相关系数R²: {linearity['correlation']**2:.4f}")
    print(f"  最大误差: {linearity['max_error']:.2f}%")
    print(f"  线性误差: {linearity['linearity_error_percent']:.2f}%")
    print(f"\n【温度分析】")
    for r in temp_analysis['temperature_analysis']:
        print(f"  {r['temperature']}K: SNR={r['snr']:.2f}, 输出={r['mean_output']*1e6:.2f}uA")
    print(f"\n【失配分析】")
    for r in mismatch_analysis['mismatch_analysis']:
        print(f"  σ/μ={r['mismatch_factor']}: CV={r['cv']:.4f}")
    print(f"\n结果已保存至: {output_file}")

    return results

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../结果"
    run_device_simulation(config_path, output_dir)
