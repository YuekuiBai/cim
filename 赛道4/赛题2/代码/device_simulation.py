import numpy as np
import yaml
from typing import Tuple, Dict, List
import json
import os
from contextlib import contextmanager


class FeFETPhysics:
    def __init__(self, config: Dict):
        self.config = config
        self.physics = config['physics']
        self.work_region = config['work_region']

        # 从config物理参数计算k值: k = mu * Cox * W / L
        mu = self.physics['electron_mobility'] * 1e-4  # cm^2/Vs -> m^2/Vs
        eps0 = 8.854e-12  # F/m
        eps_r = self.physics['permittivity']
        tox = self.physics['oxide_thickness']
        W = self.physics['channel_width']
        L = self.physics['channel_length']
        cox = eps_r * eps0 / tox
        self.k = mu * cox * W / L

        # 温度相关参数
        self.T0 = 300  # 参考温度
        self.dVth_dT = -2e-3  # V/K, HfO2典型值
        self.mu_exponent = -1.5  # 声子散射指数

    def drain_current(self, vgs: float, vds: float, vth: float,
                      temperature: float = 300) -> float:
        if vgs <= vth:
            return 1e-12

        vgt = vgs - vth
        # 温度对迁移率的影响: mu(T) = mu(300) * (T/300)^alpha
        mu_factor = (temperature / self.T0) ** self.mu_exponent
        k_eff = self.k * mu_factor

        if vds < vgt:
            ids = k_eff * ((vgt - vds / 2) * vds)
        else:
            ids = k_eff * (vgt ** 2 / 2)

        return max(ids, 1e-12)

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
    def __init__(self, weight: float, config: Dict, physics: FeFETPhysics):
        self.config = config
        self.weight = weight
        self.non_ideal = config['non_ideal']
        self.physics = physics

        vt_min, vt_max = config['work_region']['vt_range']
        vt_range = vt_max - vt_min
        self.vth = vt_min + (weight + 1) / 2 * vt_range

        self.mismatch = np.random.normal(0, self.non_ideal['mismatch_factor'] * self.vth)
        self.vth += self.mismatch
        self.noise = self.non_ideal['noise_std']

    def read_current(self, input_voltage: float, vds: float,
                     temperature: float = 300) -> float:
        # 温度引起的Vth漂移
        dVth = self.physics.dVth_dT * (temperature - self.physics.T0)
        vth_eff = self.vth + dVth

        if input_voltage <= vth_eff:
            # 亚阈值泄漏电流
            noise = np.random.normal(0, self.noise * 1e-12 * (temperature / self.physics.T0))
            return 1e-12 + noise

        vgt = input_voltage - vth_eff
        # 温度对迁移率的影响
        mu_factor = (temperature / self.physics.T0) ** self.physics.mu_exponent
        k_eff = self.physics.k * mu_factor

        if vds < vgt:
            ids = k_eff * ((vgt - vds / 2) * vds)
        else:
            ids = k_eff * (vgt ** 2 / 2)

        # 噪声随温度增加
        noise_factor = temperature / self.physics.T0
        noise = np.random.normal(0, self.noise * 1e-12 * noise_factor)
        return max(ids, 1e-12) + noise


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

    @contextmanager
    def _override_config(self, section: str, key: str, value):
        """安全地临时修改config参数，结束后自动恢复"""
        original = self.config[section][key]
        self.config[section][key] = value
        try:
            yield
        finally:
            self.config[section][key] = original

    def initialize_cells(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.weight_matrix[i, j] = np.random.uniform(-1, 1)
                self.cells[i][j] = FeFETCell(self.weight_matrix[i, j], self.config, self.fet)

    def matrix_vector_multiply(self, input_vector: np.ndarray,
                              vds: float = 0.1, temperature: float = 300) -> np.ndarray:
        if len(input_vector) != self.rows:
            raise ValueError(f"Input size {len(input_vector)} must match rows {self.rows}")

        output = np.zeros(self.cols)

        for j in range(self.cols):
            column_sum = 0.0
            for i in range(self.rows):
                vgs = input_vector[i]
                cell = self.cells[i][j]
                current = cell.read_current(vgs, vds, temperature)
                column_sum += current
            output[j] = column_sum

        return output

    def compute_ideal_output(self, input_vector: np.ndarray, vds: float = 0.1) -> np.ndarray:
        """计算理想输出（无失配、无噪声）"""
        vt_min, vt_max = self.config['work_region']['vt_range']
        vt_range = vt_max - vt_min

        output = np.zeros(self.cols)
        for j in range(self.cols):
            column_sum = 0.0
            for i in range(self.rows):
                w = self.weight_matrix[i, j]
                vth_nominal = vt_min + (w + 1) / 2 * vt_range
                vgs = input_vector[i]
                ids = self.fet.drain_current(vgs, vds, vth_nominal, temperature=300)
                column_sum += ids
            output[j] = column_sum
        return output

    def analyze_linearity(self, input_range: Tuple[float, float],
                        num_points: int = 100) -> Dict:
        inputs = np.linspace(input_range[0], input_range[1], num_points)
        vds = 0.1

        # 理想输出：无失配、无噪声的nominal电流
        ideal_outputs = []
        for inp in inputs:
            input_vec = np.full(self.rows, inp)
            ideal = self.compute_ideal_output(input_vec, vds)
            ideal_outputs.append(np.mean(ideal))

        ideal_outputs = np.array(ideal_outputs)

        # 实际输出：含失配和噪声
        actual_outputs = []
        for inp in inputs:
            input_vec = np.full(self.rows, inp)
            actual = self.matrix_vector_multiply(input_vec, vds)
            actual_outputs.append(np.mean(actual))

        actual_outputs = np.array(actual_outputs)

        # 相关系数计算
        if np.std(ideal_outputs) < 1e-20 or np.std(actual_outputs) < 1e-20:
            correlation = 0.0
        else:
            correlation = np.corrcoef(ideal_outputs, actual_outputs)[0, 1]

        # 归一化误差计算
        ideal_range = np.max(ideal_outputs) - np.min(ideal_outputs)
        if ideal_range > 1e-20:
            scaled_error = np.abs(ideal_outputs - actual_outputs)
            max_error = np.max(scaled_error) / ideal_range * 100
        else:
            max_error = 0.0

        mse = np.mean((ideal_outputs - actual_outputs) ** 2)

        return {
            'input_range': list(input_range),
            'ideal_outputs': ideal_outputs.tolist(),
            'actual_outputs': actual_outputs.tolist(),
            'correlation': float(correlation),
            'max_error': float(max_error),
            'mse': float(mse),
            'linearity_error_percent': float(max_error)
        }

    def analyze_temperature_effect(self, temperatures: List[float],
                                 input_val: float) -> Dict:
        results = []

        for temp in temperatures:
            input_vec = np.full(self.rows, input_val)
            output = self.matrix_vector_multiply(input_vec, vds=0.1, temperature=temp)
            mean_out = np.mean(output)
            std_out = np.std(output) + 1e-30

            # 计算Vth漂移量
            dVth = self.fet.dVth_dT * (temp - self.fet.T0)

            # 计算迁移率变化因子
            mu_factor = (temp / self.fet.T0) ** self.fet.mu_exponent

            results.append({
                'temperature': temp,
                'mean_output': float(mean_out),
                'std_output': float(std_out),
                'snr': float(np.abs(mean_out) / std_out),
                'vth_shift_mV': float(dVth * 1000),
                'mobility_factor': float(mu_factor)
            })

        return {'temperature_analysis': results}

    def analyze_mismatch_effect(self, mismatch_factors: List[float],
                               input_val: float, n_trials: int = 100) -> Dict:
        results = []

        for mf in mismatch_factors:
            trial_means = []

            for trial in range(n_trials):
                with self._override_config('non_ideal', 'mismatch_factor', mf):
                    self.initialize_cells()
                    input_vec = np.full(self.rows, input_val)
                    output = self.matrix_vector_multiply(input_vec, vds=0.1)
                    trial_means.append(np.mean(output))

            trial_means = np.array(trial_means)
            mean_val = np.mean(trial_means)
            std_val = np.std(trial_means)

            results.append({
                'mismatch_factor': mf,
                'mean_output': float(mean_val),
                'std_across_trials': float(std_val),
                'cv_across_trials': float(std_val / np.abs(mean_val)) if np.abs(mean_val) > 1e-30 else 0.0,
                'p5': float(np.percentile(trial_means, 5)),
                'p95': float(np.percentile(trial_means, 95)),
                'yield_5pct': float(np.mean(np.abs(trial_means - mean_val) / np.abs(mean_val) < 0.05)) if np.abs(mean_val) > 1e-30 else 0.0,
                'n_trials': n_trials
            })

        # 恢复默认值并重新初始化
        self.initialize_cells()
        return {'mismatch_analysis': results, 'n_trials': n_trials}


def run_device_simulation(config_path: str = "config.yaml",
                         output_dir: str = "../结果") -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(config['experiment']['seed'])

    sim = ArraySimulator(config)
    sim.initialize_cells()

    # 线性度分析
    vgs_range = config['work_region']['vgs_range']
    linearity = sim.analyze_linearity(
        (vgs_range[0] + 0.4, vgs_range[1] * 0.9),
        num_points=50
    )

    # 温度分析
    temps = [250, 300, 350, 400]
    temp_analysis = sim.analyze_temperature_effect(temps, input_val=0.9)

    # 失配分析 (Monte Carlo)
    mismatch_factors = [0.01, 0.05, 0.1, 0.2]
    mismatch_analysis = sim.analyze_mismatch_effect(mismatch_factors, input_val=0.9, n_trials=100)

    # FET I-V特性
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
            'input_bits': config['multi_bit']['input_bits'],
            'k_value': float(sim.fet.k)
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
    print(f"k值 (mu*Cox*W/L): {sim.fet.k:.6e}")

    print(f"\n【线性度分析】")
    print(f"  相关系数R: {linearity['correlation']:.4f}")
    print(f"  R²: {linearity['correlation']**2:.4f}")
    print(f"  最大误差: {linearity['max_error']:.2f}%")
    print(f"  线性误差: {linearity['linearity_error_percent']:.2f}%")

    print(f"\n【温度分析】")
    for r in temp_analysis['temperature_analysis']:
        print(f"  {r['temperature']}K: SNR={r['snr']:.2f}, "
              f"Vth漂移={r['vth_shift_mV']:.1f}mV, "
              f"迁移率因子={r['mobility_factor']:.3f}")

    print(f"\n【失配分析 (Monte Carlo, n={mismatch_analysis['n_trials']})】")
    for r in mismatch_analysis['mismatch_analysis']:
        print(f"  σ/μ={r['mismatch_factor']}: CV={r['cv_across_trials']:.4f}, "
              f"P5-P95=[{r['p5']:.6e}, {r['p95']:.6e}], "
              f"Yield(5%)={r['yield_5pct']:.1%}")

    print(f"\n结果已保存至: {output_file}")

    return results


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../结果"
    run_device_simulation(config_path, output_dir)
