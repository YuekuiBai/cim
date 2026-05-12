import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from typing import Dict, List, Tuple

for font in ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

ARRAY_SIZE = (128, 128)


class DeviceModel:
    def __init__(self, name: str, array_size: Tuple[int, int] = ARRAY_SIZE):
        self.name = name
        self.rows, self.cols = array_size
        self.weights = np.random.uniform(-1, 1, (self.rows, self.cols))

    def matrix_vector_multiply(self, input_vec: np.ndarray,
                               temperature: float = 300,
                               mismatch_factor: float = None) -> np.ndarray:
        raise NotImplementedError

    def compute_ideal_output(self, input_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def analyze_linearity(self, input_range: Tuple[float, float],
                         num_points: int = 50) -> Dict:
        inputs = np.linspace(input_range[0], input_range[1], num_points)
        ideal_all, actual_all = [], []

        for inp in inputs:
            inp_vec = np.full(self.rows, inp)
            ideal = np.mean(self.compute_ideal_output(inp_vec))
            actual = np.mean(self.matrix_vector_multiply(inp_vec, temperature=300))
            ideal_all.append(ideal)
            actual_all.append(actual)

        ideal_all = np.array(ideal_all)
        actual_all = np.array(actual_all)

        if np.std(ideal_all) < 1e-20 or np.std(actual_all) < 1e-20:
            corr = 0.0
        else:
            corr = np.corrcoef(ideal_all, actual_all)[0, 1]

        ideal_range = np.max(ideal_all) - np.min(ideal_all)
        if ideal_range > 1e-20:
            max_err = np.max(np.abs(ideal_all - actual_all)) / ideal_range * 100
        else:
            max_err = 0.0

        return {
            'ideal': ideal_all.tolist(),
            'actual': actual_all.tolist(),
            'correlation': float(corr),
            'r_squared': float(corr ** 2),
            'max_error': float(max_err)
        }

    def analyze_temperature(self, temperatures: List[float],
                          input_val: float = 0.9) -> Dict:
        inp_vec = np.full(self.rows, input_val)
        results = []
        for temp in temperatures:
            outputs = [self.matrix_vector_multiply(inp_vec, temperature=temp)
                      for _ in range(10)]
            mean_out = np.mean(outputs)
            std_out = np.std(outputs) + 1e-30
            results.append({
                'temperature': temp,
                'mean': float(mean_out),
                'std': float(std_out),
                'snr': float(np.abs(mean_out) / std_out)
            })
        return {'temperature_results': results}

    def analyze_mismatch(self, mismatch_factors: List[float],
                         input_val: float = 0.9, n_trials: int = 100) -> Dict:
        results = []
        for mf in mismatch_factors:
            trial_means = []
            for _ in range(n_trials):
                self.weights = np.random.uniform(-1, 1, (self.rows, self.cols))
                inp_vec = np.full(self.rows, input_val)
                out = self.matrix_vector_multiply(inp_vec, temperature=300,
                                                  mismatch_factor=mf)
                trial_means.append(np.mean(out))
            trial_means = np.array(trial_means)
            mean_val = np.mean(trial_means)
            std_val = np.std(trial_means)
            results.append({
                'mismatch_factor': mf,
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(std_val / np.abs(mean_val)) if np.abs(mean_val) > 1e-30 else 0.0,
                'yield_5pct': float(np.mean(
                    np.abs(trial_means - mean_val) / np.abs(mean_val) < 0.05
                ))
            })
        self.weights = np.random.uniform(-1, 1, (self.rows, self.cols))
        return {'mismatch_results': results}


class FeFETModel(DeviceModel):
    def __init__(self):
        super().__init__("FeFET")
        mu = 140 * 1e-4
        eps0 = 8.854e-12
        eps_r = 3.9
        tox = 1.0e-8
        W, L = 1e-7, 2.8e-8
        self.k = mu * (eps_r * eps0 / tox) * W / L
        self.dVth_dT = -2e-3
        self.mu_exp = -1.5
        self.vt_range = (0.3, 1.2)
        self.default_mismatch = 0.05

    def _weight_to_vth(self, w: float) -> float:
        vt_min, vt_max = self.vt_range
        return vt_min + (w + 1) / 2 * (vt_max - vt_min)

    def _drain_current_nominal(self, vgs: float, vds: float, vth: float) -> float:
        if vgs <= vth:
            return 1e-12
        vgt = vgs - vth
        if vds < vgt:
            ids = self.k * ((vgt - vds / 2) * vds)
        else:
            ids = self.k * (vgt ** 2 / 2)
        return max(ids, 1e-12)

    def _drain_current_real(self, vgs: float, vds: float, vth: float,
                           temperature: float, mu_factor: float,
                           mismatch: float) -> float:
        if vgs <= vth:
            return 1e-12
        vgt = vgs - vth
        if vds < vgt:
            ids = self.k * mu_factor * ((vgt - vds / 2) * vds)
        else:
            ids = self.k * mu_factor * (vgt ** 2 / 2)
        ids *= (1 - 0.08 * vgs)
        ids *= (1 + np.random.normal(0, mismatch))
        dVth = self.dVth_dT * (temperature - 300)
        ids *= (1 + dVth / vth)
        return max(ids, 1e-12)

    def matrix_vector_multiply(self, input_vec: np.ndarray,
                               temperature: float = 300,
                               mismatch_factor: float = None) -> np.ndarray:
        if mismatch_factor is None:
            mismatch_factor = self.default_mismatch
        mu_factor = (temperature / 300) ** self.mu_exp
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                vgs = np.clip(input_vec[i], 0, 1.8)
                vth = self._weight_to_vth(self.weights[i, j])
                ids = self._drain_current_real(vgs, 0.1, vth, temperature,
                                                mu_factor, mismatch_factor)
                noise = np.random.normal(0, 3e-12 * mu_factor)
                col_sum += ids + noise
            output[j] = col_sum
        return output

    def compute_ideal_output(self, input_vec: np.ndarray) -> np.ndarray:
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                vgs = np.clip(input_vec[i], 0, 1.8)
                vth = self._weight_to_vth(self.weights[i, j])
                ids = self._drain_current_nominal(vgs, 0.1, vth)
                col_sum += ids
            output[j] = col_sum
        return output


class RRAMModel(DeviceModel):
    def __init__(self):
        super().__init__("RRAM")
        self.conductance_range = (1e-6, 1e-3)
        self.alpha = 0.15
        self.default_mismatch = 0.0

    def _weight_to_conductance(self, w: float) -> float:
        g_min, g_max = self.conductance_range
        return g_min + (w + 1) / 2 * (g_max - g_min)

    def _ideal_current(self, v_in: float, g: float) -> float:
        return g * v_in

    def _real_current(self, v_in: float, g: float,
                     temperature: float, mismatch: float) -> float:
        v_eff = v_in + self.alpha * v_in ** 3
        g_adj = g * (1 + np.random.normal(0, mismatch))
        noise = np.random.normal(0, 5e-9 * (temperature / 300))
        return g_adj * v_eff + noise

    def matrix_vector_multiply(self, input_vec: np.ndarray,
                               temperature: float = 300,
                               mismatch_factor: float = None) -> np.ndarray:
        if mismatch_factor is None:
            mismatch_factor = self.default_mismatch
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                v_in = np.clip(input_vec[i], 0, 1.8)
                g = self._weight_to_conductance(self.weights[i, j])
                col_sum += self._real_current(v_in, g, temperature, mismatch_factor)
            output[j] = col_sum
        return output

    def compute_ideal_output(self, input_vec: np.ndarray) -> np.ndarray:
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                v_in = np.clip(input_vec[i], 0, 1.8)
                g = self._weight_to_conductance(self.weights[i, j])
                col_sum += self._ideal_current(v_in, g)
            output[j] = col_sum
        return output


class SRAMModel(DeviceModel):
    def __init__(self):
        super().__init__("SRAM")
        self.vdd = 1.8
        self.conductance_range = (1e-6, 1e-3)
        self.default_mismatch = 0.03

    def _weight_to_conductance(self, w: float) -> float:
        g_min, g_max = self.conductance_range
        return g_min + (w + 1) / 2 * (g_max - g_min)

    def _ideal_current(self, v_in: float, g: float) -> float:
        return g * v_in

    def _real_current(self, v_in: float, g: float,
                     temperature: float, mismatch: float) -> float:
        v_eff = v_in * (1 + 0.03 * np.sin(np.pi * v_in / self.vdd))
        g_adj = g * (1 + np.random.normal(0, mismatch))
        noise = np.random.normal(0, 5e-9 * (temperature / 300))
        return g_adj * v_eff + noise

    def matrix_vector_multiply(self, input_vec: np.ndarray,
                               temperature: float = 300,
                               mismatch_factor: float = None) -> np.ndarray:
        if mismatch_factor is None:
            mismatch_factor = self.default_mismatch
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                v_in = np.clip(input_vec[i], 0, self.vdd)
                g = self._weight_to_conductance(self.weights[i, j])
                col_sum += self._real_current(v_in, g, temperature, mismatch_factor)
            output[j] = col_sum
        return output

    def compute_ideal_output(self, input_vec: np.ndarray) -> np.ndarray:
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                v_in = np.clip(input_vec[i], 0, self.vdd)
                g = self._weight_to_conductance(self.weights[i, j])
                col_sum += self._ideal_current(v_in, g)
            output[j] = col_sum
        return output


class PCMModel(DeviceModel):
    def __init__(self):
        super().__init__("PCM")
        self.conductance_range = (1e-6, 5e-4)
        self.thermal_coeff = 0.02
        self.default_mismatch = 0.03

    def _weight_to_conductance(self, w: float) -> float:
        g_min, g_max = self.conductance_range
        return g_min + (w + 1) / 2 * (g_max - g_min)

    def _ideal_current(self, v_in: float, g: float) -> float:
        return g * v_in

    def _real_current(self, v_in: float, g: float,
                     temperature: float, mismatch: float) -> float:
        tf = 1 + self.thermal_coeff * (temperature - 300) / 100
        v_eff = v_in + 0.05 * v_in ** 2
        g_adj = g * tf * (1 + np.random.normal(0, mismatch))
        noise = np.random.normal(0, 5e-9 * tf)
        return g_adj * v_eff + noise

    def matrix_vector_multiply(self, input_vec: np.ndarray,
                               temperature: float = 300,
                               mismatch_factor: float = None) -> np.ndarray:
        if mismatch_factor is None:
            mismatch_factor = self.default_mismatch
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                v_in = np.clip(input_vec[i], 0, 1.8)
                g = self._weight_to_conductance(self.weights[i, j])
                col_sum += self._real_current(v_in, g, temperature, mismatch_factor)
            output[j] = col_sum
        return output

    def compute_ideal_output(self, input_vec: np.ndarray) -> np.ndarray:
        output = np.zeros(self.cols)
        for j in range(self.cols):
            col_sum = 0.0
            for i in range(self.rows):
                v_in = np.clip(input_vec[i], 0, 1.8)
                g = self._weight_to_conductance(self.weights[i, j])
                col_sum += self._ideal_current(v_in, g)
            output[j] = col_sum
        return output


def run_comparison(output_dir: str = "../结果") -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    devices = {
        'FeFET': FeFETModel(),
        'RRAM': RRAMModel(),
        'SRAM': SRAMModel(),
        'PCM': PCMModel()
    }

    input_range = (0.4, 1.5)
    temperatures = [250, 300, 350, 400]
    mismatch_factors = [0.01, 0.05, 0.10, 0.20]

    results = {
        'linearity': {},
        'temperature': {},
        'mismatch': {},
        'summary': {}
    }

    print("=" * 70)
    print("CIM器件横向对比实验")
    print("=" * 70)

    for name, dev in devices.items():
        print(f"\n[{name}] 开始仿真...")

        linearity = dev.analyze_linearity(input_range, num_points=50)
        results['linearity'][name] = linearity

        temp_res = dev.analyze_temperature(temperatures)
        results['temperature'][name] = temp_res

        mismatch_res = dev.analyze_mismatch(mismatch_factors)
        results['mismatch'][name] = mismatch_res

        print(f"  线性度: R2={linearity['r_squared']:.4f}, 最大误差={linearity['max_error']:.2f}%")

    results['summary'] = {
        'linearity_r2': {name: results['linearity'][name]['r_squared'] for name in devices},
        'linearity_error': {name: results['linearity'][name]['max_error'] for name in devices},
        'temperature_snr_300k': {
            name: [r for r in results['temperature'][name]['temperature_results']
                   if r['temperature'] == 300][0]['snr']
            for name in devices
        },
        'mismatch_yield_5pct': {
            name: [r for r in results['mismatch'][name]['mismatch_results']
                   if abs(r['mismatch_factor'] - 0.05) < 1e-3][0]['yield_5pct']
            for name in devices
        }
    }

    output_file = os.path.join(output_dir, 'device_comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    plot_comparison_results(results, output_dir)

    print("\n" + "=" * 70)
    print("对比实验结果汇总")
    print("=" * 70)
    print(f"\n{'Device':<10} {'R2':<10} {'MaxErr%':<12} {'SNR@300K':<12} {'Yield@5%':<10}")
    print("-" * 54)
    for name in devices:
        r2 = results['summary']['linearity_r2'][name]
        err = results['summary']['linearity_error'][name]
        snr = results['summary']['temperature_snr_300k'][name]
        yld = results['summary']['mismatch_yield_5pct'][name]
        print(f"{name:<10} {r2:<10.4f} {err:<12.2f} {snr:<12.2f} {yld:<10.1%}")

    print(f"\nResults saved to: {output_file}")
    return results


def plot_comparison_results(results: Dict, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'FeFET': '#2E86AB', 'RRAM': '#E94F37', 'SRAM': '#44AF69', 'PCM': '#F18F01'}

    ax = axes[0, 0]
    for name in results['linearity']:
        lin = results['linearity'][name]
        ideal = np.array(lin['ideal'])
        actual = np.array(lin['actual'])
        ax.scatter(ideal, actual, c=colors[name], alpha=0.5, s=15, label=name)
    max_val = max(np.max(np.array(results['linearity'][n]['ideal'])) for n in results['linearity'])
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Ideal')
    ax.set_xlabel('Ideal Output (A)', fontsize=12)
    ax.set_ylabel('Actual Output (A)', fontsize=12)
    ax.set_title('Linearity: Actual vs Ideal Output (All Devices)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    device_names = list(results['summary']['linearity_r2'].keys())
    r2_values = list(results['summary']['linearity_r2'].values())
    err_values = list(results['summary']['linearity_error'].values())
    x = np.arange(len(device_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, r2_values, width, label='R2',
                   color=[colors[n] for n in device_names], alpha=0.8)
    ax.set_ylabel('R2', fontsize=12)
    ax.set_ylim(0.85, 1.01)
    ax.set_title('Linearity Comparison (R2)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(device_names)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, r2, err in zip(bars1, r2_values, err_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{r2:.4f}\n({err:.1f}%)', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 0]
    temps = [250, 300, 350, 400]
    for name in results['temperature']:
        snrs = [r['snr'] for r in results['temperature'][name]['temperature_results']]
        ax.plot(temps, snrs, 'o-', color=colors[name], linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('SNR', fontsize=12)
    ax.set_title('SNR vs Temperature', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(temps)

    ax = axes[1, 1]
    mf_values = [0.01, 0.05, 0.10, 0.20]
    for name in results['mismatch']:
        yields = [r['yield_5pct'] for r in results['mismatch'][name]['mismatch_results']]
        ax.plot(mf_values, yields, 'o-', color=colors[name], linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Mismatch Factor (sigma/mu)', fontsize=12)
    ax.set_ylabel('Yield @ 5% Tolerance', fontsize=12)
    ax.set_title('Mismatch Tolerance (Monte Carlo, n=100)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'device_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig2, ax = plt.subplots(figsize=(12, 6))
    categories = ['Linearity\n(R2)', 'Error Suppr.\n(1/Error%)', 'Temp Stability\n(SNR)', 'Mismatch Tol.\n(Yield%)']
    normalized = {}
    for name in ['FeFET', 'RRAM', 'SRAM', 'PCM']:
        r2 = results['summary']['linearity_r2'][name]
        err = results['summary']['linearity_error'][name]
        snr = results['summary']['temperature_snr_300k'][name]
        yld = results['summary']['mismatch_yield_5pct'][name]
        norm_r2 = (r2 - 0.9) / 0.1
        norm_err = 1.0 / (err + 0.5)
        max_err = max(1.0 / (results['summary']['linearity_error'][n] + 0.5) for n in ['FeFET', 'RRAM', 'SRAM', 'PCM'])
        norm_err = norm_err / max_err
        norm_snr = min(snr / 20.0, 1.0)
        norm_yld = yld
        normalized[name] = [norm_r2, norm_err, norm_snr, norm_yld]

    x = np.arange(len(categories))
    width = 0.2
    for i, name in enumerate(['FeFET', 'RRAM', 'SRAM', 'PCM']):
        vals = normalized[name]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[name], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('CIM Device Comprehensive Comparison (Normalized)', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'device_comparison_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison charts saved to: {os.path.join(output_dir, 'device_comparison.png')}")
    print(f"Comprehensive chart saved to: {os.path.join(output_dir, 'device_comparison_radar.png')}")


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "../结果"
    run_comparison(output_dir)
