import numpy as np
import yaml
from typing import Dict, List, Tuple
import json
import os

class LayerMapper:
    def __init__(self, array_size: Tuple[int, int], bit_precision: int = 8):
        self.array_size = array_size
        self.bit_precision = bit_precision
        self.mapped_layers = []

    def map_conv_layer(self, weight: np.ndarray, stride: int = 1) -> Dict:
        out_channels, in_channels, kh, kw = weight.shape
        total_weights = out_channels * in_channels * kh * kw
        arrays_needed = int(np.ceil(total_weights / (self.array_size[0] * self.array_size[1])))

        mapping_info = {
            'layer_type': 'Conv2d',
            'weight_shape': list(weight.shape),
            'total_weights': total_weights,
            'stride': stride,
            'arrays_needed': arrays_needed,
            'utilization': (total_weights % (self.array_size[0] * self.array_size[1])) / (arrays_needed * self.array_size[0] * self.array_size[1] + 1e-10)
        }
        self.mapped_layers.append(mapping_info)
        return mapping_info

    def map_fc_layer(self, weight: np.ndarray) -> Dict:
        out_features, in_features = weight.shape
        total_weights = out_features * in_features
        arrays_needed = int(np.ceil(total_weights / (self.array_size[0] * self.array_size[1])))

        mapping_info = {
            'layer_type': 'Linear',
            'weight_shape': list(weight.shape),
            'total_weights': total_weights,
            'arrays_needed': arrays_needed,
            'utilization': (in_features * out_features % (self.array_size[0] * self.array_size[1])) / (arrays_needed * self.array_size[0] * self.array_size[1] + 1e-10)
        }
        self.mapped_layers.append(mapping_info)
        return mapping_info

class DataPathDesigner:
    def __init__(self, dac_bits: int, adc_bits: int, array_size: Tuple[int, int]):
        self.dac_bits = dac_bits
        self.adc_bits = adc_bits
        self.array_size = array_size
        self.data_paths = []

    def design_conv_path(self, in_channels: int, out_channels: int, height: int, width: int) -> Dict:
        path = {
            'type': 'Conv2d',
            'input_dac_channels': in_channels,
            'output_adc_channels': out_channels,
            'input_precision': self.dac_bits,
            'output_precision': self.adc_bits,
            'feature_map_size': [height, width],
            'word_length': height * width,
            'bandwidth_requirement': in_channels * out_channels * height * width * self.dac_bits,
            'latency_per_layer': height * width * 65e-9
        }
        self.data_paths.append(path)
        return path

    def design_fc_path(self, in_features: int, out_features: int) -> Dict:
        path = {
            'type': 'Linear',
            'input_dac_channels': 1,
            'output_adc_channels': out_features,
            'input_precision': self.dac_bits,
            'output_precision': self.adc_bits,
            'word_length': in_features,
            'bandwidth_requirement': in_features * out_features * self.dac_bits,
            'latency_per_layer': in_features * 65e-9
        }
        self.data_paths.append(path)
        return path

class NetworkMapper:
    def __init__(self, config: Dict):
        self.config = config
        self.array_size = tuple(map(int, config['device']['array_size'].split('x')))
        self.mapper = LayerMapper(self.array_size, config['device']['bit_precision'])
        self.peripherals = config['peripherals']
        self.path_designer = DataPathDesigner(
            self.peripherals['dac']['bits'],
            self.peripherals['adc']['bits'],
            self.array_size
        )

    def create_mapping_scheme(self) -> Dict:
        network_cfg = self.config['network']
        layers = network_cfg['layers']

        mapping_scheme = {
            'total_arrays': 0,
            'total_bandwidth': 0,
            'layer_mappings': [],
            'data_paths': [],
            'communication_overhead': {}
        }

        for layer in layers:
            if layer['type'] == 'Conv2d':
                out_ch = layer['out_channels']
                in_ch = layer['in_channels']
                kh = layer['kernel_size']
                kw = layer['kernel_size']
                total_weights = out_ch * in_ch * kh * kw
                arrays = int(np.ceil(total_weights / (self.array_size[0] * self.array_size[1])))

                layer_map = self.mapper.map_conv_layer(
                    np.random.randn(out_ch, in_ch, kh, kw),
                    layer['stride']
                )
                mapping_scheme['layer_mappings'].append(layer_map)

                path = self.path_designer.design_conv_path(in_ch, out_ch, 32, 32)
                mapping_scheme['data_paths'].append(path)
                mapping_scheme['total_arrays'] += arrays
                mapping_scheme['total_bandwidth'] += path['bandwidth_requirement']

            elif layer['type'] == 'Linear':
                in_feat = layer['in_features']
                out_feat = layer['out_features']
                total_weights = in_feat * out_feat
                arrays = int(np.ceil(total_weights / (self.array_size[0] * self.array_size[1])))

                layer_map = self.mapper.map_fc_layer(np.random.randn(out_feat, in_feat))
                mapping_scheme['layer_mappings'].append(layer_map)

                path = self.path_designer.design_fc_path(in_feat, out_feat)
                mapping_scheme['data_paths'].append(path)
                mapping_scheme['total_arrays'] += arrays
                mapping_scheme['total_bandwidth'] += path['bandwidth_requirement']

        mapping_scheme['communication_overhead'] = self._estimate_communication()
        return mapping_scheme

    def _estimate_communication(self) -> Dict:
        total_bandwidth = sum(p['bandwidth_requirement'] for p in self.path_designer.data_paths)
        return {
            'total_bandwidth_bits_per_inference': total_bandwidth,
            'inter_array_bandwidth': total_bandwidth * 0.1,
            'memory_access_reduction_factor': 100,
            'total_latency_per_inference': sum(p['latency_per_layer'] for p in self.path_designer.data_paths)
        }

def run_network_mapping(config_path: str = "config.yaml", output_dir: str = "../结果"):
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mapper = NetworkMapper(config)
    scheme = mapper.create_mapping_scheme()

    output_file = os.path.join(output_dir, 'network_mapping_results.json')
    with open(output_file, 'w') as f:
        json.dump(scheme, f, indent=2)

    print("=" * 60)
    print("神经网络映射方案报告")
    print("=" * 60)
    print(f"\n总阵列数量: {scheme['total_arrays']}")
    print(f"总带宽需求: {scheme['total_bandwidth']:.2e} bits/推理")

    print("\n层级映射详情:")
    print("-" * 50)
    for i, lm in enumerate(scheme['layer_mappings']):
        print(f"\nLayer {i+1}: {lm['layer_type']}")
        print(f"  Weight Shape: {lm['weight_shape']}")
        print(f"  Total Weights: {lm['total_weights']}")
        print(f"  Arrays Needed: {lm['arrays_needed']}")
        print(f"  Utilization: {lm.get('utilization', 0):.2%}")

    print("\n" + "=" * 60)
    print("通讯开销估计:")
    print("-" * 50)
    for k, v in scheme['communication_overhead'].items():
        if isinstance(v, float) and v > 1e6:
            print(f"  {k}: {v:.2e}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n结果已保存至: {output_file}")

    return scheme

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../结果"
    run_network_mapping(config_path, output_dir)
