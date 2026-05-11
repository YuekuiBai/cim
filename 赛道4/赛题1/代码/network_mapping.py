import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
import json
import os

class LayerMapper:
    def __init__(self, array_size: Tuple[int, int], bit_precision: int = 8):
        self.array_size = array_size
        self.bit_precision = bit_precision
        self.mapped_layers = []
        self.total_arrays_used = 0

    def calculate_efficient_mapping(self, weight_shape: Tuple[int, ...], layer_type: str) -> Dict:
        """Calculate optimal array mapping with tiling strategy"""
        total_elements = np.prod(weight_shape)
        array_capacity = self.array_size[0] * self.array_size[1]

        arrays_needed_h = int(np.ceil(weight_shape[0] / self.array_size[0]))
        arrays_needed_w = int(np.ceil(weight_shape[1] / self.array_size[1])) if len(weight_shape) > 1 else 1
        arrays_needed = arrays_needed_h * arrays_needed_w

        used_capacity = arrays_needed * array_capacity
        utilization = total_elements / (used_capacity + 1e-10)

        tiling_factor = min(arrays_needed_h, arrays_needed_w, 4)
        row_splits = max(1, arrays_needed_h // tiling_factor)
        col_splits = max(1, arrays_needed_w // tiling_factor)

        return {
            'tiling_factor': tiling_factor,
            'row_splits': row_splits,
            'col_splits': col_splits,
            'parallel_arrays': row_splits * col_splits
        }

    def map_conv_layer(self, weight: np.ndarray, stride: int = 1, padding: int = 0) -> Dict:
        out_channels, in_channels, kh, kw = weight.shape
        total_weights = out_channels * in_channels * kh * kw
        arrays_needed = int(np.ceil(total_weights / (self.array_size[0] * self.array_size[1])))

        efficient_mapping = self.calculate_efficient_mapping(weight.shape, 'Conv2d')

        utilized_cells = total_weights % (self.array_size[0] * self.array_size[1])
        if utilized_cells == 0:
            utilized_cells = self.array_size[0] * self.array_size[1]
        utilization = total_weights / (arrays_needed * self.array_size[0] * self.array_size[1])

        mapping_info = {
            'layer_type': 'Conv2d',
            'weight_shape': list(weight.shape),
            'total_weights': total_weights,
            'stride': stride,
            'padding': padding,
            'kernel_size': [kh, kw],
            'arrays_needed': arrays_needed,
            'utilization': float(np.clip(utilization, 0, 1)),
            'tiling': efficient_mapping,
            'input_reuse_factor': in_channels,
            'output_reuse_factor': out_channels // arrays_needed if arrays_needed > 0 else 1
        }
        self.mapped_layers.append(mapping_info)
        self.total_arrays_used += arrays_needed
        return mapping_info

    def map_fc_layer(self, weight: np.ndarray, output_bits: int = 8) -> Dict:
        out_features, in_features = weight.shape
        total_weights = out_features * in_features
        arrays_needed = int(np.ceil(total_weights / (self.array_size[0] * self.array_size[1])))

        efficient_mapping = self.calculate_efficient_mapping(weight.shape, 'Linear')

        utilization = (in_features * out_features) / (arrays_needed * self.array_size[0] * self.array_size[1])

        mapping_info = {
            'layer_type': 'Linear',
            'weight_shape': list(weight.shape),
            'total_weights': total_weights,
            'in_features': in_features,
            'out_features': out_features,
            'arrays_needed': arrays_needed,
            'utilization': float(np.clip(utilization, 0, 1)),
            'tiling': efficient_mapping,
            'output_bits': output_bits,
            'batch_processing_support': True
        }
        self.mapped_layers.append(mapping_info)
        self.total_arrays_used += arrays_needed
        return mapping_info

    def map_batchnorm_layer(self, num_features: int) -> Dict:
        mapping_info = {
            'layer_type': 'BatchNorm2d',
            'num_features': num_features,
            'total_weights': num_features * 4,
            'arrays_needed': 1,
            'utilization': 0.5,
            'tiling': {'tiling_factor': 1, 'row_splits': 1, 'col_splits': 1}
        }
        self.mapped_layers.append(mapping_info)
        return mapping_info

    def map_pooling_layer(self, kernel_size: int, pool_type: str = 'max') -> Dict:
        mapping_info = {
            'layer_type': f'Pooling({pool_type})',
            'kernel_size': kernel_size,
            'total_weights': 0,
            'arrays_needed': 0,
            'utilization': 1.0,
            'tiling': {'tiling_factor': 1, 'row_splits': 1, 'col_splits': 1},
            'inplace': True
        }
        self.mapped_layers.append(mapping_info)
        return mapping_info

class DataPathDesigner:
    def __init__(self, dac_bits: int, adc_bits: int, array_size: Tuple[int, int]):
        self.dac_bits = dac_bits
        self.adc_bits = adc_bits
        self.array_size = array_size
        self.data_paths = []

    def design_conv_path(self, in_channels: int, out_channels: int, height: int, width: int,
                        stride: int = 1, padding: int = 0) -> Dict:
        out_height = (height + 2 * padding - 1) // stride
        out_width = (width + 2 * padding - 1) // stride

        word_length = height * width
        output_word_length = out_height * out_width

        bandwidth_requirement = (
            in_channels * out_channels *
            (height * width) * self.dac_bits +
            out_channels * output_word_length * self.adc_bits
        )

        latency_dac = 10e-9
        latency_adc = 10e-9
        latency_sense = 5e-9
        latency_per_patch = latency_dac + latency_sense + latency_adc

        patches_per_output = (height * width) // (stride * stride) if stride > 1 else height * width
        total_latency = patches_per_output * latency_per_patch * out_channels

        path = {
            'type': 'Conv2d',
            'input_dac_channels': in_channels,
            'output_adc_channels': out_channels,
            'input_precision': self.dac_bits,
            'output_precision': self.adc_bits,
            'feature_map_size': [height, width],
            'output_size': [out_height, out_width],
            'word_length': word_length,
            'output_word_length': output_word_length,
            'bandwidth_requirement': bandwidth_requirement,
            'latency_per_layer': total_latency,
            'memory_access_bits': bandwidth_requirement,
            'compute_intensity': out_channels * in_channels * height * width / bandwidth_requirement
        }
        self.data_paths.append(path)
        return path

    def design_fc_path(self, in_features: int, out_features: int, batch_size: int = 1) -> Dict:
        word_length = in_features * batch_size
        output_word_length = out_features * batch_size

        bandwidth_requirement = (
            in_features * self.dac_bits +
            out_features * self.adc_bits
        ) * batch_size

        latency_dac = 10e-9
        latency_adc = 10e-9
        latency_sense = 5e-9
        latency_per_vector = latency_dac + latency_sense + latency_adc
        total_latency = word_length * latency_per_vector

        path = {
            'type': 'Linear',
            'input_dac_channels': batch_size,
            'output_adc_channels': out_features,
            'input_precision': self.dac_bits,
            'output_precision': self.adc_bits,
            'word_length': word_length,
            'output_word_length': output_word_length,
            'batch_size': batch_size,
            'bandwidth_requirement': bandwidth_requirement,
            'latency_per_layer': total_latency,
            'memory_access_bits': bandwidth_requirement,
            'compute_intensity': in_features * out_features / (bandwidth_requirement + 1e-10)
        }
        self.data_paths.append(path)
        return path

    def design_data_reuse_path(self, layer_type: str, reuse_factor: int) -> Dict:
        path = {
            'type': f'{layer_type}_with_reuse',
            'reuse_factor': reuse_factor,
            'bandwidth_reduction': reuse_factor,
            'energy_reduction': reuse_factor
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
            'total_latency': 0,
            'layer_mappings': [],
            'data_paths': [],
            'communication_overhead': {},
            'optimization_summary': {}
        }

        for layer in layers:
            if layer['type'] == 'Conv2d':
                out_ch = layer['out_channels']
                in_ch = layer['in_channels']
                kh = layer['kernel_size']
                kw = layer['kernel_size']
                stride = layer.get('stride', 1)
                padding = layer.get('padding', 0)

                layer_map = self.mapper.map_conv_layer(
                    np.random.randn(out_ch, in_ch, kh, kw),
                    stride, padding
                )
                mapping_scheme['layer_mappings'].append(layer_map)

                path = self.path_designer.design_conv_path(in_ch, out_ch, 32, 32, stride, padding)
                mapping_scheme['data_paths'].append(path)

            elif layer['type'] == 'Linear':
                in_feat = layer['in_features']
                out_feat = layer['out_features']

                layer_map = self.mapper.map_fc_layer(np.random.randn(out_feat, in_feat))
                mapping_scheme['layer_mappings'].append(layer_map)

                path = self.path_designer.design_fc_path(in_feat, out_feat)
                mapping_scheme['data_paths'].append(path)

        mapping_scheme['total_arrays'] = self.mapper.total_arrays_used
        mapping_scheme['total_bandwidth'] = sum(p['bandwidth_requirement'] for p in self.path_designer.data_paths)
        mapping_scheme['total_latency'] = sum(p['latency_per_layer'] for p in self.path_designer.data_paths)
        mapping_scheme['communication_overhead'] = self._estimate_communication()
        mapping_scheme['optimization_summary'] = self._generate_optimization_summary()

        return mapping_scheme

    def _estimate_communication(self) -> Dict:
        total_bandwidth = sum(p['bandwidth_requirement'] for p in self.path_designer.data_paths)
        total_latency = sum(p['latency_per_layer'] for p in self.path_designer.data_paths)

        avg_utilization = np.mean([lm['utilization'] for lm in self.mapper.mapped_layers])

        return {
            'total_bandwidth_bits_per_inference': total_bandwidth,
            'inter_array_bandwidth': total_bandwidth * 0.1,
            'memory_access_reduction_factor': 100 / (1 - avg_utilization + 0.1),
            'total_latency_per_inference': total_latency,
            'latency_per_inference_ms': total_latency * 1000,
            'average_array_utilization': avg_utilization
        }

    def _generate_optimization_summary(self) -> Dict:
        layers = self.mapper.mapped_layers
        total_arrays = self.mapper.total_arrays_used

        avg_utilization = np.mean([lm['utilization'] for lm in layers])

        low_util_layers = [lm for lm in layers if lm['utilization'] < 0.5]
        high_latency_paths = [p for p in self.path_designer.data_paths if p['latency_per_layer'] > 1e-6]

        return {
            'total_layers': len(layers),
            'average_utilization': float(avg_utilization),
            'utilization_bottlenecks': len(low_util_layers),
            'high_latency_layers': len(high_latency_paths),
            'recommendations': [
                f"考虑增大阵列尺寸以提高利用率(当前{avg_utilization:.1%})" if avg_utilization < 0.5 else "利用率良好",
                f"优化{len(low_util_layers)}个低利用率层的映射" if low_util_layers else "所有层利用率均衡",
                "启用数据复用以降低带宽需求" if len(high_latency_paths) > 0 else "延迟表现良好"
            ]
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
    print(f"总延迟: {scheme['total_latency']*1000:.3f} ms")

    print("\n层级映射详情:")
    print("-" * 50)
    for i, lm in enumerate(scheme['layer_mappings']):
        print(f"\nLayer {i+1}: {lm['layer_type']}")
        print(f"  Weight Shape: {lm['weight_shape']}")
        print(f"  Total Weights: {lm['total_weights']}")
        print(f"  Arrays Needed: {lm['arrays_needed']}")
        print(f"  Utilization: {lm.get('utilization', 0):.2%}")
        if 'tiling' in lm:
            print(f"  Tiling: {lm['tiling']}")

    print("\n" + "=" * 60)
    print("优化建议:")
    print("-" * 50)
    for rec in scheme['optimization_summary'].get('recommendations', []):
        print(f"  - {rec}")

    print(f"\n结果已保存至: {output_file}")

    return scheme

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../结果"
    run_network_mapping(config_path, output_dir)