"""评估模块"""

from .sensitivity import (
    sensitivity_analysis,
    layer_output_distribution_analysis,
    layer_error_accumulation_analysis,
    comprehensive_sensitivity_analysis,
    plot_sensitivity,
    plot_layer_distribution_shift,
    plot_error_accumulation
)

from .robustness import (
    RobustModelWrapper,
    PreDistortionWrapper,
    train_calibration_layer,
    evaluate_robustness_methods,
    compare_all_methods,
    comprehensive_robustness_analysis
)

from .extended import (
    GaussianNoiseInjection,
    QuantizationNoise,
    CombinedNoise,
    compare_noise_types,
    analyze_quantization_nonlinearity,
    analyze_model_architecture,
    comprehensive_extended_analysis
)

__all__ = [
    # 敏感性分析
    'sensitivity_analysis',
    'layer_output_distribution_analysis',
    'layer_error_accumulation_analysis',
    'comprehensive_sensitivity_analysis',
    'plot_sensitivity',
    'plot_layer_distribution_shift',
    'plot_error_accumulation',
    # 鲁棒性增强
    'RobustModelWrapper',
    'PreDistortionWrapper',
    'train_calibration_layer',
    'evaluate_robustness_methods',
    'compare_all_methods',
    'comprehensive_robustness_analysis',
    # 拓展研究
    'GaussianNoiseInjection',
    'QuantizationNoise',
    'CombinedNoise',
    'compare_noise_types',
    'analyze_quantization_nonlinearity',
    'analyze_model_architecture',
    'comprehensive_extended_analysis'
]
