import torch
import torch.nn.functional as F


def matmul(input, weight):
    return torch.matmul(input, weight)


def noisy_matmul(input, weight,
                 prog_noise_std=0.01,
                 drift_factor=0.005,
                 nonlinear_alpha=0.1,
                 nonlinear_beta=0.05,
                 output_noise_std=0.01,
                 quantization_bits=None,
                 crosstalk_factor=0.002,
                 temperature_factor=0.001,
                 retention_loss=0.001):
    """
    Noisy matmul modeling analog compute-in-memory effects.

    Args:
        input: Input tensor
        weight: Weight tensor
        prog_noise_std: Programming noise standard deviation
        drift_factor: Weight drift factor
        nonlinear_alpha: Positive saturation parameter
        nonlinear_beta: Negative saturation parameter
        output_noise_std: Output noise standard deviation
        quantization_bits: ADC quantization bits (None for no quantization)
        crosstalk_factor: Input crosstalk factor
        temperature_factor: Temperature noise factor
        retention_loss: Retention loss factor

    Returns:
        Noisy matmul result
    """
    prog_noise = torch.randn_like(weight) * prog_noise_std
    drift_noise = torch.randn_like(weight) * drift_factor * torch.abs(weight)
    retention_noise = weight * retention_loss * (torch.rand_like(weight) - 0.5)
    temp_noise = torch.randn_like(weight) * temperature_factor * torch.sqrt(torch.abs(weight))

    noisy_weight = weight + prog_noise + drift_noise + retention_noise + temp_noise

    if crosstalk_factor > 0:
        crosstalk = torch.randn_like(input) * crosstalk_factor * torch.norm(input, dim=-1, keepdim=True)
        noisy_input = input + crosstalk
    else:
        noisy_input = input

    result = torch.matmul(noisy_input, noisy_weight)

    pos_mask = result > 0
    neg_mask = result < 0

    result_nonlinear = torch.zeros_like(result)
    result_nonlinear[pos_mask] = torch.tanh(nonlinear_alpha * result[pos_mask]) / nonlinear_alpha
    result_nonlinear[neg_mask] = torch.tanh((nonlinear_alpha + nonlinear_beta) * result[neg_mask]) / (nonlinear_alpha + nonlinear_beta)

    if quantization_bits is not None:
        max_val = torch.max(torch.abs(result_nonlinear))
        scale = (2 ** (quantization_bits - 1) - 1) / max_val
        quantized = torch.round(result_nonlinear * scale) / scale
        quant_noise = (torch.rand_like(result_nonlinear) - 0.5) / scale
        result_nonlinear = quantized + quant_noise

    output_noise = torch.randn_like(result_nonlinear) * output_noise_std
    f_noise = torch.randn_like(result_nonlinear) * output_noise_std * 0.3
    spatial_corr = F.conv2d(f_noise.unsqueeze(0).unsqueeze(0),
                           torch.ones(1, 1, 3, 3) / 9,
                           padding=1).squeeze() if len(result_nonlinear.shape) >= 2 else f_noise

    supply_variation = (1 + (torch.randn(1) * 0.01).item())

    final_result = (result_nonlinear + output_noise + spatial_corr) * supply_variation

    return final_result