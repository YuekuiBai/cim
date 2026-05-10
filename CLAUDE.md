# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026 Inno CIM (Computing-in-Memory) University Challenge repository. CIM (存算一体) fuses computation and storage into a single hardware unit, eliminating the von Neumann memory wall for AI inference. Organized by Zhicun Technology (知存科技).

**Active work:** Track 5 (CIM Algorithms), Problems 1 and 2. Tracks 2 and 4 contain only documentation.

## Repository Structure

- `公共数据集/` — Shared datasets (CIFAR-10 downloaded, cityscapes/coco zipped)
- `赛道2/` — Track 2: Compiler Toolchain (docs only)
- `赛道4/` — Track 4: CIM Hardware Design (docs only)
- `赛道5/赛题1/` — Track 5 Problem 1: Nonlinearity error study
- `赛道5/赛题2/` — Track 5 Problem 2: STE noise-aware training

## Dependencies

```bash
pip install torch torchvision numpy matplotlib pyyaml tqdm
```

Requires CUDA GPU. Default device is `cuda:1`.

## Running Experiments

### Track 5, Problem 1 (Nonlinearity)

```bash
cd 赛道5/赛题1/代码

# Baseline training
python train_baseline.py

# Full pipeline (all tasks)
python run_sensitivity.py --task all --device cuda

# Individual tasks
python run_sensitivity.py --task task1 --device cuda   # Sensitivity analysis
python run_sensitivity.py --task task2 --alpha 0.2 --device cuda  # NAT training
python run_sensitivity.py --task task3 --device cuda   # Robustness

# Extended research
python run_extended_network_structure.py
python run_extended_gaussian_vs_nonlinear.py
python run_extended_quantization_v2.py

# Visualization
python generate_figures.py
```

### Track 5, Problem 2 (STE)

```bash
cd 赛道5/赛题2/代码

# Full pipeline
python run_ste.py --task all --device cuda:1 --save_dir results

# Individual tasks
python run_ste.py --task task1 --device cuda:1  # STE framework validation
python run_ste.py --task task2 --device cuda:1  # Noise strength training
python run_ste.py --task task3 --device cuda:1  # Comprehensive evaluation
```

## Architecture: Track 5 Problem 1

Nonlinearity model: `x' = alpha * x^3 + (1-alpha) * x` (cubic distortion).

```
代码/run_sensitivity.py  (entry point)
  ├── models/resnet.py         (get_model factory, NonLinearWrapper)
  │     └── noise/nonlinearity.py  (NonLinearInjection, wrapped layers, calibration)
  ├── training/train.py        (NonlinearityAwareTrainer, alpha scheduling)
  ├── evaluation/sensitivity.py    (accuracy degradation, per-layer analysis)
  ├── evaluation/robustness_v2.py  (pre-distortion, calibration, mixed-alpha, OVF/SAM)
  └── evaluation/extended.py       (Gaussian vs nonlinear, quantization)
```

Key classes in `noise/nonlinearity.py`:
- `NonLinearInjection` — applies cubic distortion
- `NonLinearWrapper` — wraps entire model, dynamic alpha control
- `InverseNonLinearity` — pre-distortion compensation
- `CalibrationLayer` — learnable polynomial correction

## Architecture: Track 5 Problem 2

STE framework: noise in forward pass, straight-through gradients in backward.

```
代码/run_ste.py  (entry point)
  ├── models/resnet.py       (ResNet definitions)
  ├── ste/core.py            (STEGradientEstimator, NoiseInjector, NoisyLinear, NoisyConv2d)
  ├── training/              (training utilities)
  ├── evaluation/            (robustness, sensitivity)
  └── 噪声模型/sample_noise.py  (official competition noise model)
```

`NoiseInjector` models: weight programming noise, input crosstalk, saturation nonlinearity, output noise.

## Configuration

Training configs are in `configs/config.yaml` per problem:
- Model: ResNet18, Dataset: CIFAR-10
- 30 epochs, batch_size 64, cosine LR scheduler
- Problem 1: alpha_values [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

## Notes

- No formal build system, test framework, or linter — all entry points are standalone Python scripts.
- Results, trained models (.pth), and figures are stored in `results/` directories.
- Experiment logs go to `日志/`.
- Competition problem descriptions are in markdown files at each track/problem root.
