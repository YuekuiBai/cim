# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026 Inno CIM (Computing-in-Memory) University Challenge repository. CIM (存算一体) fuses computation and storage into a single hardware unit, eliminating the von Neumann memory wall for AI inference. Organized by Zhicun Technology (知存科技).

**Active work:** All 3 tracks, 6 problems are actively developed. Status as of 2026-05-14:

| Track | Problem | Status | Completion | Innovation Level |
|-------|---------|--------|------------|------------------|
| Track 2 | Problem 1: CIM Compiler (Basic) | ✅ Complete | ~95% | High (3 patents filed) |
| Track 2 | Problem 2: CIM 3D Mapper (MoE) | ✅ Complete | ~90% | High (2 patents filed) |
| Track 4 | Problem 1: CIM Array Design | ✅ Complete | ~85% | Medium (tech report + PDF) |
| Track 4 | Problem 2: Device Comparison | ✅ Complete | ~85% | Medium (tech report + PDF) |
| Track 5 | Problem 1: Nonlinearity Error | ✅ Complete | ~95% | High (NAT+Mixed Alpha, OVF, SAM) |
| Track 5 | Problem 2: STE Noise Training | ⚠️ Partial | ~85% | High (Adaptive-STE-Sqrt) |

## Repository Structure

- `公共数据集/` — Shared datasets (CIFAR-10 downloaded, cityscapes/coco zipped)
- `赛道2/赛题1/` — Track 2 Problem 1: CIM Compiler Toolchain (ONNX→IR→ISA)
- `赛道2/赛题2/` — Track 2 Problem 2: CIM 3D Mapper for MoE models
- `赛道4/赛题1/` — Track 4 Problem 1: CIM Array Design (simulation + mapping)
- `赛道4/赛题2/` — Track 4 Problem 2: Device Comparison (RRAM vs others)
- `赛道5/赛题1/` — Track 5 Problem 1: Nonlinearity error study
- `赛道5/赛题2/` — Track 5 Problem 2: STE noise-aware training

## Deliverables Status

### Patents (5 filed)

**Track 2, Problem 1 (3 patents):**
1. `赛道2/赛题1/文档/专利一_基于区间图着色的SRAM动态分配方法.md` — Interval graph coloring for SRAM allocation
2. `赛道2/赛题1/文档/专利二_面向存算一体架构的多Pass协同优化编译方法.md` — Multi-pass collaborative optimization
3. `赛道2/赛题1/文档/专利三_面向存算一体架构的有符号数位串行计算指令生成方法.md` — Signed digit serial computation

**Track 2, Problem 2 (2 patents):**
1. `赛道2/赛题2/文档/专利一_基于饱和度图着色的MoE_Expert分组方法.md` — Saturation graph coloring for MoE expert grouping
2. `赛道2/赛题2/文档/专利二_频率感知的3D存算阵列权重放置方法.md` — Frequency-aware 3D CIM weight placement

### Academic Papers / Technical Reports

| Track/Problem | Report | Slides | Status |
|---------------|--------|--------|--------|
| Track 2 P1 | — | — | Patents only |
| Track 2 P2 | — | — | Patents only |
| Track 4 P1 | `技术报告.pdf`, `技术报告.docx` | — | Complete |
| Track 4 P2 | `技术报告.pdf`, `技术报告.docx` | — | Complete |
| Track 5 P1 | `技术报告.pdf`, `技术报告.docx`, `slides.tex` | LaTeX slides | Complete |
| Track 5 P2 | `实验报告.md`, `算法设计文档.md`, `slides.tex` | LaTeX slides | Partial (PPT missing) |

### Track 5 Problem 2 Gaps (per self-assessment 2026-05-10)
- Missing: PPT file (only prompt exists), COCO detection validation, Cityscapes segmentation validation
- CIFAR-10 accuracy: 85.30% (Adaptive-STE-Sqrt, exceeds baseline 85.15%)
- Completion: ~85%

## Dependencies

```bash
pip install torch torchvision numpy matplotlib pyyaml tqdm
```

Requires CUDA GPU. Default device is `cuda:1`.

## Running Experiments

### Track 2, Problem 1 (CIM Compiler)

```bash
cd 赛道2/赛题1/代码

# Compile ONNX model
python main.py --model <path_to_onnx> --output output

# Run all tests
python run_all_tests.py

# Generate plots
python generate_plots.py
python generate_advanced_plots.py

# Demo
python demo.py
```

### Track 2, Problem 2 (CIM 3D Mapper)

```bash
cd 赛道2/赛题2/代码

# Compile model to 3D CIM space
python main.py --model <path_to_model_json> --output output

# With MoE activation trace
python main.py --model <path> --trace <trace_path> --output output
```

### Track 4, Problem 1 (CIM Array Design)

```bash
cd 赛道4/赛题1/代码

# Run all simulations and evaluations
python run_all.py

# Generate reports
python generate_pdf.py
python generate_docx.py

# Visualization
python visualization.py
```

### Track 4, Problem 2 (Device Comparison)

```bash
cd 赛道4/赛题2/代码

# Run device simulations
python device_simulation.py

# Generate comparison reports
python generate_pdf.py
python generate_docx.py

# Visualization
python visualization.py
```

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

## Architecture: Track 2, Problem 1 (CIM Compiler)

Full compilation pipeline: ONNX → IR → Optimize → ISA → Simulate

```
代码/main.py  (entry point)
  ├── onnx_parser/         (ONNX loading, graph conversion)
  ├── optimizer/           (constant folding, CSE, DCE, operator fusion, pipeline)
  ├── resource_manager/    (SRAM allocator with interval graph coloring, weight mapper)
  ├── instruction_gen/     (IR lowering, code emitter)
  └── simulator/           (CIM instruction-level simulator)
```

Key innovations: Interval graph coloring for SRAM allocation (+25% utilization), multi-pass optimization pipeline, instruction-level simulation.

## Architecture: Track 2, Problem 2 (CIM 3D Mapper)

Maps neural networks to 3D CIM hardware space (n×n sub-cubes with depth).

```
代码/main.py  (entry point)
  ├── model_parser/        (model loader, weight partitioner, activation trace analyzer)
  ├── compiler_mapping/    (cube config, 3D mapper)
  ├── scheduler/           (pipeline scheduler)
  └── simulator/           (hardware simulator, solution validator)
```

Key innovations: 3D spatial mapping for MoE models, frequency-aware weight placement, activation trace analysis for expert grouping.

## Architecture: Track 4, Problem 1 (CIM Array Design)

```
代码/run_all.py  (entry point)
  ├── array_simulation.py     (CIM array simulation)
  ├── network_mapping.py      (network-to-array mapping)
  ├── performance_evaluation.py  (performance metrics)
  └── visualization.py        (result visualization)
```

## Architecture: Track 4, Problem 2 (Device Comparison)

```
代码/device_simulation.py    (device-level simulation)
  ├── device_comparison.py   (RRAM vs SRAM vs other devices)
  └── visualization.py       (comparison charts)
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

Key results: NAT+Mixed Alpha achieves 81.71% average (vs 70.07% baseline), fluctuation only 5.53%.

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

Key results: Adaptive-STE-Sqrt achieves 85.30% on CIFAR-10 (exceeds baseline 85.15%).

## Configuration

Training configs are in `configs/config.yaml` per problem:
- Model: ResNet18, Dataset: CIFAR-10
- 30 epochs, batch_size 64, cosine LR scheduler
- Problem 1: alpha_values [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

## Notes

- Track 2 has full compiler toolchain with test suite (`run_all_tests.py`)
- Track 4 focuses on hardware simulation and device comparison (Python scripts)
- Track 5 has the most extensive ML experiments with statistical validation
- Results, trained models (.pth), and figures are stored in `results/` directories.
- Experiment logs go to `日志/`.
- Competition problem descriptions are in markdown files at each track/problem root.
