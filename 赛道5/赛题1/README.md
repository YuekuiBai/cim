# 赛道五 - 赛题一：非线性误差对推理精度的影响研究

## 项目概述

本项目针对存算一体芯片中模拟域乘累加运算固有的输入相关非线性失真问题，系统研究非线性误差对神经网络推理精度的影响，并提出多种鲁棒性增强方法。基于 CIFAR-10 图像分类任务完成敏感性分析、非线性感知训练与鲁棒性评估。

## 项目结构

```
赛题1/
├── 代码/                     # 实验代码
│   ├── train_baseline.py     # 基准模型训练
│   ├── run_sensitivity.py    # 主入口（任务1/2/3）
│   ├── run_extended_*.py     # 拓展研究脚本
│   └── generate_figures.py   # 图表生成
├── models/                   # 神经网络模型
│   └── resnet.py             # ResNet模型 + 非线性注入包装器
├── noise/                    # 非线性误差模块
│   └── nonlinearity.py       # 非线性注入实现
├── training/                 # 训练框架
│   └── train.py              # 非线性感知训练
├── evaluation/               # 评估脚本
│   ├── sensitivity.py        # 敏感性分析（任务一）
│   ├── robustness.py         # 原始鲁棒性增强方法
│   ├── robustness_v2.py      # 改进版鲁棒性方法
│   └── robustness_advanced.py  # 高级方法（OVF/SAM/多尺度噪声）
├── configs/                  # 配置文件
│   └── config.yaml
├── 文档/                     # 项目文档
│   ├── 技术报告.md           # 完整技术报告
│   ├── 拓展研究.md           # 拓展研究成果
│   ├── ppt提示词.md          # PPT制作指南
│   └── 演示视频内容.md       # 视频脚本
├── 结果/                     # 实验结果
│   ├── figures/              # 可视化图表
│   │   ├── experiment_overview.png
│   │   ├── quantization_vs_nonlinear.png
│   │   └── summary_comparison.png
│   ├── task1_sensitivity/    # 任务一结果
│   ├── task2_training/       # 任务二结果
│   ├── task3_robustness/     # 任务三结果
│   └── extended_research/    # 拓展研究模型
├── 日志/                     # 运行日志
└── README.md                 # 本文件
```

## 技术方案

| 设计环节 | 方案 |
|---------|------|
| 非线性模型 | x' = α·x³ + (1-α)·x |
| 基准模型 | ResNet-18 on CIFAR-10 |
| 敏感性分析 | α ∈ [0, 0.5] 扫描，精度衰减评估 |
| 非线性感知训练 | NAT (Noise-Aware Training)，α ∈ [0.1, 0.3] |
| 鲁棒性增强 | NAT+混合Alpha，多尺度噪声注入 |
| 拓展研究 | 网络结构对比、高斯噪声等效、量化联合分析 |

## 核心指标

| 指标 | 数值 |
|------|------|
| 基准精度 (α=0) | 81.95% |
| 最佳鲁棒方法 | NAT+混合Alpha |
| 平均精度 (α∈[0,0.5]) | 81.71% (±5.53%) |
| 最鲁棒网络结构 | ResNet-34 (衰减3.79%) |
| 噪声等效关系 | σ=0.1 ≈ α=0.2 |

## 非线性模型

```
x' = α · x³ + (1 - α) · x
```

| 参数 | 说明 |
|------|------|
| α=0 | 无非线性（理想情况） |
| 0<α<1 | 轻度非线性 |
| α≥1 | 强非线性 |

## 运行方式

```bash
cd 代码
pip install torch torchvision numpy matplotlib pyyaml tqdm

# 训练基准模型
python train_baseline.py

# 任务一：敏感性分析
python run_sensitivity.py --task task1 --device cuda

# 任务二：非线性感知训练
python run_sensitivity.py --task task2 --alpha 0.2 --device cuda

# 任务三：鲁棒性评估
python run_sensitivity.py --task task3 --device cuda

# 拓展研究
python run_extended_network_structure.py
python run_extended_gaussian_vs_nonlinear.py
python run_extended_quantization_v2.py

# 生成可视化图表
python generate_figures.py
```

## GPU配置

**重要**：本项目统一使用 **GPU 1**，禁止使用其他显卡。

```bash
export CUDA_VISIBLE_DEVICES=1
```

## 数据集位置

CIFAR-10数据集位于：`/mnt/storage2/zyc/CIM比赛/公共数据集/cifar-10-batches-py`

## 图表说明

- `experiment_overview.png`: 实验总览（敏感性分析、方法对比、网络结构、噪声对比）
- `quantization_vs_nonlinear.png`: 量化位数与非线性联合影响
- `summary_comparison.png`: 方法对比、网络鲁棒性、噪声等效

## 评分要点

| 评分项 | 分值 |
|--------|:----:|
| 敏感性分析（非线性参数扫描、精度衰减评估） | 30 |
| 非线性感知训练（NAT框架设计与实现） | 30 |
| 鲁棒性增强方法（多方法对比与优化） | 20 |
| 拓展研究（网络结构/噪声等效/量化分析） | 10 |
| 设计报告与答辩 | 10 |
| **总分** | **100** |
