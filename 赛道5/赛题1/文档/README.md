# 赛道五-赛题一：非线性误差对推理精度的影响研究

## 项目简介

本项目针对存算一体芯片中**模拟域乘累加运算**固有的输入相关非线性失真问题，系统研究非线性误差对神经网络推理精度的影响，并提出多种鲁棒性增强方法。

### 非线性模型

```
x' = α · x³ + (1 - α) · x
```

| 参数 | 说明 |
| --- | --- |
| α=0 | 无非线性（理想情况） |
| 0<α<1 | 轻度非线性 |
| α≥1 | 强非线性 |

***

## 项目结构

```
赛题1/
├── models/                     # 神经网络模型
│   └── resnet.py               # ResNet模型定义 + 非线性注入包装器
├── noise/                      # 非线性误差模块
│   └── nonlinearity.py         # 非线性注入实现
├── training/                   # 训练脚本
│   └── train.py                # 非线性感知训练框架
├── evaluation/                 # 评估脚本
│   ├── sensitivity.py          # 敏感性分析（任务一）
│   ├── robustness.py           # 原始鲁棒性增强方法
│   ├── robustness_v2.py        # 改进版鲁棒性方法
│   └── robustness_advanced.py   # 高级鲁棒性方法（OVF/SAM/多尺度噪声）
├── configs/
│   └── config.yaml             # 配置文件
├── results/                    # 实验结果
│   ├── figures/                # 可视化图表
│   ├── task1_sensitivity/      # 任务一结果
│   ├── task2_training/         # 任务二结果
│   ├── task3_robustness/       # 任务三结果
│   ├── extended_research/       # 拓展研究模型
│   └── 图表.md                  # 实验数据汇总
├── data/                       # CIFAR-10数据集
├── 代码/                        # Python脚本
│   ├── train_baseline.py       # 基准模型训练
│   ├── run_sensitivity.py       # 主入口脚本
│   ├── run_extended_*.py       # 拓展研究脚本
│   └── generate_figures.py     # 图表生成
├── 文档/                        # 项目文档
│   ├── README.md               # 项目总览
│   ├── 技术报告.md              # 完整技术报告
│   ├── 拓展研究.md              # 拓展研究成果
│   ├── ppt提示词.md             # PPT制作指南
│   ├── 演示视频内容.md          # 视频脚本
│   └── 赛题1.md                # 赛题说明
└── 日志/                        # 实验日志
    └── *.log                   # 运行日志
```

***

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision numpy matplotlib pyyaml tqdm
```

### 2. 训练基准模型

```bash
cd 代码 && python train_baseline.py
```

### 3. 运行任务一（敏感性分析）

```bash
cd 代码 && python run_sensitivity.py --task task1 --device cuda
```

### 4. 运行任务二（非线性感知训练）

```bash
cd 代码 && python run_sensitivity.py --task task2 --alpha 0.2 --device cuda
```

### 5. 运行任务三（鲁棒性评估）

```bash
cd 代码 && python run_sensitivity.py --task task3 --device cuda
```

### 6. 拓展研究

```bash
cd 代码
# 网络结构影响
python run_extended_network_structure.py

# 高斯噪声 vs 非线性
python run_extended_gaussian_vs_nonlinear.py

# 量化 + 非线性分析
python run_extended_quantization_v2.py
```

### 7. 生成可视化图表

```bash
cd 代码 && python generate_figures.py
```

***

## 实验结果摘要

### 任务一：敏感性分析

| α值 | 精度 | 衰减 |
| --- | --- | --- |
| 0.0 | 81.95% | 基准 |
| 0.1 | 80.99% | -1.0% |
| 0.2 | 77.63% | -4.3% |
| 0.3 | 70.95% | -11.0% |
| 0.4 | 60.40% | -21.5% |
| 0.5 | 48.49% | -33.5% |

### 任务二：非线性感知训练

| 训练α | Clean精度 | 噪声下精度 |
| --- | --- | --- |
| 0.1 | 86.33% | 86.98% |
| 0.2 | 84.34% | 87.30% |
| 0.3 | 81.72% | 87.31% |

### 任务三：最佳方法 - NAT+混合Alpha

| α范围 | 平均精度 | 波动 |
| --- | --- | --- |
| 0.0-0.5 | **81.71%** | ±5.53% |

### 拓展研究结论

| 研究方向 | 核心发现 |
| --- | --- |
| 网络结构 | ResNet34最鲁棒（衰减3.79%） |
| 噪声等效 | σ=0.1≈α=0.2 |
| 量化影响 | INT8几乎无影响 |

***

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [技术报告.md](技术报告.md) | 完整学术报告（引言、原理、设计、实验、总结） |
| [拓展研究.md](拓展研究.md) | 三个拓展研究方向详细分析 |
| [图表.md](../results/图表.md) | 精炼实验数据表格汇总 |
| [ppt提示词.md](ppt提示词.md) | PPT每页内容指导 |
| [演示视频内容.md](演示视频内容.md) | 视频录制脚本和时间线 |

***

## 可视化图表

| 图表 | 文件 | 说明 |
| --- | --- | --- |
| 实验总览 | [experiment_overview.png](../results/figures/experiment_overview.png) | 敏感性分析、方法对比、网络结构、噪声对比 |
| 量化分析 | [quantization_vs_nonlinear.png](../results/figures/quantization_vs_nonlinear.png) | 量化位数与非线性联合影响 |
| 对比汇总 | [summary_comparison.png](../results/figures/summary_comparison.png) | 方法对比、网络鲁棒性、噪声等效 |

***

## GPU配置

**重要**：本项目统一使用 **GPU 1**，禁止使用其他显卡。

```bash
# 查看GPU状态
nvidia-smi

# 设置默认GPU
export CUDA_VISIBLE_DEVICES=1
```

***

## 交付物清单

| 交付物 | 状态 | 文件 |
| --- | --- | --- |
| 设计报告 | 草稿 | 技术报告.md |
| 介绍PPT | 待制作 | ppt提示词.md |
| 演示视频 | 待录制 | 演示视频内容.md |
| 代码 | ✅ 完成 | *.py |