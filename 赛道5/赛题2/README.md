# 赛道五 - 赛题二：基于直通估计器（STE）的噪声感知训练框架设计

## 项目概述

本项目针对存算一体芯片的噪声特性，设计并实现基于直通估计器（STE）的噪声感知训练框架。通过前向注入噪声、反向使用STE绕过不可微操作，在 CIFAR-10 图像分类任务上验证框架的有效性，并提供完整的统计分析与消融实验。

## 项目结构

```
赛题2/
├── 代码/                     # 实验代码
│   ├── run_ste.py            # 主入口（任务1/2/3）
│   ├── train_baseline.py     # 基准训练脚本
│   ├── run_statistics.py     # 统计分析脚本
│   ├── run_innovation.py     # 创新方法验证
│   ├── generate_charts.py    # 图表生成
│   └── generate_ppt.py       # PPT生成
├── ste/                      # STE核心框架
│   ├── core.py               # STE核心实现
│   ├── noisy_linear.py       # 噪声Linear层
│   ├── noisy_conv2d.py       # 噪声Conv2d层
│   └── innovation.py         # 创新方法实现
├── models/                   # 神经网络模型
│   └── resnet.py             # ResNet模型
├── training/                 # 训练框架
│   └── train.py              # 噪声感知训练
├── evaluation/               # 评估脚本
│   ├── sensitivity.py        # 敏感性分析
│   └── robustness.py         # 鲁棒性评估
├── 噪声模型/                  # 主办方噪声模型
│   └── sample_noise.py
├── configs/                  # 配置文件
│   └── config.yaml
├── 文档/                     # 项目文档
│   ├── 实验报告.md           # 完整实验报告
│   ├── 算法设计文档.md       # STE算法设计
│   ├── 专利技术交底书.md     # 3篇发明专利（自适应STE/层次化噪声注入/偏差校正正则化）
│   ├── 噪声模型文档.md       # 噪声模型说明
│   ├── 创新点总结.md         # 创新方法总结
│   ├── 创新理论证明.md       # 理论证明
│   ├── 用户手册.md           # 使用指南
│   ├── 完成情况自查.md       # 交付物自查
│   ├── ppt提示词.md          # PPT制作指南
│   └── 算法演示视频.md       # 视频脚本
├── 结果/                     # 实验结果
│   ├── figures/              # 可视化图表
│   │   ├── 图1_框架架构图.png
│   │   ├── 图2_训练曲线.png
│   │   ├── 图9_鲁棒性对比.png
│   │   ├── 图12_统计分析.png
│   │   ├── 图10_消融实验.png
│   │   └── 图4_创新算法对比.png
│   ├── task1_ste_design/     # 任务一结果
│   ├── task2_ste_nat/        # 任务二结果
│   ├── task2_validation/     # 任务二验证
│   ├── task3_evaluation/     # 任务三结果
│   └── innovation_gpu1/      # 创新方法结果
└── README.md                 # 本文件
```

## 技术方案

| 设计环节 | 方案 |
|---------|------|
| STE核心 | 前向噪声注入 + 反向直通估计器 |
| 噪声类型 | 加性噪声、乘性噪声、量化效应 |
| 梯度优化 | 梯度裁剪、自适应缩放、偏差校正 |
| 适配架构 | Conv2d、Linear 操作 |
| 训练策略 | STE-NAT (Noise-Aware Training) |
| 统计分析 | t检验、方差分析、置信区间 |

## 核心指标

| 指标 | 数值 |
|------|------|
| 基准精度 (无噪声) | 85.66% |
| STE-NAT (ns=0.5) | 84.78% |
| STE-NAT (ns=1.0) | 85.11% |
| STE-NAT (ns=1.5) | 85.00% |

## STE核心思想

```
前向传播：使用带噪声的矩阵乘法模拟存算芯片特性
    ↓
反向传播：使用STE绕过不可微的噪声操作
    ↓
梯度更新：维持训练的可行性
```

## 运行方式

```bash
cd 代码
pip install torch torchvision numpy matplotlib pyyaml tqdm

# GPU配置（统一使用GPU 1）
export CUDA_VISIBLE_DEVICES=1

# 运行完整实验
python run_ste.py --task all --device cuda:1 --save_dir results

# 任务一：STE框架设计与验证
python run_ste.py --task task1 --device cuda:1

# 任务二：领域任务验证
python run_ste.py --task task2 --device cuda:1

# 任务三：综合性能评估
python run_ste.py --task task3 --device cuda:1

# 统计分析
python run_statistics.py

# 创新方法验证
python run_innovation.py

# 生成图表
python generate_charts.py
```

## GPU配置

**重要**：本项目统一使用 **GPU 1**。

```bash
export CUDA_VISIBLE_DEVICES=1
```

## 数据集位置

CIFAR-10数据集位于：`/mnt/storage2/zyc/CIM比赛/公共数据集/cifar-10-batches-py`

## 图表说明

- `图1_框架架构图.png`: STE框架架构图
- `图2_训练曲线.png`: 训练曲线
- `图9_鲁棒性对比.png`: 鲁棒性对比
- `图12_统计分析.png`: 统计分析结果
- `图10_消融实验.png`: 消融实验
- `图4_创新算法对比.png`: 创新方法对比

## 评分要点

| 评分项 | 分值 |
|--------|:----:|
| STE框架设计（核心算法、多架构适配、梯度优化） | 40 |
| 领域任务验证（图像分类必选、目标检测/语义分割加分） | 25 |
| 综合性能评估（统计分析、消融实验、机理分析） | 20 |
| 创新性与技术先进性 | 5 |
| 设计报告与答辩 | 10 |
| **总分** | **100** |
