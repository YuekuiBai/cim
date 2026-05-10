# 赛道五：存算算法 - 赛题二

## 基于直通估计器（STE）的噪声感知训练框架设计

### 项目简介

本项目针对存算一体芯片的噪声特性，设计并实现基于直通估计器（STE）的噪声感知训练框架，在CIFAR-10图像分类任务上验证框架的有效性。

### STE核心思想

```
前向传播：使用带噪声的矩阵乘法模拟存算芯片特性
    ↓
反向传播：使用STE绕过不可微的噪声操作
    ↓
梯度更新：维持训练的可行性
```

***

## 任务说明

### 任务一：通用STE框架设计与实现

| 子任务       | 说明                       |
| --------- | ------------------------ |
| 核心STE算法设计 | 处理加性噪声、乘性噪声、量化效应的梯度估计策略  |
| 多架构适配机制   | 支持Conv2d、Linear等操作的STE适配 |
| 梯度估计优化策略  | 梯度裁剪、自适应缩放、偏差校正、方差稳定化    |

### 任务二：领域任务验证实现

| 任务   | 要求  | 数据集        |
| ---- | --- | ---------- |
| 图像分类 | 必选项 | CIFAR-10   |
| 目标检测 | 加分项 | COCO       |
| 语义分割 | 加分项 | Cityscapes |

### 任务三：综合性能评估与分析

| 内容   | 说明              |
| ---- | --------------- |
| 统计分析 | t检验、方差分析、置信区间分析 |
| 消融实验 | 分析各组件的独立贡献      |
| 机理分析 | 深入分析STE机制的作用机理  |

***

## 项目结构

```
赛题2/
├── models/                     # 神经网络模型
│   └── resnet.py              # ResNet模型
├── ste/                        # STE核心框架
│   ├── core.py                 # STE核心实现
│   ├── noisy_linear.py         # (保留接口)
│   └── noisy_conv2d.py        # (保留接口)
├── configs/
│   └── config.yaml            # 配置文件
├── 噪声模型/
│   └── sample_noise.py        # 主办方提供的噪声模型
├── 代码/
│   ├── run_ste.py             # 主入口脚本
│   └── train_baseline.py      # 基准训练脚本
├── results/                    # 实验结果
│   ├── task1_ste_design/      # 任务一结果
│   ├── task2_validation/      # 任务二结果
│   └── task3_evaluation/      # 任务三结果
└── 文档/
    ├── README.md              # 项目总览
    └── 实验报告.md              # 完整实验报告
```

***

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision numpy matplotlib pyyaml tqdm
```

### 2. GPU配置

**重要**：本项目统一使用 **GPU 1**

```bash
export CUDA_VISIBLE_DEVICES=1
```

### 3. 运行完整实验

```bash
cd 代码 && python run_ste.py --task all --device cuda:1 --save_dir results
```

### 4. 分步运行

```bash
# 任务一：STE框架设计与验证
python run_ste.py --task task1 --device cuda:1

# 任务二：领域任务验证
python run_ste.py --task task2 --device cuda:1

# 任务三：综合性能评估
python run_ste.py --task task3 --device cuda:1
```

***

## 实验结果

详见 [results/图表.md](../results/图表.md)

### 任务一：STE框架验证

- ✅ STE框架注入成功
- ✅ 前向传播验证通过
- ✅ 反向传播验证通过

### 任务二：训练结果

| 训练噪声强度   | 测试精度       |
| -------- | ---------- |
| 0.0 (基准) | **85.66%** |
| 0.5      | 84.78%     |
| 1.0      | 85.11%     |
| 1.5      | 85.00%     |

### 任务三：评估结论

- 基准模型在干净环境下表现最好
- STE-NAT可以接近基准性能
- 不同噪声强度下训练模型表现一致

***

## 文档索引

| 文档                        | 内容       |
| ------------------------- | -------- |
| [实验报告.md](实验报告.md)        | 完整实验报告   |
| [图表.md](../results/图表.md) | 精炼实验数据汇总 |

***

## GPU配置

默认使用GPU 1：

```bash
export CUDA_VISIBLE_DEVICES=1
```

***

## 数据集说明

CIFAR-10数据集位于：`/mnt/storage2/zyc/CIM比赛/公共数据集/cifar-10-batches-py`
