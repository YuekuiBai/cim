# 赛道四 - 赛题一：神经网络在模拟存内计算系统的映射

## 项目概述

本项目针对赛题一要求，完成从器件选型到性能评估的全流程方案设计。

## 项目结构

```
赛题1/
├── 代码/                     # 实验代码
│   ├── config.yaml           # 配置文件
│   ├── array_simulation.py   # 存算阵列仿真
│   ├── network_mapping.py   # 网络映射算法
│   ├── performance_evaluation.py  # 性能评估
│   ├── visualization.py      # 可视化脚本
│   └── run_all.py            # 运行所有实验
├── 文档/                     # 设计报告
│   ├── 01_器件原理.md        # 第1章
│   ├── 02_神经网络模型.md    # 第2章
│   ├── 03_存算阵列规格.md    # 第3章
│   ├── 04_网络映射方案.md    # 第4章
│   ├── 05_性能评估.md        # 第5章
│   └── 06_总结讨论.md        # 第6章
├── 图表/                     # 生成的图表
│   ├── nonlinearity_analysis.png
│   ├── ppa_metrics.png
│   ├── layer_mapping.png
│   ├── latency_breakdown.png
│   └── array_simulation.png
├── 结果/                     # 实验结果
└── README.md                 # 本文件
```

## 技术方案

| 设计环节 | 方案 |
|---------|------|
| 器件选型 | RRAM (阻变存储器) |
| 阵列规模 | 128×128 |
| 神经网络 | ResNet-18 |
| 外围电路 | 8-bit DAC + 10-bit ADC |

## 核心指标

| 指标 | 数值 |
|------|------|
| 峰值吞吐 | 52.4 TOPS |
| 能效 | 90.9 TOPS/W |
| 功耗 | 0.58 W |
| 面积 | 10.49 mm² |
| 精度损失 | ~6% |

## 运行方式

```bash
cd 代码
conda activate base
python run_all.py
```

## 图表说明

- `nonlinearity_analysis.png`: 非线性参数扫描结果
- `ppa_metrics.png`: PPA综合指标
- `layer_mapping.png`: 层级映射统计
- `latency_breakdown.png`: 延迟分解
- `array_simulation.png`: 阵列仿真结果
