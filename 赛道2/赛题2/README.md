# 赛道二 - 赛题二：权重排布优化

## 项目概述

面向 3D 存算一体硬件的智能编译器/调度器。将深度学习模型（特别是 MoE 大模型）的权重映射到分层 3D 异构资源空间，优化端到端推理延迟，支持 DeepSeek-671B 等超大模型。

## 项目结构

```
赛题2/
├── 代码/                     # 实验代码
│   ├── main.py               # 主入口（编译映射流水线）
│   ├── generate_test_models.py  # 测试模型生成器
│   ├── run_all_tests.py      # 批量测试脚本
│   ├── requirements.txt      # Python依赖
│   ├── model_parser/         # 模型解析
│   │   ├── model_loader.py   # JSON/ONNX模型加载
│   │   ├── trace_analyzer.py # MoE激活轨迹分析
│   │   ├── weight_extractor.py  # 权重提取
│   │   └── weight_partitioner.py  # 权重切分
│   ├── compiler_mapping/     # 编译映射
│   │   ├── weight_cube.py    # Weight-Cube定义
│   │   ├── sub_cube.py       # Sub-Cube定义
│   │   └── mapper.py         # 3D映射算法（Greedy First-Fit）
│   ├── scheduler/            # 调度器
│   │   ├── dependency_graph.py  # 依赖图构建
│   │   └── pipeline_scheduler.py  # 流水线调度
│   ├── simulator/            # 模拟器
│   │   ├── hardware_sim.py   # 硬件模拟
│   │   ├── latency_calculator.py  # 延迟计算
│   │   └── validator.py      # 约束验证
│   ├── tests/                # 测试用例
│   ├── test_models/          # 测试模型
│   │   ├── simple_model.json # 简单模型
│   │   ├── moe_model.json    # MoE模型
│   │   └── activation_trace.json  # 激活轨迹
│   └── utils/                # 工具函数
├── 文档/                     # 设计报告
├── 图表/                     # 生成的图表
│   ├── 图4.png               # 3D资源空间示意图
│   ├── 图5.png               # 映射算法流程图
│   └── 图6.png               # 调度甘特图
├── 结果/                     # 编译输出结果
│   ├── output_simple/        # 简单模型编译结果
│   └── output_moe/           # MoE模型编译结果
└── README.md                 # 本文件
```

## 技术方案

| 设计环节 | 方案 |
|---------|------|
| 模型解析 | JSON/ONNX加载，提取算子形状与权重参数 |
| 权重切分 | 按Sub-Cube容量切分为Section，支持行列切分 |
| MoE分析 | 激活轨迹分析，共现矩阵，专家优先级排序 |
| 3D映射 | Greedy First-Fit算法，Sub-Cube负载均衡 |
| 流水线调度 | 依赖图拓扑排序，Barrier同步 |
| 延迟计算 | 计算周期 + Sub-Cube切换开销(D周期) |

## 核心指标

| 指标 | 数值 |
|------|------|
| Sub-Cube数量 | N×N (N∈[2,4]) |
| Sub-Cube尺寸 | H×W (4096~16384) |
| Z轴深度 | 自定义（自动适配模型大小） |
| 优化目标 | 端到端推理延迟（总周期数） |
| 约束条件 | Weight Stationary, 依赖严格同步 |
| 测试模型 | 简单模型 + MoE模型 |

## 硬件模型

```
Global Space (N×N Sub-Cubes)
├── Sub-Cube 0 (H×W×D)
│   ├── Weight-Cube 0
│   ├── Weight-Cube 1
│   └── ...
├── Sub-Cube 1 (H×W×D)
│   └── ...
└── ...
```

## 运行方式

```bash
cd 代码
pip install -r requirements.txt

# 生成测试模型
python generate_test_models.py

# 编译单个模型
python main.py --model test_models/simple_model.json --output output_test

# 运行全部测试
python run_all_tests.py
```

## 输出说明

每个模型编译后生成以下中间结果：

| 文件 | 说明 |
|------|------|
| `parsed_operators.json` | 解析后的算子列表 |
| `weight_sections.json` | 权重切分结果 |
| `trace_analysis.json` | MoE激活轨迹分析（仅MoE模型） |
| `solution.json` | 完整3D映射与调度方案 |

## 评分要点

| 评分项 | 分值 |
|--------|:----:|
| 部署框架（完整实现模型解析、映射、调度、模拟） | 60 |
| 优化排序效果（延迟优化、空间利用率） | 15 |
| 技术路线先进性 | 10 |
| 设计报告 | 15 |
| **总分** | **100** |
