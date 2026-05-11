# 赛道二 - 赛题一：编译架构设计

## 项目概述

面向存算一体（CIM）硬件的编译器中间表示（IR）设计与编译框架实现。将 ONNX 神经网络模型编译为目标存算硬件的 ISA 指令序列，支持 linear 和 elementwise 算子的高效映射。

## 项目结构

```
赛题1/
├── 代码/                     # 实验代码
│   ├── main.py               # 主入口（编译流水线）
│   ├── generate_test_models.py  # 测试ONNX模型生成器
│   ├── run_all_tests.py      # 批量测试脚本
│   ├── requirements.txt      # Python依赖
│   ├── ir/                   # IR中间表示设计
│   │   ├── ir_nodes.py       # IR节点定义（Linear/Elementwise/Tensor）
│   │   └── ir_builder.py     # IR构建器
│   ├── onnx_parser/          # ONNX模型解析
│   │   └── onnx_loader.py    # ONNX加载与计算图转换
│   ├── resource_manager/     # 硬件资源管理
│   │   ├── sram_allocator.py # 512KB SRAM分配器
│   │   └── weight_mapper.py  # 权重映射到1024×4096 CIM阵列
│   ├── instruction_gen/      # 指令生成
│   │   ├── lowering.py       # IR降级到ISA（cim.bit/elt/mem.copy）
│   │   └── code_emitter.py   # 汇编代码输出
│   ├── optimizer/            # 图优化
│   │   ├── constant_folding.py  # 常量折叠
│   │   └── operator_fusion.py   # 算子融合
│   ├── tests/                # 测试用例
│   ├── test_models/          # 测试ONNX模型
│   └── utils/                # 工具函数
├── 文档/                     # 设计报告
├── 图表/                     # 生成的图表
│   ├── 图1.png               # 知存模拟存算计算方案原理图
│   ├── 图2.png               # No-bias linear ONNX
│   └── 图3.png               # 资源分配示意图
├── 结果/                     # 编译输出结果
│   ├── output_linear/        # 纯Linear模型编译结果
│   ├── output_linear_bias/   # Linear+Bias模型编译结果
│   ├── output_linear_add/    # Linear+Add模型编译结果
│   └── ...
└── README.md                 # 本文件
```

## 技术方案

| 设计环节 | 方案 |
|---------|------|
| IR设计 | 自定义IRGraph，支持Linear/Elementwise节点，拓扑排序 |
| 模型解析 | ONNX MatMul/Gemm→Linear，Add/Sub/Mul/Div→Elementwise |
| SRAM管理 | 512KB拓扑序分配，input/output/acc/tmp四区规划 |
| 权重映射 | 映射到1024×4096 CIM阵列，支持行列切分 |
| 指令生成 | Bit-serial cim.bit.i8 + elt.mul/add/sub + mem.copy |
| 数据类型 | int8输入 → int32输出（8bit逐位计算+移位累加） |

## 核心指标

| 指标 | 数值 |
|------|------|
| 支持算子 | Linear, Add, Sub, Mul, Div |
| CIM阵列 | 1024bit(row) × 4096bit(col) |
| SRAM容量 | 512KB |
| 指令集 | cim.bit.type, elt.op.type.mode, mem.copy |
| 测试模型 | 6个（纯Linear/带Bias/多算子组合） |

## 编译流程

```
ONNX模型 → 前端解析 → IR构建 → 图优化 → SRAM分配 → 权重映射 → 指令生成 → ISA输出
```

## 运行方式

```bash
cd 代码
pip install -r requirements.txt

# 生成测试模型
python generate_test_models.py

# 编译单个模型
python main.py --model test_models/model_linear.onnx --output output_test

# 运行全部测试
python run_all_tests.py
```

## 输出说明

每个模型编译后生成以下中间结果：

| 文件 | 说明 |
|------|------|
| `ir.json` | IR中间表示（节点、张量、连接关系） |
| `sram_layout.json` | SRAM地址分配方案 |
| `weight_mapping.json` | 权重到CIM阵列的映射 |
| `output.asm` | 生成的汇编指令序列 |
| `output.json` | 完整编译结果（JSON格式） |

## 评分要点

| 评分项 | 分值 |
|--------|:----:|
| IR设计（表达能力、简洁性、扩展性） | 20 |
| 模型解析（完成ONNX→IR转换） | 10 |
| 指令生成（生成正确的指令序列） | 15 |
| 资源管理策略 | 15 |
| Transform/Lowering | 20 |
| 技术路线先进性 | 10 |
| 设计报告 | 5 |
| 现场答辩 | 5 |
| **总分** | **100** |
