"""
Model Parser for Problem 2
Loads model.json / ONNX and extracts operators and weights
"""

import json
import os
from typing import List, Dict, Optional
from compiler_mapping.weight_cube import Operator


class ModelParser:
    """Parses model description from JSON"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.operators: List[Operator] = []
        self.is_moe = False
        self.num_experts = 0
        self.top_k = 0
        self.shared_experts: List[int] = []
        self.total_params = 0

    def parse(self) -> List[Operator]:
        ext = os.path.splitext(self.model_path)[1]
        if ext == ".json":
            return self._parse_json()
        elif ext == ".onnx":
            return self._parse_onnx()
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _parse_json(self) -> List[Operator]:
        with open(self.model_path, 'r') as f:
            data = json.load(f)

        self.is_moe = data.get("is_moe", False)
        self.num_experts = data.get("num_experts", 0)
        self.top_k = data.get("top_k", 0)
        self.shared_experts = data.get("shared_experts", [])
        self.total_params = data.get("total_params", 0)

        self.operators = []
        for op_data in data.get("operators", []):
            op = Operator(
                name=op_data["name"],
                op_type=op_data.get("op_type", "MatMul"),
                weight_shape=op_data.get("weight_shape", [0, 0]),
                dependencies=op_data.get("dependencies", []),
                is_sparse=op_data.get("is_sparse", False),
                expert_id=op_data.get("expert_id"),
                is_shared_expert=op_data.get("is_shared_expert", False),
            )
            self.operators.append(op)
        return self.operators

    def _parse_onnx(self) -> List[Operator]:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(self.model_path)
        onnx.checker.check_model(model)

        weights = {}
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            weights[init.name] = arr

        self.operators = []
        node_counter = 0
        for node in model.graph.node:
            if node.op_type in ("MatMul", "Gemm"):
                weight_name = node.input[1]
                if weight_name in weights:
                    shape = list(weights[weight_name].shape)
                else:
                    shape = [0, 0]
                op = Operator(
                    name=f"matmul_{node_counter}",
                    op_type="MatMul",
                    weight_shape=shape,
                    dependencies=[node.input[0]],
                )
                self.operators.append(op)
                node_counter += 1
        return self.operators
