"""
ONNX Parser - Converts ONNX models to CIM IR
Handles MatMul/Gemm -> Linear and Add/Sub/Mul/Div -> Elementwise
"""

import onnx
from onnx import numpy_helper
import numpy as np
from typing import List, Dict, Optional

from ir.ir_nodes import (
    IRGraph, IRNode, LinearNode, ElementwiseNode,
    TensorInfo, DataType, ElementwiseOp, DTYPE_BYTES
)

ONNX_TO_DTYPE = {
    1: DataType.INT32,   # FLOAT -> treat as int32
    2: DataType.INT8,    # UINT8
    3: DataType.INT8,    # INT8
    5: DataType.INT16,   # INT16
    6: DataType.INT32,   # INT32
    7: DataType.INT32,   # INT64
}


class ONNXParser:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.graph = None
        self.ir = None

    def parse(self) -> IRGraph:
        self.model = onnx.load(self.model_path)
        onnx.checker.check_model(self.model)
        self.graph = self.model.graph

        self.ir = IRGraph(name=self.graph.name or "imported_model")
        self._load_weights()
        self._add_inputs()
        self._convert_nodes()
        self._add_outputs()
        return self.ir

    def _load_weights(self):
        for init in self.graph.initializer:
            arr = numpy_helper.to_array(init)
            self.ir.weights[init.name] = arr
            dtype = ONNX_TO_DTYPE.get(init.data_type, DataType.INT8)
            self.ir.add_tensor(init.name, list(arr.shape), dtype)

    def _get_tensor_info(self, tp) -> TensorInfo:
        name = tp.name
        dtype = ONNX_TO_DTYPE.get(tp.type.tensor_type.elem_type, DataType.INT8)
        shape = []
        if tp.type.tensor_type.HasField('shape'):
            for dim in tp.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
        t = self.ir.add_tensor(name, shape, dtype)
        return t

    def _add_inputs(self):
        for inp in self.graph.input:
            if inp.name in self.ir.weights:
                continue
            self._get_tensor_info(inp)
            self.ir.input_names.append(inp.name)

    def _convert_nodes(self):
        node_counter = 0
        for onnx_node in self.graph.node:
            nid = f"node_{node_counter}"
            node_counter += 1
            op = onnx_node.op_type

            if op in ("MatMul", "Gemm"):
                self._convert_linear(nid, onnx_node)
            elif op == "Add":
                self._convert_elementwise(nid, onnx_node, ElementwiseOp.ADD)
            elif op == "Sub":
                self._convert_elementwise(nid, onnx_node, ElementwiseOp.SUB)
            elif op == "Mul":
                self._convert_elementwise(nid, onnx_node, ElementwiseOp.MUL)
            elif op == "Div":
                self._convert_elementwise(nid, onnx_node, ElementwiseOp.DIV)
            elif op == "Relu":
                continue
            elif op == "Transpose":
                continue
            else:
                print(f"[WARN] Unsupported op: {op}, skipping")

    def _convert_linear(self, nid, onnx_node):
        inp = onnx_node.input[0]
        weight_name = onnx_node.input[1]
        out = onnx_node.output[0]

        bias_name = None
        if len(onnx_node.input) > 2:
            bias_name = onnx_node.input[2]

        weight = self.ir.weights.get(weight_name)
        cin, cout = 0, 0
        if weight is not None:
            if weight.ndim == 2:
                cin, cout = weight.shape
            else:
                cin = weight.shape[0] if weight.shape else 0
                cout = 1

        node = LinearNode(
            node_id=nid,
            node_type=None,
            inputs=[inp],
            outputs=[out],
            weight_name=weight_name,
            bias_name=bias_name,
            cin=cin,
            cout=cout,
        )
        self.ir.nodes.append(node)

        out_shape = [-1, cout]
        self.ir.add_tensor(out, out_shape, DataType.INT32)

    def _convert_elementwise(self, nid, onnx_node, op):
        inputs = list(onnx_node.input)
        out = onnx_node.output[0]

        is_vi = False
        imm = None
        for inp in inputs:
            if inp in self.ir.weights:
                arr = self.ir.weights[inp]
                if arr.size == 1:
                    is_vi = True
                    imm = int(arr.flatten()[0])
                    break

        node = ElementwiseNode(
            node_id=nid,
            node_type=None,
            inputs=inputs,
            outputs=[out],
            op=op,
            mode="vi" if is_vi else "vv",
            immediate_value=imm,
        )
        self.ir.nodes.append(node)

        inp0 = inputs[0]
        if inp0 in self.ir.tensors:
            t = self.ir.tensors[inp0]
            self.ir.add_tensor(out, t.shape, t.dtype)
        else:
            self.ir.add_tensor(out, [-1], DataType.INT32)

    def _add_outputs(self):
        for outp in self.graph.output:
            name = outp.name
            if name not in self.ir.tensors:
                self._get_tensor_info(outp)
            self.ir.output_names.append(name)
