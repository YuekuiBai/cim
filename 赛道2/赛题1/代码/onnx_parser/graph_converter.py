"""
Graph Converter - Converts ONNX model to IR
"""

import onnx
from onnx import ModelProto, NodeProto
import numpy as np
from typing import Dict, List

from ir.ir_nodes import (
    IRGraph, IRNode, LinearNode, ElementwiseNode, 
    TensorInfo, DataType, IRNodeType
)
from ir.ir_builder import IRBuilder
from onnx_parser.onnx_loader import ONNXLoader


# ONNX data type mapping
ONNX_DTYPE_MAP = {
    1: DataType.INT8,    # Actually FLOAT, but we assume INT8 per problem spec
    2: DataType.INT8,    # UINT8
    3: DataType.INT8,    # INT8
    4: DataType.INT16,   # UINT16
    5: DataType.INT16,   # INT16
    6: DataType.INT32,   # INT32
    7: DataType.INT32,   # INT64
}


class GraphConverter:
    """Converts ONNX computational graph to IR"""
    
    def __init__(self, model: ModelProto):
        self.model = model
        self.builder = IRBuilder(name=model.graph.name if model.graph.name else "model")
        self.weights = {}
        
        # Extract weights from initializers
        for initializer in model.graph.initializer:
            self.weights[initializer.name] = onnx.numpy_helper.to_array(initializer)
    
    def convert(self) -> IRGraph:
        """Convert ONNX model to IR graph"""
        self._add_inputs()
        self._convert_nodes()
        self._add_outputs()
        return self.builder.build()
    
    def _add_inputs(self):
        """Add model inputs to IR"""
        input_names = []
        for inp in self.model.graph.input:
            # Skip initializers (weights) - they are not inputs
            if inp.name in self.weights:
                continue
            
            name = inp.name
            shape = self._get_shape(inp)
            dtype = self._get_dtype(inp)
            
            self.builder.add_tensor(name, shape, dtype)
            input_names.append(name)
        
        self.builder.set_inputs(input_names)
    
    def _convert_nodes(self):
        """Convert ONNX nodes to IR nodes"""
        for node in self.model.graph.node:
            op_type = node.op_type

            if op_type == "MatMul" or op_type == "Gemm":
                self._convert_linear(node)
            elif op_type in ["Add", "Sub", "Mul", "Div"]:
                self._convert_elementwise(node, op_type.lower())
            elif op_type == "Relu":
                self._convert_relu(node)
            elif op_type == "Reshape":
                self._convert_reshape(node)
            elif op_type == "Transpose":
                self._convert_transpose(node)
            elif op_type == "Flatten":
                self._convert_flatten(node)
            elif op_type == "Sigmoid":
                self._convert_activation(node, "sigmoid")
            elif op_type == "Tanh":
                self._convert_activation(node, "tanh")
            elif op_type == "Softmax":
                self._convert_softmax(node)
            elif op_type == "BatchNormalization":
                self._convert_batchnorm(node)
            elif op_type == "Conv":
                self._convert_conv(node)
            elif op_type == "MaxPool" or op_type == "AveragePool":
                self._convert_pool(node)
            elif op_type == "Concat":
                self._convert_concat(node)
            elif op_type == "Split":
                self._convert_split(node)
            elif op_type == "Constant":
                self._convert_constant(node)
            else:
                print(f"Warning: Unsupported op type {op_type}, skipping")
    
    def _convert_linear(self, node: NodeProto):
        """Convert MatMul/Gemm to Linear IR node"""
        input_name = node.input[0]
        weight_name = node.input[1]
        output_name = node.output[0]
        
        # Get weight shape
        if weight_name in self.weights:
            weight_shape = list(self.weights[weight_name].shape)
        else:
            weight_shape = None
        
        # Check for bias
        bias = None
        if len(node.input) > 2:
            bias = node.input[2]
        
        self.builder.add_linear(
            inputs=[input_name],
            outputs=[output_name],
            weight=weight_name,
            bias=bias,
            weight_shape=weight_shape
        )
        
        # Add output tensor info
        if output_name not in self.builder.tensors:
            self.builder.add_tensor(output_name, [1, weight_shape[1] if weight_shape else 0], DataType.INT32)
    
    def _convert_elementwise(self, node: NodeProto, op_type: str):
        """Convert elementwise operations to IR"""
        inputs = list(node.input)
        outputs = list(node.output)

        self.builder.add_elementwise(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            mode="vv"
        )

    def _convert_relu(self, node: NodeProto):
        """Convert ReLU activation"""
        self.builder.add_activation(
            op_type="relu",
            inputs=[node.input[0]],
            outputs=[node.output[0]]
        )

    def _convert_activation(self, node: NodeProto, op_type: str):
        """Convert activation functions (sigmoid, tanh)"""
        self.builder.add_activation(
            op_type=op_type,
            inputs=[node.input[0]],
            outputs=[node.output[0]]
        )

    def _convert_reshape(self, node: NodeProto):
        """Convert Reshape operation"""
        shape_attr = None
        for attr in node.attribute:
            if attr.name == "shape":
                shape_attr = list(attr.ints)
                break
        self.builder.add_reshape(
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            shape=shape_attr or node.input[1] if len(node.input) > 1 else None
        )

    def _convert_transpose(self, node: NodeProto):
        """Convert Transpose operation"""
        perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
                break
        self.builder.add_node(
            node_type="transpose",
            inputs=list(node.input),
            outputs=[node.output[0]],
            attributes={"perm": perm}
        )

    def _convert_flatten(self, node: NodeProto):
        """Convert Flatten operation"""
        self.builder.add_node(
            node_type="flatten",
            inputs=[node.input[0]],
            outputs=[node.output[0]]
        )

    def _convert_softmax(self, node: NodeProto):
        """Convert Softmax operation"""
        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        self.builder.add_node(
            node_type="softmax",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            attributes={"axis": axis}
        )

    def _convert_batchnorm(self, node: NodeProto):
        """Convert BatchNormalization - treat as elementwise mul + add"""
        inputs = list(node.input)
        outputs = [node.output[0]]
        self.builder.add_elementwise(
            op_type="batchnorm",
            inputs=inputs,
            outputs=outputs
        )

    def _convert_conv(self, node: NodeProto):
        """Convert Convolution operation"""
        kernel_shape = [3, 3]
        strides = [1, 1]
        pads = [0, 0, 0, 0]
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
            elif attr.name == "strides":
                strides = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)
        self.builder.add_node(
            node_type="conv",
            inputs=list(node.input),
            outputs=[node.output[0]],
            attributes={
                "kernel_shape": kernel_shape,
                "strides": strides,
                "pads": pads
            }
        )

    def _convert_pool(self, node: NodeProto):
        """Convert Pooling operation (MaxPool/AveragePool)"""
        pool_type = "max" if node.op_type == "MaxPool" else "avg"
        kernel_shape = [2, 2]
        strides = [2, 2]
        pads = [0, 0, 0, 0]
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
            elif attr.name == "strides":
                strides = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)
        self.builder.add_node(
            node_type="pool",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            attributes={
                "pool_type": pool_type,
                "kernel_shape": kernel_shape,
                "strides": strides,
                "pads": pads
            }
        )

    def _convert_concat(self, node: NodeProto):
        """Convert Concat operation"""
        axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        self.builder.add_node(
            node_type="concat",
            inputs=list(node.input),
            outputs=[node.output[0]],
            attributes={"axis": axis}
        )

    def _convert_split(self, node: NodeProto):
        """Convert Split operation"""
        axis = 0
        split = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
            elif attr.name == "split":
                split = list(attr.ints)
        self.builder.add_node(
            node_type="split",
            inputs=[node.input[0]],
            outputs=list(node.output),
            attributes={"axis": axis, "split": split}
        )

    def _convert_constant(self, node: NodeProto):
        """Convert Constant operation"""
        value = None
        for attr in node.attribute:
            if attr.name == "value":
                value = attr.t
        self.builder.add_constant(
            outputs=[node.output[0]],
            value=value
        )

    def _add_outputs(self):
        """Add model outputs to IR"""
        output_names = []
        for out in self.model.graph.output:
            name = out.name
            shape = self._get_shape(out)
            dtype = self._get_dtype(out)
            
            if name not in self.builder.tensors:
                self.builder.add_tensor(name, shape, dtype)
            output_names.append(name)
        
        self.builder.set_outputs(output_names)
    
    def _get_shape(self, tensor) -> List[int]:
        """Extract shape from tensor"""
        shape = []
        if tensor.type.tensor_type.HasField('shape'):
            for dim in tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(-1)  # Dynamic dimension
        return shape
    
    def _get_dtype(self, tensor) -> DataType:
        """Extract data type from tensor"""
        onnx_dtype = tensor.type.tensor_type.elem_type
        return ONNX_DTYPE_MAP.get(onnx_dtype, DataType.INT8)
