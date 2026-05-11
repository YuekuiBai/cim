"""
IR Nodes for CIM Compiler - Problem 1
Supports linear and elementwise (add/sub/mul/div) operators
Extended with Placeholder, Constant, Input/Output nodes for complete graph representation
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import numpy as np


class IRNodeType(Enum):
    LINEAR = "linear"
    ELEMENTWISE = "elementwise"
    INPUT = "input"
    OUTPUT = "output"
    CONSTANT = "constant"
    PLACEHOLDER = "placeholder"
    RELU = "relu"
    RESHAPE = "reshape"
    CONCAT = "concat"
    SPLIT = "split"


class DataType(Enum):
    INT8 = "i8"
    INT16 = "i16"
    INT32 = "i32"
    FLOAT32 = "f32"


class ElementwiseOp(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"


DTYPE_BYTES = {
    DataType.INT8: 1,
    DataType.INT16: 2,
    DataType.INT32: 4,
    DataType.FLOAT32: 4,
}

DTYPE_BITS = {
    DataType.INT8: 8,
    DataType.INT16: 16,
    DataType.INT32: 32,
    DataType.FLOAT32: 32,
}


@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: DataType
    sram_address: Optional[int] = None

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            if d > 0:
                n *= d
        return n

    @property
    def size_bytes(self) -> int:
        return self.num_elements * DTYPE_BYTES[self.dtype]

    def flattened_size(self) -> int:
        return self.num_elements


@dataclass
class IRNode:
    node_id: str
    node_type: IRNodeType
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"{self.node_id}: {self.node_type.value}({self.inputs} -> {self.outputs})"


@dataclass
class LinearNode(IRNode):
    weight_name: str = ""
    bias_name: Optional[str] = None
    cin: int = 0
    cout: int = 0

    def __post_init__(self):
        self.node_type = IRNodeType.LINEAR


@dataclass
class ElementwiseNode(IRNode):
    op: ElementwiseOp = ElementwiseOp.ADD
    mode: str = "vv"
    immediate_value: Optional[int] = None

    def __post_init__(self):
        self.node_type = IRNodeType.ELEMENTWISE
        self.attributes["op"] = self.op.value
        self.attributes["mode"] = self.mode


@dataclass
class PlaceholderNode(IRNode):
    """Represents input placeholder node"""
    shape: List[int] = field(default_factory=list)
    dtype: DataType = DataType.INT8

    def __post_init__(self):
        self.node_type = IRNodeType.PLACEHOLDER
        self.attributes["shape"] = self.shape
        self.attributes["dtype"] = self.dtype.value


@dataclass
class ConstantNode(IRNode):
    """Represents constant value node (bias, scale, etc.)"""
    value: Optional[np.ndarray] = None
    dtype: DataType = DataType.INT32

    def __post_init__(self):
        self.node_type = IRNodeType.CONSTANT
        if self.value is not None:
            self.attributes["shape"] = list(self.value.shape)
            self.attributes["dtype"] = self.dtype.value
            self.attributes["num_elements"] = self.value.size

    @property
    def num_elements(self) -> int:
        if self.value is not None:
            return self.value.size
        return 0

    @property
    def size_bytes(self) -> int:
        if self.value is not None:
            return self.value.nbytes
        return 0


@dataclass
class OutputNode(IRNode):
    """Represents output node marking graph outputs"""
    shape: List[int] = field(default_factory=list)
    dtype: DataType = DataType.INT32

    def __post_init__(self):
        self.node_type = IRNodeType.OUTPUT
        self.attributes["shape"] = self.shape
        self.attributes["dtype"] = self.dtype.value


@dataclass
class ReLUNode(IRNode):
    """Represents ReLU activation node"""
    def __post_init__(self):
        self.node_type = IRNodeType.RELU


@dataclass
class ReshapeNode(IRNode):
    """Represents reshape operation"""
    target_shape: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = IRNodeType.RESHAPE
        self.attributes["target_shape"] = self.target_shape


@dataclass
class ConcatNode(IRNode):
    """Represents concatenation operation"""
    axis: int = 0

    def __post_init__(self):
        self.node_type = IRNodeType.CONCAT
        self.attributes["axis"] = self.axis


@dataclass
class SplitNode(IRNode):
    """Represents split operation"""
    axis: int = 0
    split_sizes: Optional[List[int]] = None

    def __post_init__(self):
        self.node_type = IRNodeType.SPLIT
        self.attributes["axis"] = self.axis
        if self.split_sizes:
            self.attributes["split_sizes"] = self.split_sizes


@dataclass
class IRGraph:
    name: str
    nodes: List[IRNode] = field(default_factory=list)
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    weights: Dict[str, Any] = field(default_factory=dict)

    def add_tensor(self, name, shape, dtype):
        t = TensorInfo(name=name, shape=shape, dtype=dtype)
        self.tensors[name] = t
        return t

    def get_node_by_output(self, output_name):
        for node in self.nodes:
            if output_name in node.outputs:
                return node
        return None

    def toposort(self):
        visited = set()
        result = []
        node_map = {n.outputs[0]: n for n in self.nodes if n.outputs}

        def visit(node):
            if node.node_id in visited:
                return
            visited.add(node.node_id)
            for inp in node.inputs:
                dep = node_map.get(inp)
                if dep:
                    visit(dep)
            result.append(node)

        for node in self.nodes:
            visit(node)
        return result

    def serialize(self, path):
        data = {
            "name": self.name,
            "inputs": self.input_names,
            "outputs": self.output_names,
            "tensors": {
                n: {"shape": t.shape, "dtype": t.dtype.value, "addr": t.sram_address}
                for n, t in self.tensors.items()
            },
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type.value,
                    "inputs": n.inputs,
                    "outputs": n.outputs,
                    "attrs": n.attributes,
                }
                for n in self.nodes
            ],
            "statistics": self.get_statistics(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics for analysis"""
        node_counts = {}
        for node in self.nodes:
            node_type = node.node_type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        total_params = 0
        total_ops = 0
        for node in self.nodes:
            if node.node_type == IRNodeType.LINEAR:
                cin = node.cin if hasattr(node, 'cin') else 0
                cout = node.cout if hasattr(node, 'cout') else 0
                total_params += cin * cout
                total_ops += 2 * cin * cout

        tensor_memory = sum(t.size_bytes for t in self.tensors.values())

        return {
            "total_nodes": len(self.nodes),
            "node_type_distribution": node_counts,
            "total_tensors": len(self.tensors),
            "total_parameters": total_params,
            "total_operations": total_ops,
            "estimated_memory_bytes": tensor_memory,
        }

    def validate(self) -> List[str]:
        """Validate IR graph integrity"""
        errors = []
        defined_tensors = set()

        for inp in self.input_names:
            defined_tensors.add(inp)

        for node in self.nodes:
            for inp in node.inputs:
                if inp not in defined_tensors:
                    errors.append(f"Node {node.node_id}: input tensor '{inp}' not defined")

            for out in node.outputs:
                defined_tensors.add(out)

        for out in self.output_names:
            if out not in defined_tensors:
                errors.append(f"Output tensor '{out}' not defined by any node")

        return errors
