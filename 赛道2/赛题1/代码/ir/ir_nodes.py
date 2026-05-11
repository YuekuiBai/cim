"""
IR Nodes for CIM Compiler - Problem 1
Supports linear and elementwise (add/sub/mul/div) operators
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


class IRNodeType(Enum):
    LINEAR = "linear"
    ELEMENTWISE = "elementwise"
    INPUT = "input"
    OUTPUT = "output"
    CONSTANT = "constant"


class DataType(Enum):
    INT8 = "i8"
    INT16 = "i16"
    INT32 = "i32"


class ElementwiseOp(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"


DTYPE_BYTES = {
    DataType.INT8: 1,
    DataType.INT16: 2,
    DataType.INT32: 4,
}

DTYPE_BITS = {
    DataType.INT8: 8,
    DataType.INT16: 16,
    DataType.INT32: 32,
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
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
