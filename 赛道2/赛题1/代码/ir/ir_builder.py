"""
IR Builder for constructing IR graphs
"""

from ir.ir_nodes import IRGraph, IRNode, LinearNode, ElementwiseNode, TensorInfo, DataType
from typing import List, Dict


class IRBuilder:
    """Builds IR graphs from parsed model information"""
    
    def __init__(self, name: str = "model"):
        self.name = name
        self.nodes: List[IRNode] = []
        self.tensors: Dict[str, TensorInfo] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []
    
    def add_tensor(self, name: str, shape: List[int], dtype: DataType) -> TensorInfo:
        """Add tensor info to the graph"""
        tensor = TensorInfo(name=name, shape=shape, dtype=dtype)
        self.tensors[name] = tensor
        return tensor
    
    def add_linear(self, inputs: List[str], outputs: List[str], 
                   weight: str, bias: str = None, weight_shape: List[int] = None) -> LinearNode:
        """Add a linear layer node"""
        node = LinearNode(
            node_type=None,  # Will be set by dataclass
            inputs=inputs,
            outputs=outputs,
            attributes={},
            weight=weight,
            bias=bias,
            weight_shape=weight_shape
        )
        self.nodes.append(node)
        return node
    
    def add_elementwise(self, op_type: str, inputs: List[str], outputs: List[str],
                       mode: str = "vv", immediate_value: int = None) -> ElementwiseNode:
        """Add an elementwise operation node"""
        node = ElementwiseNode(
            node_type=None,
            inputs=inputs,
            outputs=outputs,
            attributes={"op_type": op_type, "mode": mode},
            op_type=op_type,
            mode=mode,
            immediate_value=immediate_value
        )
        self.nodes.append(node)
        return node
    
    def set_inputs(self, input_names: List[str]):
        """Set graph inputs"""
        self.input_names = input_names
    
    def set_outputs(self, output_names: List[str]):
        """Set graph outputs"""
        self.output_names = output_names
    
    def build(self) -> IRGraph:
        """Build and return the complete IR graph"""
        return IRGraph(
            name=self.name,
            nodes=self.nodes,
            tensors=self.tensors,
            input_names=self.input_names,
            output_names=self.output_names
        )
