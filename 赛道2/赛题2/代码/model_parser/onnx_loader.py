"""
ONNX Loader for Problem 2
"""

import onnx
from typing import List, Dict


class ONNXLoader:
    """Loads ONNX models for 3D mapping"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
    
    def load(self):
        """Load ONNX model"""
        model = onnx.load(self.model_path)
        onnx.checker.check_model(model)
        return model
    
    def extract_operators(self):
        """Extract operators from ONNX model"""
        operators = []
        for node in self.model.graph.node:
            operators.append({
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output)
            })
        return operators
