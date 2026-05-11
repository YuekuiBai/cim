"""
Model Loader - Unified interface for loading models
Compatible with existing Operator and WeightSection data structures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compiler_mapping.weight_cube import Operator
from typing import List
import json


class ModelLoader:
    """Unified model loader supporting JSON format"""
    
    def load_json(self, json_path: str) -> List[Operator]:
        """Load model from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        operators = []
        for op_data in data.get("operators", []):
            op = Operator(
                name=op_data["name"],
                op_type=op_data.get("type", op_data.get("op_type", "MatMul")),
                weight_shape=op_data.get("weight_shape", [0, 0]),
                dependencies=op_data.get("predecessors", op_data.get("dependencies", [])),
                is_sparse=op_data.get("is_sparse", False),
                expert_id=op_data.get("expert_id"),
                is_shared_expert=op_data.get("is_shared", op_data.get("is_shared_expert", False)),
            )
            operators.append(op)
        
        return operators
    
    def load_simple_model(self, json_path: str) -> List[Operator]:
        """Load simple model format"""
        return self.load_json(json_path)
