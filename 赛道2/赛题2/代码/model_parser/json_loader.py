"""
Model JSON Loader
Loads model description from JSON format
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Operator:
    """Represents a model operator"""
    name: str
    op_type: str  # MatMul, MoE, etc.
    weight_shape: List[int]
    dependencies: List[str]
    is_sparse: bool = False
    expert_id: Optional[int] = None


@dataclass
class ModelDescription:
    """Complete model description"""
    name: str
    operators: List[Operator]
    total_params: int
    is_moe: bool = False
    num_experts: int = 0
    top_k: int = 0
    shared_experts: List[int] = None


class ModelJSONLoader:
    """Loads model description from JSON"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
    
    def load(self) -> ModelDescription:
        """Load model description from JSON file"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        operators = []
        for op_data in data.get('operators', []):
            op = Operator(
                name=op_data['name'],
                op_type=op_data['op_type'],
                weight_shape=op_data['weight_shape'],
                dependencies=op_data.get('dependencies', []),
                is_sparse=op_data.get('is_sparse', False),
                expert_id=op_data.get('expert_id')
            )
            operators.append(op)
        
        return ModelDescription(
            name=data.get('name', 'model'),
            operators=operators,
            total_params=data.get('total_params', 0),
            is_moe=data.get('is_moe', False),
            num_experts=data.get('num_experts', 0),
            top_k=data.get('top_k', 0),
            shared_experts=data.get('shared_experts', [])
        )
