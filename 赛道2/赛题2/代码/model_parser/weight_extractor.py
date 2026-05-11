"""
Weight Extractor
Extracts and partitions weights for 3D mapping
"""

from dataclasses import dataclass
from typing import List
from model_parser.json_loader import ModelDescription, Operator


@dataclass
class WeightSection:
    """Represents a section of weight"""
    name: str
    shape: List[int]
    parent_operator: str
    section_id: int


class WeightExtractor:
    """Extracts weights from model and partitions them"""
    
    def __init__(self, model_desc: ModelDescription):
        self.model_desc = model_desc
    
    def extract(self) -> List[WeightSection]:
        """Extract all weight sections from model"""
        sections = []
        
        for op in self.model_desc.operators:
            # For each operator, create weight sections
            # TODO: Implement weight partitioning based on Sub-Cube size
            section = WeightSection(
                name=f"{op.name}_weight",
                shape=op.weight_shape,
                parent_operator=op.name,
                section_id=0
            )
            sections.append(section)
        
        return sections
