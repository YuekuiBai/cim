"""
Latency Calculator
Calculates execution latency for operations
"""

from compiler_mapping.sub_cube import SubCube
from compiler_mapping.weight_cube import WeightCube


class LatencyCalculator:
    """Calculates execution latency"""
    
    @staticmethod
    def calculate_activation_latency(cube: WeightCube, depth: int) -> int:
        """Calculate latency for activating a Weight-Cube"""
        # 1 + D cycles per activation
        return 1 + depth
    
    @staticmethod
    def calculate_switching_latency(prev_cube: WeightCube, next_cube: WeightCube, 
                                   same_subcube: bool, depth: int) -> int:
        """Calculate switching latency between cubes"""
        if not same_subcube:
            return 0  # Inter-Sub-Cube, no switching penalty
        if prev_cube is None or prev_cube.name == next_cube.name:
            return 0  # Same cube, no switching
        return depth  # Intra-Sub-Cube switching penalty
