"""
Sub-Cube - Represents a compute core in 3D space
"""

from dataclasses import dataclass, field
from typing import List, Optional
from compiler_mapping.weight_cube import WeightCube


@dataclass
class SubCube:
    """Represents a Sub-Cube (compute core)"""
    id: int
    hw_size: int  # H x W
    depth: int    # D
    weight_cubes: List[WeightCube] = field(default_factory=list)
    active_cube: Optional[WeightCube] = None
    
    @property
    def total_volume(self) -> int:
        """Calculate total volume"""
        return self.hw_size * self.hw_size * self.depth
    
    @property
    def used_volume(self) -> int:
        """Calculate used volume"""
        return sum(wc.volume for wc in self.weight_cubes)
    
    @property
    def utilization(self) -> float:
        """Calculate space utilization"""
        if self.total_volume == 0:
            return 0.0
        return self.used_volume / self.total_volume
    
    def can_place(self, cube: WeightCube) -> bool:
        """Check if a cube can be placed in this Sub-Cube"""
        # Check if cube fits in dimensions
        if not cube.fits_in_subcube(self.hw_size, self.depth):
            return False
        
        # Check if there's enough space (simplified)
        if self.used_volume + cube.volume > self.total_volume:
            return False
        
        # TODO: Implement detailed 3D placement check
        return True
    
    def place_cube(self, cube: WeightCube) -> bool:
        """Place a weight cube in this Sub-Cube"""
        if not self.can_place(cube):
            return False
        
        cube.subcube_id = self.id
        self.weight_cubes.append(cube)
        return True
    
    def activate_cube(self, cube: WeightCube):
        """Activate a specific cube for computation"""
        if cube in self.weight_cubes:
            self.active_cube = cube
        else:
            raise ValueError(f"Cube {cube.name} not in Sub-Cube {self.id}")
