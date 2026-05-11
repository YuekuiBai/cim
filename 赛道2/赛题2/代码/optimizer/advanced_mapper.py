"""
Advanced 3D Mapping Optimizer
- ILP-based optimal mapping
- Simulated Annealing heuristic
- Frequency-aware z-axis placement
- Conflict minimization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import random
import math


class WeightCube:
    """Represents a weight cube to be placed"""
    
    def __init__(self, name: str, rows: int, cols: int, depth: int,
                 frequency: float = 0.0, expert_id: Optional[int] = None):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.depth = depth
        self.frequency = frequency
        self.expert_id = expert_id
        self.volume = rows * cols * depth
    
    def __repr__(self):
        return f"WeightCube({self.name}, {self.rows}x{self.cols}x{self.depth})"


class SubCubeState:
    """Tracks the state of a Sub-Cube"""
    
    def __init__(self, subcube_id: int, hw: int, max_depth: int):
        self.subcube_id = subcube_id
        self.hw = hw
        self.max_depth = max_depth
        self.placements: List[Tuple[int, int, int, WeightCube]] = []  # (z_start, z_end, rows*cols, cube)
        self.used_depth = 0
        self.used_2d_area = 0
    
    def can_place(self, cube: WeightCube) -> Optional[int]:
        """Check if cube can be placed, return z_start if possible"""
        if self.used_depth + cube.depth > self.max_depth:
            return None
        
        # Find first free z-position
        if not self.placements:
            return 0
        
        sorted_placements = sorted(self.placements, key=lambda x: x[0])
        
        # Try to fit between existing placements
        z_start = 0
        for z_end_prev, _, _, _ in sorted_placements:
            if z_start + cube.depth <= z_end_prev:
                return z_start
            z_start = max(z_start, z_end_prev)
        
        # Try at the end
        if z_start + cube.depth <= self.max_depth:
            return z_start
        
        return None
    
    def place(self, cube: WeightCube, z_start: int):
        """Place a cube at specified z position"""
        z_end = z_start + cube.depth
        self.placements.append((z_start, z_end, cube.rows * cube.cols, cube))
        self.used_depth = max(p[1] for p in self.placements)
        self.used_2d_area += cube.rows * cube.cols
    
    def get_utilization(self) -> Tuple[float, float]:
        """Return (depth_utilization, area_utilization)"""
        depth_util = self.used_depth / self.max_depth if self.max_depth > 0 else 0
        max_area = self.hw * self.hw
        area_util = self.used_2d_area / max_area if max_area > 0 else 0
        return depth_util, area_util


class ILPBasedMapper:
    """
    Integer Linear Programming based mapper
    Formulates the mapping as an optimization problem
    
    Since we may not have ILP solver, we provide a greedy approximation
    that mimics ILP behavior
    """
    
    def __init__(self, num_subcubes: int, subcube_hw: int, max_depth: int,
                 cooccurrence_matrix: Optional[np.ndarray] = None,
                 activation_freq: Optional[np.ndarray] = None):
        self.num_subcubes = num_subcubes
        self.subcube_hw = subcube_hw
        self.max_depth = max_depth
        self.cooccurrence = cooccurrence_matrix
        self.freq = activation_freq
        
        self.subcubes = [
            SubCubeState(i, subcube_hw, max_depth)
            for i in range(num_subcubes)
        ]
    
    def map(self, cubes: List[WeightCube],
            expert_assignment: Optional[Dict[int, int]] = None) -> Dict[str, Tuple[int, int]]:
        """
        Map weight cubes to subcubes
        Returns: cube_name -> (subcube_id, z_start)
        """
        # Sort cubes by priority: volume * frequency (descending)
        sorted_cubes = sorted(cubes, key=lambda c: -c.volume * (1 + c.frequency))
        
        placement_result = {}
        
        for cube in sorted_cubes:
            placed = False
            
            # If expert assignment is provided, respect it
            if expert_assignment is not None and cube.expert_id is not None:
                preferred_subcube = expert_assignment.get(cube.expert_id)
                if preferred_subcube is not None:
                    z_start = self.subcubes[preferred_subcube].can_place(cube)
                    if z_start is not None:
                        self.subcubes[preferred_subcube].place(cube, z_start)
                        placement_result[cube.name] = (preferred_subcube, z_start)
                        placed = True
            
            if not placed:
                # Try all subcubes, pick best fit
                best_subcube = None
                best_z = None
                best_score = float('inf')
                
                for sc in self.subcubes:
                    z_start = sc.can_place(cube)
                    if z_start is not None:
                        # Score: prefer subcubes with less fragmentation
                        score = sc.used_depth + cube.depth
                        if score < best_score:
                            best_score = score
                            best_subcube = sc
                            best_z = z_start
                
                if best_subcube is not None:
                    best_subcube.place(cube, best_z)
                    placement_result[cube.name] = (best_subcube.subcube_id, best_z)
                    placed = True
            
            if not placed:
                print(f"Warning: Could not place {cube.name}")
        
        return placement_result
    
    def get_statistics(self) -> Dict:
        """Get mapping statistics"""
        total_volume = 0
        total_capacity = self.num_subcubes * self.subcube_hw * self.subcube_hw * self.max_depth
        
        for sc in self.subcubes:
            for _, _, area, cube in sc.placements:
                total_volume += area * cube.depth
        
        utilizations = [sc.get_utilization() for sc in self.subcubes]
        avg_depth_util = np.mean([u[0] for u in utilizations])
        avg_area_util = np.mean([u[1] for u in utilizations])
        
        return {
            'total_volume': total_volume,
            'total_capacity': total_capacity,
            'space_utilization': total_volume / total_capacity if total_capacity > 0 else 0,
            'avg_depth_utilization': avg_depth_util,
            'avg_area_utilization': avg_area_util,
            'subcube_utilizations': utilizations
        }


class SimulatedAnnealingMapper:
    """
    Simulated Annealing based mapper for 3D placement
    Optimizes for:
    1. Minimize conflicts (co-occurring experts in same subcube)
    2. Balance load across subcubes
    3. Place high-frequency cubes at low z positions
    """
    
    def __init__(self, num_subcubes: int, subcube_hw: int, max_depth: int,
                 cooccurrence_matrix: Optional[np.ndarray] = None):
        self.num_subcubes = num_subcubes
        self.subcube_hw = subcube_hw
        self.max_depth = max_depth
        self.cooccurrence = cooccurrence_matrix
    
    def optimize(self, cubes: List[WeightCube],
                 expert_assignment: Optional[Dict[int, int]] = None,
                 initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 iterations: int = 10000) -> Dict:
        """Run SA optimization"""
        
        # Initial solution: greedy placement
        current_placement = self._greedy_initial_placement(cubes, expert_assignment)
        current_cost = self._calculate_cost(current_placement, cubes)
        
        best_placement = current_placement.copy()
        best_cost = current_cost
        
        temp = initial_temp
        
        for i in range(iterations):
            # Generate neighbor: move one cube to different subcube/z
            neighbor = self._perturb(current_placement, cubes)
            if neighbor is None:
                continue
            
            neighbor_cost = self._calculate_cost(neighbor, cubes)
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_placement = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_placement = current_placement.copy()
                    best_cost = current_cost
            
            temp *= cooling_rate
        
        return {
            'placement': best_placement,
            'cost': best_cost,
            'method': 'Simulated Annealing'
        }
    
    def _greedy_initial_placement(self, cubes: List[WeightCube],
                                   expert_assignment: Optional[Dict[int, int]]) -> Dict[str, Tuple[int, int]]:
        """Generate initial greedy placement"""
        mapper = ILPBasedMapper(
            self.num_subcubes, self.subcube_hw, self.max_depth,
            cooccurrence_matrix=self.cooccurrence
        )
        return mapper.map(cubes, expert_assignment)
    
    def _perturb(self, placement: Dict[str, Tuple[int, int]],
                 cubes: List[WeightCube]) -> Optional[Dict[str, Tuple[int, int]]]:
        """Generate neighbor solution by moving one cube"""
        if not placement:
            return None
        
        cube_name = random.choice(list(placement.keys()))
        cube = next(c for c in cubes if c.name == cube_name)
        
        new_placement = placement.copy()
        
        # Try random subcube
        new_subcube = random.randint(0, self.num_subcubes - 1)
        new_placement[cube_name] = (new_subcube, 0)  # Simplified z assignment
        
        return new_placement
    
    def _calculate_cost(self, placement: Dict[str, Tuple[int, int]],
                        cubes: List[WeightCube]) -> float:
        """Calculate cost of placement"""
        cost = 0.0
        
        # Group cubes by subcube
        subcube_cubes = defaultdict(list)
        for cube_name, (sc_id, z) in placement.items():
            cube = next(c for c in cubes if c.name == cube_name)
            subcube_cubes[sc_id].append(cube)
        
        # Conflict cost
        if self.cooccurrence is not None:
            for sc_id, sc_cubes in subcube_cubes.items():
                for i in range(len(sc_cubes)):
                    for j in range(i + 1, len(sc_cubes)):
                        if sc_cubes[i].expert_id is not None and sc_cubes[j].expert_id is not None:
                            e1, e2 = sc_cubes[i].expert_id, sc_cubes[j].expert_id
                            cost += self.cooccurrence[e1][e2]
        
        # Load balance cost
        loads = [len(sc_cubes) for sc_cubes in subcube_cubes.values()]
        if loads:
            cost += np.var(loads) * 0.1
        
        # Frequency cost: high-freq cubes should be at low z
        for cube_name, (sc_id, z) in placement.items():
            cube = next(c for c in cubes if c.name == cube_name)
            cost += cube.frequency * z * 0.01
        
        return cost
