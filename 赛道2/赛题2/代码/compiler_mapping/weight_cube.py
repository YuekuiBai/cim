"""
Data structures for Problem 2 - 3D Weight Mapping & Scheduling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class Operator:
    """Represents a neural network operator"""
    name: str
    op_type: str  # MatMul, MoE, etc.
    weight_shape: List[int]  # [rows, cols]
    dependencies: List[str] = field(default_factory=list)
    is_sparse: bool = False
    expert_id: Optional[int] = None
    is_shared_expert: bool = False

    @property
    def num_params(self) -> int:
        n = 1
        for d in self.weight_shape:
            n *= d
        return n


@dataclass
class WeightSection:
    """A section of weight after partitioning"""
    name: str
    shape_2d: List[int]  # [rows, cols]
    depth: int  # z-axis depth needed
    parent_operator: str
    section_idx: int  # section index within parent
    total_sections: int  # total sections this weight was split into
    is_replicated: bool = False
    replication_idx: int = 0

    @property
    def volume(self) -> int:
        return self.shape_2d[0] * self.shape_2d[1] * self.depth

    @property
    def num_params(self) -> int:
        return self.shape_2d[0] * self.shape_2d[1]


@dataclass
class Placement3D:
    """3D placement of a Weight-Cube in a Sub-Cube"""
    section: WeightSection
    subcube_id: int
    z_start: int
    z_end: int
    y_start: int = 0
    y_end: int = 0
    x_start: int = 0
    x_end: int = 0

    @property
    def volume(self) -> int:
        return (self.z_end - self.z_start) * (self.y_end - self.y_start) * (self.x_end - self.x_start)


@dataclass
class SubCubeConfig:
    """Configuration of a single Sub-Cube"""
    id: int
    hw: int  # H x W (square)
    depth: int  # D

    @property
    def total_volume(self) -> int:
        return self.hw * self.hw * self.depth


@dataclass
class CubeConfig:
    """Global 3D Cube configuration"""
    n: int  # N x N Sub-Cubes (2~4)
    hw: int  # Sub-Cube H x W (4096~16384)
    depth: int  # Z-axis depth

    @property
    def num_subcubes(self) -> int:
        return self.n * self.n

    @property
    def total_volume(self) -> int:
        return self.num_subcubes * self.hw * self.hw * self.depth

    def validate(self) -> bool:
        if not (2 <= self.n <= 4):
            return False
        if not (4096 <= self.hw <= 16384):
            return False
        if self.depth <= 0:
            return False
        return True


@dataclass
class ScheduleEntry:
    """A scheduled execution step"""
    operator: str
    subcube_id: int
    start_cycle: int
    end_cycle: int
    is_barrier: bool = False
    switching_penalty: int = 0
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """Complete mapping + scheduling solution"""
    cube_config: CubeConfig
    placements: List[Placement3D] = field(default_factory=list)
    schedule: List[ScheduleEntry] = field(default_factory=list)
    total_latency: int = 0
    space_utilization: float = 0.0
    total_params_mapped: int = 0

    def save(self, path: str):
        data = {
            "cube_config": {
                "n": self.cube_config.n,
                "hw": self.cube_config.hw,
                "depth": self.cube_config.depth,
            },
            "placements": [
                {
                    "section": p.section.name,
                    "subcube_id": p.subcube_id,
                    "z": [p.z_start, p.z_end],
                    "y": [p.y_start, p.y_end],
                    "x": [p.x_start, p.x_end],
                    "volume": p.volume,
                }
                for p in self.placements
            ],
            "schedule": [
                {
                    "operator": s.operator,
                    "subcube_id": s.subcube_id,
                    "start_cycle": s.start_cycle,
                    "end_cycle": s.end_cycle,
                    "switching_penalty": s.switching_penalty,
                    "depends_on": s.depends_on,
                }
                for s in self.schedule
            ],
            "metrics": {
                "total_latency": self.total_latency,
                "space_utilization": self.space_utilization,
                "total_params_mapped": self.total_params_mapped,
                "total_capacity": self.cube_config.total_volume,
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return data

    def report(self) -> str:
        lines = [
            "=== 3D Mapping Solution ===",
            f"  Cube: {self.cube_config.n}x{self.cube_config.n} Sub-Cubes, HW={self.cube_config.hw}, D={self.cube_config.depth}",
            f"  Total capacity: {self.cube_config.total_volume:,} cells",
            f"  Total params mapped: {self.total_params_mapped:,}",
            f"  Space utilization: {self.space_utilization*100:.1f}%",
            f"  End-to-end latency: {self.total_latency:,} cycles",
            f"  Placements: {len(self.placements)}",
            f"  Schedule entries: {len(self.schedule)}",
        ]
        return "\n".join(lines)
