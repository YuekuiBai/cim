"""
3D Mapper - Core algorithm for placing Weight-Cubes in 3D resource space
Uses greedy first-fit with MoE-aware placement optimization
"""

from typing import List, Dict, Optional, Tuple
from compiler_mapping.weight_cube import (
    CubeConfig, Placement3D, WeightSection, ScheduleEntry, Solution
)
from model_parser.trace_analyzer import ActivationTraceAnalyzer


class Mapper3D:
    """Maps weight sections to 3D CIM space"""

    def __init__(self, config: CubeConfig):
        self.config = config
        # Track used z-ranges per Sub-Cube
        self.subcube_usage: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(config.num_subcubes)
        }

    def map(
        self,
        sections: List[WeightSection],
        analyzer: Optional[ActivationTraceAnalyzer] = None,
    ) -> Solution:
        """Place all sections into 3D space"""
        placements = []
        failed = []

        if analyzer:
            analyzer.analyze()
            freq = analyzer.get_activation_frequency()
            sorted_sections = self._sort_sections_by_priority(sections, freq, analyzer)
        else:
            sorted_sections = sorted(sections, key=lambda s: s.volume, reverse=True)

        for section in sorted_sections:
            placement = self._place_section(section)
            if placement:
                placements.append(placement)
            else:
                failed.append(section)

        if failed:
            print(f"  Warning: {len(failed)} sections could not be placed within depth limit")

        total_params = sum(s.num_params for s in sections)
        total_capacity = self.config.total_volume
        used_capacity = sum(p.volume for p in placements)

        solution = Solution(
            cube_config=self.config,
            placements=placements,
            total_params_mapped=total_params,
        )
        solution.space_utilization = used_capacity / total_capacity if total_capacity > 0 else 0

        return solution

    def _sort_sections_by_priority(self, sections, freq, analyzer):
        shared_set = analyzer.shared_experts if analyzer else set()

        def priority_key(s):
            expert_id = s.section_idx
            is_shared = expert_id in shared_set
            if is_shared:
                return (0, 0, s.volume)
            f = freq[expert_id] if expert_id < len(freq) else 0
            return (1, -f, s.volume)

        return sorted(sections, key=priority_key)

    def _place_section(self, section: WeightSection) -> Optional[Placement3D]:
        """Find the best Sub-Cube and z-position for a section.
        Never exceeds config.depth."""
        # Phase 1: Try to find contiguous free z-space
        best_sc = None
        best_z = None
        for sc_id in range(self.config.num_subcubes):
            z_pos = self._find_free_z(sc_id, section.depth)
            if z_pos is not None:
                if best_z is None or z_pos < best_z:
                    best_sc = sc_id
                    best_z = z_pos

        if best_sc is not None:
            return self._create_placement(section, best_sc, best_z)

        # Phase 2: Try to pack more tightly by using least-used Sub-Cube
        # Find Sub-Cube with lowest max z-end
        sc_loads = {}
        for sc_id in range(self.config.num_subcubes):
            usage = self.subcube_usage[sc_id]
            max_end = max((end for _, end in usage), default=0)
            remaining = self.config.depth - max_end
            if remaining >= section.depth:
                sc_loads[sc_id] = max_end

        if sc_loads:
            best_sc = min(sc_loads, key=sc_loads.get)
            z_pos = sc_loads[best_sc]
            return self._create_placement(section, best_sc, z_pos)

        # Phase 3: Cannot fit within depth - return None
        return None

    def _create_placement(self, section, sc_id, z_pos):
        z_end = z_pos + section.depth
        if z_end > self.config.depth:
            return None
        placement = Placement3D(
            section=section,
            subcube_id=sc_id,
            z_start=z_pos,
            z_end=z_end,
            y_start=0,
            y_end=min(section.shape_2d[0], self.config.hw),
            x_start=0,
            x_end=min(section.shape_2d[1], self.config.hw),
        )
        self.subcube_usage[sc_id].append((z_pos, z_end))
        return placement

    def _find_free_z(self, sc_id: int, depth: int) -> Optional[int]:
        """Find a contiguous free z-range within config.depth"""
        usage = sorted(self.subcube_usage[sc_id])
        if not usage:
            if depth <= self.config.depth:
                return 0
            return None

        # Check from z=0
        if usage[0][0] >= depth:
            return 0

        # Check gaps between used ranges
        for i in range(1, len(usage)):
            gap_start = usage[i - 1][1]
            gap_end = usage[i][0]
            if gap_end - gap_start >= depth:
                return gap_start

        # Check after last used range
        last_end = usage[-1][1]
        if last_end + depth <= self.config.depth:
            return last_end

        return None
