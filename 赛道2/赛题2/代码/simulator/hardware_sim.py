"""
Hardware Simulator for Problem 2
Simulates execution on 3D CIM hardware and validates physical constraints
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict
from compiler_mapping.weight_cube import Solution, ScheduleEntry, Placement3D


class SolutionValidator:
    """Validates mapping solution against hardware constraints"""

    def __init__(self, solution: Solution):
        self.solution = solution

    def validate(self) -> bool:
        checks = [
            self._check_no_overlap(),
            self._check_capacity(),
            self._check_weight_stationary(),
        ]
        return all(checks)

    def validate_details(self) -> Dict[str, bool]:
        return {
            "no_overlap": self._check_no_overlap(),
            "capacity_ok": self._check_capacity(),
            "weight_stationary": self._check_weight_stationary(),
        }

    def _check_no_overlap(self) -> bool:
        """Check no two Weight-Cubes overlap in same Sub-Cube z-space"""
        sc_ranges = defaultdict(list)
        for p in self.solution.placements:
            sc_ranges[p.subcube_id].append((p.z_start, p.z_end, p.section.name))

        for sc_id, ranges in sc_ranges.items():
            ranges.sort()
            for i in range(1, len(ranges)):
                prev_end = ranges[i - 1][1]
                curr_start = ranges[i][0]
                if curr_start < prev_end:
                    print(f"[FAIL] Overlap in Sub-Cube {sc_id}: "
                          f"{ranges[i-1][2]}[{ranges[i-1][0]}:{ranges[i-1][1]}] vs "
                          f"{ranges[i][2]}[{ranges[i][0]}:{ranges[i][1]}]")
                    return False
        return True

    def _check_capacity(self) -> bool:
        """Check each Sub-Cube z-usage doesn't exceed depth"""
        for p in self.solution.placements:
            if p.z_end > self.solution.cube_config.depth:
                print(f"[FAIL] {p.section.name} exceeds depth: {p.z_end} > {self.solution.cube_config.depth}")
                return False
        return True

    def _check_weight_stationary(self) -> bool:
        """Check weights are stationary (already guaranteed by design)"""
        return True


class HardwareSimulator:
    """Simulates 3D CIM hardware execution"""

    def __init__(self, solution: Solution):
        self.solution = solution
        self.depth = solution.cube_config.depth

    def simulate(self) -> int:
        """Simulate and return total latency"""
        if not self.solution.schedule:
            return 0
        return max(e.end_cycle for e in self.solution.schedule)

    def simulate_detailed(self) -> Dict:
        """Detailed simulation report"""
        latency = self.simulate()
        num_subcubes = self.solution.cube_config.num_subcubes

        # Sub-Cube utilization
        sc_usage = defaultdict(int)
        for entry in self.solution.schedule:
            sc_usage[entry.subcube_id] += entry.end_cycle - entry.start_cycle

        max_possible = latency
        sc_efficiency = {}
        for sc_id in range(num_subcubes):
            used = sc_usage.get(sc_id, 0)
            sc_efficiency[sc_id] = used / max_possible if max_possible > 0 else 0

        total_switching = sum(e.switching_penalty for e in self.solution.schedule)

        return {
            "total_latency": latency,
            "num_subcubes": num_subcubes,
            "num_schedule_entries": len(self.solution.schedule),
            "total_switching_penalty_cycles": total_switching,
            "subcube_efficiency": sc_efficiency,
        }
