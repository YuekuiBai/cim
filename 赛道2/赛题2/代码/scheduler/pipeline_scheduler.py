"""
Pipeline Scheduler - Schedules operator execution with dependency and switching awareness
Respects barrier synchronization and intra-Sub-Cube switching penalty
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict, deque
from compiler_mapping.weight_cube import Solution, ScheduleEntry, Placement3D
from model_parser.model_loader import Operator


class PipelineScheduler:
    """Schedules operator execution across Sub-Cubes"""

    def __init__(self, solution: Solution, operators: List[Operator]):
        self.solution = solution
        self.operators = {op.name: op for op in operators}
        self.depth = solution.cube_config.depth

    def schedule(self) -> List[ScheduleEntry]:
        """Generate execution schedule based on dependencies and placement"""
        placement_map = self._build_placement_map()
        dep_map = self._build_dependency_map()

        # Track Sub-Cube readiness: when each Sub-Cube becomes free
        subcube_free_at = {i: 0 for i in range(self.solution.cube_config.num_subcubes)}

        # Track operator completion times
        operator_ready_at = {}
        schedule = []
        completed = set()

        # Topological order with barrier sync
        order = self._topological_order(dep_map)

        for op_name in order:
            if op_name not in self.operators:
                continue

            op = self.operators[op_name]
            deps = dep_map.get(op_name, [])

            # Wait for all dependencies to complete (barrier sync)
            earliest_start = 0
            for dep in deps:
                if dep in operator_ready_at:
                    earliest_start = max(earliest_start, operator_ready_at[dep])

            # Find Sub-Cubes that contain this operator's weights
            subcubes_for_op = placement_map.get(op_name, [0])
            primary_sc = subcubes_for_op[0]

            # Wait for Sub-Cube to be free
            sc_free = subcube_free_at[primary_sc]
            start_cycle = max(earliest_start, sc_free)

            # Add switching penalty if Sub-Cube was doing something else
            last_op = self._last_operator_on_sc(primary_sc, schedule)
            switching_penalty = 0
            if last_op and last_op != op_name:
                switching_penalty = self.depth

            start_cycle += switching_penalty

            # Execution time: 1 + D cycles per section
            num_sections = len(subcubes_for_op)
            exec_time = (1 + self.depth) * num_sections
            end_cycle = start_cycle + exec_time

            entry = ScheduleEntry(
                operator=op_name,
                subcube_id=primary_sc,
                start_cycle=start_cycle,
                end_cycle=end_cycle,
                switching_penalty=switching_penalty,
                depends_on=deps,
            )
            schedule.append(entry)

            operator_ready_at[op_name] = end_cycle
            subcube_free_at[primary_sc] = end_cycle
            completed.add(op_name)

        self.solution.schedule = schedule
        self.solution.total_latency = max((e.end_cycle for e in schedule), default=0)
        return schedule

    def _build_placement_map(self) -> Dict[str, List[int]]:
        """Map operator name to list of Sub-Cube IDs where it's placed"""
        op_to_sc = defaultdict(list)
        for p in self.solution.placements:
            sc_id = p.subcube_id
            if sc_id not in op_to_sc[p.section.parent_operator]:
                op_to_sc[p.section.parent_operator].append(sc_id)
        return dict(op_to_sc)

    def _build_dependency_map(self) -> Dict[str, List[str]]:
        """Build operator dependency map"""
        return {op.name: op.dependencies for op in self.operators.values()}

    def _topological_order(self, dep_map: Dict[str, List[str]]) -> List[str]:
        """Topological sort respecting dependencies"""
        in_degree = defaultdict(int)
        all_nodes = set(dep_map.keys())

        for node, deps in dep_map.items():
            for dep in deps:
                if dep in all_nodes:
                    in_degree[node] += 1

        queue = deque()
        for node in all_nodes:
            if in_degree[node] == 0:
                queue.append(node)

        order = []
        while queue:
            node = queue.popleft()
            order.append(node)

            for other, deps in dep_map.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        # Add any nodes not in dep_map
        for node in all_nodes:
            if node not in order:
                order.append(node)

        return order

    def _last_operator_on_sc(self, sc_id: int, schedule: List[ScheduleEntry]) -> Optional[str]:
        """Get the last operator scheduled on a Sub-Cube"""
        for entry in reversed(schedule):
            if entry.subcube_id == sc_id:
                return entry.operator
        return None
