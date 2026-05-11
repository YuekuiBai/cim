"""
Advanced Scheduling Strategies
- Pipeline Parallelism
- Data Parallelism with weight replication
- Critical Path aware scheduling
- Dynamic conflict resolution
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import heapq


class Operator:
    """Represents a computational operator"""
    
    def __init__(self, name: str, op_type: str, 
                 weight_shape: Tuple[int, int],
                 predecessors: List[str] = None,
                 expert_id: Optional[int] = None,
                 is_shared: bool = False):
        self.name = name
        self.op_type = op_type
        self.weight_shape = weight_shape
        self.predecessors = predecessors or []
        self.expert_id = expert_id
        self.is_shared = is_shared
        self.subcube_id: Optional[int] = None
        self.z_start: Optional[int] = None
        self.depth: int = 0


class ScheduledStep:
    """Represents a scheduled execution step"""
    
    def __init__(self, operator_id: str, start_time: int, end_time: int,
                 subcube_id: int, is_switch: bool = False, switch_cycles: int = 0):
        self.operator_id = operator_id
        self.start_time = start_time
        self.end_time = end_time
        self.subcube_id = subcube_id
        self.is_switch = is_switch
        self.switch_cycles = switch_cycles
    
    def duration(self) -> int:
        return self.end_time - self.start_time
    
    def __repr__(self):
        switch_str = f" [SWITCH +{self.switch_cycles} cycles]" if self.is_switch else ""
        return f"Step({self.operator_id}@[SC{self.subcube}] {self.start_time}-{self.end_time}{switch_str})"


class DependencyGraph:
    """Manages operator dependencies and scheduling constraints"""
    
    def __init__(self):
        self.nodes: Dict[str, Operator] = {}
        self.edges: List[Tuple[str, str]] = []  # (predecessor, successor)
        self.adj: Dict[str, List[str]] = defaultdict(list)
        self.rev_adj: Dict[str, List[str]] = defaultdict(list)
    
    def add_operator(self, op: Operator):
        self.nodes[op.name] = op
    
    def add_edge(self, pred: str, succ: str):
        self.edges.append((pred, succ))
        self.adj[pred].append(succ)
        self.rev_adj[succ].append(pred)
    
    def topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological sorting"""
        in_degree = {node: 0 for node in self.nodes}
        for pred, succ in self.edges:
            in_degree[succ] += 1
        
        queue = [node for node, deg in in_degree.items() if deg == 0]
        heapq.heapify(queue)  # Use heap for deterministic ordering
        result = []
        
        while queue:
            node = heapq.heappop(queue)
            result.append(node)
            for succ in self.adj[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    heapq.heappush(queue, succ)
        
        return result
    
    def get_critical_path(self) -> List[str]:
        """Find the critical path (longest path in DAG)"""
        topo_order = self.topological_sort()
        
        # Calculate longest path to each node
        dist = {node: 0 for node in self.nodes}
        predecessor = {node: None for node in self.nodes}
        
        for node in topo_order:
            for succ in self.adj[node]:
                new_dist = dist[node] + 1  # Assume unit weight
                if new_dist > dist[succ]:
                    dist[succ] = new_dist
                    predecessor[succ] = node
        
        # Backtrack from node with maximum distance
        end_node = max(dist, key=dist.get)
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessor[current]
        
        return list(reversed(path))
    
    def get_predecessors(self, node: str) -> List[str]:
        return self.rev_adj.get(node, [])
    
    def get_successors(self, node: str) -> List[str]:
        return self.adj.get(node, [])


class PipelineScheduler:
    """
    Advanced pipeline scheduler supporting:
    1. Basic topological scheduling with barrier sync
    2. Pipeline parallelism across layers
    3. Data parallelism with replicated weights
    """
    
    def __init__(self, num_subcubes: int, subcube_depth: int):
        self.num_subcubes = num_subcubes
        self.subcube_depth = subcube_depth
    
    def schedule_basic(self, graph: DependencyGraph,
                       placement: Dict[str, Tuple[int, int]]) -> List[ScheduledStep]:
        """
        Basic scheduling: topological order with barrier synchronization
        """
        topo_order = graph.topological_sort()
        schedule = []
        end_times = {}
        subcube_last_end = {}  # Track when each subcube becomes free
        
        for op_name in topo_order:
            op = graph.nodes[op_name]
            
            # Barrier sync: wait for all predecessors
            pred_ends = [end_times[pred] for pred in graph.get_predecessors(op_name) 
                        if pred in end_times]
            ready_time = max(pred_ends) if pred_ends else 0
            
            # Get placement
            if op_name in placement:
                sc_id, z_start = placement[op_name]
            else:
                sc_id = 0
                z_start = 0
            
            # Switching penalty
            switch_cycles = 0
            if sc_id in subcube_last_end:
                last_op_end = subcube_last_end[sc_id]
                # If there was a previous operation, add switching penalty
                if last_op_end > 0:
                    switch_cycles = self.subcube_depth
            
            # Execution time: 1 + depth cycles
            exec_time = 1 + self.subcube_depth
            
            # Start time: max of ready time and subcube availability + switch
            start_time = max(ready_time, 
                           subcube_last_end.get(sc_id, 0) + switch_cycles)
            end_time = start_time + exec_time
            
            # Add switch step if needed
            if switch_cycles > 0:
                schedule.append(ScheduledStep(
                    operator_id=f"SWITCH_{op_name}",
                    start_time=subcube_last_end.get(sc_id, 0),
                    end_time=start_time,
                    subcube_id=sc_id,
                    is_switch=True,
                    switch_cycles=switch_cycles
                ))
            
            schedule.append(ScheduledStep(
                operator_id=op_name,
                start_time=start_time,
                end_time=end_time,
                subcube_id=sc_id
            ))
            
            end_times[op_name] = end_time
            subcube_last_end[sc_id] = end_time
        
        return schedule
    
    def schedule_pipeline(self, graph: DependencyGraph,
                          placement: Dict[str, Tuple[int, int]],
                          pipeline_stages: int = 2) -> List[ScheduledStep]:
        """
        Pipeline parallelism: operators execute as soon as their predecessors complete
        No barrier sync - maximizes parallelism across different paths in DAG
        """
        topo_order = graph.topological_sort()
        schedule = []
        end_times = {}
        subcube_last_end = {}

        for op_name in topo_order:
            op = graph.nodes[op_name]

            # NO barrier sync: only wait for immediate predecessor to finish
            pred_ends = []
            for pred in graph.get_predecessors(op_name):
                if pred in end_times:
                    pred_ends.append(end_times[pred])

            ready_time = max(pred_ends) if pred_ends else 0

            sc_id, z_start = placement.get(op_name, (0, 0))

            # Reduced switching penalty for pipeline mode
            switch_cycles = 0
            if sc_id in subcube_last_end and subcube_last_end[sc_id] > ready_time:
                switch_cycles = self.subcube_depth // 2

            exec_time = 1 + self.subcube_depth
            start_time = max(ready_time,
                           subcube_last_end.get(sc_id, 0) + switch_cycles)
            end_time = start_time + exec_time

            if switch_cycles > 0:
                schedule.append(ScheduledStep(
                    operator_id=f"SWITCH_{op_name}",
                    start_time=subcube_last_end.get(sc_id, 0),
                    end_time=start_time,
                    subcube_id=sc_id,
                    is_switch=True,
                    switch_cycles=switch_cycles
                ))

            schedule.append(ScheduledStep(
                operator_id=op_name,
                start_time=start_time,
                end_time=end_time,
                subcube_id=sc_id
            ))

            end_times[op_name] = end_time
            subcube_last_end[sc_id] = end_time

        return schedule
    
    def schedule_data_parallel(self, graph: DependencyGraph,
                                placement: Dict[str, Tuple[int, int]],
                                num_replicas: int = 2) -> List[ScheduledStep]:
        """
        Data parallelism: execute multiple inference requests in parallel
        using replicated weights
        """
        all_schedules = []
        
        for replica_id in range(num_replicas):
            # Offset each replica's execution
            replica_schedule = self.schedule_basic(graph, placement)
            
            # Adjust times for this replica
            offset = replica_id * 100  # Stagger start times
            for step in replica_schedule:
                step.start_time += offset
                step.end_time += offset
            
            all_schedules.extend(replica_schedule)
        
        # Sort by start time
        all_schedules.sort(key=lambda s: s.start_time)
        return all_schedules
    
    def _assign_pipeline_stages(self, graph: DependencyGraph,
                                 num_stages: int) -> Dict[int, List[str]]:
        """Assign operators to pipeline stages based on topological level"""
        topo_order = graph.topological_sort()
        levels = {}
        
        for op_name in topo_order:
            preds = graph.get_predecessors(op_name)
            if not preds:
                levels[op_name] = 0
            else:
                levels[op_name] = max(levels.get(p, 0) for p in preds) + 1
        
        # Map levels to stages
        max_level = max(levels.values()) if levels else 0
        stages = defaultdict(list)
        
        for op_name, level in levels.items():
            stage = min(int(level * num_stages / (max_level + 1)), num_stages - 1)
            stages[stage].append(op_name)
        
        return dict(stages)
    
    def analyze_schedule(self, schedule: List[ScheduledStep]) -> Dict:
        """Analyze schedule quality"""
        if not schedule:
            return {'total_time': 0, 'compute_time': 0, 'switch_time': 0, 'utilization': 0}
        
        total_time = max(s.end_time for s in schedule)
        compute_steps = [s for s in schedule if not s.is_switch]
        switch_steps = [s for s in schedule if s.is_switch]
        
        compute_time = sum(s.duration() for s in compute_steps)
        switch_time = sum(s.duration() for s in switch_steps)
        
        # Subcube utilization
        subcube_active_time = defaultdict(int)
        for step in schedule:
            subcube_active_time[step.subcube_id] += step.duration()
        
        max_possible = total_time * len(subcube_active_time)
        actual = sum(subcube_active_time.values())
        utilization = actual / max_possible if max_possible > 0 else 0
        
        return {
            'total_time': total_time,
            'compute_time': compute_time,
            'switch_time': switch_time,
            'switch_overhead_ratio': switch_time / total_time if total_time > 0 else 0,
            'subcube_utilization': utilization,
            'num_steps': len(schedule),
            'num_switches': len(switch_steps)
        }
