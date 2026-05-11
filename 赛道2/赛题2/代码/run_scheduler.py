"""
Run Pipeline Scheduler and generate statistics for visualization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from scheduler.advanced_scheduler import PipelineScheduler, DependencyGraph, Operator

def load_placement(path: str) -> dict:
    """Load and convert placement to scheduler format"""
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert to {cube_name: (subcube_id, z_start)} format
    converted = {}
    for cube_name, info in data['placement'].items():
        converted[cube_name] = (info['subcube_id'], info['z_start'])
    return converted

def create_dependency_graph(placement: dict) -> tuple:
    """Create a dependency graph from placement - models MoE layer execution"""
    graph = DependencyGraph()

    subcube_ops = {}

    def get_sc_id(info):
        if isinstance(info, tuple):
            return info[0]
        return info['subcube_id']

    # Group by subcube
    for cube_name, info in placement.items():
        sc_id = get_sc_id(info)
        if sc_id not in subcube_ops:
            subcube_ops[sc_id] = []
        subcube_ops[sc_id].append(cube_name)

    # Add all operators
    for cube_name in placement.keys():
        parts = cube_name.split('_sec_')
        expert_name = parts[0]
        try:
            expert_id = int(expert_name.split('_')[1])
        except:
            expert_id = 0
        op = Operator(
            name=cube_name,
            op_type='Linear',
            weight_shape=(512, 512),
            expert_id=expert_id
        )
        graph.add_operator(op)

    # Create dependency graph for pipeline parallelism
    # Within each subcube: sequential dependencies (experts in same SC must be sequential)
    # Across subcubes: no dependencies (different SCs can run in parallel)
    for sc_id, ops in subcube_ops.items():
        for i in range(len(ops) - 1):
            graph.add_edge(ops[i], ops[i + 1])

    return graph, subcube_ops

def run_scheduler():
    """Run both basic and pipeline scheduling"""
    # Load placement
    placement_path = '../结果/output_moe_advanced/solution.json'
    placement = load_placement(placement_path)

    # Create dependency graph
    graph, subcube_ops = create_dependency_graph(placement)

    # Create scheduler
    scheduler = PipelineScheduler(num_subcubes=4, subcube_depth=64)

    # Basic scheduling
    basic_schedule = scheduler.schedule_basic(graph, placement)

    # Pipeline scheduling
    pipeline_schedule = scheduler.schedule_pipeline(graph, placement)

    # Calculate statistics
    def calc_stats(schedule):
        total_time = max(s.end_time for s in schedule) if schedule else 0
        compute_time = sum(s.duration() for s in schedule if not s.is_switch)
        switch_time = sum(s.switch_cycles for s in schedule if s.is_switch)

        return {
            'total_time': total_time,
            'compute_time': compute_time,
            'switch_time': switch_time
        }

    basic_stats = calc_stats(basic_schedule)
    pipeline_stats = calc_stats(pipeline_schedule)

    # Print results
    print("Basic Scheduling:")
    print(f"  Total: {basic_stats['total_time']}")
    print(f"  Compute: {basic_stats['compute_time']}")
    print(f"  Switch: {basic_stats['switch_time']}")

    print("\nPipeline Scheduling:")
    print(f"  Total: {pipeline_stats['total_time']}")
    print(f"  Compute: {pipeline_stats['compute_time']}")
    print(f"  Switch: {pipeline_stats['switch_time']}")

    print(f"\nSpeedup: {basic_stats['total_time'] / pipeline_stats['total_time']:.2f}x")

    # Save to solution.json
    output_path = '../结果/output_moe_advanced/solution.json'
    with open(output_path, 'r') as f:
        solution = json.load(f)

    solution['statistics'] = {
        'basic': basic_stats,
        'pipeline': pipeline_stats
    }

    with open(output_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"\nStatistics saved to {output_path}")

if __name__ == '__main__':
    run_scheduler()