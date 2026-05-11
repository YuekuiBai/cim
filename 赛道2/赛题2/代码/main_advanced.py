"""
Main entry point for advanced CIM compiler with optimization
Integrates all advanced algorithms:
- Expert grouping (DSatur + SA)
- Advanced mapping (ILP-based + SA)
- Pipeline scheduling
- Comprehensive analysis
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_parser.model_loader import ModelLoader
from model_parser.weight_partitioner import WeightPartitioner
from model_parser.trace_analyzer import ActivationTraceAnalyzer
from optimizer.expert_grouping import HybridExpertGrouper
from optimizer.advanced_mapper import ILPBasedMapper, WeightCube
from scheduler.advanced_scheduler import DependencyGraph, Operator, PipelineScheduler


def run_advanced_compiler(model_path: str,
                          trace_path: str = None,
                          output_dir: str = "../results/advanced",
                          num_subcubes: int = 4,
                          subcube_hw: int = 8192,
                          max_depth: int = 64,
                          use_optimization: bool = True):
    """
    Run the advanced CIM compiler pipeline
    
    Args:
        model_path: Path to model JSON file
        trace_path: Path to activation trace JSON (for MoE models)
        output_dir: Output directory for results
        num_subcubes: Number of Sub-Cubes (N×N)
        subcube_hw: Sub-Cube 2D dimensions (H×W)
        max_depth: Maximum depth (D axis)
        use_optimization: Whether to use advanced optimization
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Advanced CIM Compiler for 3D Resource Space")
    print("=" * 70)
    
    # Step 1: Parse model
    print("\n[Step 1] Parsing model...")
    loader = ModelLoader()
    operators = loader.load_json(model_path)
    print(f"  Loaded {len(operators)} operators")
    
    # Save parsed operators
    parsed_ops = [
        {
            'name': op.name,
            'op_type': op.op_type,
            'weight_shape': op.weight_shape,
            'dependencies': op.dependencies,
            'is_sparse': op.is_sparse,
            'expert_id': op.expert_id,
            'is_shared_expert': op.is_shared_expert,
        }
        for op in operators
    ]
    with open(os.path.join(output_dir, "parsed_operators.json"), 'w') as f:
        json.dump(parsed_ops, f, indent=2)
    
    # Step 2: Analyze activation traces (if available)
    trace_analysis = None
    expert_assignment = None
    
    if trace_path and os.path.exists(trace_path):
        print("\n[Step 2] Analyzing activation traces...")
        analyzer = ActivationTraceAnalyzer(trace_path)
        analyzer.analyze()
        print(analyzer.report())
        
        # Save trace analysis
        trace_result = {
            'cooccurrence_matrix': analyzer.cooccurrence.tolist() if analyzer.cooccurrence is not None else None,
            'activation_frequency': analyzer.activation_freq.tolist() if analyzer.activation_freq is not None else None,
            'mutex_pairs': analyzer.mutex_pairs,
            'expert_groups': analyzer.get_expert_groups()
        }
        with open(os.path.join(output_dir, "trace_analysis.json"), 'w') as f:
            json.dump(trace_result, f, indent=2, default=str)
        
        trace_analysis = analyzer
        
        # Step 3: Expert grouping (if optimization enabled)
        if use_optimization:
            print("\n[Step 3] Optimizing expert grouping...")
            
            cooccurrence = analyzer.cooccurrence
            freq = analyzer.activation_freq
            
            grouper = HybridExpertGrouper(
                cooccurrence_matrix=cooccurrence,
                activation_freq=freq,
                num_subcubes=num_subcubes,
                shared_experts=analyzer.shared_experts
            )
            
            grouping_result = grouper.optimize(use_sa=True, sa_iterations=5000)
            expert_assignment = grouping_result['assignment']
            
            print(f"  Method: {grouping_result['method']}")
            print(f"  Cost: {grouping_result['cost']:.2f}")
            
            # Save grouping result
            with open(os.path.join(output_dir, "expert_grouping.json"), 'w') as f:
                json.dump({
                    'assignment': {str(k): v for k, v in expert_assignment.items()},
                    'method': grouping_result['method'],
                    'cost': grouping_result['cost']
                }, f, indent=2)
    
    # Step 4: Partition weights
    print("\n[Step 4] Partitioning weights...")
    partitioner = WeightPartitioner(subcube_hw, max_depth)
    sections = partitioner.partition(operators)
    print(f"  Generated {len(sections)} weight sections")
    
    # Save sections
    sections_data = [
        {
            'name': s.name,
            'shape_2d': s.shape_2d,
            'depth': s.depth,
            'parent_operator': s.parent_operator,
            'section_idx': s.section_idx,
            'total_sections': s.total_sections,
            'is_replicated': s.is_replicated,
            'replication_idx': s.replication_idx,
        }
        for s in sections
    ]
    with open(os.path.join(output_dir, "weight_sections.json"), 'w') as f:
        json.dump(sections_data, f, indent=2)
    
    # Step 5: Advanced 3D mapping
    print("\n[Step 5] Mapping to 3D resource space...")
    
    # Convert sections to WeightCubes
    cubes = []
    for section in sections:
        # Extract expert_id from operator name if present
        expert_id = None
        freq_val = 0.0
        
        if trace_analysis and trace_analysis.activation_freq is not None:
            # Try to match operator to expert
            for op in operators:
                if op.name == section.parent_operator:
                    if hasattr(op, 'expert_id') and op.expert_id is not None:
                        expert_id = op.expert_id
                        if expert_id < len(trace_analysis.activation_freq):
                            freq_val = float(trace_analysis.activation_freq[expert_id])
                        break
        
        cube = WeightCube(
            name=section.name,
            rows=section.shape_2d[0],
            cols=section.shape_2d[1],
            depth=section.depth,
            frequency=freq_val,
            expert_id=expert_id
        )
        cubes.append(cube)
    
    # Run mapping
    mapper = ILPBasedMapper(
        num_subcubes=num_subcubes,
        subcube_hw=subcube_hw,
        max_depth=max_depth,
        cooccurrence_matrix=trace_analysis.cooccurrence if trace_analysis else None,
        activation_freq=trace_analysis.activation_freq if trace_analysis else None
    )
    
    placement = mapper.map(cubes, expert_assignment)
    
    stats = mapper.get_statistics()
    print(f"  Space utilization: {stats['space_utilization']:.2%}")
    print(f"  Avg depth utilization: {stats['avg_depth_utilization']:.2%}")
    
    # Save placement
    placement_data = {
        cube_name: {'subcube_id': sc_id, 'z_start': z}
        for cube_name, (sc_id, z) in placement.items()
    }
    with open(os.path.join(output_dir, "placement.json"), 'w') as f:
        json.dump(placement_data, f, indent=2)
    
    # Step 6: Build dependency graph and schedule
    print("\n[Step 6] Scheduling...")
    
    dep_graph = DependencyGraph()
    for op in operators:
        dep_op = Operator(
            name=op.name,
            op_type=op.op_type if hasattr(op, 'op_type') else 'unknown',
            weight_shape=op.weight_shape,
            predecessors=op.predecessors if hasattr(op, 'predecessors') else [],
            expert_id=op.expert_id if hasattr(op, 'expert_id') else None,
            is_shared=op.is_shared if hasattr(op, 'is_shared') else False
        )
        dep_graph.add_operator(dep_op)
        
        for pred in (op.predecessors if hasattr(op, 'predecessors') else []):
            dep_graph.add_edge(pred, op.name)
    
    # Convert placement to operator-level
    op_placement = {}
    for cube_name, (sc_id, z) in placement.items():
        # Extract operator name from cube name
        op_name = cube_name.rsplit('_sec_', 1)[0]
        if op_name not in op_placement:
            op_placement[op_name] = (sc_id, z)
    
    scheduler = PipelineScheduler(num_subcubes, max_depth)
    
    # Basic scheduling
    print("  Running basic scheduling...")
    basic_schedule = scheduler.schedule_basic(dep_graph, op_placement)
    basic_stats = scheduler.analyze_schedule(basic_schedule)
    print(f"    Total time: {basic_stats['total_time']} cycles")
    print(f"    Switch overhead: {basic_stats['switch_overhead_ratio']:.2%}")
    
    # Pipeline scheduling
    print("  Running pipeline scheduling...")
    pipeline_schedule = scheduler.schedule_pipeline(dep_graph, op_placement, pipeline_stages=2)
    pipeline_stats = scheduler.analyze_schedule(pipeline_schedule)
    print(f"    Total time: {pipeline_stats['total_time']} cycles")
    print(f"    Switch overhead: {pipeline_stats['switch_overhead_ratio']:.2%}")
    
    # Save schedules
    basic_schedule_data = [
        {
            'operator_id': s.operator_id,
            'start_time': s.start_time,
            'end_time': s.end_time,
            'subcube_id': s.subcube_id,
            'is_switch': s.is_switch
        }
        for s in basic_schedule
    ]
    
    pipeline_schedule_data = [
        {
            'operator_id': s.operator_id,
            'start_time': s.start_time,
            'end_time': s.end_time,
            'subcube_id': s.subcube_id,
            'is_switch': s.is_switch
        }
        for s in pipeline_schedule
    ]
    
    with open(os.path.join(output_dir, "schedule_basic.json"), 'w') as f:
        json.dump({'schedule': basic_schedule_data, 'stats': basic_stats}, f, indent=2)
    
    with open(os.path.join(output_dir, "schedule_pipeline.json"), 'w') as f:
        json.dump({'schedule': pipeline_schedule_data, 'stats': pipeline_stats}, f, indent=2)
    
    # Step 7: Generate comprehensive solution
    print("\n[Step 7] Generating solution...")
    
    solution = {
        'model': model_path,
        'hardware_config': {
            'num_subcubes': num_subcubes,
            'subcube_hw': subcube_hw,
            'max_depth': max_depth
        },
        'placement': placement_data,
        'schedule': {
            'basic': basic_schedule_data,
            'pipeline': pipeline_schedule_data
        },
        'statistics': {
            'basic': basic_stats,
            'pipeline': pipeline_stats,
            'mapping': stats
        }
    }
    
    with open(os.path.join(output_dir, "solution.json"), 'w') as f:
        json.dump(solution, f, indent=2, default=str)
    
    print(f"\n  Solution saved to: {output_dir}/solution.json")
    
    # Summary
    print("\n" + "=" * 70)
    print("Compilation Summary")
    print("=" * 70)
    print(f"  Operators: {len(operators)}")
    print(f"  Weight sections: {len(sections)}")
    print(f"  Sub-Cubes used: {num_subcubes}")
    print(f"  Space utilization: {stats['space_utilization']:.2%}")
    print(f"  Basic schedule time: {basic_stats['total_time']} cycles")
    print(f"  Pipeline schedule time: {pipeline_stats['total_time']} cycles")
    print(f"  Speedup (pipeline vs basic): {basic_stats['total_time'] / max(pipeline_stats['total_time'], 1):.2f}x")
    print("=" * 70)
    
    return solution


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced CIM Compiler")
    parser.add_argument("--model", required=True, help="Path to model JSON file")
    parser.add_argument("--trace", help="Path to activation trace JSON")
    parser.add_argument("--output", default="../results/advanced", help="Output directory")
    parser.add_argument("--num-subcubes", type=int, default=4, help="Number of Sub-Cubes")
    parser.add_argument("--subcube-hw", type=int, default=8192, help="Sub-Cube 2D size")
    parser.add_argument("--max-depth", type=int, default=64, help="Maximum depth")
    parser.add_argument("--no-optimize", action="store_true", help="Disable optimization")
    
    args = parser.parse_args()
    
    run_advanced_compiler(
        model_path=args.model,
        trace_path=args.trace,
        output_dir=args.output,
        num_subcubes=args.num_subcubes,
        subcube_hw=args.subcube_hw,
        max_depth=args.max_depth,
        use_optimization=not args.no_optimize
    )
