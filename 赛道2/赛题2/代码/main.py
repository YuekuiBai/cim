"""
CIM 3D Mapper - Main Entry Point (Track 2 Problem 2)
Maps neural network models to 3D CIM hardware space
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_parser.model_loader import ModelParser
from model_parser.weight_partitioner import WeightPartitioner
from model_parser.trace_analyzer import ActivationTraceAnalyzer
from compiler_mapping.weight_cube import CubeConfig
from compiler_mapping.mapper import Mapper3D
from scheduler.pipeline_scheduler import PipelineScheduler
from simulator.hardware_sim import HardwareSimulator, SolutionValidator


def compile_model(model_path, trace_path=None, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CIM 3D Mapper - Track 2 Problem 2")
    print("=" * 60)

    # Step 1: Parse model
    print("\n[Step 1] Parsing model...")
    parser = ModelParser(model_path)
    operators = parser.parse()
    print(f"  Operators: {len(operators)}")
    print(f"  Total params: {parser.total_params:,}")
    print(f"  Is MoE: {parser.is_moe}")

    parsed_path = os.path.join(output_dir, "parsed_operators.json")
    parsed_data = {
        "operators": [
            {
                "name": op.name,
                "op_type": op.op_type,
                "weight_shape": op.weight_shape,
                "params": op.num_params,
                "dependencies": op.dependencies,
                "is_sparse": op.is_sparse,
            }
            for op in operators
        ],
        "total_params": parser.total_params,
    }
    with open(parsed_path, 'w') as f:
        json.dump(parsed_data, f, indent=2)

    # Step 2: Analyze activation trace (MoE)
    analyzer = None
    if trace_path and os.path.exists(trace_path):
        print("\n[Step 2] Analyzing activation trace...")
        analyzer = ActivationTraceAnalyzer(trace_path)
        analyzer.analyze()
        print(analyzer.report())

        trace_report_path = os.path.join(output_dir, "trace_analysis.json")
        trace_report = {
            "mutex_pairs_count": len(analyzer.mutex_pairs),
            "top5_active": np.argsort(analyzer.activation_freq)[-5:][::-1].tolist(),
        }
        with open(trace_report_path, 'w') as f:
            json.dump(trace_report, f, indent=2)

    # Step 3: Configure 3D space
    print("\n[Step 3] Configuring 3D resource space...")
    total_params = parser.total_params
    hw = 4096
    n = 2
    min_depth = (total_params + (n * n * hw * hw - 1)) // (n * n * hw * hw)
    depth = max(64, min_depth + 16)  # Extra headroom for fragmentation

    config = CubeConfig(n=n, hw=hw, depth=depth)
    assert config.validate(), "Invalid config"
    print(f"  Config: {n}x{n} Sub-Cubes, HW={hw}, D={depth}")
    print(f"  Total capacity: {config.total_volume:,} cells")
    print(f"  Model size: {total_params:,} params ({total_params/config.total_volume*100:.1f}%)")

    # Step 4: Partition weights
    print("\n[Step 4] Partitioning weights...")
    partitioner = WeightPartitioner(subcube_hw=hw, depth=depth)
    sections = partitioner.partition(operators)
    print(f"  Sections created: {len(sections)}")

    sections_path = os.path.join(output_dir, "weight_sections.json")
    sections_data = [
        {
            "name": s.name,
            "shape_2d": s.shape_2d,
            "depth": s.depth,
            "parent": s.parent_operator,
            "section_idx": s.section_idx,
            "params": s.num_params,
        }
        for s in sections
    ]
    with open(sections_path, 'w') as f:
        json.dump(sections_data, f, indent=2)

    # Step 5: 3D Mapping
    print("\n[Step 5] Mapping to 3D space...")
    mapper = Mapper3D(config)
    solution = mapper.map(sections, analyzer)
    print(solution.report())

    # Step 6: Scheduling
    print("\n[Step 6] Scheduling execution...")
    scheduler = PipelineScheduler(solution, operators)
    schedule = scheduler.schedule()
    print(f"  Schedule entries: {len(schedule)}")

    # Step 7: Validation
    print("\n[Step 7] Validating solution...")
    validator = SolutionValidator(solution)
    details = validator.validate_details()
    all_ok = all(details.values())
    print(f"  Validation: {'PASSED' if all_ok else 'FAILED'}")
    for k, v in details.items():
        print(f"    {k}: {'OK' if v else 'FAIL'}")

    # Step 8: Simulation
    print("\n[Step 8] Simulating execution...")
    simulator = HardwareSimulator(solution)
    sim_report = simulator.simulate_detailed()
    print(f"  End-to-end latency: {sim_report['total_latency']:,} cycles")

    # Step 9: Save solution
    solution.total_latency = sim_report['total_latency']
    sol_path = os.path.join(output_dir, "solution.json")
    solution.save(sol_path)
    print(f"\n  Solution saved to: {sol_path}")

    print("\n" + "=" * 60)
    print("Compilation completed!")
    print(f"  Latency: {sim_report['total_latency']:,} cycles")
    print(f"  Utilization: {solution.space_utilization*100:.1f}%")
    print("=" * 60)

    return solution


def main():
    p = argparse.ArgumentParser(description="CIM 3D Mapper - Problem 2")
    p.add_argument("--model", required=True, help="Path to model JSON")
    p.add_argument("--trace", default=None, help="Activation trace for MoE")
    p.add_argument("--output", default="output", help="Output directory")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    compile_model(args.model, args.trace, args.output)


if __name__ == "__main__":
    main()
