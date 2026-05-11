"""
Calculate real space utilization from placement solution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np

def calculate_utilization():
    """Calculate real space utilization from solution"""
    solution_path = '../结果/output_moe_advanced/solution.json'

    with open(solution_path, 'r') as f:
        solution = json.load(f)

    placement = solution['placement']
    hw = solution['hardware_config']['subcube_hw']
    max_depth = solution['hardware_config']['max_depth']

    num_subcubes = solution['hardware_config']['num_subcubes']

    # Track usage per subcube
    subcube_depth_used = {i: 0 for i in range(num_subcubes)}
    subcube_area_used = {i: 0 for i in range(num_subcubes)}

    # Assume each weight cube has shape (rows, cols, depth)
    # For MoE experts, typically (512, 512, depth) or similar
    # We'll estimate based on the number of placements

    for cube_name, info in placement.items():
        sc_id = info['subcube_id']
        z_start = info['z_start']

        # Estimate weight volume (simplified model)
        # Each expert weight is roughly 512x512xD
        rows, cols = 512, 512
        depth = 1  # z-axis depth for this weight

        # Update usage
        subcube_area_used[sc_id] += rows * cols
        subcube_depth_used[sc_id] = max(subcube_depth_used[sc_id], z_start + depth)

    # Calculate utilization
    max_area = hw * hw
    max_depth_per_sc = max_depth

    depth_utils = []
    area_utils = []

    for i in range(num_subcubes):
        depth_util = subcube_depth_used[i] / max_depth_per_sc if max_depth_per_sc > 0 else 0
        area_util = subcube_area_used[i] / max_area if max_area > 0 else 0
        depth_utils.append(min(1.0, depth_util))
        area_utils.append(min(1.0, area_util))

    # Calculate overall
    avg_depth = np.mean(depth_utils)
    avg_area = np.mean(area_utils)

    print("Sub-Cube Utilization:")
    for i in range(num_subcubes):
        print(f"  SC{i}: Depth={depth_utils[i]:.2%}, Area={area_utils[i]:.2%}")

    print(f"\nOverall:")
    print(f"  Avg Depth: {avg_depth:.2%}")
    print(f"  Avg Area: {avg_area:.2%}")
    print(f"  Space Utilization: {(avg_depth + avg_area) / 2:.2%}")

    # Save to solution
    solution['utilization'] = {
        'subcube_depth_utils': depth_utils,
        'subcube_area_utils': area_utils,
        'avg_depth_util': avg_depth,
        'avg_area_util': avg_area
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"\nSaved to {solution_path}")

if __name__ == '__main__':
    calculate_utilization()