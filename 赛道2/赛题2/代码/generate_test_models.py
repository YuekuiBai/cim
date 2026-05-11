"""
Test model generator for Problem 2
Generates model.json and activation_trace.json for testing
"""

import json
import os
import random
import numpy as np


def generate_simple_model(output_dir="test_models"):
    """Simple 3-layer model: Linear1 -> Linear2 -> Linear3"""
    os.makedirs(output_dir, exist_ok=True)

    model = {
        "name": "simple_3layer",
        "is_moe": False,
        "total_params": 0,
        "operators": [
            {
                "name": "linear_1",
                "op_type": "MatMul",
                "weight_shape": [512, 256],
                "dependencies": [],
            },
            {
                "name": "linear_2",
                "op_type": "MatMul",
                "weight_shape": [256, 128],
                "dependencies": ["linear_1"],
            },
            {
                "name": "linear_3",
                "op_type": "MatMul",
                "weight_shape": [128, 64],
                "dependencies": ["linear_2"],
            },
        ],
    }

    total = sum(
        op["weight_shape"][0] * op["weight_shape"][1]
        for op in model["operators"]
    )
    model["total_params"] = total

    with open(os.path.join(output_dir, "simple_model.json"), 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Created simple_model.json ({total:,} params)")


def generate_moe_model(output_dir="test_models"):
    """MoE model with shared + routed experts (256 experts, top-8)"""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    num_experts = 256
    top_k = 8
    shared_experts = [0]
    n_inferences = 10000
    expert_dim = 1024
    hidden_dim = 4096

    operators = []

    # Shared attention layer
    operators.append({
        "name": "shared_attention",
        "op_type": "MatMul",
        "weight_shape": [hidden_dim, hidden_dim],
        "dependencies": [],
    })

    # MoE experts (256 routed experts + 1 shared)
    for i in range(num_experts):
        is_shared = i in shared_experts
        operators.append({
            "name": f"expert_{i}",
            "op_type": "MoE",
            "weight_shape": [hidden_dim, expert_dim],
            "dependencies": ["shared_attention"],
            "is_sparse": not is_shared,
            "expert_id": i,
            "is_shared_expert": is_shared,
        })

    # Shared MLP after experts
    operators.append({
        "name": "shared_mlp",
        "op_type": "MatMul",
        "weight_shape": [hidden_dim, hidden_dim],
        "dependencies": [f"expert_{i}" for i in range(num_experts)],
    })

    total_params = sum(
        op["weight_shape"][0] * op["weight_shape"][1]
        for op in operators
    )

    model = {
        "name": "moe_256experts",
        "is_moe": True,
        "num_experts": num_experts,
        "top_k": top_k,
        "shared_experts": shared_experts,
        "total_params": total_params,
        "operators": operators,
    }

    with open(os.path.join(output_dir, "moe_model.json"), 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Created moe_model.json ({total_params:,} params, {num_experts} experts)")

    # Generate activation traces
    traces = []
    non_shared = [i for i in range(num_experts) if i not in shared_experts]
    for _ in range(n_inferences):
        chosen = random.sample(non_shared, top_k - len(shared_experts))
        trace = sorted(shared_experts + chosen)
        traces.append(trace)

    trace_data = {
        "num_experts": num_experts,
        "top_k": top_k,
        "shared_experts": shared_experts,
        "n_inferences": n_inferences,
        "traces": traces,
    }

    with open(os.path.join(output_dir, "activation_trace.json"), 'w') as f:
        json.dump(trace_data, f, indent=2)
    print(f"Created activation_trace.json ({n_inferences} traces)")


def generate_all(output_dir="test_models"):
    generate_simple_model(output_dir)
    generate_moe_model(output_dir)
    print(f"\nAll test models generated in: {output_dir}/")


if __name__ == "__main__":
    generate_all()
