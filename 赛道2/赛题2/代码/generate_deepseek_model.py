"""
Generate DeepSeek-671B simplified model for testing
- 256 experts with fine-grained MoE architecture
- Top-K=8 routing mechanism
- Shared experts
- Realistic activation traces
"""

import json
import numpy as np
import os
from typing import List, Dict


def generate_deepseek_671b_simplified(output_dir: str,
                                       num_experts: int = 256,
                                       num_layers: int = 64,
                                       expert_hidden: int = 1024,
                                       model_dim: int = 4096,
                                       top_k: int = 8,
                                       num_shared_experts: int = 1,
                                       n_inferences: int = 10000):
    """
    Generate a simplified DeepSeek-671B model description
    
    Architecture:
    - 64 MoE layers
    - 256 routed experts per layer
    - 1 shared expert per layer
    - Top-K=8 routing
    - Model dimension: 4096
    - Expert hidden dimension: 1024
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate model.json
    model_data = {
        "name": "DeepSeek-671B-Simplified",
        "total_parameters": 671_000_000_000,
        "architecture": {
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "num_shared_experts": num_shared_experts,
            "expert_hidden": expert_hidden
        },
        "operators": []
    }
    
    operator_id = 0
    
    for layer_idx in range(num_layers):
        # Shared expert (always activated)
        shared_expert = {
            "id": f"op_{operator_id}",
            "name": f"layer{layer_idx}_shared_expert",
            "type": "moe_expert",
            "weight_shape": [model_dim, expert_hidden],
            "bias_shape": [expert_hidden],
            "predecessors": [f"op_{operator_id - 1}"] if operator_id > 0 else [],
            "expert_id": 0,
            "is_shared": True,
            "layer": layer_idx
        }
        model_data["operators"].append(shared_expert)
        operator_id += 1
        
        # Routed experts (conditionally activated)
        for expert_idx in range(1, num_experts + 1):
            expert = {
                "id": f"op_{operator_id}",
                "name": f"layer{layer_idx}_expert{expert_idx}",
                "type": "moe_expert",
                "weight_shape": [model_dim, expert_hidden],
                "bias_shape": [expert_hidden],
                "predecessors": [f"op_{operator_id - 1}"] if operator_id > 0 else [],
                "expert_id": expert_idx,
                "is_shared": False,
                "layer": layer_idx
            }
            model_data["operators"].append(expert)
            operator_id += 1
        
        # Output projection
        output_proj = {
            "id": f"op_{operator_id}",
            "name": f"layer{layer_idx}_output_proj",
            "type": "linear",
            "weight_shape": [expert_hidden, model_dim],
            "bias_shape": [model_dim],
            "predecessors": [f"op_{operator_id - 1}"],
            "layer": layer_idx
        }
        model_data["operators"].append(output_proj)
        operator_id += 1
    
    # Save model.json
    model_path = os.path.join(output_dir, "deepseek_671b_simplified.json")
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"Generated model: {model_path}")
    print(f"  Total operators: {len(model_data['operators'])}")
    print(f"  Layers: {num_layers}")
    print(f"  Experts per layer: {num_experts + num_shared_experts}")
    
    # Generate activation_trace.json
    trace_data = generate_activation_traces(
        num_experts=num_experts + num_shared_experts,
        top_k=top_k,
        shared_experts=list(range(num_shared_experts)),
        n_inferences=n_inferences,
        layer_count=num_layers
    )
    
    trace_path = os.path.join(output_dir, "deepseek_activation_trace.json")
    with open(trace_path, 'w') as f:
        json.dump(trace_data, f, indent=2)
    
    print(f"Generated activation trace: {trace_path}")
    print(f"  Traces: {n_inferences}")
    print(f"  Top-K: {top_k}")
    
    return model_path, trace_path


def generate_activation_traces(num_experts: int,
                                top_k: int,
                                shared_experts: List[int],
                                n_inferences: int,
                                layer_count: int,
                                seed: int = 42) -> Dict:
    """
    Generate realistic MoE activation traces
    
    Simulates:
    - Shared experts always activated
    - Top-K routing with realistic distribution
    - Expert co-occurrence patterns
    """
    np.random.seed(seed)
    
    # Expert activation frequencies follow power law (some experts are "hot")
    frequencies = np.random.power(2.0, num_experts)
    frequencies = frequencies / frequencies.sum()
    
    # Ensure shared experts have highest frequency
    for e in shared_experts:
        frequencies[e] = frequencies.max() * 1.5
    
    traces = []
    routed_experts = [e for e in range(num_experts) if e not in shared_experts]
    
    for _ in range(n_inferences):
        # Always include shared experts
        activated = list(shared_experts)
        
        # Select top-k routed experts based on frequency
        k_routed = min(top_k, len(routed_experts))
        routed_selected = np.random.choice(
            routed_experts,
            size=k_routed,
            replace=False,
            p=frequencies[routed_experts] / frequencies[routed_experts].sum()
        )
        
        activated.extend(routed_selected.tolist())
        traces.append(sorted(activated))
    
    return {
        "num_experts": num_experts,
        "top_k": top_k,
        "shared_experts": shared_experts,
        "n_inferences": n_inferences,
        "traces": traces
    }


def generate_small_test_model(output_dir: str):
    """Generate a smaller test model for quick validation"""
    return generate_deepseek_671b_simplified(
        output_dir=output_dir,
        num_experts=32,
        num_layers=8,
        expert_hidden=256,
        model_dim=512,
        top_k=4,
        num_shared_experts=1,
        n_inferences=1000
    )


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "../test_models/deepseek_671b"
    
    print("=" * 60)
    print("DeepSeek-671B Simplified Model Generator")
    print("=" * 60)
    
    model_path, trace_path = generate_deepseek_671b_simplified(output_dir)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
