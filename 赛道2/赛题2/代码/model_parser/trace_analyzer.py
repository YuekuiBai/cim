"""
MoE Activation Trace Analyzer
Builds co-occurrence matrix, identifies mutex pairs, computes activation frequency
"""

import json
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class ActivationTraceAnalyzer:
    """Analyzes MoE activation traces for optimal placement"""

    def __init__(self, trace_path: str):
        with open(trace_path, 'r') as f:
            self.data = json.load(f)

        self.num_experts = self.data["num_experts"]
        self.top_k = self.data["top_k"]
        self.shared_experts = set(self.data.get("shared_experts", []))
        self.traces = self.data["traces"]
        self.n_inferences = self.data["n_inferences"]

        self.cooccurrence = None
        self.activation_freq = None
        self.mutex_pairs = None

    def analyze(self):
        """Run full analysis"""
        self._build_cooccurrence()
        self._compute_activation_frequency()
        self._find_mutex_pairs()

    def _build_cooccurrence(self):
        """Build expert co-occurrence matrix"""
        n = self.num_experts
        cooc = np.zeros((n, n), dtype=np.int64)

        for trace in self.traces:
            for i in range(len(trace)):
                for j in range(i + 1, len(trace)):
                    e1, e2 = trace[i], trace[j]
                    cooc[e1][e2] += 1
                    cooc[e2][e1] += 1

        self.cooccurrence = cooc

    def _compute_activation_frequency(self):
        """Compute how often each expert is activated"""
        n = self.num_experts
        freq = np.zeros(n, dtype=np.int64)

        for trace in self.traces:
            for e in trace:
                freq[e] += 1

        self.activation_freq = freq

    def _find_mutex_pairs(self) -> List[Tuple[int, int]]:
        """Find pairs of experts that never co-occur (can share Sub-Cube)"""
        pairs = []
        n = self.num_experts
        for i in range(n):
            for j in range(i + 1, n):
                if i in self.shared_experts or j in self.shared_experts:
                    continue
                if self.cooccurrence[i][j] == 0:
                    pairs.append((i, j))
        self.mutex_pairs = pairs
        return pairs

    def get_cooccurrence_matrix(self) -> np.ndarray:
        return self.cooccurrence

    def get_activation_frequency(self) -> np.ndarray:
        return self.activation_freq

    def get_mutex_pairs(self) -> List[Tuple[int, int]]:
        return self.mutex_pairs

    def get_expert_groups(self) -> List[List[int]]:
        """
        Group experts using greedy coloring on co-occurrence graph.
        Experts that can co-occur should be in different Sub-Cubes.
        Experts that never co-occur can share a Sub-Cube.
        """
        n = self.num_experts
        assigned = [-1] * n
        group_id = 0

        # Shared experts get their own group (always active)
        for e in self.shared_experts:
            assigned[e] = -2  # special marker

        # Greedy: for unassigned experts, find mutex-compatible groups
        expert_list = [i for i in range(n) if assigned[i] == -1]
        # Sort by activation frequency descending
        expert_list.sort(key=lambda e: self.activation_freq[e], reverse=True)

        groups = defaultdict(list)

        for expert in expert_list:
            placed = False
            for gid in range(group_id):
                can_place = True
                for other in groups[gid]:
                    if self.cooccurrence[expert][other] > 0:
                        can_place = False
                        break
                if can_place:
                    groups[gid].append(expert)
                    assigned[expert] = gid
                    placed = True
                    break
            if not placed:
                groups[group_id].append(expert)
                assigned[expert] = group_id
                group_id += 1

        # Add shared experts as special group
        shared_group = list(self.shared_experts)

        result_groups = [shared_group] if shared_group else []
        for gid in range(group_id):
            if groups[gid]:
                result_groups.append(groups[gid])

        return result_groups

    def report(self) -> str:
        lines = [
            "=== MoE Activation Trace Analysis ===",
            f"  Experts: {self.num_experts}, Top-K: {self.top_k}",
            f"  Traces: {self.n_inferences}",
            f"  Shared experts: {sorted(self.shared_experts)}",
            f"  Mutex pairs: {len(self.mutex_pairs)}",
            f"  Expert groups: {len(self.get_expert_groups())}",
        ]
        if self.activation_freq is not None:
            top5 = np.argsort(self.activation_freq)[-5:][::-1]
            lines.append("  Top-5 most active experts:")
            for e in top5:
                lines.append(f"    Expert {e}: {self.activation_freq[e]} activations")
        return "\n".join(lines)
