"""
Advanced Expert Grouping Algorithms
- Graph Coloring with DSatur
- Simulated Annealing based grouping
- Conflict-aware placement optimization
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import random
import math


class ExpertGraph:
    """Graph representation of expert co-occurrence relationships"""
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        self.weights: Dict[Tuple[int, int], int] = {}
    
    def add_edge(self, u: int, v: int, weight: int = 1):
        self.adjacency[u].add(v)
        self.adjacency[v].add(u)
        self.weights[(u, v)] = weight
        self.weights[(v, u)] = weight
    
    def get_degree(self, node: int) -> int:
        return len(self.adjacency[node])
    
    def get_saturation_degree(self, node: int, coloring: Dict[int, int]) -> int:
        """Number of different colors used by neighbors"""
        neighbor_colors = set()
        for neighbor in self.adjacency[node]:
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])
        return len(neighbor_colors)


class DSaturColoring:
    """
    DSatur (Degree of Saturation) graph coloring algorithm
    More effective than greedy coloring for expert grouping
    """
    
    def __init__(self, cooccurrence_matrix: np.ndarray, 
                 activation_freq: np.ndarray,
                 num_subcubes: int):
        self.cooccurrence = cooccurrence_matrix
        self.freq = activation_freq
        self.num_experts = cooccurrence_matrix.shape[0]
        self.num_subcubes = num_subcubes
        self.graph = self._build_graph()
    
    def _build_graph(self) -> ExpertGraph:
        """Build conflict graph from co-occurrence matrix"""
        graph = ExpertGraph(self.num_experts)
        
        # Add edges for experts that co-occur (conflict)
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                if self.cooccurrence[i][j] > 0:
                    graph.add_edge(i, j, int(self.cooccurrence[i][j]))
        
        return graph
    
    def color(self, shared_experts: Optional[Set[int]] = None) -> Dict[int, int]:
        """
        Apply DSatur algorithm to color the graph
        Returns: expert_id -> subcube_id mapping
        """
        coloring: Dict[int, int] = {}
        uncolored = set(range(self.num_experts))
        
        # Handle shared experts first - they get dedicated subcubes
        if shared_experts:
            for idx, expert in enumerate(shared_experts):
                coloring[expert] = idx
                uncolored.remove(expert)
        
        next_color = len(shared_experts) if shared_experts else 0
        
        while uncolored:
            # Select uncolored vertex with maximum saturation degree
            # Break ties by degree, then by activation frequency
            best_node = max(
                uncolored,
                key=lambda n: (
                    self.graph.get_saturation_degree(n, coloring),
                    self.graph.get_degree(n),
                    self.freq[n]
                )
            )
            
            # Find smallest available color
            neighbor_colors = set()
            for neighbor in self.graph.adjacency[best_node]:
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            color = 0
            while color in neighbor_colors:
                color += 1
            
            # Respect subcube limit
            if color >= self.num_subcubes:
                color = color % self.num_subcubes
            
            coloring[best_node] = color
            uncolored.remove(best_node)
        
        return coloring


class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing for expert-to-subcube assignment optimization
    Minimizes conflict probability while balancing load
    """
    
    def __init__(self, cooccurrence_matrix: np.ndarray,
                 activation_freq: np.ndarray,
                 num_subcubes: int,
                 shared_experts: Optional[Set[int]] = None):
        self.cooccurrence = cooccurrence_matrix
        self.freq = activation_freq
        self.num_experts = cooccurrence_matrix.shape[0]
        self.num_subcubes = num_subcubes
        self.shared_experts = shared_experts or set()
        
        # Normalize frequency
        self.freq_norm = self.freq / (self.freq.max() + 1e-8)
    
    def _cost(self, assignment: List[int]) -> float:
        """
        Cost function combining:
        1. Conflict cost: co-occurring experts in same subcube
        2. Load imbalance: variance of subcube usage
        3. Frequency penalty: high-freq experts at high z positions
        """
        # Conflict cost
        conflict_cost = 0.0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                if assignment[i] == assignment[j]:
                    conflict_cost += self.cooccurrence[i][j] * self.freq_norm[i] * self.freq_norm[j]
        
        # Load balance cost
        load_counts = np.zeros(self.num_subcubes)
        for expert_id in range(self.num_experts):
            load_counts[assignment[expert_id]] += 1
        
        load_variance = np.var(load_counts)
        load_cost = load_variance * 0.1
        
        # Total cost
        return conflict_cost + load_cost
    
    def optimize(self, initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 iterations: int = 10000) -> List[int]:
        """
        Run simulated annealing optimization
        Returns: expert_id -> subcube_id assignment
        """
        # Initialize: random assignment respecting shared experts
        current = self._random_initial_solution()
        current_cost = self._cost(current)
        
        best = current.copy()
        best_cost = current_cost
        
        temp = initial_temp
        
        for i in range(iterations):
            # Generate neighbor solution
            neighbor = current.copy()
            
            # Random move: change one expert's subcube
            expert_idx = random.randint(0, self.num_experts - 1)
            if expert_idx not in self.shared_experts:
                neighbor[expert_idx] = random.randint(0, self.num_subcubes - 1)
            
            neighbor_cost = self._cost(neighbor)
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
            
            temp *= cooling_rate
        
        return best
    
    def _random_initial_solution(self) -> List[int]:
        """Generate random initial solution"""
        solution = [0] * self.num_experts
        for expert_id in range(self.num_experts):
            if expert_id in self.shared_experts:
                solution[expert_id] = expert_id  # Dedicated subcube
            else:
                solution[expert_id] = random.randint(0, self.num_subcubes - 1)
        return solution


class HybridExpertGrouper:
    """
    Hybrid approach combining DSatur and Simulated Annealing
    1. DSatur for initial grouping
    2. SA for refinement
    """
    
    def __init__(self, cooccurrence_matrix: np.ndarray,
                 activation_freq: np.ndarray,
                 num_subcubes: int,
                 shared_experts: Optional[Set[int]] = None):
        self.cooccurrence = cooccurrence_matrix
        self.freq = activation_freq
        self.num_subcubes = num_subcubes
        self.shared_experts = shared_experts or set()
    
    def optimize(self, use_sa: bool = True,
                 sa_iterations: int = 5000) -> Dict[str, any]:
        """
        Run hybrid optimization
        Returns: detailed grouping results
        """
        # Phase 1: DSatur coloring
        dsatur = DSaturColoring(
            self.cooccurrence, self.freq, self.num_subcubes
        )
        dsatur_result = dsatur.color(self.shared_experts)
        
        if not use_sa:
            return {
                'assignment': dsatur_result,
                'method': 'DSatur',
                'cost': self._evaluate_assignment(dsatur_result)
            }
        
        # Phase 2: Simulated Annealing refinement
        assignment_list = [dsatur_result.get(i, 0) for i in range(self.cooccurrence.shape[0])]
        
        sa_optimizer = SimulatedAnnealingOptimizer(
            self.cooccurrence, self.freq, self.num_subcubes, self.shared_experts
        )
        sa_result = sa_optimizer.optimize(
            initial_temp=500.0,
            cooling_rate=0.998,
            iterations=sa_iterations
        )
        
        # Compare and choose best
        dsatur_cost = self._evaluate_assignment(dsatur_result)
        sa_cost = self._evaluate_assignment_list(sa_result)
        
        if sa_cost < dsatur_cost:
            final_assignment = {i: sa_result[i] for i in range(len(sa_result))}
            method = 'DSatur+SA'
            cost = sa_cost
        else:
            final_assignment = dsatur_result
            method = 'DSatur'
            cost = dsatur_cost
        
        return {
            'assignment': final_assignment,
            'method': method,
            'cost': cost,
            'dsatur_cost': dsatur_cost,
            'sa_cost': sa_cost
        }
    
    def _evaluate_assignment(self, assignment: Dict[int, int]) -> float:
        """Evaluate assignment quality"""
        cost = 0.0
        subcube_experts = defaultdict(list)
        
        for expert_id, subcube_id in assignment.items():
            subcube_experts[subcube_id].append(expert_id)
        
        # Conflict cost
        for subcube_id, experts in subcube_experts.items():
            for i in range(len(experts)):
                for j in range(i + 1, len(experts)):
                    cost += self.cooccurrence[experts[i]][experts[j]]
        
        # Load balance
        loads = [len(experts) for experts in subcube_experts.values()]
        if loads:
            cost += np.var(loads) * 0.1
        
        return cost
    
    def _evaluate_assignment_list(self, assignment: List[int]) -> float:
        """Evaluate list-based assignment"""
        return self._evaluate_assignment({i: assignment[i] for i in range(len(assignment))})
