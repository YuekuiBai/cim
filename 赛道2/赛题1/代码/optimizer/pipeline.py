"""
Optimization Pipeline for CIM Compiler
Combines multiple optimization passes into a unified pipeline
"""

from ir.ir_nodes import IRGraph
from optimizer.constant_folding import ConstantFolding
from optimizer.operator_fusion import OperatorFusion
from optimizer.cse import CommonSubexpressionElimination
from optimizer.dce import DeadCodeElimination


class OptimizationPipeline:
    """Manages the optimization pipeline"""
    
    def __init__(self):
        self.passes = [
            ("ConstantFolding", ConstantFolding()),
            ("OperatorFusion", OperatorFusion()),
            ("CSE", CommonSubexpressionElimination()),
            ("DCE", DeadCodeElimination()),
        ]
    
    def optimize(self, graph: IRGraph) -> IRGraph:
        """Run all optimization passes"""
        stats = {"initial_nodes": len(graph.nodes), "passes": {}}
        
        for name, optimizer in self.passes:
            before = len(graph.nodes)
            graph = optimizer.optimize(graph)
            after = len(graph.nodes)
            stats["passes"][name] = {
                "nodes_before": before,
                "nodes_after": after,
                "removed": before - after,
            }
        
        stats["final_nodes"] = len(graph.nodes)
        stats["total_removed"] = stats["initial_nodes"] - stats["final_nodes"]
        
        return graph, stats
    
    def get_report(self, stats: dict) -> str:
        """Generate optimization report"""
        lines = ["=== Optimization Report ==="]
        lines.append(f"  Initial nodes: {stats['initial_nodes']}")
        lines.append(f"  Final nodes: {stats['final_nodes']}")
        lines.append(f"  Total removed: {stats['total_removed']}")
        lines.append("")
        for name, pass_stats in stats["passes"].items():
            lines.append(f"  {name}:")
            lines.append(f"    Nodes: {pass_stats['nodes_before']} -> {pass_stats['nodes_after']}")
            lines.append(f"    Removed: {pass_stats['removed']}")
        return "\n".join(lines)
