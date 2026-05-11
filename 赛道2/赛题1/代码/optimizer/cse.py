"""
Common Subexpression Elimination (CSE) Optimizer
Eliminates redundant computations by detecting identical subexpressions
"""

from ir.ir_nodes import IRGraph, IRNode, IRNodeType


class CommonSubexpressionElimination:
    """Detects and eliminates common subexpressions"""
    
    def optimize(self, graph: IRGraph) -> IRGraph:
        """Apply CSE optimization"""
        expr_map = {}
        nodes_to_remove = set()
        output_redirects = {}

        for node in graph.nodes:
            if self._is_pure(node):
                signature = self._compute_signature(node)
                if signature in expr_map:
                    original_node = expr_map[signature]
                    for i, out in enumerate(node.outputs):
                        output_redirects[out] = original_node.outputs[i]
                    nodes_to_remove.add(node.node_id)
                else:
                    expr_map[signature] = node

        if output_redirects:
            self._redirect_uses(graph, output_redirects)
            graph.nodes = [n for n in graph.nodes if n.node_id not in nodes_to_remove]

        return graph

    def _is_pure(self, node: IRNode) -> bool:
        """Check if node is pure (no side effects)"""
        return node.node_type in [IRNodeType.ELEMENTWISE, IRNodeType.CONSTANT]

    def _compute_signature(self, node: IRNode) -> tuple:
        """Compute a unique signature for a node based on its operation and inputs"""
        inputs = tuple(sorted(node.inputs))
        attrs = tuple(sorted((k, str(v)) for k, v in node.attributes.items()))
        return (node.node_type.value, inputs, attrs)

    def _redirect_uses(self, graph: IRGraph, redirects: dict):
        """Redirect all uses of eliminated outputs to original outputs"""
        for node in graph.nodes:
            new_inputs = []
            for inp in node.inputs:
                new_inputs.append(redirects.get(inp, inp))
            node.inputs = new_inputs
