"""
Dead Code Elimination (DCE) Optimizer
Removes nodes whose outputs are never used
"""

from ir.ir_nodes import IRGraph, IRNode, IRNodeType


class DeadCodeElimination:
    """Removes dead code from IR graph"""
    
    def optimize(self, graph: IRGraph) -> IRGraph:
        """Apply DCE optimization"""
        live_outputs = set(graph.output_names)
        live_nodes = set()

        self._mark_live(graph, graph.output_names, live_outputs, live_nodes)

        graph.nodes = [n for n in graph.nodes if n.node_id in live_nodes]

        self._remove_dead_tensors(graph, live_outputs)

        return graph

    def _mark_live(self, graph: IRGraph, outputs, live_outputs, live_nodes):
        """Mark nodes that produce live outputs"""
        node_map = {}
        for node in graph.nodes:
            for out in node.outputs:
                node_map[out] = node

        worklist = list(outputs)
        while worklist:
            out_name = worklist.pop()
            live_outputs.add(out_name)
            node = node_map.get(out_name)
            if node and node.node_id not in live_nodes:
                live_nodes.add(node.node_id)
                for inp in node.inputs:
                    worklist.append(inp)

    def _remove_dead_tensors(self, graph: IRGraph, live_outputs):
        """Remove tensors that are no longer referenced"""
        live_tensors = set(graph.input_names) | set(graph.output_names)
        for node in graph.nodes:
            live_tensors.update(node.inputs)
            live_tensors.update(node.outputs)

        graph.tensors = {k: v for k, v in graph.tensors.items() if k in live_tensors}
