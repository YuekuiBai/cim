"""
Operator Fusion Optimizer
Fuses consecutive operations for efficiency
Supports: Linear+Elementwise fusion, Elementwise chain fusion
"""

from ir.ir_nodes import IRGraph, IRNode, IRNodeType, ElementwiseNode, ElementwiseOp


class OperatorFusion:
    """Optimizes IR by fusing compatible operators"""
    
    def optimize(self, graph: IRGraph) -> IRGraph:
        """Apply operator fusion optimization"""
        graph = self._fuse_linear_elementwise(graph)
        graph = self._fuse_elementwise_chains(graph)
        return graph

    def _fuse_linear_elementwise(self, graph: IRGraph) -> IRGraph:
        """Fuse Linear + Elementwise patterns where Elementwise uses Linear output as first input"""
        new_nodes = []
        fused_outputs = set()
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.node_type == IRNodeType.LINEAR and i + 1 < len(graph.nodes):
                next_node = graph.nodes[i + 1]
                if (next_node.node_type == IRNodeType.ELEMENTWISE and 
                    node.outputs[0] == next_node.inputs[0] and
                    next_node.op in [ElementwiseOp.ADD, ElementwiseOp.MUL]):
                    fused_node = self._create_fused_node(node, next_node)
                    new_nodes.append(fused_node)
                    fused_outputs.add(node.outputs[0])
                    i += 2
                    continue
            if node.outputs[0] not in fused_outputs:
                new_nodes.append(node)
            i += 1
        
        graph.nodes = new_nodes
        return graph

    def _fuse_elementwise_chains(self, graph: IRGraph) -> IRGraph:
        """Fuse consecutive elementwise operations into a single composite operation"""
        new_nodes = []
        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]
            if node.node_type == IRNodeType.ELEMENTWISE:
                chain = [node]
                j = i + 1
                while j < len(graph.nodes):
                    next_node = graph.nodes[j]
                    if (next_node.node_type == IRNodeType.ELEMENTWISE and
                        chain[-1].outputs[0] == next_node.inputs[0]):
                        chain.append(next_node)
                        j += 1
                    else:
                        break
                
                if len(chain) > 1:
                    fused = self._create_chain_fused_node(chain)
                    new_nodes.append(fused)
                else:
                    new_nodes.append(chain[0])
                i = j
            else:
                new_nodes.append(node)
                i += 1
        
        graph.nodes = new_nodes
        return graph

    def _create_fused_node(self, linear_node, ew_node):
        """Create a fused Linear+Elementwise node"""
        fused = ElementwiseNode(
            node_id=f"fused_{linear_node.node_id}_{ew_node.node_id}",
            node_type=IRNodeType.LINEAR,
            inputs=linear_node.inputs + ew_node.inputs[1:],
            outputs=ew_node.outputs,
            attributes={
                "fused": True,
                "linear_op": "matmul",
                "elementwise_op": ew_node.op.value,
                "weight": linear_node.weight_name,
                "bias": linear_node.bias_name,
                "cin": linear_node.cin,
                "cout": linear_node.cout,
            }
        )
        fused.weight_name = linear_node.weight_name
        fused.bias_name = linear_node.bias_name
        fused.cin = linear_node.cin
        fused.cout = linear_node.cout
        return fused

    def _create_chain_fused_node(self, chain):
        """Create a fused node from elementwise chain"""
        first = chain[0]
        last = chain[-1]
        fused = ElementwiseNode(
            node_id=f"fused_{'_'.join(n.node_id for n in chain)}",
            node_type=IRNodeType.ELEMENTWISE,
            inputs=first.inputs,
            outputs=last.outputs,
            op=first.op,
            mode=first.mode,
            attributes={
                "fused": True,
                "chain_length": len(chain),
                "operations": [n.op.value for n in chain],
            }
        )
        return fused
