"""
SRAM Allocator - Manages 512KB SRAM for CIM hardware
Allocates addresses for input, output, accumulator, bias, and temporary buffers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ir.ir_nodes import IRGraph, TensorInfo, DataType, DTYPE_BYTES, IRNodeType


@dataclass
class MemoryRegion:
    name: str
    start: int
    size: int
    tensor_name: str = ""


@dataclass
class SRAMLayout:
    regions: List[MemoryRegion] = field(default_factory=list)
    total_used: int = 0

    def add_region(self, name, start, size, tensor_name=""):
        r = MemoryRegion(name=name, start=start, size=size, tensor_name=tensor_name)
        self.regions.append(r)
        self.total_used += size
        return r

    def report(self) -> str:
        lines = ["=== SRAM Layout (512KB) ==="]
        for r in self.regions:
            end = r.start + r.size - 1
            lines.append(f"  {r.start:6d} ~ {end:6d}  | {r.name:10s} | {r.size:6d}B | {r.tensor_name}")
        lines.append(f"  Total used: {self.total_used} / 524288 bytes ({self.total_used/524288*100:.1f}%)")
        return "\n".join(lines)


class SRAMAllocator:
    CAPACITY = 512 * 1024  # 512KB

    def __init__(self):
        self.next_addr = 0
        self.layout = SRAMLayout()
        self.allocations: Dict[str, MemoryRegion] = {}

    def allocate(self, graph: IRGraph) -> SRAMLayout:
        self.next_addr = 0
        self.layout = SRAMLayout()
        self.allocations = {}

        sorted_nodes = graph.toposort()

        # Phase 1: allocate input tensors
        for inp_name in graph.input_names:
            if inp_name in graph.tensors:
                t = graph.tensors[inp_name]
                self._alloc_tensor(t, f"input_{inp_name}")

        # Phase 2: for each node, allocate output and temp buffers
        for node in sorted_nodes:
            if node.node_type == IRNodeType.LINEAR:
                self._alloc_linear_buffers(node, graph)
            elif node.node_type == IRNodeType.ELEMENTWISE:
                self._alloc_elementwise_buffers(node, graph)

        # Phase 3: allocate final output tensors if not already allocated
        for out_name in graph.output_names:
            if out_name in graph.tensors and out_name not in self.allocations:
                t = graph.tensors[out_name]
                self._alloc_tensor(t, f"output_{out_name}")

        return self.layout

    def _alloc_tensor(self, tensor: TensorInfo, label: str) -> MemoryRegion:
        size = tensor.size_bytes
        aligned = (size + 3) & ~3  # 4-byte align
        region = self.layout.add_region(label, self.next_addr, aligned, tensor.name)
        tensor.sram_address = self.next_addr
        self.allocations[tensor.name] = region
        self.next_addr += aligned
        return region

    def _alloc_linear_buffers(self, node, graph: IRGraph):
        """Allocate SRAM for linear node: input already allocated, need output, acc, tmp"""
        out_name = node.outputs[0]
        if out_name in graph.tensors:
            out_tensor = graph.tensors[out_name]
            if out_name not in self.allocations:
                self._alloc_tensor(out_tensor, f"output_{out_name}")

        # Allocate accumulator buffer (same size as output, INT32)
        acc_size = out_tensor.num_elements * 4  # INT32
        if acc_size > 0 and f"acc_{out_name}" not in self.allocations:
            aligned = (acc_size + 3) & ~3
            region = self.layout.add_region("acc", self.next_addr, aligned, out_name)
            self.next_addr += aligned

        # Allocate temporary buffer for shifted partial results
        tmp_size = out_tensor.num_elements * 4
        if tmp_size > 0 and f"tmp_{out_name}" not in self.allocations:
            aligned = (tmp_size + 3) & ~3
            region = self.layout.add_region("tmp", self.next_addr, aligned, out_name)
            self.next_addr += aligned

        # Allocate bias if present
        if node.bias_name and node.bias_name in graph.tensors:
            bias_t = graph.tensors[node.bias_name]
            if node.bias_name not in self.allocations:
                self._alloc_tensor(bias_t, f"bias_{node.bias_name}")

    def _alloc_elementwise_buffers(self, node, graph: IRGraph):
        out_name = node.outputs[0]
        if out_name in graph.tensors and out_name not in self.allocations:
            self._alloc_tensor(graph.tensors[out_name], f"ew_out_{out_name}")

    def get_address(self, tensor_name: str) -> int:
        if tensor_name in self.allocations:
            return self.allocations[tensor_name].start
        return 0

    def get_acc_address(self, output_name: str) -> int:
        for r in self.layout.regions:
            if r.name == "acc" and r.tensor_name == output_name:
                return r.start
        return 0

    def get_tmp_address(self, output_name: str) -> int:
        for r in self.layout.regions:
            if r.name == "tmp" and r.tensor_name == output_name:
                return r.start
        return 0
