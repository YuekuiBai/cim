"""
SRAM Allocator - Manages 512KB SRAM for CIM hardware
Enhanced with Interval Graph Coloring for optimal tensor reuse
Allocates addresses for input, output, accumulator, bias, and temporary buffers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from ir.ir_nodes import IRGraph, TensorInfo, DataType, DTYPE_BYTES, IRNodeType


@dataclass
class MemoryRegion:
    name: str
    start: int
    size: int
    tensor_name: str = ""


@dataclass
class TensorInterval:
    """Represents the live interval of a tensor"""
    tensor_name: str
    start_point: int
    end_point: int
    size: int
    dtype: DataType


@dataclass
class SRAMLayout:
    regions: List[MemoryRegion] = field(default_factory=list)
    total_used: int = 0
    reuse_count: int = 0

    def add_region(self, name, start, size, tensor_name=""):
        r = MemoryRegion(name=name, start=start, size=size, tensor_name=tensor_name)
        self.regions.append(r)
        self.total_used += size
        return r

    def report(self) -> str:
        lines = ["=== SRAM Layout (512KB) ==="]
        for r in self.regions:
            end = r.start + r.size - 1
            lines.append(f"  {r.start:6d} ~ {end:6d}  | {r.name:12s} | {r.size:6d}B | {r.tensor_name}")
        lines.append(f"  Total used: {self.total_used} / 524288 bytes ({self.total_used/524288*100:.1f}%)")
        lines.append(f"  Tensor reuse count: {self.reuse_count}")
        return "\n".join(lines)


class IntervalGraph:
    """Interval graph for tensor lifetime analysis"""
    
    def __init__(self):
        self.intervals: List[TensorInterval] = []
        self.adjacency: Dict[str, Set[str]] = {}
    
    def add_interval(self, interval: TensorInterval):
        self.intervals.append(interval)
        self.adjacency[interval.tensor_name] = set()
    
    def build_conflicts(self):
        """Build conflict graph based on overlapping intervals"""
        for i, interval1 in enumerate(self.intervals):
            for j, interval2 in enumerate(self.intervals):
                if i >= j:
                    continue
                if self._overlaps(interval1, interval2):
                    self.adjacency[interval1.tensor_name].add(interval2.tensor_name)
                    self.adjacency[interval2.tensor_name].add(interval1.tensor_name)
    
    def _overlaps(self, i1: TensorInterval, i2: TensorInterval) -> bool:
        """Check if two intervals overlap"""
        return i1.start_point < i2.end_point and i2.start_point < i1.end_point
    
    def color_greedy(self) -> Dict[str, int]:
        """Greedy graph coloring for address assignment"""
        colors: Dict[str, int] = {}
        for interval in sorted(self.intervals, key=lambda x: -x.size):
            name = interval.tensor_name
            used_colors = set()
            for neighbor in self.adjacency[name]:
                if neighbor in colors:
                    used_colors.add(colors[neighbor])
            
            color = 0
            while color in used_colors:
                color += 1
            colors[name] = color
        return colors


class SRAMAllocator:
    CAPACITY = 512 * 1024  # 512KB

    def __init__(self, use_interval_coloring=True):
        self.next_addr = 0
        self.layout = SRAMLayout()
        self.allocations: Dict[str, MemoryRegion] = {}
        self.use_interval_coloring = use_interval_coloring

    def allocate(self, graph: IRGraph) -> SRAMLayout:
        self.next_addr = 0
        self.layout = SRAMLayout()
        self.allocations = {}

        if self.use_interval_coloring:
            return self._allocate_with_coloring(graph)
        else:
            return self._allocate_simple(graph)

    def _allocate_with_coloring(self, graph: IRGraph) -> SRAMLayout:
        """Allocate SRAM using interval graph coloring for tensor reuse"""
        intervals = self._compute_live_intervals(graph)
        
        ig = IntervalGraph()
        for interval in intervals:
            ig.add_interval(interval)
        ig.build_conflicts()
        
        colors = ig.color_greedy()
        
        color_addresses: Dict[int, int] = {}
        color_max_end: Dict[int, int] = {}
        
        sorted_intervals = sorted(intervals, key=lambda x: x.start_point)
        
        for interval in sorted_intervals:
            color = colors[interval.tensor_name]
            
            if color in color_addresses and color_max_end.get(color, 0) <= interval.start_point:
                addr = color_addresses[color]
                color_addresses[color] = addr
                color_max_end[color] = interval.end_point
                self.layout.reuse_count += 1
            else:
                addr = self.next_addr
                color_addresses[color] = addr
                color_max_end[color] = interval.end_point
                self.next_addr += (interval.size + 3) & ~3
            
            region = self.layout.add_region(
                f"tensor_{interval.tensor_name}", 
                addr, 
                (interval.size + 3) & ~3, 
                interval.tensor_name
            )
            self.allocations[interval.tensor_name] = region
            
            if interval.tensor_name in graph.tensors:
                graph.tensors[interval.tensor_name].sram_address = addr
        
        # Allocate acc and tmp buffers for linear nodes (these need separate space)
        sorted_nodes = graph.toposort()
        for node in sorted_nodes:
            if node.node_type == IRNodeType.LINEAR:
                self._alloc_linear_buffers_for_coloring(node, graph)
            elif node.node_type == IRNodeType.ELEMENTWISE:
                self._alloc_elementwise_buffers_for_coloring(node, graph)
        
        return self.layout

    def _alloc_linear_buffers_for_coloring(self, node, graph: IRGraph):
        """Allocate acc and tmp buffers for linear nodes in coloring mode"""
        out_name = node.outputs[0]
        if out_name in graph.tensors:
            out_tensor = graph.tensors[out_name]
            
            acc_size = out_tensor.num_elements * 4
            if acc_size > 0 and f"acc_{out_name}" not in self.allocations:
                aligned = (acc_size + 3) & ~3
                region = self.layout.add_region("acc", self.next_addr, aligned, out_name)
                self.next_addr += aligned
                self.allocations[f"acc_{out_name}"] = region

            tmp_size = out_tensor.num_elements * 4
            if tmp_size > 0 and f"tmp_{out_name}" not in self.allocations:
                aligned = (tmp_size + 3) & ~3
                region = self.layout.add_region("tmp", self.next_addr, aligned, out_name)
                self.next_addr += aligned
                self.allocations[f"tmp_{out_name}"] = region

            if node.bias_name and node.bias_name in graph.tensors:
                bias_t = graph.tensors[node.bias_name]
                if node.bias_name not in self.allocations:
                    self._alloc_tensor(bias_t, f"bias_{node.bias_name}")

    def _alloc_elementwise_buffers_for_coloring(self, node, graph: IRGraph):
        """Allocate buffers for elementwise nodes in coloring mode"""
        out_name = node.outputs[0]
        if out_name in graph.tensors and out_name not in self.allocations:
            self._alloc_tensor(graph.tensors[out_name], f"ew_out_{out_name}")

    def _compute_live_intervals(self, graph: IRGraph) -> List[TensorInterval]:
        """Compute live intervals for all tensors"""
        intervals = []
        tensor_first_use: Dict[str, int] = {}
        tensor_last_use: Dict[str, int] = {}
        
        for i, node in enumerate(graph.nodes):
            for inp in node.inputs:
                if inp not in tensor_first_use:
                    tensor_first_use[inp] = i
                tensor_last_use[inp] = i
            
            for out in node.outputs:
                tensor_first_use[out] = i
                tensor_last_use[out] = i
        
        for out_name in graph.output_names:
            if out_name in tensor_last_use:
                tensor_last_use[out_name] = len(graph.nodes)
        
        all_tensors = set(tensor_first_use.keys()) | set(tensor_last_use.keys())
        
        for tensor_name in all_tensors:
            if tensor_name in graph.tensors:
                t = graph.tensors[tensor_name]
                start = tensor_first_use.get(tensor_name, 0)
                end = tensor_last_use.get(tensor_name, start + 1)
                intervals.append(TensorInterval(
                    tensor_name=tensor_name,
                    start_point=start,
                    end_point=end,
                    size=t.size_bytes,
                    dtype=t.dtype
                ))
        
        return intervals

    def _allocate_simple(self, graph: IRGraph) -> SRAMLayout:
        """Simple linear allocation without tensor reuse"""
        self.next_addr = 0
        self.layout = SRAMLayout()
        self.allocations = {}

        sorted_nodes = graph.toposort()

        for inp_name in graph.input_names:
            if inp_name in graph.tensors:
                t = graph.tensors[inp_name]
                self._alloc_tensor(t, f"input_{inp_name}")

        for node in sorted_nodes:
            if node.node_type == IRNodeType.LINEAR:
                self._alloc_linear_buffers(node, graph)
            elif node.node_type == IRNodeType.ELEMENTWISE:
                self._alloc_elementwise_buffers(node, graph)

        for out_name in graph.output_names:
            if out_name in graph.tensors and out_name not in self.allocations:
                t = graph.tensors[out_name]
                self._alloc_tensor(t, f"output_{out_name}")

        return self.layout

    def _alloc_tensor(self, tensor: TensorInfo, label: str) -> MemoryRegion:
        size = tensor.size_bytes
        aligned = (size + 3) & ~3
        region = self.layout.add_region(label, self.next_addr, aligned, tensor.name)
        tensor.sram_address = self.next_addr
        self.allocations[tensor.name] = region
        self.next_addr += aligned
        return region

    def _alloc_linear_buffers(self, node, graph: IRGraph):
        out_name = node.outputs[0]
        if out_name in graph.tensors:
            out_tensor = graph.tensors[out_name]
            if out_name not in self.allocations:
                self._alloc_tensor(out_tensor, f"output_{out_name}")

        acc_size = out_tensor.num_elements * 4
        if acc_size > 0 and f"acc_{out_name}" not in self.allocations:
            aligned = (acc_size + 3) & ~3
            region = self.layout.add_region("acc", self.next_addr, aligned, out_name)
            self.next_addr += aligned

        tmp_size = out_tensor.num_elements * 4
        if tmp_size > 0 and f"tmp_{out_name}" not in self.allocations:
            aligned = (tmp_size + 3) & ~3
            region = self.layout.add_region("tmp", self.next_addr, aligned, out_name)
            self.next_addr += aligned

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
