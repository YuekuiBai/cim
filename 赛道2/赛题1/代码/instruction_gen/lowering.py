"""
IR Lowering - Converts IR to CIM ISA instructions
Generates cim.bit.type, elt.op.type.mode, mem.copy instructions
"""

from dataclasses import dataclass, field
from typing import List, Optional
from ir.ir_nodes import (
    IRGraph, IRNode, LinearNode, ElementwiseNode,
    IRNodeType, ElementwiseOp, DataType, DTYPE_BITS
)
from resource_manager.sram_allocator import SRAMAllocator
from resource_manager.weight_mapper import WeightMapper, WeightMapping


@dataclass
class Instruction:
    opcode: str
    operands: list
    comment: str = ""

    def __str__(self):
        ops = ", ".join(str(o) for o in self.operands)
        if self.comment:
            return f"{self.opcode} {ops}  // {self.comment}"
        return f"{self.opcode} {ops}"


class IRLowering:
    def __init__(self, allocator: SRAMAllocator, weight_mappings: List[WeightMapping]):
        self.allocator = allocator
        self.weight_mappings = {m.weight_name: m for m in weight_mappings}
        self.instructions: List[Instruction] = []

    def lower(self, graph: IRGraph) -> List[Instruction]:
        self.instructions = []
        sorted_nodes = graph.toposort()

        for node in sorted_nodes:
            if node.node_type == IRNodeType.LINEAR:
                self._lower_linear(node, graph)
            elif node.node_type == IRNodeType.ELEMENTWISE:
                self._lower_elementwise(node, graph)

        return self.instructions

    def _lower_linear(self, node: LinearNode, graph: IRGraph):
        """
        Lower a linear node to CIM bit-serial instructions.
        For int8 input: 8 cim.bit.i8 instructions (bits 0-6 add, bit 7 sub for signed).
        Each bit result is shifted and accumulated.
        
        Algorithm:
          output = sum(input[i][bit_k] * weight << k) for k=0..6
          output -= (input[i][bit_7] * weight << 7)  // sign bit
        """
        inp_name = node.inputs[0]
        out_name = node.outputs[0]
        weight_name = node.weight_name

        # Get SRAM addresses
        input_addr = self.allocator.get_address(inp_name)
        output_addr = self.allocator.get_address(out_name)
        acc_addr = self.allocator.get_acc_address(out_name)
        tmp_addr = self.allocator.get_tmp_address(out_name)

        # Get weight mapping
        wm = self.weight_mappings.get(weight_name)
        if wm is None:
            weight_pos = [0, 0, 1024, 4096]
            bit_width = 8
        else:
            weight_pos = [wm.row_start, wm.col_start, wm.row_end, wm.col_end]
            bit_width = wm.bit_width

        # Get element count from input tensor
        if inp_name in graph.tensors:
            num_elements = graph.tensors[inp_name].num_elements
        else:
            num_elements = node.cin

        # Determine input dtype
        if inp_name in graph.tensors:
            inp_dtype = graph.tensors[inp_name].dtype
        else:
            inp_dtype = DataType.INT8
        input_bits = DTYPE_BITS[inp_dtype]

        # Determine output dtype (always INT32 for linear)
        output_dtype_str = "i32"

        # Step 1: bit 0 - direct accumulation to acc
        self._comment(f"Process bit 0 of {inp_name} ({inp_dtype.value})")
        self._emit_cim_bit(acc_addr, input_addr, 0, weight_pos, inp_dtype.value)

        # Step 2: bits 1 to N-2 - accumulate with shift (add)
        for k in range(1, input_bits - 1):
            self._comment(f"Process bit {k} of {inp_name}, shift by {k}")
            self._emit_cim_bit(output_addr, input_addr, k, weight_pos, inp_dtype.value)
            shift_value = 1 << k
            self._emit_elt_mul_vi(tmp_addr, output_addr, shift_value, num_elements, output_dtype_str)
            self._emit_elt_add_vv(acc_addr, acc_addr, tmp_addr, num_elements, output_dtype_str)

        # Step 3: last bit (sign bit for signed int8) - subtract
        k = input_bits - 1
        self._comment(f"Process bit {k} (sign bit) of {inp_name}, shift by {k} and subtract")
        self._emit_cim_bit(output_addr, input_addr, k, weight_pos, inp_dtype.value)
        shift_value = 1 << k
        self._emit_elt_mul_vi(tmp_addr, output_addr, shift_value, num_elements, output_dtype_str)
        self._emit_elt_sub_vv(acc_addr, acc_addr, tmp_addr, num_elements, output_dtype_str)

        # Step 4: copy acc to output
        self._comment(f"Copy accumulated result to {out_name}")
        self._emit_mem_copy(output_addr, acc_addr, num_elements, output_dtype_str, output_dtype_str)

        # Step 5: add bias if present
        if node.bias_name and node.bias_name in graph.tensors:
            bias_addr = self.allocator.get_address(node.bias_name)
            self._comment(f"Add bias {node.bias_name}")
            self._emit_elt_add_vv(output_addr, output_addr, bias_addr, num_elements, output_dtype_str)

    def _lower_elementwise(self, node: ElementwiseNode, graph: IRGraph):
        """Lower elementwise to elt.op.type.mode instructions"""
        out_name = node.outputs[0]
        src1_name = node.inputs[0]
        src2_name = node.inputs[1] if len(node.inputs) > 1 else None

        out_addr = self.allocator.get_address(out_name)
        src1_addr = self.allocator.get_address(src1_name)

        if out_name in graph.tensors:
            num_elements = graph.tensors[out_name].num_elements
            dtype = graph.tensors[out_name].dtype.value
        else:
            num_elements = 128
            dtype = "i32"

        op_str = node.op.value

        if node.mode == "vi" and node.immediate_value is not None:
            self._comment(f"Elementwise {op_str} immediate {node.immediate_value}")
            self._emit_elt(op_str, out_addr, src1_addr, node.immediate_value, num_elements, dtype, "vi")
        elif src2_name:
            src2_addr = self.allocator.get_address(src2_name)
            self._comment(f"Elementwise {op_str} {src1_name} {src2_name}")
            self._emit_elt(op_str, out_addr, src1_addr, src2_addr, num_elements, dtype, "vv")

    # --- Instruction emitters ---

    def _comment(self, text):
        self.instructions.append(Instruction("//", [], text))

    def _emit_cim_bit(self, dst, src, index, weight_pos, dtype="i8"):
        opcode = f"cim.bit.{dtype}"
        self.instructions.append(Instruction(
            opcode,
            [f"/*out*/{dst}", f"/*input*/{src}", f"/*index*/{index}", f"/*weightPos*/{weight_pos}"],
        ))

    def _emit_elt(self, op, dst, src1, src2, length, dtype="i32", mode="vv"):
        opcode = f"elt.{op}.{dtype}.{mode}"
        self.instructions.append(Instruction(
            opcode,
            [f"/*dst*/{dst}", f"/*src1*/{src1}", f"/*src2*/{src2}", f"/*len*/{length}"],
        ))

    def _emit_elt_mul_vi(self, dst, src1, src2, length, dtype="i32"):
        self._emit_elt("mul", dst, src1, src2, length, dtype, "vi")

    def _emit_elt_add_vv(self, dst, src1, src2, length, dtype="i32"):
        self._emit_elt("add", dst, src1, src2, length, dtype, "vv")

    def _emit_elt_sub_vv(self, dst, src1, src2, length, dtype="i32"):
        self._emit_elt("sub", dst, src1, src2, length, dtype, "vv")

    def _emit_mem_copy(self, dst, src, length, dst_type="i32", src_type="i32"):
        opcode = f"mem.copy.{dst_type}.{src_type}"
        self.instructions.append(Instruction(
            opcode,
            [f"/*dst*/{dst}", f"/*src*/{src}", f"/*len*/{length}"],
        ))
