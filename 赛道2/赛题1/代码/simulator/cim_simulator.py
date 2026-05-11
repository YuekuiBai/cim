"""
CIM Instruction-Level Simulator
Simulates the execution of generated CIM ISA instructions for correctness verification
Supports: cim.bit.*, elt.*.*, mem.copy.*.*
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SimInstruction:
    opcode: str
    operands: list
    comment: str = ""


@dataclass
class SimMemory:
    """Simulated SRAM memory"""
    capacity: int = 512 * 1024
    data: np.ndarray = field(default_factory=lambda: np.zeros(512 * 1024, dtype=np.uint8))
    
    def read_bytes(self, addr: int, size: int, dtype_str: str) -> np.ndarray:
        np_dtype = self._dtype_from_str(dtype_str)
        return self.data[addr:addr+size].view(np_dtype).copy()
    
    def write_bytes(self, addr: int, data: np.ndarray, dtype_str: str):
        np_dtype = self._dtype_from_str(dtype_str)
        self.data[addr:addr+data.nbytes] = data.view(np.uint8)
    
    def _dtype_from_str(self, dtype_str: str) -> np.dtype:
        mapping = {
            "i8": np.int8, "u8": np.uint8,
            "i16": np.int16, "u16": np.uint16,
            "i32": np.int32, "u32": np.uint32,
            "f32": np.float32,
        }
        return np.dtype(mapping.get(dtype_str, np.int8))


@dataclass
class SimCIMArray:
    """Simulated CIM array for bit-serial computation"""
    row_bits: int = 1024
    col_bits: int = 4096
    weights: Optional[np.ndarray] = None
    
    def set_weights(self, weights: np.ndarray, row_start: int, col_start: int):
        if self.weights is None:
            self.weights = np.zeros((self.row_bits, self.col_bits), dtype=np.int8)
        h, w = weights.shape
        self.weights[row_start:row_start+h, col_start:col_start+w] = weights
    
    def compute_bit_serial(self, input_data: np.ndarray, bit_index: int, 
                          weight_row_start: int, weight_row_end: int,
                          weight_col_start: int, weight_col_end: int) -> np.ndarray:
        input_bits = np.array([(x >> bit_index) & 1 for x in input_data], dtype=np.int32)
        
        if self.weights is None:
            cin = weight_row_end - weight_row_start
            cout = (weight_col_end - weight_col_start) // 8
            self.weights = np.random.RandomState(42).randint(-128, 127, (cin, cout), dtype=np.int8)
        
        weight_slice = self.weights[weight_row_start:weight_row_end, weight_col_start//8:weight_col_end//8]
        cin = weight_slice.shape[0]
        cout = weight_slice.shape[1]
        
        input_reshaped = input_bits[:cin].reshape(cin, 1)
        result = np.dot(input_reshaped.T, weight_slice.astype(np.int32)).flatten()
        return result


class CIMSimulator:
    """Instruction-level simulator for CIM ISA"""
    
    def __init__(self, cim_array: SimCIMArray = None):
        self.memory = SimMemory()
        self.cim_array = cim_array or SimCIMArray()
        self.execution_log: List[str] = []
        self.instruction_count = 0
    
    def load_program(self, instructions: List[SimInstruction]):
        self.program = instructions
    
    def execute(self) -> Dict[str, any]:
        self.execution_log = []
        self.instruction_count = 0
        
        for instr in self.program:
            if instr.opcode.startswith("//"):
                continue
            self._execute_instruction(instr)
            self.instruction_count += 1
        
        return {
            "total_instructions": self.instruction_count,
            "log": self.execution_log,
        }
    
    def _execute_instruction(self, instr: SimInstruction):
        if instr.opcode.startswith("cim.bit."):
            self._exec_cim_bit(instr)
        elif instr.opcode.startswith("elt."):
            self._exec_elementwise(instr)
        elif instr.opcode.startswith("mem.copy."):
            self._exec_mem_copy(instr)
        else:
            self.execution_log.append(f"Unknown instruction: {instr.opcode}")
    
    def _exec_cim_bit(self, instr: SimInstruction):
        dst_addr = int(str(instr.operands[0]).replace("/*out*/", ""))
        src_addr = int(str(instr.operands[1]).replace("/*input*/", ""))
        bit_index = int(str(instr.operands[2]).replace("/*index*/", ""))
        weight_pos_str = str(instr.operands[3]).replace("/*weightPos*/", "")
        weight_pos = eval(weight_pos_str)
        
        dtype_str = instr.opcode.split(".")[-1]
        np_dtype = self.memory._dtype_from_str(dtype_str)
        
        input_data = self.memory.data[src_addr:src_addr+1024].view(np_dtype)[:1024].astype(np.int32)
        
        row_start, col_start, row_end, col_end = weight_pos
        result = self.cim_array.compute_bit_serial(
            input_data, bit_index, row_start, row_end, col_start, col_end
        )
        
        self.memory.data[dst_addr:dst_addr+len(result)*4] = result.view(np.uint8)
        self.execution_log.append(f"CIM_BIT: bit={bit_index}, result_len={len(result)}")
    
    def _exec_elementwise(self, instr: SimInstruction):
        parts = instr.opcode.split(".")
        op = parts[1]
        dtype_str = parts[2]
        mode = parts[3]
        
        dst_addr = int(str(instr.operands[0]).replace("/*dst*/", "").strip())
        src1_addr = int(str(instr.operands[1]).replace("/*src1*/", "").strip())
        length = int(str(instr.operands[3]).replace("/*len*/", "").strip())
        
        np_dtype = self.memory._dtype_from_str(dtype_str)
        src1 = self.memory.data[src1_addr:src1_addr+length*np_dtype.itemsize].view(np_dtype).copy()
        
        if mode == "vv":
            src2_addr = int(str(instr.operands[2]).replace("/*src2*/", "").strip())
            src2 = self.memory.data[src2_addr:src2_addr+length*np_dtype.itemsize].view(np_dtype).copy()
            result = self._elementwise_op(op, src1, src2)
        elif mode == "vi":
            imm_str = str(instr.operands[2]).replace("/*src2*/", "").strip()
            imm = int(imm_str)
            result = self._elementwise_op_imm(op, src1, imm)
        else:
            result = src1
        
        self.memory.data[dst_addr:dst_addr+result.nbytes] = result.view(np.uint8)
        self.execution_log.append(f"ELEMENTWISE: {op}.{dtype_str}.{mode}, len={length}")
    
    def _elementwise_op(self, op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if op == "add":
            return a + b
        elif op == "sub":
            return a - b
        elif op == "mul":
            return a * b
        elif op == "div":
            return np.where(b != 0, a // b, 0)
        return a
    
    def _elementwise_op_imm(self, op: str, a: np.ndarray, imm: int) -> np.ndarray:
        if op == "mul":
            return a * imm
        elif op == "add":
            return a + imm
        elif op == "sub":
            return a - imm
        return a
    
    def _exec_mem_copy(self, instr: SimInstruction):
        parts = instr.opcode.split(".")
        dst_type = parts[2]
        src_type = parts[3]
        
        dst_addr = int(str(instr.operands[0]).replace("/*dst*/", ""))
        src_addr = int(str(instr.operands[1]).replace("/*src*/", ""))
        length = int(str(instr.operands[2]).replace("/*len*/", ""))
        
        src_dtype = self.memory._dtype_from_str(src_type)
        dst_dtype = self.memory._dtype_from_str(dst_type)
        
        src_data = self.memory.data[src_addr:src_addr+length*src_dtype.itemsize].view(src_dtype).copy()
        dst_data = src_data.astype(dst_dtype)
        
        self.memory.data[dst_addr:dst_addr+dst_data.nbytes] = dst_data.view(np.uint8)
        self.execution_log.append(f"MEM_COPY: {src_type}->{dst_type}, len={length}")
    
    def get_tensor(self, addr: int, length: int, dtype_str: str) -> np.ndarray:
        np_dtype = self.memory._dtype_from_str(dtype_str)
        return self.memory.data[addr:addr+length*np_dtype.itemsize].view(np_dtype).copy()
    
    def verify_result(self, expected: np.ndarray, actual_addr: int, length: int, dtype_str: str) -> Tuple[bool, float]:
        actual = self.get_tensor(actual_addr, length, dtype_str)
        if len(expected) != len(actual):
            return False, 0.0
        max_diff = np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))
        is_correct = max_diff == 0
        return is_correct, float(max_diff)
