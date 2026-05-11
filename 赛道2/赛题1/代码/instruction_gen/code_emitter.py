"""
Code Emitter - Outputs assembly instructions to file
"""

from typing import List
from instruction_gen.lowering import Instruction
import json


class CodeEmitter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def emit(self, instructions: List[Instruction], extra_info=None):
        with open(self.output_path, 'w') as f:
            f.write("// " + "=" * 60 + "\n")
            f.write("// CIM Compiler Generated Assembly\n")
            f.write("// Track 2 Problem 1 - CNN Compiler\n")
            f.write("// " + "=" * 60 + "\n\n")

            for instr in instructions:
                ops = ", ".join(str(o) for o in instr.operands)
                if ops:
                    line = f"{instr.opcode} {ops}"
                else:
                    line = instr.opcode
                if instr.comment:
                    line += f"  // {instr.comment}"
                f.write(line + "\n")

            f.write("\n// End of generated code\n")

    def emit_json(self, json_path: str, instructions: List[Instruction], extra=None):
        data = {
            "instructions": [
                {
                    "opcode": i.opcode,
                    "operands": [str(o) for o in i.operands],
                    "comment": i.comment,
                }
                for i in instructions
                if i.opcode != "//"
            ],
        }
        if extra:
            data.update(extra)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
