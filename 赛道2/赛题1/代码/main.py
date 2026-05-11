"""
CIM Compiler - Main Entry Point (Track 2 Problem 1)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx_parser.onnx_loader import ONNXParser
from optimizer.constant_folding import ConstantFolding
from resource_manager.sram_allocator import SRAMAllocator
from resource_manager.weight_mapper import WeightMapper
from instruction_gen.lowering import IRLowering
from instruction_gen.code_emitter import CodeEmitter


def compile_model(model_path, output_dir="output"):
    """Full compilation pipeline: ONNX -> IR -> ISA"""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CIM Compiler - Track 2 Problem 1")
    print("=" * 60)

    # Step 1: Parse ONNX -> IR
    print("\n[Step 1] Parsing ONNX model...")
    parser = ONNXParser(model_path)
    ir = parser.parse()
    print(f"  Nodes: {len(ir.nodes)}")
    print(f"  Tensors: {len(ir.tensors)}")
    print(f"  Inputs: {ir.input_names}")
    print(f"  Outputs: {ir.output_names}")

    # Save IR intermediate result
    ir_path = os.path.join(output_dir, "ir.json")
    ir.serialize(ir_path)
    print(f"  IR saved to: {ir_path}")

    # Step 2: Optimize
    print("\n[Step 2] Optimizing IR...")
    optimizer = ConstantFolding()
    ir = optimizer.optimize(ir)
    print("  Optimization complete")

    # Step 3: SRAM Allocation
    print("\n[Step 3] Allocating SRAM (512KB)...")
    allocator = SRAMAllocator()
    layout = allocator.allocate(ir)
    print(layout.report())

    # Save SRAM layout
    layout_path = os.path.join(output_dir, "sram_layout.json")
    import json
    layout_data = {
        "regions": [
            {"name": r.name, "start": r.start, "size": r.size, "tensor": r.tensor_name}
            for r in layout.regions
        ],
        "total_used": layout.total_used,
    }
    with open(layout_path, 'w') as f:
        json.dump(layout_data, f, indent=2)

    # Step 4: Weight Mapping
    print("\n[Step 4] Mapping weights to CIM array...")
    w_mapper = WeightMapper()
    mappings = w_mapper.map_weights(ir)
    print(w_mapper.report())

    # Save weight mapping
    wm_path = os.path.join(output_dir, "weight_mapping.json")
    wm_data = {
        "array": {"row_bits": 1024, "col_bits": 4096},
        "mappings": [m.to_dict() for m in mappings],
    }
    with open(wm_path, 'w') as f:
        json.dump(wm_data, f, indent=2)

    # Step 5: Lowering IR -> ISA
    print("\n[Step 5] Lowering IR to ISA...")
    lowering = IRLowering(allocator, mappings)
    instructions = lowering.lower(ir)
    print(f"  Generated {len(instructions)} lines (including comments)")
    real_instrs = [i for i in instructions if i.opcode != "//"]
    print(f"  Real instructions: {len(real_instrs)}")

    # Step 6: Emit
    asm_path = os.path.join(output_dir, "output.asm")
    emitter = CodeEmitter(asm_path)
    emitter.emit(instructions)
    print(f"  Assembly saved to: {asm_path}")

    json_out_path = os.path.join(output_dir, "output.json")
    emitter.emit_json(json_out_path, instructions, extra={
        "sram_layout": layout_data,
        "weight_mapping": wm_data,
    })
    print(f"  JSON saved to: {json_out_path}")

    print("\n" + "=" * 60)
    print("Compilation completed!")
    print("=" * 60)
    return instructions


def main():
    parser = argparse.ArgumentParser(description="CIM Compiler - Problem 1")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    compile_model(args.model, args.output)


if __name__ == "__main__":
    main()
