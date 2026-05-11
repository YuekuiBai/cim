"""
Run all tests for Problem 1
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import compile_model

test_models = [
    "test_models/model_linear.onnx",
    "test_models/model_linear_bias.onnx",
    "test_models/model_linear_add.onnx",
    "test_models/model_two_linear_add.onnx",
    "test_models/model_linear_mul.onnx",
    "test_models/model_linear_sub_div.onnx",
]

print("=" * 60)
print("Running all Problem 1 tests")
print("=" * 60)

for model_path in test_models:
    name = os.path.basename(model_path).replace(".onnx", "")
    print(f"\n{'='*40}")
    print(f"Testing: {name}")
    print(f"{'='*40}")

    output_dir = f"output_{name}"
    try:
        instructions = compile_model(model_path, output_dir)
        real_count = len([i for i in instructions if i.opcode != "//"])
        print(f"  PASS: {real_count} real instructions generated")
    except Exception as e:
        print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("All Problem 1 tests completed")
print("=" * 60)
