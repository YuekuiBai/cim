"""
Run all tests for Problem 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import compile_model

print("=" * 60)
print("Running all Problem 2 tests")
print("=" * 60)

# Test 1: Simple model (no MoE)
print("\n[Test 1] Simple 3-layer model")
try:
    compile_model("test_models/simple_model.json", output_dir="output_simple")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: MoE model with activation trace
print("\n[Test 2] MoE model (256 experts) + activation trace")
try:
    compile_model(
        "test_models/moe_model.json",
        trace_path="test_models/activation_trace.json",
        output_dir="output_moe"
    )
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("All Problem 2 tests completed")
print("=" * 60)
