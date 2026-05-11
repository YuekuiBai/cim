"""
Test ONNX Model Generator for Problem 1
Generates: linear-only, linear+add, linear+sub, multi-layer models
All with int8/integer types as per competition spec
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import os

def create_linear_model(onnx_path, batch=1, cin=128, cout=128, has_bias=False):
    """Create a simple linear (MatMul) model"""

    x = helper.make_tensor_value_info('input', TensorProto.INT8, [batch, 1, cin])
    y = helper.make_tensor_value_info('output', TensorProto.INT32, [batch, 1, cout])

    weight_data = np.random.randint(-5, 5, size=(cin, cout)).astype(np.int8)
    weight_tensor = numpy_helper.from_array(weight_data, name='weight_B')

    nodes = [
        helper.make_node('MatMul', ['input', 'weight_B'], ['output'], name='matmul_0'),
    ]

    graph = helper.make_graph(
        nodes, 'test_linear',
        [x],
        [y],
        [weight_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model = shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    print(f"Created: {onnx_path}  [Linear {cin}->{cout}, batch={batch}, bias={has_bias}]")


def create_linear_with_bias_model(onnx_path, batch=1, cin=128, cout=64):
    """Create a linear model with bias (Gemm)"""

    x = helper.make_tensor_value_info('input', TensorProto.INT8, [batch, 1, cin])
    y = helper.make_tensor_value_info('output', TensorProto.INT32, [batch, 1, cout])

    weight_data = np.random.randint(-3, 3, size=(cin, cout)).astype(np.int8)
    bias_data = np.random.randint(-10, 10, size=(cout,)).astype(np.int32)

    weight_tensor = numpy_helper.from_array(weight_data, name='W')
    bias_tensor = numpy_helper.from_array(bias_data, name='bias')

    nodes = [
        helper.make_node('Gemm', ['input', 'W', 'bias'], ['output'], name='gemm_0', alpha=1.0, beta=1.0, transB=1),
    ]

    graph = helper.make_graph(
        nodes, 'test_linear_bias',
        [x],
        [y],
        [weight_tensor, bias_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model = shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    print(f"Created: {onnx_path}  [Gemm {cin}->{cout}+bias, batch={batch}]")


def create_linear_add_model(onnx_path, batch=1, cin=128, cout=64):
    """Create: Linear -> Add"""

    x = helper.make_tensor_value_info('input', TensorProto.INT8, [batch, 1, cin])
    y = helper.make_tensor_value_info('output', TensorProto.INT32, [batch, 1, cout])

    weight_data = np.random.randint(-5, 5, size=(cin, cout)).astype(np.int8)
    weight_tensor = numpy_helper.from_array(weight_data, name='W')

    add_const = np.array(1, dtype=np.int32)
    add_tensor = numpy_helper.from_array(add_const, name='add_const')

    nodes = [
        helper.make_node('MatMul', ['input', 'W'], ['matmul_out'], name='matmul_0'),
        helper.make_node('Add', ['matmul_out', 'add_const'], ['output'], name='add_0'),
    ]

    graph = helper.make_graph(
        nodes, 'test_linear_add',
        [x],
        [y],
        [weight_tensor, add_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model = shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    print(f"Created: {onnx_path}  [Linear->Add {cin}->{cout}]")


def create_two_linear_add_model(onnx_path, batch=1, cin=128, hidden=64, cout=32):
    """Create: Linear1 -> Linear2 -> Add"""

    x = helper.make_tensor_value_info('input', TensorProto.INT8, [batch, 1, cin])
    y = helper.make_tensor_value_info('output', TensorProto.INT32, [batch, 1, cout])

    w1 = np.random.randint(-3, 3, size=(cin, hidden)).astype(np.int8)
    w2 = np.random.randint(-3, 3, size=(hidden, cout)).astype(np.int8)
    w1_t = numpy_helper.from_array(w1, name='W1')
    w2_t = numpy_helper.from_array(w2, name='W2')

    add_const = np.array(1, dtype=np.int32)
    add_tensor = numpy_helper.from_array(add_const, name='add_const')

    nodes = [
        helper.make_node('MatMul', ['input', 'W1'], ['hidden'], name='matmul_0'),
        helper.make_node('MatMul', ['hidden', 'W2'], ['matmul_out'], name='matmul_1'),
        helper.make_node('Add', ['matmul_out', 'add_const'], ['output'], name='add_0'),
    ]

    graph = helper.make_graph(
        nodes, 'test_two_linear_add',
        [x],
        [y],
        [w1_t, w2_t, add_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model = shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    print(f"Created: {onnx_path}  [Linear->Linear->Add {cin}->{hidden}->{cout}]")


def create_linear_mul_model(onnx_path, batch=1, cin=64, cout=64):
    """Create: Linear -> Mul (vector * immediate)"""

    x = helper.make_tensor_value_info('input', TensorProto.INT8, [batch, 1, cin])
    y = helper.make_tensor_value_info('output', TensorProto.INT32, [batch, 1, cout])

    weight_data = np.random.randint(-3, 3, size=(cin, cout)).astype(np.int8)
    weight_tensor = numpy_helper.from_array(weight_data, name='W')

    mul_const = np.array(2, dtype=np.int32)
    mul_tensor = numpy_helper.from_array(mul_const, name='mul_const')

    nodes = [
        helper.make_node('MatMul', ['input', 'W'], ['matmul_out'], name='matmul_0'),
        helper.make_node('Mul', ['matmul_out', 'mul_const'], ['output'], name='mul_0'),
    ]

    graph = helper.make_graph(
        nodes, 'test_linear_mul',
        [x],
        [y],
        [weight_tensor, mul_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model = shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    print(f"Created: {onnx_path}  [Linear->Mul {cin}->{cout}]")


def create_linear_sub_div_model(onnx_path, batch=1, cin=64, cout=64):
    """Create: Linear -> Sub -> Div"""

    x = helper.make_tensor_value_info('input', TensorProto.INT8, [batch, 1, cin])
    y = helper.make_tensor_value_info('output', TensorProto.INT32, [batch, 1, cout])

    weight_data = np.random.randint(-3, 3, size=(cin, cout)).astype(np.int8)
    weight_tensor = numpy_helper.from_array(weight_data, name='W')

    sub_const = np.array(10, dtype=np.int32)
    sub_tensor = numpy_helper.from_array(sub_const, name='sub_const')

    div_const = np.array(2, dtype=np.int32)
    div_tensor = numpy_helper.from_array(div_const, name='div_const')

    nodes = [
        helper.make_node('MatMul', ['input', 'W'], ['matmul_out'], name='matmul_0'),
        helper.make_node('Sub', ['matmul_out', 'sub_const'], ['sub_out'], name='sub_0'),
        helper.make_node('Div', ['sub_out', 'div_const'], ['output'], name='div_0'),
    ]

    graph = helper.make_graph(
        nodes, 'test_linear_sub_div',
        [x],
        [y],
        [weight_tensor, sub_tensor, div_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model = shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)
    print(f"Created: {onnx_path}  [Linear->Sub->Div {cin}->{cout}]")


def generate_all(output_dir="test_models"):
    """Generate all test models"""
    os.makedirs(output_dir, exist_ok=True)

    create_linear_model(os.path.join(output_dir, "model_linear.onnx"), batch=1, cin=128, cout=128)
    create_linear_with_bias_model(os.path.join(output_dir, "model_linear_bias.onnx"), batch=1, cin=128, cout=64)
    create_linear_add_model(os.path.join(output_dir, "model_linear_add.onnx"), batch=1, cin=128, cout=64)
    create_two_linear_add_model(os.path.join(output_dir, "model_two_linear_add.onnx"), batch=1, cin=128, hidden=64, cout=32)
    create_linear_mul_model(os.path.join(output_dir, "model_linear_mul.onnx"), batch=1, cin=64, cout=64)
    create_linear_sub_div_model(os.path.join(output_dir, "model_linear_sub_div.onnx"), batch=1, cin=64, cout=64)

    print(f"\nAll test models generated in: {output_dir}/")


if __name__ == "__main__":
    generate_all()
