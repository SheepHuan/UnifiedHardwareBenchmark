import onnx
import numpy as np
import onnxruntime as ort
from typing import List


def make_matmul_4d_node(input_a_shape: List[int], input_b_shape: List[int], type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT, index=0):
    pass
    # input_c_tensor_info = onnx.helper.make_tensor_value_info(input_a_tensor_name,type,input_a_shape)


def make_matmul_3d_node(input_a_shape: List[int], input_b_shape: List[int], type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT, index=0):
    pass


def make_matmul_2d_node(input_a_shape: List[int], input_b_shape: List[int], type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT, index=0):
    input_a_tensor_name = f'matmul_input_a.{index}'
    input_b_tensor_name = f'matmul_input_b.{index}'
    output_c_tensor_name = f'conv_output_c.{index}'

    height_a, width_a = input_a_shape
    height_b, width_b = input_b_shape
    # 确保[height_a,width_a] * [height_b,width_b]
    assert width_a == height_b
    output_c_shape = height_a, width_b
    # TODO 检查a * b
    input_a_tensor_info = onnx.helper.make_tensor_value_info(
        input_a_tensor_name, type, input_a_shape)
    input_b_tensor_info = onnx.helper.make_tensor_value_info(
        input_b_tensor_name, type, input_b_shape)
    output_c_tensor_info = onnx.helper.make_tensor_value_info(
        output_c_tensor_name, type, output_c_shape)

    matmul_node = onnx.helper.make_node(
        op_type='MatMul',  # 节点类型
        inputs=[input_a_tensor_name, input_b_tensor_name],  # 输入张量的名称
        outputs=[output_c_tensor_name],  # 输出张量的名称
        name=f"matmul.{index}"
    )
    return matmul_node, [input_a_tensor_info, input_b_tensor_info], [output_c_tensor_info]




def run_matmul_node(node, input_tensors_info, output_tensor_info):
    # 先构图
    graph_def = onnx.helper.make_graph(
        [node],  # 节点列表
        'test_model',  # 图的名称
        input_tensors_info,  # 输入张量列表
        output_tensor_info,  # 输出张量列表
    )

    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-example')
    onnx.save_model(model_def, "tmp/model.onnx")

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = f"tmp/matmul"

    sess = ort.InferenceSession("tmp/model.onnx", so)
    input_a, input_b = sess.get_inputs()
    output_c = sess.get_outputs()[0]
    input_a_shape = input_a.shape
    input_b_shape = input_b.shape

    input_a_tensor = np.random.random(input_a_shape).astype(np.float32)
    input_b_tensor = np.random.random(input_b_shape).astype(np.float32)
    # 运行指定节点
    for i in range(10):
        c = sess.run([output_c.name], {input_a.name: input_a_tensor,input_b.name : input_b_tensor})


if __name__ == "__main__":

    matmul_node, input_tensors_info, output_tensors_info = make_matmul_2d_node([
                                                                               128, 256], [256, 128])
    run_matmul_node(matmul_node, input_tensors_info, output_tensors_info)
