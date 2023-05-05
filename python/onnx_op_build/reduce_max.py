import onnx
import numpy as np
import onnxruntime as ort
from typing import List
import random

def make_reduce_max_node(input_shape: List[int], axes_shape: List[int], keepdims: int = 0, type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT, index=0):

    input_tensor_name = f'reduce_max_input.{index}'
    axes_tensor_name = f'reduce_max_axes.{index}'
    output_tensor_name = f'reduce_max_output.{index}'
    input_tensor_info = onnx.helper.make_tensor_value_info(
        input_tensor_name, type, input_shape)
    axes_tensor_info = onnx.helper.make_tensor_value_info(
        axes_tensor_name, type, axes_shape)

    input_data = np.array(input_shape, dtype=np.float32)
    axes = np.array([random.randint(0, len(input_shape)-1) for i in range(len(axes_shape))], dtype=np.int64)
    reduced = np.maximum.reduce(
        input_data, axis=tuple(axes), keepdims=keepdims == 1)

    output_tensor_info = onnx.helper.make_tensor_value_info(
        output_tensor_name, type, reduced.shape)

    # 创建卷积节点
    reduce_max_node = onnx.helper.make_node(
        op_type='ReduceMax',  # 节点类型
        inputs=[input_tensor_name, axes_tensor_name],  # 输入张量的名称
        outputs=[output_tensor_name],  # 输出张量的名称
        name=f"reduce_max.{index}",
        keepdims=keepdims
    )

    return reduce_max_node, [input_tensor_info,axes_tensor_info], [output_tensor_info], []


def run_reduce_max_node(node, input_tensors_info, output_tensors_info):
    graph_def = onnx.helper.make_graph(
        [node],  # 节点列表
        'test_model',  # 图的名称
        input_tensors_info,  # 输入张量列表
        output_tensors_info,  # 输出张量列表
       
    )
    #
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-example')
    onnx.save_model(model_def, "tmp/model.onnx")
    # so = ort.SessionOptions()
    # so.enable_profiling = True
    # so.profile_file_prefix = f"tmp/conv"
    # sess = ort.InferenceSession("tmp/model.onnx", so)
    # input = sess.get_inputs()[0]
    # output = sess.get_outputs()[0]
    # input_shape = input.shape
    # input_tensor = np.random.random(input_shape).astype(np.float32)
    # for i in range(10):
    #     out = sess.run([output.name], {input.name: input_tensor})


if __name__ == "__main__":
    reduce_max_node, input_tensors_info, output_tensors_info, _ = make_reduce_max_node([6,2],[1])
    run_reduce_max_node(reduce_max_node,input_tensors_info,output_tensors_info)
    # onnx.save(model_def, 'test_model.onnx')
