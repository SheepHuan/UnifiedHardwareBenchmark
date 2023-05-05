import onnx
from onnx.checker import check_model
import numpy as np
import onnxruntime as ort
from typing import List
import itertools

def make_transpose_node(input_shape: List[int], permutation: List[int], type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT, index=0):
    input_tensor_name = f'transpose_input.{index}'
    output_tensor_name = f'transpose_output.{index}'
    
    # # TODO 检查a * b
    input_tensor_info = onnx.helper.make_tensor_value_info(
        input_tensor_name, type, input_shape)

    # 显然outout_shape是重排列之后的
    data = np.random.random_sample(input_shape).astype(np.float32)
    transposed = np.transpose(data, permutation)
    output_tensor_info = onnx.helper.make_tensor_value_info(
        output_tensor_name, type, transposed.shape)

    matmul_node = onnx.helper.make_node(
        op_type='Transpose',  # 节点类型
        inputs=[input_tensor_name],  # 输入张量的名称
        outputs=[output_tensor_name],  # 输出张量的名称
        name=f"transpose.{index}"
    )
    return matmul_node, [input_tensor_info], [output_tensor_info]




def run_transpose_node(node, input_tensors_info, output_tensor_info):
    # 先构图
    graph_proto = onnx.helper.make_graph(
        [node],  # 节点列表
        'test_model',  # 图的名称
        input_tensors_info,  # 输入张量列表
        output_tensor_info,  # 输出张量列表
    )

    model_proto = onnx.helper.make_model(graph_proto, producer_name='onnx-example')
    # model_proto.opset_import[0].version = 15
    onnx.save_model(model_proto, "tmp/model.onnx")
    # model = onnx.load_model( "tmp/model.onnx")
 
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = f"tmp/transpose"

    sess = ort.InferenceSession("tmp/model.onnx", so)
    input= sess.get_inputs()[0]
    output = sess.get_outputs()[0]
    input_shape = input.shape

    input_tensor = np.random.random(input_shape).astype(np.float32)
    # 运行指定节点
    for i in range(10):
        c = sess.run([output.name], {input.name: input_tensor})
        print(c[0].shape)
 

if __name__ == "__main__":  
    # transpose_node, input_tensors_info, output_tensors_info = make_transpose_node([100, 256,3], [0,2,1])
    # run_transpose_node(transpose_node, input_tensors_info, output_tensors_info)
    shape =[100, 256,3]
    permutations = list(itertools.permutations(np.arange(len(shape))))
    transpose_node, input_tensors_info, output_tensors_info = make_transpose_node([3,10], [0,1])
    run_transpose_node(transpose_node, input_tensors_info, output_tensors_info)
    print(permutations)
