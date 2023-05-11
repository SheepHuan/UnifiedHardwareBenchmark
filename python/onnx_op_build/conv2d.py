import onnx
import numpy as np
import onnxruntime as ort
from typing import List


def make_conv2d_node(input_shape: List[int], conv_kernel_size: int, conv_stride: int, conv_kernel_num: int, padding: int, type : onnx.TensorProto.DataType = onnx.TensorProto.FLOAT, index=0):

    # 创建输入、卷积核和输出张量（形状为[N, C, H, W]）
    input_batch, input_channel, input_height, input_width = input_shape
    # 输出参数
    out_w = int((input_width - conv_kernel_size +
                2 * padding) / conv_stride) + 1
    out_h = int((input_height - conv_kernel_size +
                2 * padding) / conv_stride) + 1
    out_n = input_batch
    out_c = conv_kernel_num

    input_tensor_name = f'conv_input.{index}'
    weight_tensor_name = f'conv_weight.{index}'
    output_tensor_name = f'conv_output.{index}'
    input_tensor_info = onnx.helper.make_tensor_value_info(input_tensor_name, onnx.TensorProto.FLOAT, [
                                                           input_batch, input_channel, input_height, input_width])
    output_tensor_info = onnx.helper.make_tensor_value_info(
        output_tensor_name, onnx.TensorProto.FLOAT, [out_n, out_c, out_h, out_w])
    weight_tensor = onnx.helper.make_tensor(weight_tensor_name, onnx.TensorProto.FLOAT, [
                                            conv_kernel_num, input_channel, conv_kernel_size, conv_kernel_size], np.random.random([conv_kernel_num, input_channel, conv_kernel_size, conv_kernel_size]))

    # 创建卷积节点
    conv_node = onnx.helper.make_node(
        op_type='Conv',  # 节点类型
        inputs=[input_tensor_name, weight_tensor_name],  # 输入张量的名称
        outputs=[output_tensor_name],  # 输出张量的名称
        name=f"conv.{index}",
        kernel_shape=[conv_kernel_size, conv_kernel_size],  # 卷积核大小
        pads=[padding, padding, padding, padding],  # 填充大小
        strides=[conv_stride, conv_stride]  # 步长大小
    )

    return conv_node, [input_tensor_info], [output_tensor_info], [weight_tensor]


def run_conv_node(node,input_tensors_info,output_tensors_info,weight_tensors):
    # 先构图
    graph_def = onnx.helper.make_graph(
        [node],  # 节点列表
        'conv2d_model',  # 图的名称
        input_tensors_info,  # 输入张量列表
        output_tensors_info,  # 输出张量列表
        weight_tensors  # 可选的初始化张量列表
    )
    # 
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-example')
    model_def.ir_version = 8
    model_def.opset_import[0].version = 18
    onnx.save_model(model_def, f"{workspace}/model.onnx")
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = f"{workspace}/conv"
    sess = ort.InferenceSession(f"{workspace}/model.onnx", so)
    input = sess.get_inputs()[0]
    output = sess.get_outputs()[0]
    input_shape = input.shape
    input_tensor = np.random.random(input_shape).astype(np.float32)
    for i in range(10):
        out = sess.run([output.name], {input.name: input_tensor})

def save_conv_node(node,input_tensors_info,output_tensors_info,weight_tensors,hw,ksize,stride):
    graph_def = onnx.helper.make_graph(
        [node],  # 节点列表
        'conv2d_model',  # 图的名称
        input_tensors_info,  # 输入张量列表
        output_tensors_info,  # 输出张量列表
        weight_tensors  # 可选的初始化张量列表
    )
    # 
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-example')
    model_def.ir_version = 8
    model_def.opset_import[0].version = 18
    # hw_ksize_stride_padding_dilation
    onnx.save_model(model_def, f"{workspace}/conv2d_{hw}_{ksize}_{stride}_{1}_{1}.onnx")


if __name__ == "__main__":
    workspace = "D:/yanghuan/code/UnifiedHardwareBenchmark/workspace/ort_models"
    input_hw = [8192,4096,2048,1024,512,384]
    input_hws = [512,384,256,192,128]
    conv2d_ksizes = [3,5,7,9,11,13,15,17,19,21,23,25]
    conv2d_strides = [1,2,3,4,5,7,9,11]
    
    # input_hws = [128]
    # conv2d_ksizes = [3]
    # conv2d_strides = [1]

    for input_hw in input_hws:
        for conv2d_ksize in conv2d_ksizes:
            for conv2d_stride in conv2d_strides:
                conv2d_node, input_tensors_info, output_tensors_info, weight_tensors = make_conv2d_node(
                    [1, 3, input_hw, input_hw], conv2d_ksize, conv2d_stride, 1, 0, index=0)
                # run_conv_node(conv2d_node,input_tensors_info,output_tensors_info,weight_tensors)
                save_conv_node(conv2d_node,input_tensors_info,output_tensors_info,weight_tensors,input_hw,conv2d_ksize,conv2d_stride)
   
    
    # onnx.save(model_def, 'test_model.onnx')
