import onnx
import numpy as np
import onnxruntime as ort


def make_conv_node():
    # 创建输入、卷积核和输出张量（形状为[N, C, H, W]）
    input_tensor = onnx.helper.make_tensor_value_info('conv_input', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = onnx.helper.make_tensor_value_info('conv_output', onnx.TensorProto.FLOAT, [1, 64, 112, 112])
    kernel_tensor = onnx.helper.make_tensor('conv_weight', onnx.TensorProto.FLOAT, [64, 3, 7, 7], np.random.random([64, 3, 7, 7]))

    # 创建卷积节点
    conv_node = onnx.helper.make_node(
        op_type='Conv', # 节点类型
        inputs=['conv_input', 'conv_weight'], # 输入张量的名称
        outputs=['conv_output'], # 输出张量的名称
        name = "0.conv",
        kernel_shape=[7, 7], # 卷积核大小
        pads=[3, 3, 3, 3], # 填充大小
        strides=[2, 2] # 步长大小
    )
    # return conv_node

    # 创建ONNX图并添加节点
    graph_def = onnx.helper.make_graph(
        [conv_node], # 节点列表
        'test_model', # 图的名称
        [input_tensor], # 输入张量列表
        [output_tensor], # 输出张量列表
        [kernel_tensor] # 可选的初始化张量列表
    )

    # 创建ONNX模型
    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-example')

    # # 保存模型
    onnx.save(model_def, 'test_model.onnx')
    return np.random.random([1, 3, 224, 224]).astype(np.float32), np.random.random( [1, 64, 112, 112])



so = ort.SessionOptions()
so.enable_profiling = True
so.profile_file_prefix = "profile"

ro = ort.RunOptions()
ro.only_execute_path_to_fetches = True
input_tensor,output_tensor = make_conv_node()

sess = ort.InferenceSession('test_model.onnx',so)

# 运行指定节点
sess.run(["conv_output"], {"conv_input": input_tensor},ro)



