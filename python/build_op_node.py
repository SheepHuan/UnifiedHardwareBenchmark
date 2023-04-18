import onnx
import numpy as np
import onnxruntime as ort
# 创建输入、卷积核和输出张量（形状为[N, C, H, W]）
input_tensor = onnx.helper.make_tensor_value_info('conv_input', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
output_tensor = onnx.helper.make_tensor_value_info('conv_output', onnx.TensorProto.FLOAT, [1, 64, 112, 112])
kernel_tensor = onnx.helper.make_tensor('conv_weight', onnx.TensorProto.FLOAT, [64, 3, 7, 7], np.random.random([64, 3, 7, 7]))

# 创建卷积节点
conv_node = onnx.helper.make_node(
    op_type='Conv', # 节点类型
    inputs=['conv_input', 'conv_weight'], # 输入张量的名称
    outputs=['conv_output'], # 输出张量的名称
    kernel_shape=[7, 7], # 卷积核大小
    pads=[3, 3, 3, 3], # 填充大小
    strides=[2, 2] # 步长大小
)

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
