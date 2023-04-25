import onnx
import numpy as np
import onnxruntime as ort


# input_hw = [8192,4096,2048,1024,512,384]
input_hw = [6144,7168,3072,5120,896,768]
conv_ksize = [3,5,7,9,11,13,15,17,19,21,23,25]
conv_stride = [1,2,3,4,5,7,9,11]

# 不考虑填充
def make_and_run_conv_node(in_n,in_c,in_h,in_w,kernel_size,kernel_stride,kernel_num,padding):
    # 创建输入、卷积核和输出张量（形状为[N, C, H, W]）
    
    # 输入参数
    # in_n,in_c,in_h,in_w = 1,3,224,224
    # kernel_size = 3
    # kernel_stride = 2
    # kernel_num = 1
    # padding = 0
    
    # 输出参数
    out_w =int((in_w - kernel_size + 2 *padding) / kernel_stride) + 1
    out_h =int((in_h - kernel_size+ 2 *padding) / kernel_stride )+ 1
    out_n = in_n
    out_c = kernel_num
    
    input_tensor = onnx.helper.make_tensor_value_info('conv_input', onnx.TensorProto.FLOAT, [in_n, in_c, in_h, in_w])
    output_tensor = onnx.helper.make_tensor_value_info('conv_output', onnx.TensorProto.FLOAT, [out_n, out_c,out_h, out_w])
    kernel_tensor = onnx.helper.make_tensor('conv_weight', onnx.TensorProto.FLOAT, [kernel_num, in_c, kernel_size, kernel_size], np.random.random([kernel_num, in_c, kernel_size, kernel_size]))

    # 创建卷积节点
    conv_node = onnx.helper.make_node(
        op_type='Conv', # 节点类型
        inputs=['conv_input', 'conv_weight'], # 输入张量的名称
        outputs=['conv_output'], # 输出张量的名称
        name = "0.conv",
        kernel_shape=[kernel_size,kernel_size], # 卷积核大小
        pads=[padding,padding,padding,padding], # 填充大小
        strides=[kernel_stride, kernel_stride] # 步长大小
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
    
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = f"profile/hw_{in_h}-ksize_{kernel_size}-knum_{kernel_num}-kstride_{kernel_stride}"

    ro = ort.RunOptions()
    ro.only_execute_path_to_fetches = True

    sess = ort.InferenceSession('test_model.onnx',so)
    input_tensor =np.random.random([in_n, in_c, in_h, in_w]).astype(np.float32)
    # 运行指定节点
    for i in range(10):
         sess.run(["conv_output"], {"conv_input": input_tensor},ro)



in_n = 1
in_c = 3

for hw in input_hw:
    for ksize in conv_ksize:
        for kstride in conv_stride:
            if ksize >= hw:
                continue
            make_and_run_conv_node(in_n,in_c,hw,hw,ksize,kstride,1,0)