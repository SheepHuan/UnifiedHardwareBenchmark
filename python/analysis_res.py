import json
import glob
import matplotlib.pyplot as plt
import numpy as np

files = glob.glob("/root/workspace/UnifiedModelBenchmark/tmp/conv2_profile/profile/*.json")

# 1. 选择一个ksize查看图像shape的影响
o_ksize = 3
o_ksize_res = []
o_ksize_res_time = []
# 2. 选择一个shape查看ksize的影响
o_shape = 512
o_shape_res = []
o_shape_res_time = []
for file in files:
    # print(file)
    try:
        res = json.load(open(file,"r"))
    
        items = file.split("_")
        hw = int(items[2])
        ksize = int(items[3])
        stride = int(items[4])
        padding = int(items[5])
       
        conv_time = []
        for node in res:
            if node["cat"]=="Node" and node["name"]=="conv.0_kernel_time":
                conv_time.append(node["dur"])
        conv_time = np.average( np.asanyarray(conv_time[2:]))
        
        if ksize == o_ksize  and stride==1:
            o_ksize_res.append(hw)
            print(hw,ksize,1,stride,conv_time)
            o_ksize_res_time.append(conv_time)
        if hw == o_shape and stride==1:
            o_shape_res.append(ksize)
            
            o_shape_res_time.append(conv_time)
    except Exception as e:
        print(e)

# print(1)


plt.plot(o_shape_res, o_shape_res_time,'ro')
# 设置 x 轴坐标刻度为整数
plt.xticks(range(min(o_shape_res), max(o_shape_res)+1))
# 设置 X 轴和 Y 轴标签
plt.xlabel('kernel size')
plt.ylabel('time (ms)')

# 设置图标标题
plt.title(f'shape={o_shape}')

# 显示图像
plt.savefig(f'shape={o_shape}.png')


# plt.plot(o_ksize_res, o_ksize_res_time,'ro')
# # 设置 x 轴坐标刻度为整数
# # plt.xticks(range(min(o_ksize_res), max(o_ksize_res)+1))
# # 设置 X 轴和 Y 轴标签
# plt.xlabel('image size')
# plt.ylabel('time (ms)')

# # 设置图标标题
# plt.title(f'kernel size={o_ksize}')

# # 显示图像
# plt.savefig(f"kernel size={o_ksize}.png")