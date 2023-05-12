import json
import glob
import matplotlib.pyplot as plt
import numpy as np

files = glob.glob("/root/workspace/UnifiedModelBenchmark/tmp/matmul_profile/profile/*.json")

# 1. 选择一个ksize查看图像shape的影响
shape = []
time  = []
for file in files:
    # print(file)
    try:
        res = json.load(open(file,"r"))
    
        items = file.split("_")
        hw1 = items[2]
        hw2 = items[3]
       
        conv_time = []
        for node in res:
            if node["cat"]=="Node" and node["name"]=="matmul.0_kernel_time":
                conv_time.append(node["dur"])
        conv_time = np.average( np.asanyarray(conv_time[2:]))
        
        shape.append(f"{hw1} * {hw2}")
        time.append(conv_time)
    except Exception as e:
        print(e)

# print(1)


plt.plot(shape, time,'ro')
# 设置 x 轴坐标刻度为整数
# plt.xticks(range(min(o_shape_res), max(o_shape_res)+1))
# 设置 X 轴和 Y 轴标签
plt.xlabel('shape')
plt.ylabel('time (ms)')

# 设置图标标题
plt.title(f'matmul')

# 显示图像
plt.savefig(f'matmul.png')


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