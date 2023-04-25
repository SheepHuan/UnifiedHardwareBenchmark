import json
import glob
import matplotlib.pyplot as plt
import numpy as np

files = glob.glob("profile/*.json")

# 1. 选择一个ksize查看图像shape的影响
o_ksize = 11
o_ksize_res = []
o_ksize_res_time = []
# 2. 选择一个shape查看ksize的影响
o_shape = 8192
o_shape_res = []
o_shape_res_time = []
for file in files:
    res = json.load(open(file,"r"))
    try:
        items = file.split("-")
        hw = int(items[0].split("_")[-1])
        ksize = int(items[1].split("_")[-1])
        knum = int(items[2].split("_")[-1])
        kstride = int(items[3].split("_")[-2])
       
        conv_time = []
        for node in res:
            if node["cat"]=="Node" and node["name"]=="conv_output_nchwc_kernel_time":
                conv_time.append(node["dur"])
        conv_time = np.average( np.asanyarray(conv_time[2:])) / 1000.0
        
        if ksize == o_ksize  and kstride==1:
            o_ksize_res.append(hw)
            print(hw,ksize,knum,kstride,conv_time)
            o_ksize_res_time.append(conv_time)
        if hw == o_shape and kstride==1:
            o_shape_res.append(ksize)
            
            o_shape_res_time.append(conv_time)
    except:
        print(file)

print(1)


# plt.plot(o_shape_res, o_shape_res_time,'ro')
# # 设置 x 轴坐标刻度为整数
# plt.xticks(range(min(o_shape_res), max(o_shape_res)+1))
# # 设置 X 轴和 Y 轴标签
# plt.xlabel('kernel size')
# plt.ylabel('time (ms)')

# # 设置图标标题
# plt.title(f'shape={o_shape}')

# # 显示图像
# plt.savefig("1.png")


plt.plot(o_ksize_res, o_ksize_res_time,'ro')
# 设置 x 轴坐标刻度为整数
# plt.xticks(range(min(o_ksize_res), max(o_ksize_res)+1))
# 设置 X 轴和 Y 轴标签
plt.xlabel('image size')
plt.ylabel('time (ms)')

# 设置图标标题
plt.title(f'kernel size={11}')

# 显示图像
plt.savefig("2.png")