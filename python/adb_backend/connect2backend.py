"""
1. 先暴力生成onnx模型到/mnt/scrard
2. 基于benchmark获得数据
3. 获取到数据后集中分析
"""
from ppadb.client import Client as AdbClient
# Default is "127.0.0.1" and 5037
client = AdbClient(host="127.0.0.1", port=5037)
device = client.device("3a9c4f5")
# device.shell("mkdir -p /mnt/sdcard/ort_models")
workspace = "D:/yanghuan/code/workspace/ort_models/conv2d/*.onnx"
device.shell(f"adb push --sync -z {workspace} /mnt/sdcard/ort_models")
