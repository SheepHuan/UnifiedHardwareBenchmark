"""
1. 先暴力生成onnx模型到/mnt/scrard
2. 基于benchmark获得数据
3. 获取到数据后集中分析
"""
from ppadb.client import Client as AdbClient
# Default is "127.0.0.1" and 5037
client = AdbClient(host="127.0.0.1", port=5037)
device = client.device("emulator-5554")
