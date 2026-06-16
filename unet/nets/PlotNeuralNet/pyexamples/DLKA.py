# 导入必要的库
import matplotlib.pyplot as plt
from nets.PlotNeuralNet import PlotNeuralNet

# 定义网络结构
network_structure = [
    {"name": "Input", "type": "rectangle", "width": 4, "height": 3},
    {"name": "Conv2D 1x1", "type": "conv", "filters": 64, "size": 1},
    {"name": "GELU", "type": "gelu"},
    {"name": "Conv 5x5 offsets", "type": "conv", "filters": 128, "size": 5},
    {"name": "Weighted Summation", "type": "sum"},
    {"name": "Conv 7x7 offsets", "type": "conv", "filters": 256, "size": 7},
    {"name": "Weighted Summation", "type": "sum"},
    {"name": "Pointwise Multiply", "type": "mul"},
    {"name": "Conv2D 1x1", "type": "conv", "filters": 64, "size": 1},
    {"name": "Conv2D 1x1", "type": "conv", "filters": 64, "size": 1},
    {"name": "Add(Shortcut)", "type": "add"},
    {"name": "Shortcut(Clone)", "type": "clone"},
    {"name": "Output", "type": "rectangle", "width": 4, "height": 3}
]

# 创建绘图对象
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制网络结构图
plot = PlotNeuralNet(network_structure, figsize=(12, 8))
plot.draw(ax)

# 显示图形
plt.show()
