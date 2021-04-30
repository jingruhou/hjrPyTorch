# coding: utf-8
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

# 汉字字体相关
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

"""
    @Time    : 2021/4/29 15:26
    @Author  : houjingru@semptian.com
    @FileName: 2_Datasets.py
    @Software: PyCharm
"""

"""
    一：加载数据集
"""

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
"""
    二：迭代和可视化数据
"""
labels_map = {
    0: "T-Shirt:T恤",
    1: "Trouser:长裤",
    2: "Pullover:套衫",
    3: "Dress:着装",
    4: "Coat:外套",
    5: "Sandal:凉鞋",
    6: "Shirt:衬衫",
    7: "Sneaker:运动鞋",
    8: "Bag:包",
    9: "Ankle Boot:踝靴",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()