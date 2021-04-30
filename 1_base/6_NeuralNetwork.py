# coding: utf-8
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
    @Time    : 2021/4/30 10:41
    @Author  : houjingru@semptian.com
    @FileName: 6_NeuralNetwork.py
    @Software: PyCharm
"""

"""
    一：获取训练设备
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

"""
    二：模型构建
"""


class NeuralNetwork(nn.Module):
    """
        神经网络模型构建

        1：使用nn.Module子类定义神经网络，并在__init__中初始化网络层
        2：每个nn.Module子类都在forward()方法中实现对输入函数的操作
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


"""
    三：创建模型实例，并且将其移到训练设备上
"""
model = NeuralNetwork().to(device)
print(model)  # 打印模型结构

"""
    四：模型调用
"""
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
