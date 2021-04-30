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

# 模型层
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten 展平、扁平化
# 我们初始化nn.Flatten层,将每个28*28的二维图像转化为784个像素值的连续数组【minibatch的维度被保持】
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear  线性函数
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# 非线性激活会在模型的输入和输出之间创建复杂的映射。
# 在线性变换之后应用引入非线性，从而帮助神经网络学习各种各样的现象
# 在此模型中，我们在线性层之间使用了nn.ReLU,但还有其他激活方式可以在你的模型中引入非线性

print(f"Before ReLU: {hidden1} \n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


# nn.Sequential
# nn.Sequential是模块的有序容器。数据以定义的相同顺序通过所有的模块。
# 你可以使用顺序容器将seq_modules之类的快速网络组合再一起
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)


# nn.Softmax
# 神经网络的最后一个线性层返回对数[-infty, infty]中的原始值，将其传递到nn.Softmax模块。
# logit被缩放为数值[0,1]之间，该值表示每个类别的预测概率。
# dim参数表示将值相加为1的维度

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


# 模型参数
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
