# coding: utf-8
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
"""
    @Time    : 2021/4/30 10:24
    @Author  : houjingru@semptian.com
    @FileName: 5_Transforms.py
    @Software: PyCharm
"""
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # ToTensor()将PIL图像或NumPy ndarray转化为FloatTensor。并在[0,1]范围内缩放图像的像素强度值
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

"""
    Lambda变换
    
    Lambda变换会应用任何用户定义的Lambda函数。
    在这里，我们定义了一个将整数转换为单次one-hot编码张量的函数。
    它首先创建一个大小为10（数据集中的标签数）的零张量，并调用scatter_,该标签在标签y所给定的索引上分配了一个值value=1
"""
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

print(target_transform)