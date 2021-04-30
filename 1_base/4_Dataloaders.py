# coding: utf-8
import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import DataLoader
"""
    @Time    : 2021/4/29 17:37
    @Author  : houjingru@semptian.com
    @FileName: 4_Dataloaders.py
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

"""
    三：创建自定义数据集
"""


class CustomImageDataset(Dataset):
    """
        创建自定义数据集
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": image, "label": label}
        return sample


"""
    四：准备数据，使用DataLoader进行模型训练
"""
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


"""
    五：使用DataLoader遍历数据
"""
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
