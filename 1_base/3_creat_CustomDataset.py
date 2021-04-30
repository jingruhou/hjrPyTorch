# coding: utf-8
import os
import pandas as pd
from torchvision.io import read_image

"""
    @Time    : 2021/4/29 15:56
    @Author  : houjingru@semptian.com
    @FileName: 3_creat_CustomDataset.py
    @Software: PyCharm
"""


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        实例化数据对象 - 运行一次
        初始化包含图像、标注文件的文件夹，以及两种转换

        labels.csv eg:

        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ......
        ankleboot999.jpg, 9

        :param annotations_file:
        :param img_dir:
        :param transform:
        :param target_transform:
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        返回样本集中的样本数量
        :return:
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        从给定索引idx的数据集中加载并返回一个样本
        根据这个索引，它标识图像在磁盘上的位置。
        使用read_image()将其转化为张量
        从self.img_labels()中的CSV数据中获取相应的标签，对其调用transform转化函数（如果适用），
        并在一个Python Dict中返回张量图像和相应的标签
        :param idx:
        :return:
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": image, "label": label}
        return sample
