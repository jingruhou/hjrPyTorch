# coding: utf-8
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

"""
    @Time    : 2021/4/30 16:34
    @Author  : houjingru@semptian.com
    @FileName: 8_OptimizingModelParameters.py
    @Software: PyCharm
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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeunalNetwork(nn.Module):
    def __init__(self):
        super(NeunalNetwork, self).__init__()
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


model = NeunalNetwork()

"""
    超参数
    
    超参数是可调整的参数，可让您控制模型优化过程。不同的超参数值可能会影响模型训练和收敛速度
"""
learning_rate = 1e-3  # 学习率
batch_size = 64  # 模型在每个epoch中看到的样本数量
epochs = 5  # 遍历数据集的次数

"""
    循环优化
    
    设置超参后，我们可以使用循环来训练和优化模型。
    优化循环的每次迭代都被称为一个周期-Epoch
    
    每个Epoch包括两个主要部分：
    
    1、训练循环：遍历训练数据集，并尝试收敛到最佳参数
    2、验证/测试循环：遍历测试数据集，以检查模型性能是否有所改善
"""


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测与损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向计算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss: >7f} [{current:>5d} / {size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error :\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss: >8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10

for t in range(epochs):
    print(f"Epoch {t + 1} \n --------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
