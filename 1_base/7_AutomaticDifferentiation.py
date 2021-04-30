# coding: utf-8
import torch
"""
    @Time    : 2021/4/30 12:10
    @Author  : houjingru@semptian.com
    @FileName: 7_AutomaticDifferentiation.py
    @Software: PyCharm
"""

"""
    1、当训练神经网络的时候，最常用的算法是反向传播。
    在该算法中，根据损失函数相对于给定参数的梯度来调整参数（模型权重）
    
    2、为了计算这些梯度，PyTorch具有一个内置的差异化引擎-torch.autograd，它支持任何计算图的梯度自动计算
    
    3、考虑最简单的一层神经网络，它具有输入x，参数w和b以及一些损失函数，
    它可以通过以下方式定义:
    
    4、这个网络中，w和b是参数，我们需要进行优化。
    因此，我们需要能够计算损失函数的梯度对这些变量，为了做到这点，我们设置了这些张量的requires_grad属性
"""
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("Gradient function for z = ", z.grad_fn)
print("Gradient function for loss = ", loss.grad_fn)

# 梯度计算

# 1、我们只能获取计算图的叶子结点的grad属性，这些节点的requires_grad属性设置为True
# 2、对于图中的所有其他节点，梯度不可用
# 3、由于性能原因，我们只能在给定的图形上使用backward执行梯度计算。
# 4、如果需要在同一图形上进行多个backward调用，则需要将retain_graph=True传递给反向调用
loss.backward()
print(w.grad)
print(b.grad)

# 禁用梯度跟踪

# 默认情况下，所有具有requires_grad=True的张量都在跟踪其计算历史并支持梯度计算。但是，在某些情况下我们不需要这样做
# eg:当我们训练模型并将其应用于某些输入数据时，即我们只想通过网络进行正向计算，我们可以通过torch.no_grad()块包围计算代码来停止跟踪计算
# 另外一种实现方式-在张量上使用detach()方法

z = torch.matmul(x, w) + b
print(z.requires_grad)

# 你可能要禁用梯度跟踪，有以下原因：
# 1、要将神经网络中的某些参数标记为冻结参数。这是微调预训练网络的非常常见的情况
# 2、仅仅在进行正向传递的时候可以加快计算速度，因为在不跟踪梯度的张量上进行计算会更有效

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# 另外一种实现方式-在张量上使用detach()方法
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

# 从概念上讲，autograd再由函数对象组成的有向无环图（DAG）中记录数据（张量）和所有已经执行的操作（以及由此产生的新张量）。
# 在DAG中，叶子是输入张量，根是输出张量。通过从根到叶跟踪该图，可以使用链规则自动计算梯度。

# 一：在前向计算中，autograd同时执行两项操作-
# 【1】运行所请求的操作，计算出一个结果张量 【2】在DAG中保持该操作的梯度函数
# 二：当在DAG根上调用.backward()时，向后传递开始，然后autograd
# 【1】从每一个.grad_fn计算梯度【2】将它们累计在各自的张量的.grad属性中【3】使用链规则，一直传递到叶子张量

# DAG在PyTorch中是动态的，需要注意的重要一点是，该图是从头开始重新创建的。
# 在每个.backward()调用之后，autograd开始填充新图。这正是允许您在该模型中使用控制流语句的原因。
# 可以根据需要在每次迭代中更改形状、大小、和操作

"""
    Tensor Gradients and Jacobian Products 张量梯度 和 雅可比积
    
    在许多情况下，我们具有标量损失函数，并且需要针对某些参数计算梯度。
    但是，在某些情况下，输出函数是任意张量，在这种情况下，PyTorch允许
    计算所谓的雅克比积，而不是实际的梯度
"""

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)

out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)

inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)

# 注意
# 当我们第二次使用相同的参数调用.backward()时，梯度的值是不同的。发生这种情况是因为PyTorch在进行向后传播的会累积梯度，
# 即将计算出的梯度值添加到计算图所有节点的grad属性中。如果要计算适当的梯度，则需要先将grad属性清零。
# 在现实中的训练中，优化程序可以帮助我们做到这一点。
