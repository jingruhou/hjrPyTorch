# coding: utf-8
import torch
import numpy as np
"""
    @Time    : 2021/4/28 15:49
    @Author  : houjingru@semptian.com
    @FileName: 1_tensors.py
    @Software: PyCharm
"""

# 初始化tensor

#  1.直接从数据
data = [[1,2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

#  2.直接从NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#  3.从另外的tensor加载数据
x_ones = torch.ones_like(x_data)
print(f"Ones Tensors:\n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖这些数据
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor:\n {ones_tensor} \n")
print(f"Zeros Tensor:\n {zeros_tensor}")


# Tensor的属性

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor的操作

# 标准类似numpy的索引和切片
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])

tensor[:, 1] =0
print(tensor)

#  tensor链接 - Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 算数运算 - Arithmetic operations
y1 = tensor @ tensor.T
Y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
print(z1)

z2 = tensor.mul(tensor)
print(z2)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 单元素张量
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(tensor, "\n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")

n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
print(t)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

