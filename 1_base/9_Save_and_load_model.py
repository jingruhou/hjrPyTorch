# coding: utf-8
import torch
import torch.onnx as onnx
import torchvision.models as models
"""
    @Time    : 2021/5/6 12:09
    @Author  : houjingru@semptian.com
    @FileName: 9_Save_and_load_model.py
    @Software: PyCharm
"""
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'models\\model_weight.pth')

model = models.vgg16()
model.load_state_dict(torch.load('models\\model_weight.pth'))
model.eval()

#  导出模型至ONNX
input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'models\\model.onnx')