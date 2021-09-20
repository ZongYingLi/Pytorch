# 在这里了解 unsqueeze() 增加维度的效果，具体的可以看书《深度学习之Pytorch》62页
import torch

g=torch.FloatTensor([0.5,3,2.4])
print(g.shape)
a=g.unsqueeze(1)
print(a)
print(a.shape)