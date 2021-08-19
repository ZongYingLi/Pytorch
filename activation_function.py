# 用激活函数来使得曲线非线性，常见的有relu,sigmoid,tanh,softplus

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt     # 用来画图的

# 不然报错Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# fake data
x = torch.linspace(-5, 5, 200)  # linespace()就是把线段分成一点一点的，这里是从-5到5的区间内取200个点作为数据
x = Variable(x)
x_np = x.data.numpy()   # 画图的时候要把torch的格式转换成numpy的数据，tensor是存在data里面的

# following are popular activation functions 激活值
y_relu = torch.relu(x).data.numpy() # 输入值是x，为了画图转化为numpy的数据
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy() # there's no softplus in torch，softmax不是用来做线图的，是用来做概率图的
# y_softmax = torch.softmax(x, dim=0).data.numpy() softmax is a special kind of activation function, it is about probability

# plt to visualize these activation function  画4个图
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()