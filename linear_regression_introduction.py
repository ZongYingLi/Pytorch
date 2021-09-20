import numpy as np

import torch
from torch.autograd import Variable

import torch.nn as nn

# torchvision包括了数据库，也包括了图片的数据库
import torchvision
import matplotlib.pyplot as plt

# 不然报错Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 遇到一个报错Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking arugment for argument mat1 in method wrapper_addmm)
# 报错的本意就是：希望所有的tensor都在同一个设备上，而不是一会cpu，一会儿gpu

# array是数组类型数据
x_train=np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[7.59],[2.167],[7.042],[10.791]
                  ,[5.313],[7.997],[3.1]],dtype = np.float32)
y_train=np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.827],[3.465]
                     ,[1.65],[2.904],[1.3]],dtype = np.float32)

# 通过matplotlib画出散点图

# plt.scatter(x_train,y_train)
# plt.show()


# 找一条直线去逼近这些点，希望这条直线距离这些点的距离之和最小
# 先将numpy.array转换为tensor，因为Pytorch里面的处理单元是tensor
x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)

# 接着需要建立模型
# 定义一个简单的模型

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1)      # input and output is 1 demension
        # 输入参数是一维，输出参数也是一维，这就是一条直线，即 y = w * x + b

    def forward(self,x):
        out = self.linear(x)
        return out

# 判断是否支持cpu加速，如果支持，可以通过model.cuda()将模型放在GPU上
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else :
    model = LinearRegression()

# 定义损失函数和优化函数，这里用均方误差作为优化函数，使用梯度下降进行优化,我们希望通过优化参数w和b是直线尽可能接近这些点
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3) # lr表示学习率

# 训练模型
num_epochs = 1000        # 定义好要跑的epoch个数

# 把数据变成Variable放入计算图中
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)

    # forward
    out=model(inputs)       # 得到网络前向传播的结果
    loss=criterion(out,target)      # 得到损失函数

    # backward
    optimizer.zero_grad()       # 归零梯度，可以辨析一下PyTorch中model.zero_grad()和optimizer.zero_grad()用法的区别

# 注意，每次做反向传播之前都要归零梯度
# 不然梯队会累加在一起，造成结果不收敛
# 在训练的过程中，隔一段时间就将损失函数的值打印出来看看，确保模型误差越来越小
    loss.backward()         # 反向传播
    optimizer.step()           # 更新参数

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1,num_epochs, loss.item()))
        # 报错 invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
        # print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1,num_epochs, loss.item()))
        # 由于以上报错，把loss.data[0]改为loss.item()[0]
        # 报错'builtin_function_or_method' object is not subscriptable

# loss.data[0]中，首先loss是一个variable，所以通过loss.data可以取出一个tensor
# 再通过loss.data[0]得到一个int或者float型的数据，这样我们才可以打印出相应的数据

# 做完可以预测一下结果
model.eval()        # 首先将模型变成测试模型，这是因为有些层的操作
# 比如Dropout和BatchNormalization在训练和测试的时候是不一样的，所以我们需要这样一个操作来转换这些不一样的层操作
predict = model(Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(),y_train.numpy() , 'ro' , label='Original data')
plt.plot(x_train.numpy(),predict,label='Fitting Line')

plt.legend()
plt.show()

