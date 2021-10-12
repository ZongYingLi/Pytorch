import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np



# torchvision包括了数据库，也包括了图片的数据库
import torchvision
import matplotlib.pyplot as plt

# 不然报错Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# pytorch里面使用torch.cat()函数来实现tensor的拼接
def make_features(x):
    x=x.unsqueeze(1)        # unsqueeze()用于添加维度
    return torch.cat([x**i for i in range (1,4)],1)   # 对于输入的n个数据，先将其扩展成为n*1的矩阵形式（这里说矩阵只是为了方便理解）

# 想要拟合的方程是y=0.9+ 0.5*x +3*x^2+2.4*x^3
# 设置方程参数 y=b+w1*x+w2*x^2+w3*x^3

# 下面就定义好真实的函数
b_target=torch.FloatTensor([0.9])
w_target=torch.FloatTensor([0.5,3,2.4]).unsqueeze(1)       # 将原来的tensor大小从3变成（3,1）即([0.5,3,2.4])变成([[5],[3],[2.4]])

# f(x)就是每次输入一个x得到一个y的真实函数
def  f(x):
    return x.mm(w_target)+b_target[0]       # x.mm(w_target)表示矩阵乘法

# 在进行训练队时候我们需要采样一些点，可以随机生成一些数来得到每次的训练集

# 通过以下的函数，我们每次都去batch_size这么多个数据点，然后将其转换成矩阵的形式，再把这个值通过函数之后的记过也返回作为真正的目标
def get_batch(batch_size=32):
    random=torch.randn(batch_size)
    x=make_features(random)
    y=f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(),Variable(y).cuda()
    else:
        return Varaible(x),Variable(y)

# 定义多项式模型

class poly_model(nn.Module):
    def __init__(self):
        super(poly_model,self).__init__()
        self.poly=nn.Linear(3,1)        # 输入参数是3维,输出参数是1维

    def forward(self,x):
        out=self.poly(x)
        return out

if torch.cuda.is_available():
    model=poly_model().cuda()
else:
    model=poly_model()

# 定义损失函数和优化器
criterion=nn.MSELoss()          # 使用均方误差来衡量模型的好坏
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)       # 使用随机梯度下降来优化模型

# 训练模型
epoch=0
while True:
    # get data
    batch_x,batch_y=get_batch()

    # forward pass
    output=model(batch_x)
    loss=criterion(output,batch_y)
    print_loss=loss.item()
    # print_loss = loss.data[0]报错，固改成loss.item()
    # invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number

    # reset gradients
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    optimizer.step()
    epoch+=1
    if epoch % 100 == 0:
        print('epoch : {}loss : {}'.format(epoch, print_loss))
    if print_loss<1e-3:     # 我们希望模型能够不断优化，知道我们设立的条件，取出的32个点的均方误差能够小于0.001
        break

# 测试模型

x = np.arange(-1,1,0.1)
x = torch.from_numpy(x).float()
x = make_features(x).squeeze()


plt.plot(x[:,0],f(x),'ro',label = 'real')

a = np.arange(-1,1,0.1)
plt.plot(a,b_target + w_target[0]*a + w_target[1]*a*a+ w_target[2]*a*a*a, label = 'fitting')
plt.legend()
plt.show()