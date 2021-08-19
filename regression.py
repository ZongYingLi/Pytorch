import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 不然报错Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# regression回归，就是说数据是连续的，分类数据是离散的

# fake data
# unsqueeze()就是把一维的数据变成二维的数据，因为在torch中只会处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # x的平方加上一些燥点的影响 noisy y data (tensor), shape=(100, 1)

#神经网络只能输入Variable
# x, y = Variable(x), Variable(y)
#
#
# plt.scatter(x.data.numpy(), y.data.numpy()) # scatter()打印散点图
# plt.show()


class Net(torch.nn.Module):
    # 搭建层所需要的信息
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() # 继承Net到torch.nn.Module这个模块，并输出，官方操作
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        # 神经网络前向传播的过程
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

# define the network
net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)  # net architecture，可以展示层结构

#优化器,lr=learning rate，即学习效率，一般小于1
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # 均方差

plt.ion()   # something about plotting

for t in range(200):        # 训练的步数设为200步
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)，y是真实值，prediction是预测值，前后不能调换

    optimizer.zero_grad()   #把有优化器的梯度全部降为0, clear gradients for next train
    loss.backward()         # 反向传递，backpropagation, compute gradients
    optimizer.step()        # 用优化器来优化梯度，apply gradients

#设置一个实时打印的过程，每5步就打印图上的信息，原始数据，学习程度，误差
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()