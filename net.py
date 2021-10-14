import torch
import torch.nn as nn

# 定义一个三层全连接神经网络
class simpleNet(nn.Module):
    # 对于一个三层网络，需要传递进去的参数包括：输入的维度，第一层网络的神经元个数，第二层网络的神经元个数，以及第三层网络（输出层）神经元的个数
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

# 增加激活函数增加函数的非线性
class Activatiton_Net(nn.Module):
    # 这里只需要在每层网络的输出部分添加激活函数就可以了
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(NeuralNetwork,self).__init__()
        # nn.Sequential()是将网络的层组合在一起
        self.layer1=nn.Sequential(
            nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(
            nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        # 最后一层输出层不能添加激活函数，因为输出的结果表示的是实际的得分
        self.layer3=nn.Sequential(
            nn.Linear(n_hidden_2,out_dim))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# 添加一个加快收敛速度的方法——批标准化
class Batch_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Batch_Net,self).__init__()
        # 同样使用nn.Sequential()将nn.BatchNorm1d()组合到网络层中
        # 批标准化一般放在全连接层的后面、非线性层（激活函数）的前面
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x