# pytorch里编写神经网络，所有的层结构和损失函数都来自于toch.nn
# 所有的模型构建都是从基类nn.Module继承的，于是有了以下模板

class net_name(nn.Module):
    def __init__(self,other_arguments):
        super(net_name,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernnel_size)

        # other network layer

    def forward(self,x):
        x=self.conv1(x)
        return x

# 这样就定义了一个计算图，且这个结构可以反复用多次，每次调用相当于用该计算图定氮仪的相同参数做一次前向传播
# 由于Pytorch的自动求导功能，所以我们不用自己编写反向传播，所有的网络层都是由nn这个包得到，如线性层nn.Linear


# 定义完模型之后，我们要通过nn这个包来定义损失函数
# 常见的损失函数都定义在了nn中了，比如均方误差，多分类的交叉熵，二分类的交叉熵等等

# 调用这些已经定义好的损失函数

criterion=nn.CrossEntropyLoss()
loss=criterion(output,target)