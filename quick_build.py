import torch
import torch.nn.functional as F

# replace following class code with an easy sequential network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

#前面calssification的建立神经网络的方法
net1 = Net(1, 10, 1)

# 下面是快速建立神经网络
# easy and fast way to build your network
# Sequential()就是指在括号里一层一层垒神经层
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(), # 算出前一层的结果以后过一层ReLu的激励函数，然后到接下来的一层
    torch.nn.Linear(10, 1)
)

# 两种方法不同，但是效果是一样的
print(net1)     # net1 architecture

print(net2)     # net2 architecture