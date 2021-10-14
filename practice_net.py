import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import time
import net      # 之间定义网络的文件net.py

# 定义超参数Hyperparameters
batch_size=64
learning_rate=1e-2
num_epoches=20          # 总共训练批次

# 将数据标准化的的函数是torchvision.transforms,它提供了很多图片预处理方法
# 这里使用两个方法，一个是transforms.ToTensor(),另一个是transforms.Normalize()
# transforms.ToTensor()就是将图片转换成Pytorch中的处理对象Tensor,在转化的过程中Pytorch自动将图片标准化了，Tensor的范围是0--1
# transforms.Normalize()需要传入两个参数，一个是均值，一个是方差，做的处理就是减去均值再除以方差，这样就把图片转化到-1到1之间，
# 因为图片是灰度图，所以只有一个通道，如果是彩色的图片，有三通道，那么用transforms.Normalize([a,b,c],[d,e,f])来表示每个通道对应的均值和方差
data_tf=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])


# 读取数据集
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)
# 通过DataLoader来建立一个数据迭代器，传入数据集和batch_size,通过shuffle=True来表示每次迭代数据的时候是否将数据打乱
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# 导入网络，定义损失函数和优化方法
# 因为这是一个分类问题，一共有0~9这10个数字，所以是10分类
model=net.simpleNet(28*28,300,100,10)       # 输入图片的大小是28*28，定义两个隐藏层分别是300和100，最后输出的结果是10
if torch.cuda.is_available():
    model=model.cuda()

# 损失函数定义为分类函数中最常见的损失函数交叉熵
criterion=nn.CrossEntropyLoss()
# 使用随机梯度下降来优化损失函数
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

# 训练模型
for epoch in range(num_epoches):
    print('*' * 20)
    print('epoch {}'.format(epoch + 1))
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        # print("img.size(0)", img.size(0), "size: ", img.data.size())
        img = img.view(img.size(0), -1)  # 将图片展开成 28x28

    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    # 向前传播
    out = model(img)
    loss = criterion(out, label)  # 计算loss
    running_loss += loss.item() * label.size(0)  # 把每一次的loss值加起来 后面求平均loss,最后得到的loss = running_loss/i
    _, pred = torch.max(out, 1)  # 取张量的最大值 组成一个一维矩阵
    # print("pred: ", pred)
    num_correct = (pred == label).sum()
    running_acc += num_correct.item()

    # 向后传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 300 == 0:
        print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, num_epoches, running_loss / (batch_size * i),
            running_acc / (batch_size * i)))

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
    epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
        train_dataset))))
    print("loss: ", loss.data.item())

# 测试网络
model.eval()
eval_loss = 0.
eval_acc = 0.
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))
print('Time:{:.1f} s'.format(time.time() - since))
print()