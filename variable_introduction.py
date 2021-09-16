# Variable 变量，提供了自动求导的功能
# Variable 和 Tensor本质上没有区别，Variable会被放入一个计算图中，然后进行前向传播，反向传播，自动求导。
# Variable 在torch.autograd.Variable中, 若要将tensor a 转换为Variable ，只用Variable(a)即可

# Variable有三个重要的组成属性，data,grad 和grad_fn
# data可以取出Variable里面的tensor数值
# grad_fn 小时的是得到Variable的操作
# grad是这个Variable反向传播的梯度

# 在单变量函数中，梯度可以理解为只是导数
# 函数f的梯度方向是函数f的值增长最快的方向，最陡的方向，在一个场中，函数在某一点的梯度即为这个点方向导数最大值

import torch
from torch.autograd import Variable

# create Variable
x=Variable(torch.Tensor([1]),requires_grad=True)        # requires_grad=True表示是否对这个变量求梯度，默认是 False
w=Variable(torch.Tensor([2]),requires_grad=True)
b=Variable(torch.Tensor([3]),requires_grad=True)


# build a computational graph
y = w * x + b

# 如果需要计算导数，可以在Tensor上调用.backward()。
#  1. 如果Tensor是一个标量（即它包含一个元素的数据），则不需要为backward()指定任何参数
#  2. 但是如果它有更多的元素，则需要指定一个gradient参数，它是形状匹配的张量。

# compute gradients
y.backward()  # 自动求导,等价于 y.backward(torch.FloatTensor([1]))
# 不过对于标量求导里面的参数就可以不写了，自动求导不需要再写哪个函数对哪个函数求导，直接通过这行代码就可以对所有的需要梯度的变量进行求导
# ，得到梯度，然后通过 x.grad 可以得到 x 的梯度

# print out the gradients
print(x.grad)       # y 对 x 求导 即 y = 2 * x + 3 求导，即为 2
print(w.grad)       # y 对 w 求导 即 y = 1 * w + 3 求导，即为 1
print(b.grad)       # y 对 b 求导 即 y = 1 * 2 + b 求导，即为 1



# 除了以上了标量求导，也可以做矩阵求导

# torch.randn()返回一个张量，包含了从标准正态分布（均值为0，方差为1）中抽取的一组随机数。张量的形状由参数sis定义
z = torch.randn(3)                  # 取一个长度为3的正态分布张量
z = Variable(z,requires_grad=True)
f = z * 2
print(f)

# 相当于给出了一个三维向量去做运算，这时结果f就是一个向量，此时对向量求导就不能直接是f.backward()
# 此时要传入参数声明
f.backward(torch.FloatTensor([1,0.1,0.01]))     # 此时得到的梯度就是他们原本的梯度分别乘上1,0,1和0,01
print(z.grad)