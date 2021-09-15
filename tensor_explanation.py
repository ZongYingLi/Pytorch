import torch
import numpy as np

# Pytorch的基本操作对象就是 tensor ，表示的是一个多维的矩阵

# 定义一个三行两列的矩阵
a =torch.Tensor([[2,3],[4,8],[7,9]])
print("a is : {}".format(a))  # torch.Tensor默认的是torch.FloatTensor数据类型,所以输出后是小数
print("a size is {}".format(a.size()))


# 也可以定义我们想要的数据类型
b=torch.LongTensor([[2,3],[4,8],[7,9]])
print("b is {}".format(b))

# 也可以取一个全是0的空tensor
c=torch.zeros((3,2))  # 去一个3*2的张量
print("zero tensor:{}".format(c))  # 输出还是0.


# 取一个正太分布作为随机初始值
d=torch.randn((3,2))  # 去一个随机的3*2的张量
print("normal randon is:{}".format(d))

# 可以通过索引取得其中的元素,同时可以改变它的值
a[0,1]=100
print("changed a is {}".format(a))

# Tensor 和 numpy.ndarray之间相互转换
numpy_b=b.numpy() # 将张量b转化为numpy类型数据,也可以通过torch.from_numpy()将numpy转换为tensor
print("conver to numpy is \n {}".format(numpy_b))

e=np.array([[2,3],[4,5]])
torch_e=torch.from_numpy(e)     # 将numpy类型的e转换为tensor为torch_e
print("from numpy to torch.Tensor is {}".format(torch_e))
f_torch_e=torch_e.float()   # 将转换后的tensor加上想要的类型，这里是将张量e转化为float类型的数据
print("change data type to float tensor:{}".format(f_torch_e))

# 如果电脑支持gpu加速，就可以将tensor放在gpu上
if torch.cuda.is_available():       # 判断一下是否支持cpu
    a_cuda=a.cuda()         # 将tensor类型的a放在gpu上，即a.cuda()就可以把张量a放在gpu上了
    print(a_cuda)