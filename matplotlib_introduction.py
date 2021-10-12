# torchvision包括了数据库，也包括了图片的数据库
import torchvision
import matplotlib.pyplot as plt

# 不然报错Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import numpy as np
# 定义 x 变量的范围 (-6，6) 数量 50
x=np.linspace(-6,6,50)
y=1/(1+e^(-x))


# Figure 并指定大小

plt.figure(num=3,figsize=(8,5))

# 绘制 y=x^2 的图像，设置 color 为 red，线宽度是 1，线的样式是 --

plt.plot(x,y,color='red',linewidth=1.0,linestyle='--')


# 显示图像

plt.show()