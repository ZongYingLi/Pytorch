# 如何进行批训练，把大量的数据分成一批一批地处理，提高训练速度
import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

# 定义批训练的容量是5个，这里为了能成功运行增加了D盘的虚拟内存，教程收藏在了csdn里
BATCH_SIZE = 5
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

# 用torch来定义数据库
# data_tensor=x, target_tensor=y
torch_dataset = Data.TensorDataset(x, y)
# loader来使得训练呈一批一批的
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training，定义是否打乱数据
    num_workers=2,              # subprocesses for loading data，定义提取数据的线程/进程的数量
)


def show_batch():
    # 把数据整体的训练3次，每一次训练都会分成5批
    for epoch in range(3):   # train entire dataset 3 times
        # 把整个数据分batch指定的批进行训练,也可以用loader决定是否打乱数据顺序
        # for each training step，enumerate()是在loader每一次提取的时候都加上索引
        for step, (batch_x, batch_y) in enumerate(loader):

            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()