# 在处理任何机器学习问题之前都需要数据获取并进行预处理
# torch.utils.data.Dataset是pytorch提供的一个抽象类，使数据读取和预处理更加容易
# 可以自定义数据类型集成和重写这个抽象类，只需定义__len__和__getitem__这两个函数即可


import torch
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self,csv_file,txt_file,root_dir,other_file):
        self.csv_data=pd.read_csv(csv_file)
        with open(txt_file,'r') as f:
            data_list=f.readlines()
        self.txt_data=data_list
        self.root_dir=root_dir

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self,idx):
        data=(self,csv_data[idx],self.txt_data[idx])
        return data

# 通过上面的方法，可以定义我们需要的数据类型，通过迭代的方式来取得每一个数据，但是这样很难实现取batch,shuffle或者是多线程去读取数据

# pytorch提供了一个更简单的方法，通过torch.utils.data.DataLoader来定义一个新的迭代器
# collate_fn表示是如何取样本的
dataiter=DataLoader(myDataset,batch_size=32,shuffle=True,collate_fn=default_collate)

# 在torchvision这个包中海油一个更高级的有关于计算机视觉的数据读取类，ImageFolder
# 主要功能是处理图片，且要求图片是下面的存放形式root/dog/xxx.png
# 之后这样定义这个类


# 其中root是根目录，在这个目录下有几个文件夹，每个文件夹都表示一个类别
dset=ImageFolder(root='root_path',transform=None,loader=default_folder)
# loader是图片读取的方法，因为我们读取的是图片的名字，然后通过loader将图片转换为我们需要的图片类型进入神经网络