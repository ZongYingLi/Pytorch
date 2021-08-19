from torch.utils.data import Dataset
from PIL import Image
import os

class Mydata(Dataset):

    # 初始化类，为class提供全局变量
    def __init__(self,root_dir,label_dir):
        # 训练集路径,一般设为相对地址
        self.root_dir = root_dir
        # 训练集的名字就是标签
        self.label_dir = label_dir
        # 训练集所有图片的地址是训练集路径和标签名字的连接
        self.path = os.path.join(self.root_dir,self.root_dir)
        # 所有训练集图片的地址
        self.img_path = os.listdir(self.path)


#     通过idx--索引来获取图片的地址
    def __getitem__(self,idx):
        # 加self即为全局的
        img_name = self.img_path[idx]
        # 获取每个图片的路径
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants"
ants_dataset = Mydata(root_dir,label_dir)

