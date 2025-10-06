
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import config1
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk    
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from patchify import patchify
from scipy.ndimage import binary_fill_holes
class GetTrainSet(Dataset):
    def __init__(self, data_root, img_size):    # 这里传入一些必要的config参数
        # 设置一些基本的类变量
        self.data_root = data_root
        self.img_size = img_size

        # 合成完整的文件路径
        self.input_root = os.path.join(self.data_root, "train", "input")     # "./data_root/train/input/"
        self.gt_root = os.path.join(self.data_root, "train", "gt")           # "./data_root/train/gt/"


        # 获取图片名列表
        self.input_name_list = os.listdir(self.input_root)
        self.input_name_list.sort()
        self.gt_name_list = os.listdir(self.gt_root)
        self.gt_name_list.sort()
        # 图片数量
        self.img_num = len(self.input_name_list)
        # print(self.input_name_list)

        # 设置transforms (只能对PIL格式的图像或numpy数组用)
        # todo 做transform之前，图像数据已经被norm到[0, 1]
        # self.to_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        #         # todo 对于自然图像: (1)PIL类型的图 (2)uint8类型的numpy数组 ===>>> transforms.ToTensor() 会自动完成： [0, 255] -> [0, 1]
        #         # todo 对于医学图像: dicom -> uint16, 不会改变值的范围, 因此需要提前将图片的像素值范围预处理好, 通常是归到 [0, 1]
        #     # transforms.Normalize(mean=0.5, std=0.5),
        #         # todo 可以实现: [0, 1] -> [-1, 1]
        #         # todo Normalize(mean=0.5, std=0.5)的使用与否，需要与model输出层的激活函数保持一致（tanh）
        #         # todo 演示一下不一致会怎么样
        #     #transforms.Resize((self.img_size, self.img_size), antialias=True)
        # ])
    
    def __len__(self):
        return self.img_num

    # normalize -> [0, 1]
    def MinMax(self,image):
        # image = image - np.mean(image)  # 零均值
        if np.max(image) - np.min(image) != 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # 归一化

        return image
    def zsorce(self,image):
        image = (image - np.mean(image)) / np.std(image)
        return image
    def Crop(self,image, label, output_size):
        if image.shape[0] <= output_size[0] or image.shape[1] <= output_size[1] or image.shape[2] <= output_size[2]:
            pw = max((output_size[2] - image.shape[2]) // 2 + 3, 0)
            ph = max((output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((output_size[0] - image.shape[0]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph),(pw, pw) ], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph),(pw, pw) ], mode='constant', constant_values=0)

        (d,w, h ) = image.shape

        w1 = int(round((w - output_size[2]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[0]) / 2.))

        # print(image.shape, output_size, get_center(label), w1, h1, d1)
        image = image[ 180: 180+output_size[0],w1:w1 + output_size[2], h1:h1 + output_size[1]]
        label = label[180: 180+output_size[0], w1:w1 + output_size[2],h1:h1 + output_size[1] ]

        return image, label

    # todo 在train.py里读入数据的时候是通过调用__getitem__实现的，每调用一次就读一条
    def __getitem__(self, index):   # index为待取图像的序号
        # todo 读入每张图片都需要获取每张图片的完整路径 = 文件夹路径 + 图片名
        # print(index)
        input_img_path = os.path.join(self.input_root, self.input_name_list[index])
        gt_img_path = os.path.join(self.gt_root, self.gt_name_list[index])
        # print(self.input_name_list[index])
        # load img
        input_img = sitk.ReadImage(input_img_path)      # sitk类型的图像
        input_img = sitk.GetArrayFromImage(input_img)  # sitk -> np, tpye: uint16
        input_img = np.float32(input_img)               # todo uint16 -> float32

        gt_img = sitk.ReadImage(gt_img_path)
        gt_img = sitk.GetArrayFromImage(gt_img)
        gt_img = np.float32(gt_img)

        # norm
        input_img=self.MinMax(input_img)
        gt_img = self.MinMax(gt_img)
        # print("in:",np.shape(input_img))


        #crop
        #input_img, gt_img=self.Crop(input_img,gt_img,[121,512,512])
        # print("out:", np.shape(input_img))

        # divide into patches
        # This will split the image into small images of shape [3,3,3]
        # input_img = patchify(input_img, (3, 3, 3), step=1)

        #expand_dim
        # input_img=np.expand_dims(input_img,axis=1)
        # gt_img=np.expand_dims(gt_img, axis=1)


        # to tensor
        input_img=(torch.from_numpy(input_img))
        gt_img=(torch.from_numpy(gt_img))
        # print(gt_img.size())


        return input_img, gt_img


class GetTestSet(Dataset):
    def __init__(self, data_root, img_size):  # 这里传入一些必要的config参数
        # 设置一些基本的类变量
        self.data_root = data_root
        self.img_size = img_size

        # 合成完整的文件路径
        self.input_root = os.path.join(self.data_root, "test", "input")      # todo "./data_root/test/input/"
        self.gt_root = os.path.join(self.data_root, "test", "gt")            # todo "./data_root/test/gt/"
        # 获取图片名列表
        self.input_name_list = os.listdir(self.input_root)
        self.input_name_list.sort()

        self.gt_name_list = os.listdir(self.gt_root)
        self.gt_name_list.sort()
        # 图片数量
        self.img_num = len(self.input_name_list)

    def __len__(self):
        return self.img_num

    def MinMax(self, image):
        # image = image - np.mean(image)  # 零均值
        if np.max(image) - np.min(image) != 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # 归一化

        return image

    def zsorce(self, image):
        image = (image - np.mean(image)) / np.std(image)
        return image

    def Crop(self, image, label, output_size):
        if image.shape[0] <= output_size[0] or image.shape[1] <= output_size[1] or image.shape[2] <= output_size[2]:
            pw = max((output_size[2] - image.shape[2]) // 2 + 3, 0)
            ph = max((output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((output_size[0] - image.shape[0]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        (d, w, h) = image.shape

        w1 = int(round((w - output_size[2]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[0]) / 2.))

        # print(image.shape, output_size, get_center(label), w1, h1, d1)
        image = image[180: 180 + output_size[0], w1:w1 + output_size[2], h1:h1 + output_size[1]]
        label = label[180: 180 + output_size[0], w1:w1 + output_size[2], h1:h1 + output_size[1]]

        return image, label

    # 在train.py里读入数据的时候是通过调用__getitem__实现的
    def __getitem__(self, index):  # index为待取图像的序号
        # 获取每张图片的完整路径 = 文件夹路径 + 图片名
        input_img_path = os.path.join(self.input_root, self.input_name_list[index])
        gt_img_path = os.path.join(self.gt_root, self.gt_name_list[index])
        # print( self.input_name_list[index],self.gt_name_list[index])
        # load img
        input_img = sitk.ReadImage(input_img_path)  # sitk类型的图像
        input_img = sitk.GetArrayFromImage(input_img).squeeze()  # sitk -> np, tpye: uint16
        input_img = np.float32(input_img)  # uint16 -> float32
        gt_img = sitk.ReadImage(gt_img_path)
        gt_img = sitk.GetArrayFromImage(gt_img).squeeze()
        gt_img = np.float32(gt_img)

        # low-dose ct denoise
        # norm
        input_img = self.MinMax(input_img)
        gt_img = self.MinMax(gt_img)
        # print("in:",np.shape(input_img))

        # to tensor
        input_img = (torch.from_numpy(input_img))
        gt_img = (torch.from_numpy(gt_img))

        # # fixme seg
        # # norm
        # input_img = self.norm_brats_seg(input_img)
        # # to tensor
        # input_img = self.to_tensor(input_img)
        # gt_img = self.to_tensor(gt_img)

        return input_img, gt_img,self.input_name_list[index]


class GetTrainSet_9MASK(Dataset):
    def __init__(self, data_root, img_size):  # 这里传入一些必要的config参数
        # 设置一些基本的类变量
        self.data_root = data_root
        self.img_size = img_size

        # 合成完整的文件路径
        self.input_root = os.path.join(self.data_root, "train", "input")  # "./data_root/train/input/"
        self.gt_root = os.path.join(self.data_root, "train", "gt")  # "./data_root/train/gt/"

        # 获取图片名列表
        self.input_name_list = os.listdir(self.input_root)
        self.input_name_list.sort()
        self.gt_name_list = os.listdir(self.gt_root)
        self.gt_name_list.sort()
        # 图片数量
        self.img_num = len(self.input_name_list)
        # print(self.input_name_list)

        # 设置transforms (只能对PIL格式的图像或numpy数组用)
        # todo 做transform之前，图像数据已经被norm到[0, 1]
        # self.to_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        #         # todo 对于自然图像: (1)PIL类型的图 (2)uint8类型的numpy数组 ===>>> transforms.ToTensor() 会自动完成： [0, 255] -> [0, 1]
        #         # todo 对于医学图像: dicom -> uint16, 不会改变值的范围, 因此需要提前将图片的像素值范围预处理好, 通常是归到 [0, 1]
        #     # transforms.Normalize(mean=0.5, std=0.5),
        #         # todo 可以实现: [0, 1] -> [-1, 1]
        #         # todo Normalize(mean=0.5, std=0.5)的使用与否，需要与model输出层的激活函数保持一致（tanh）
        #         # todo 演示一下不一致会怎么样
        #     #transforms.Resize((self.img_size, self.img_size), antialias=True)
        # ])

    def __len__(self):
        return self.img_num

    # normalize -> [0, 1]
    def MinMax(self, image):
        # image = image - np.mean(image)  # 零均值
        if np.max(image) - np.min(image) != 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # 归一化

        return image

    def zsorce(self, image):
        image = (image - np.mean(image)) / np.std(image)
        return image

    def Crop(self, image, label, output_size):
        if image.shape[0] <= output_size[0] or image.shape[1] <= output_size[1] or image.shape[2] <= output_size[2]:
            pw = max((output_size[2] - image.shape[2]) // 2 + 3, 0)
            ph = max((output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((output_size[0] - image.shape[0]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        (d, w, h) = image.shape

        w1 = int(round((w - output_size[2]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[0]) / 2.))

        # print(image.shape, output_size, get_center(label), w1, h1, d1)
        image = image[180: 180 + output_size[0], w1:w1 + output_size[2], h1:h1 + output_size[1]]
        label = label[180: 180 + output_size[0], w1:w1 + output_size[2], h1:h1 + output_size[1]]

        return image, label

    # todo 在train.py里读入数据的时候是通过调用__getitem__实现的，每调用一次就读一条
    def __getitem__(self, index):  # index为待取图像的序号
        # todo 读入每张图片都需要获取每张图片的完整路径 = 文件夹路径 + 图片名
        # print(index)
        input_img_path = os.path.join(self.input_root, self.input_name_list[index])
        gt_img_path = os.path.join(self.gt_root, self.gt_name_list[index])
        # print(self.input_name_list[index])
        # load img
        input_img = sitk.ReadImage(input_img_path)  # sitk类型的图像
        input_img = sitk.GetArrayFromImage(input_img)  # sitk -> np, tpye: uint16
        input_img = np.float32(input_img)  # todo uint16 -> float32

        gt_img = sitk.ReadImage(gt_img_path)
        gt_img = sitk.GetArrayFromImage(gt_img)
        gt_img = np.float32(gt_img)

        # norm
        input_img = self.MinMax(input_img)
        # gt_img = self.MinMax(gt_img)
        # print("in:",np.shape(input_img))

        # crop
        # input_img, gt_img=self.Crop(input_img,gt_img,[121,512,512])
        # print("out:", np.shape(input_img))

        # divide into patches
        # This will split the image into small images of shape [3,3,3]
        # input_img = patchify(input_img, (3, 3, 3), step=1)

        # expand_dim
        # input_img=np.expand_dims(input_img,axis=1)
        # gt_img=np.expand_dims(gt_img, axis=1)

        # to tensor
        input_img = (torch.from_numpy(input_img))
        gt_img = (torch.from_numpy(gt_img))
        # print(gt_img.size())

        return input_img, gt_img

class GetTestSet_9MASK(Dataset):
    def __init__(self, data_root, img_size):  # 这里传入一些必要的config参数
        # 设置一些基本的类变量
        self.data_root = data_root
        self.img_size = img_size

        # 合成完整的文件路径
        self.input_root = os.path.join(self.data_root, "test", "input")      # todo "./data_root/test/input/"
        self.gt_root = os.path.join(self.data_root, "test", "gt")            # todo "./data_root/test/gt/"
        # 获取图片名列表
        self.input_name_list = os.listdir(self.input_root)
        self.input_name_list.sort()

        self.gt_name_list = os.listdir(self.gt_root)
        self.gt_name_list.sort()
        # 图片数量
        self.img_num = len(self.input_name_list)

    def __len__(self):
        return self.img_num

    def MinMax(self, image):
        # image = image - np.mean(image)  # 零均值
        if np.max(image) - np.min(image) != 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # 归一化

        return image

    def zsorce(self, image):
        image = (image - np.mean(image)) / np.std(image)
        return image

    def Crop(self, image, label, output_size):
        if image.shape[0] <= output_size[0] or image.shape[1] <= output_size[1] or image.shape[2] <= output_size[2]:
            pw = max((output_size[2] - image.shape[2]) // 2 + 3, 0)
            ph = max((output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((output_size[0] - image.shape[0]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        (d, w, h) = image.shape

        w1 = int(round((w - output_size[2]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[0]) / 2.))

        # print(image.shape, output_size, get_center(label), w1, h1, d1)
        image = image[180: 180 + output_size[0], w1:w1 + output_size[2], h1:h1 + output_size[1]]
        label = label[180: 180 + output_size[0], w1:w1 + output_size[2], h1:h1 + output_size[1]]

        return image, label

    # 在train.py里读入数据的时候是通过调用__getitem__实现的
    def __getitem__(self, index):  # index为待取图像的序号
        # 获取每张图片的完整路径 = 文件夹路径 + 图片名
        input_img_path = os.path.join(self.input_root, self.input_name_list[index])
        gt_img_path = os.path.join(self.gt_root, self.gt_name_list[index])
        # print( self.input_name_list[index],self.gt_name_list[index])
        # load img
        input_img = sitk.ReadImage(input_img_path)  # sitk类型的图像
        input_img = sitk.GetArrayFromImage(input_img).squeeze()  # sitk -> np, tpye: uint16
        input_img = np.float32(input_img)  # uint16 -> float32
        gt_img = sitk.ReadImage(gt_img_path)
        gt_img = sitk.GetArrayFromImage(gt_img).squeeze()
        gt_img = np.float32(gt_img)

        # low-dose ct denoise
        # norm
        input_img = self.MinMax(input_img)
        # gt_img = self.MinMax(gt_img)
        # print("in:",np.shape(input_img))

        # to tensor
        input_img = (torch.from_numpy(input_img))
        gt_img = (torch.from_numpy(gt_img))

        # # fixme seg
        # # norm
        # input_img = self.norm_brats_seg(input_img)
        # # to tensor
        # input_img = self.to_tensor(input_img)
        # gt_img = self.to_tensor(gt_img)

        return input_img, gt_img,self.input_name_list[index]

if __name__ == '__main__':
    cfg = config1.get_config_2d()

    train_dataloader = DataLoader(GetTrainSet(data_root=cfg.data_root, img_size=cfg.img_size),
                                  batch_size=cfg.batch_size, shuffle=True,num_workers=cfg.num_workers) #=0
    train_iterator = tqdm(train_dataloader)
    for index,data in enumerate(train_iterator):
        print("dataset_test")


