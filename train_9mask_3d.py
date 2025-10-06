#!/usr/bin/python3
import os
from dask.sizeof import sizeof
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import xlwt
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt
import SimpleITK as sitk

# local
from model import UNet2d,Unet3d
import config1
import dataset
import utils
import test


def train_multimasks_3d(cfg,unet_1mask,unet_3mask,unet):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device =  "cpu"

    print(f"device -> {device}")

    # set optimizers
    optimizer = optim.Adam(unet.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    # lr decay  todo 学习率衰减
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(20 / 100 * cfg.epoch_num) - 1,
                                                                      int(40 / 100 * cfg.epoch_num) - 1,
                                                                      int(60 / 100 * cfg.epoch_num) - 1,
                                                                      int(80 / 100 * cfg.epoch_num) - 1],
                                               gamma=0.5)

    # loss functions
    # 图像生成任务
    #l1_loss = nn.L1Loss().to(device)
    #mse_loss = nn.MSELoss().to(device)
    # 分类 or 分割任务
    # bce_loss = nn.BCELoss().to(device)
    CrossEntropy_loss = nn.CrossEntropyLoss().to(device)
    # load training set
    train_dataloader = DataLoader(dataset.GetTrainSet_9MASK(data_root=cfg.data_root, img_size=cfg.img_size),
                                  batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
                                                            # todo shuffle: train->True, test->False

    # 训练断点恢复设置
    if os.path.exists(cfg.weight_path):
        weight = torch.load(cfg.weight_path)
        epoch_begin = weight["epoch"] + 1
        unet.load_state_dict(weight["unet"], strict=False)
        print("Weights load successfully!")
    else:
        epoch_begin = 0
        print("No previous weight.")
    # 创建存储量化指标的表
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("metrics", cell_overwrite_ok=True)
    sheet.write(0, 0, "epoch"), sheet.write(0, 1, "loss")
    sheet.write(0, 2, "Accuracy1"), sheet.write(0, 3, "Accuracy2")
    sheet.write(0, 4, "Accuracy3"), sheet.write(0, 5, "Accuracy4")
    sheet.write(0, 6, "Accuracy5"), sheet.write(0, 7, "Accuracy6")
    sheet.write(0, 8, "Accuracy7"), sheet.write(0, 9, "Accuracy8")
    sheet.write(0, 10, "Accuracy9")

    sheet.write(0, 11, "F1_1"), sheet.write(0, 12, "F1_2")
    sheet.write(0, 13, "F1_3"), sheet.write(0, 14, "F1_4")
    sheet.write(0, 15, "F1_5"), sheet.write(0, 16, "F1_6")
    sheet.write(0, 17, "F1_7"), sheet.write(0, 18, "F1_8")
    sheet.write(0, 19, "F1_9")

    sheet.write(0, 20, "Macro_P"), sheet.write(0, 21, "Macro_R")
    sheet.write(0, 22, "Macro_F1"), sheet.write(0, 23, "Micro_P")
    sheet.write(0, 24, "Micro_R"), sheet.write(0, 25, "Micro_F1")



    book.save(cfg.metric_path)

    # start train
    for epoch in range(epoch_begin, cfg.epoch_num):
        train_iterator = tqdm(train_dataloader)
        patch_total=0
        # init total loss
        loss_total = 0
        # set train
        unet.train()
        # train one epoch
        for index, data in enumerate(train_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)  # [d,512,512]
            gt_img = (data[1].squeeze()).to(device)  # [d,512,512]
            patch_input = input_img.unsqueeze(0).unsqueeze(0)  # [1,1,d,512,512]
            patch_gt = gt_img.unsqueeze(0).long()#[1,1,d,H,W]

            with torch.no_grad():
                # preprocess
                input_mask = unet_1mask(patch_input)
                input_mask[input_mask > 0.7] = 1
                input_mask[input_mask <= 0.7] = 0
                patch_input = input_mask * patch_input

                input_mask2 = unet_3mask(patch_input)  # [1,4,D,H,W]
                patch_input = torch.cat((patch_input,input_mask2), dim=1)  # [1,5,D,H,W]
            # 清空optimizer的梯度
            optimizer.zero_grad()
            # forward 前向传播
            if patch_gt.max()!=0:
                patch_total=patch_total+1
                pred =unet(patch_input)
                # 计算loss
                loss = CrossEntropy_loss(pred, patch_gt)
                loss_total += loss.item()    # todo 这个item一定要加，否则会因为计算图不释放导致训练越来越慢以及Out of memory
                # backward 反向传播
                loss.backward()
                # 更新网络参数
                optimizer.step()
                    # 存一些训练过程中的效果图
                # with torch.no_grad():
                #     plt.imshow(slice_gt[0,0,:,:].cpu().numpy())
                #     plt.show()
                # if patch_total % 100== 0:


                # 经常保存权重
                if patch_total % 200 == 0:
                    torch.save({
                        "epoch": epoch,
                        "unet": unet.state_dict()
                    }, cfg.weight_path)




        # 一个epoch结束
        # todo 更新学习率
        scheduler.step()

        # 保存每一轮的权重文件
        weight_save_path = f"{cfg.weight_save_root}/epoch{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "unet": unet.state_dict()
        }, weight_save_path)

        # 计算每一轮的平均loss
        loss_mean = loss_total /patch_total #len(train_dataloader)
        sheet.write(epoch + 1, 0, epoch)
        sheet.write(epoch + 1, 1, loss_mean)
        book.save(cfg.metric_path)
        train_iterator.desc = "epoch{}  loss:{}".format(epoch, loss_mean)
        # test
        with torch.no_grad():
            metrics =test.test_9label_3d(cfg, unet_1mask,unet_3mask,unet)

            for i in range(9):
                sheet.write(epoch + 1, 2 + i, str(metrics[i]))

            # 写入 F1 指标
            for i in range(9):
                sheet.write(epoch + 1, 11 + i, str(metrics[9 + i]))
        book.save(cfg.metric_path)


if __name__ == '__main__':
  
    cfg3 = config1.get_config_1mask_3d()
    cfg4 = config1.get_config_3mask_3d()
    cfg5=config1.get_config_9mask_3d()
    print(cfg3)
    print(cfg4)
    print(cfg5)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_1mask_3d = Unet3d.UNet().to(device)
    unet_3mask_3d= Unet3d_3.UNet().to(device)
    unet_9mask_3d=Unet3d_9.UNet().to(device)
    #load weight
    weight3=torch.load(cfg3.weight_path)
    unet_1mask_3d.load_state_dict(weight3["unet"])
    weight4 = torch.load(cfg4.weight_path)
    unet_3mask_3d.load_state_dict(weight4["unet"])
    unet_9mask_3d.train()
    train_multimasks_3d(cfg5,unet_1mask_3d,unet_3mask_3d,unet_9mask_3d)





