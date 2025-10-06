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



def train_3mask_2d(cfg,unet_1mask,unet):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    bce_loss = nn.BCELoss().to(device)
    CrossEntropy_loss=nn.CrossEntropyLoss().to(device)


    # load training set
    train_dataloader = DataLoader(dataset.GetTrainSet(data_root=cfg.data_root, img_size=cfg.img_size),
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
    sheet.write(0, 2, "psnr"), sheet.write(0, 3, "std_psnr")
    sheet.write(0, 4, "ssim"), sheet.write(0, 5, "std_ssim")
    sheet.write(0, 6, "Accuracy1"), sheet.write(0, 7, "Accuracy2")
    sheet.write(0, 8, "Accuracy3"), sheet.write(0, 9, "F1_1")
    sheet.write(0, 10, "F1_2"), sheet.write(0, 11, "F1_3")
    sheet.write(0, 12, " Macro_P"), sheet.write(0, 13, "Macro_R")
    sheet.write(0, 14, " Macro_F1"), sheet.write(0, 15, "Micro_P")
    sheet.write(0, 16, " Micro_R"), sheet.write(0, 17, "Micro_F1")



    book.save(cfg.metric_path)

    # start train
    for epoch in range(epoch_begin, cfg.epoch_num):
        train_iterator = tqdm(train_dataloader)
        img_total=0
        # init total loss
        loss_total = 0
        # set train
        unet.train()
        # train one epoch
        for index, data in enumerate(train_iterator):
            # load imgs
            img_total=img_total+1
            input_img = (data[0].squeeze()).to(device)
            gt_img = (data[1].squeeze()).to(device)
            input_img=input_img.unsqueeze(0)
            gt_img=gt_img.unsqueeze(0)
            slice_input = input_img.unsqueeze(0)
            slice_gt = gt_img.clone()
            # 清空optimizer的梯度
            optimizer.zero_grad()
            with torch.no_grad():
                slice_gt[slice_gt==1]=3
                slice_gt[slice_gt==0.25]=1
                slice_gt[slice_gt==0.5]=2
                slice_gt=slice_gt.long()

                input_mask=unet_1mask(slice_input)
                input_mask[input_mask>0.7]=1
                input_mask[input_mask <= 0.7] = 0
                slice_input=input_mask*slice_input
            # forward 前向传播
            pred=unet(slice_input)#[1,4,512,512]
            # 计算loss
            loss = CrossEntropy_loss(pred, slice_gt)
            loss_total += loss.item()    # todo 这个item一定要加，否则会因为计算图不释放导致训练越来越慢以及Out of memory
            # backward 反向传播
            loss.backward()
            # 更新网络参数
            optimizer.step()
                # 存一些训练过程中的效果图
            # with torch.no_grad():
            #     plt.imshow(slice_gt[0,0,:,:].cpu().numpy())
            #     plt.show()
            if img_total % 800== 0:
                img1=utils.minmax(slice_input)
                img2=torch.argmax(pred,dim=1,keepdim=True)
                img2=utils.minmax(img2)
                img3=utils.minmax(slice_gt)
                save_path = f"{cfg.train_img_root}/epoch_{epoch}img_idx{index}.png"
                # 为了方便对比，把多个图像cat起来
                img_show = torch.cat([img1, img2,img3], dim=-1)#
                # 调窗
                img_show = utils.trans_to_display_window(img_show)
                cv2.imwrite(save_path, img_show)

            # 经常保存权重
            if img_total % 200 == 0:
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
        loss_mean = loss_total /img_total #len(train_dataloader)
        sheet.write(epoch + 1, 0, epoch)
        sheet.write(epoch + 1, 1, loss_mean)
        book.save(cfg.metric_path)
        train_iterator.desc = "epoch{}  loss:{}".format(epoch, loss_mean)
        # test
        # with torch.no_grad():
        #     (psnr_mean, psnr_std, ssim_mean, ssim_std, A1_mean, A2_mean, A3_mean, F1_1_mean,
        #      F1_2_mean, F1_3_mean, Macro_P_mean,Macro_R_mean , Macro_F1_mean, Micro_P_mean,
        #      Micro_R_mean, Micro_F1_mean) =test.test_3label(cfg, unet_1mask,unet)
        #     ##A1, A2, A3, F1_1, F1_2, F1_3, Macro_P, Macro_R, Macro_F1, Micro_P,
        #     ##Micro_R, Micro_F1
        # sheet.write(epoch + 1, 2, str(psnr_mean))
        # sheet.write(epoch + 1, 3, str(psnr_std))
        # sheet.write(epoch + 1, 4, str(ssim_mean))
        # sheet.write(epoch + 1, 5, str(ssim_std))
        # sheet.write(epoch + 1, 6, str(A1_mean))
        # sheet.write(epoch + 1, 7, str(A2_mean))
        # sheet.write(epoch + 1, 8, str(A3_mean))
        # sheet.write(epoch + 1, 9, str(F1_1_mean))
        # sheet.write(epoch + 1, 10, str(F1_2_mean))
        # sheet.write(epoch + 1, 11, str(F1_3_mean))
        # sheet.write(epoch + 1, 12, str(Macro_P_mean))
        # sheet.write(epoch + 1, 13, str(Macro_R_mean))
        # sheet.write(epoch + 1, 14, str(Macro_F1_mean))
        # sheet.write(epoch + 1, 15, str(Micro_P_mean))
        # sheet.write(epoch + 1, 16, str(Micro_R_mean))
        # sheet.write(epoch + 1, 17, str(Micro_F1_mean))
        # book.save(cfg.metric_path)

if __name__ == '__main__':
    # cfg1 = config1.get_config_1mask_2d()
    # print(cfg1)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_1mask_2d = UNet2d.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    # unet_1mask_2d.train()
    # train_1mask_2d(cfg1, unet_1mask_2d)
    #
    # cfg2= config1.get_config_3mask_2d()
    # print(cfg2)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_3mask_2d = UNet2d_3.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    # unet_3mask_2d.train()
    # train_3mask_2d(cfg2,unet_1mask_2d,unet_3mask_2d)
    #
   