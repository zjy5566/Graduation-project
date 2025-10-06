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


def train_1mask_2d(cfg,unet):
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

    sheet.write(0, 6, " dice"), sheet.write(0, 7, "std_dice")
    sheet.write(0, 8, "iou"), sheet.write(0, 9, "std_iou")
    sheet.write(0, 10, "hausdorff"), sheet.write(0, 11, "std_hausdorff")
    sheet.write(0, 12, " volume_similarity"), sheet.write(0, 13, "std_volume_similarity")
    # sheet.write(0, 14, " false_negative"), sheet.write(0, 15, "std_false_negative")
    # sheet.write(0, 16, " false_positive"), sheet.write(0, 17, "std_false_positive")

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
            slice_gt = gt_img.unsqueeze(0)
            #3mask 2 1mask
            slice_gt[slice_gt>0]=1
            # 清空optimizer的梯度
            optimizer.zero_grad()
            # forward 前向传播
            # sigmoid =nn.Sigmoid()
            pred=unet(slice_input)

            # 计算loss
            loss = bce_loss(pred, slice_gt)
            loss_total += loss.item()    # todo 这个item一定要加，否则会因为计算图不释放导致训练越来越慢以及Out of memory
            # backward 反向传播
            loss.backward()
            # 更新网络参数
            optimizer.step()
                # 存一些训练过程中的效果图
            if img_total % 500== 0:
                save_path = f"{cfg.train_img_root}/epoch_{epoch}img_idx{index}.png"
                # 为了方便对比，把多个图像cat起来
                img_show = torch.cat([slice_input, pred,slice_gt], dim=-1)
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
        with torch.no_grad():
            (psnr_mean, psnr_std, ssim_mean,ssim_std, dice_mean, dice_std, iou_mean,iou_std, hausdorff_mean
             , hausdorff_std, volume_similarity_mean, volume_similarity_std) =test.test(cfg, unet)
        sheet.write(epoch + 1, 2, str(psnr_mean))
        sheet.write(epoch + 1, 3, str(psnr_std))
        sheet.write(epoch + 1, 4, str(ssim_mean))
        sheet.write(epoch + 1, 5, str(ssim_std))
        sheet.write(epoch + 1, 6, str(dice_mean))
        sheet.write(epoch + 1, 7, str(dice_std))
        sheet.write(epoch + 1, 8, str(iou_mean))
        sheet.write(epoch + 1, 9, str(iou_std))
        sheet.write(epoch + 1, 10, str(hausdorff_mean))
        sheet.write(epoch + 1, 11, str(hausdorff_std))
        sheet.write(epoch + 1, 12, str(volume_similarity_mean))
        sheet.write(epoch + 1, 13, str(volume_similarity_std))
        # sheet.write(epoch + 1, 14, false_negative_mean)
        # sheet.write(epoch + 1, 15, false_negative_std)
        # sheet.write(epoch + 1, 16, false_positive_mean)
        # sheet.write(epoch + 1, 17, false_positive_std)
        book.save(cfg.metric_path)


if __name__ == '__main__':
    # cfg1 = config1.get_config_1mask_2d()
    # print(cfg1)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_1mask_2d = UNet2d.UNet().to(device)  
    # unet_1mask_2d.train()
    # train_1mask_2d(cfg1, unet_1mask_2d)