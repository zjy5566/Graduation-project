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

def  train_1mask_3d(cfg,unet):
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
        patch_train_total=0
        # init total loss
        loss_total = 0
        # set train
        unet.train()
        # train one epoch
        for index, data in enumerate(train_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)#[3,512,512]
            gt_img = (data[1].squeeze()).to(device)#[3,512,512]
            patch_input = input_img.unsqueeze(0).unsqueeze(0)#[1,1,3,512,512]
            patch_gt = gt_img.unsqueeze(0).unsqueeze(0)
            #3mask 2 1mask
            patch_gt[patch_gt>0]=1
            if patch_gt.max()!=0:
                # 清空optimizer的梯度
                optimizer.zero_grad()
                # forward 前向传播
                pred=unet(patch_input)
                # 计算loss
                loss = bce_loss(pred, patch_gt)
                loss_total += loss.item()    # todo 这个item一定要加，否则会因为计算图不释放导致训练越来越慢以及Out of memory
                # backward 反向传播
                loss.backward()
                # 更新网络参数
                optimizer.step()
                patch_train_total = patch_train_total + 1
                # 存一些训练过程中的效果图
                with torch.no_grad():
                    if patch_train_total % 200== 0:
                        save_path = f"{cfg.train_img_root}/epoch_{epoch}patch_idx{index}.nii.gz"
                        # 为了方便对比，把多个图像cat起来
                        # img1=torch.cat([patch_input[:,:,0,:,:],patch_input[:,:,1,:,:],patch_input[:,:,2,:,:]],dim=2)
                        # img2=torch.cat([pred[:,:,0,:,:],pred[:,:,1,:,:],pred[:,:,2,:,:]],dim=2)
                        # img3=torch.cat([patch_gt[:,:,0,:,:],patch_gt[:,:,1,:,:],patch_gt[:,:,2,:,:]],dim=2)
                        # img_show = torch.cat([img1, img2,img3], dim=3)
                        # 为了方便对比，把多个PATCH CAT起来
                        patch_input=patch_input.squeeze()
                        patch_gt=patch_gt.squeeze()
                        pred=pred.squeeze()
                        # patch_show_1=torch.cat([patch_input,pred ,patch_gt],dim=1)#[D,H,W]
                        patch_show_np=pred.cpu().numpy()
                        patch_show_2=sitk.GetImageFromArray(patch_show_np)
                        sitk.WriteImage(patch_show_2,save_path)

                # # 调窗
                # img_show = utils.trans_to_display_window(img_show)
                # cv2.imwrite(save_path, img_show)

            # 经常保存权重
            if patch_train_total % 200 == 0:
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
        loss_mean = loss_total /patch_train_total #len(train_dataloader)
        sheet.write(epoch + 1, 0, epoch)
        sheet.write(epoch + 1, 1, loss_mean)
        book.save(cfg.metric_path)
        train_iterator.desc = "epoch{}  loss:{}".format(epoch, loss_mean)

        # test
        with torch.no_grad():
            (psnr_mean, psnr_std, ssim_mean,ssim_std, dice_mean, dice_std, iou_mean,iou_std, hausdorff_mean
             , hausdorff_std, volume_similarity_mean, volume_similarity_std) =test.test_1label_3d(cfg, unet)
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
        book.save(cfg.metric_path)\

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

def train_3mask_3d(cfg,unet_1mask,unet):
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
        patch_train_total=0
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
            patch_gt = gt_img.unsqueeze(0)#[1,d,512,512]
            # 清空optimizer的梯度
            optimizer.zero_grad()
            # if ((torch.isnan(patch_gt).any().item()) == False):
            if patch_gt.max()!=0:

                #划分面部区域 Label处理
                with torch.no_grad():
                    patch_gt[patch_gt == 1] = 3
                    patch_gt[patch_gt == 0.25] = 1
                    patch_gt[patch_gt == 0.5] = 2
                    patch_gt = patch_gt.long()

                    input_mask=unet_1mask(patch_input)
                    input_mask[input_mask>0.7]=1
                    input_mask[input_mask <= 0.7] = 0
                    patch_input=input_mask*patch_input
                if patch_input.max()!=0:
                    # forward 前向传播
                    patch_train_total =patch_train_total + 1
                    pred =unet(patch_input)#[1,4,d,512,512]
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
                    with torch.no_grad():
                        if patch_train_total % 800== 0:
                            save_path = f"{cfg.train_img_root}/epoch_{epoch}_patch_idx{index}.nii.gz"
                            # 为了方便对比，把多个图像cat起来
                            img1 = patch_input.squeeze()
                            img2 = patch_gt.squeeze()
                            img3 = utils.minmax(torch.argmax(pred,dim=1,keepdim=True))
                            # patch_show_1 = torch.cat([patch_input, pred, patch_gt], dim=1)  # [D,H,W]
                            patch_show_np = img3.cpu().numpy()
                            patch_show_2 = sitk.GetImageFromArray(patch_show_np)
                            sitk.WriteImage(patch_show_2, save_path)


                # 经常保存权重
            if patch_train_total % 200 == 0:
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
        loss_mean = loss_total /patch_train_total #len(train_dataloader)
        sheet.write(epoch + 1, 0, epoch)
        sheet.write(epoch + 1, 1, loss_mean)
        book.save(cfg.metric_path)
        train_iterator.desc = "epoch{}  loss:{}".format(epoch, loss_mean)
        # test
        # with torch.no_grad():
        #     (psnr_mean, psnr_std, ssim_mean, ssim_std, A1_mean, A2_mean, A3_mean, F1_1_mean,
        #      F1_2_mean, F1_3_mean, Macro_P_mean,Macro_R_mean , Macro_F1_mean, Micro_P_mean,
        #      Micro_R_mean, Micro_F1_mean) =test.test_3label_3d(cfg, unet_1mask,unet)
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


def test_only_3m_2d():
    #model1
    cfg1 = config1.get_config_1mask_2d()
    print(cfg1)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_1mask_2d = UNet2d.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    weight1 = torch.load(cfg1.weight_path)
    unet_1mask_2d.load_state_dict(weight1["unet"])
    # model2
    cfg2 = config1.get_config_3mask_2d()
    print(cfg2)
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("metrics", cell_overwrite_ok=True)
    sheet.write(0, 0, "epoch"), sheet.write(0, 1, "loss")
    sheet.write(0, 2, "psnr"), sheet.write(0, 3, "std_psnr")
    sheet.write(0, 4, "ssim"), sheet.write(0, 5, "std_ssim")
    sheet.write(0, 6, "Precision1"), sheet.write(0, 7, "Precision2")
    sheet.write(0, 8, "Precision3"), sheet.write(0, 9, "F1_1")
    sheet.write(0, 10, "F1_2"), sheet.write(0, 11, "F1_3")
    sheet.write(0, 12, " Macro_P"), sheet.write(0, 13, "Macro_R")
    sheet.write(0, 14, " Macro_F1"), sheet.write(0, 15, "Micro_P")
    sheet.write(0, 16, " Micro_R"), sheet.write(0, 17, "Micro_F1")
    book.save(cfg2.metric_path)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_3mask_2d = UNet2d_3.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    # weight2_list = os.listdir(cfg2.weight_save_root)
    # weight2_list.sort()
    for i in range(10,100):
        weight_epoch=f'epoch{i}.pth'
        weight2 = torch.load(os.path.join(cfg2.weight_save_root,weight_epoch))
        print(os.path.join(cfg2.weight_save_root,weight_epoch))
        unet_3mask_2d.load_state_dict(weight2["unet"], strict=False)
        print("Weights load successfully!")
        (psnr_mean, psnr_std, ssim_mean, ssim_std, A1_mean, A2_mean, A3_mean, F1_1_mean,
         F1_2_mean, F1_3_mean, Macro_P_mean, Macro_R_mean, Macro_F1_mean, Micro_P_mean,
         Micro_R_mean, Micro_F1_mean) = test.test_3label(cfg2, unet_1mask_2d, unet_3mask_2d)
        sheet.write(i + 1, 2, str(psnr_mean))
        sheet.write(i + 1, 3, str(psnr_std))
        sheet.write(i + 1, 4, str(ssim_mean))
        sheet.write(i + 1, 5, str(ssim_std))
        sheet.write(i + 1, 6, str(A1_mean))
        sheet.write(i + 1, 7, str(A2_mean))
        sheet.write(i + 1, 8, str(A3_mean))
        sheet.write(i + 1, 9, str(F1_1_mean))
        sheet.write(i + 1, 10, str(F1_2_mean))
        sheet.write(i + 1, 11, str(F1_3_mean))
        sheet.write(i + 1, 12, str(Macro_P_mean))
        sheet.write(i + 1, 13, str(Macro_R_mean))
        sheet.write(i + 1, 14, str(Macro_F1_mean))
        sheet.write(i + 1, 15, str(Micro_P_mean))
        sheet.write(i + 1, 16, str(Micro_R_mean))
        sheet.write(i + 1, 17, str(Micro_F1_mean))
        book.save(cfg2.metric_path)
    print("test of train_1mask_3d have finished!!")

def test_only_3m_3d():
    #model1
    cfg1 = config1.get_config_1mask_3d()
    print(cfg1)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_1mask_3d = Unet3d.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    weight1 = torch.load(cfg1.weight_path)
    unet_1mask_3d.load_state_dict(weight1["unet"])
    # model2
    cfg2 = config1.get_config_3mask_3d()
    print(cfg2)
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("metrics", cell_overwrite_ok=True)
    sheet.write(0, 0, "epoch"), sheet.write(0, 1, "loss")
    sheet.write(0, 2, "psnr"), sheet.write(0, 3, "std_psnr")
    sheet.write(0, 4, "ssim"), sheet.write(0, 5, "std_ssim")
    sheet.write(0, 6, "Precision1"), sheet.write(0, 7, "Precision2")
    sheet.write(0, 8, "Precision3"), sheet.write(0, 9, "F1_1")
    sheet.write(0, 10, "F1_2"), sheet.write(0, 11, "F1_3")
    sheet.write(0, 12, " Macro_P"), sheet.write(0, 13, "Macro_R")
    sheet.write(0, 14, " Macro_F1"), sheet.write(0, 15, "Micro_P")
    sheet.write(0, 16, " Micro_R"), sheet.write(0, 17, "Micro_F1")
    book.save(cfg2.metric_path)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_3mask_3d = Unet3d_3.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    # weight2_list = os.listdir(cfg2.weight_save_root)
    # weight2_list.sort()
    for i in range(5, 100):
        weight_epoch = f'epoch{i}.pth'
        weight2 = torch.load(os.path.join(cfg2.weight_save_root,weight_epoch))
        unet_3mask_3d.load_state_dict(weight2["unet"], strict=False)
        print("Weights load successfully!")
        (psnr_mean, psnr_std, ssim_mean, ssim_std, A1_mean, A2_mean, A3_mean, F1_1_mean,
         F1_2_mean, F1_3_mean, Macro_P_mean, Macro_R_mean, Macro_F1_mean, Micro_P_mean,
         Micro_R_mean, Micro_F1_mean) = test.test_3label_3d(cfg2, unet_1mask_3d, unet_3mask_3d)
        sheet.write(i + 1, 2, str(psnr_mean))
        sheet.write(i + 1, 3, str(psnr_std))
        sheet.write(i + 1, 4, str(ssim_mean))
        sheet.write(i + 1, 5, str(ssim_std))
        sheet.write(i + 1, 6, str(A1_mean))
        sheet.write(i + 1, 7, str(A2_mean))
        sheet.write(i + 1, 8, str(A3_mean))
        sheet.write(i + 1, 9, str(F1_1_mean))
        sheet.write(i + 1, 10, str(F1_2_mean))
        sheet.write(i + 1, 11, str(F1_3_mean))
        sheet.write(i + 1, 12, str(Macro_P_mean))
        sheet.write(i + 1, 13, str(Macro_R_mean))
        sheet.write(i + 1, 14, str(Macro_F1_mean))
        sheet.write(i + 1, 15, str(Micro_P_mean))
        sheet.write(i + 1, 16, str(Micro_R_mean))
        sheet.write(i + 1, 17, str(Micro_F1_mean))
        book.save(cfg2.metric_path)
        print("metric successfully saved!!")

    print("test of train_1mask_3d have finished!!")


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
    #
    #
    # cfg3 = config1.get_config_1mask_3d()
    # print(cfg3)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_1mask_3d = Unet3d.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    # unet_1mask_3d.train()
    # train_1mask_3d(cfg3, unet_1mask_3d)
    # # test.test_1label_3d(cfg3,unet_1mask_3d)
    # #
    # cfg4 = config1.get_config_3mask_3d()
    # print(cfg4)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # # device = "cpu"
    # unet_3mask_3d= Unet3d_3.UNet().to(device)  # todo 一定要记住 .to(device) 或 .cuda() in_channels=1, out_channels=1
    # unet_3mask_3d.train()
    # train_3mask_3d(cfg4, unet_1mask_3d,unet_3mask_3d)
    # unet_3mask_3d.eval()

    # test_only_3m_2d()
    # test_only_3m_3d()
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





