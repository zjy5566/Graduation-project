import os
from dask.sizeof import sizeof
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import sys
import xlwt
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# local
from model import UNet2d, Unet3d, UNet2d_3, Unet3d_3, Unet3d_9
import config1
import dataset
import utils
import  train
import SimpleITK as sitk
import  re

def test(cfg,unet):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device -> {device}")

    test_dataloader = DataLoader(dataset.GetTestSet(data_root=cfg.data_root, img_size=cfg.img_size),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_iterator = tqdm(test_dataloader)

    # 保存每一轮的测试图片
    test_epoch_root = f"{cfg.test_img_root}"  #/epoch{epoch}"
    if not os.path.exists(test_epoch_root):
        os.makedirs(test_epoch_root)

    # start test
    with torch.no_grad():
        # init

        psnr_list = []
        ssim_list = []
        dice_list = []
        iou_list = []
        hausdorff_list = []
        volume_similarity_list = []
        false_negative_list = []
        false_positive_list = []
        for index, data in enumerate(test_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)
            gt_img = (data[1].squeeze()).to(device)
            input_img = input_img.unsqueeze(0)
            gt_img = gt_img.unsqueeze(0)
            slice_input = input_img.unsqueeze(0)
            slice_gt = gt_img.unsqueeze(0)
            slice_gt[slice_gt>0]=1
            # forward 前向传播
            pred = unet(slice_input)
            # 计算指标
            gt_calculate=utils.trans_to_calculate_window_1mask(slice_gt)
            pred_calculate=utils.trans_to_calculate_window_1mask(pred)
            if gt_calculate.cpu().numpy().sum()!=0:
                psnr, ssim, _ = utils.compute_measure(gt_calculate, pred_calculate,
                                                      data_range=gt_calculate.max() - gt_calculate.min())
                dice, iou, hausdorff, volume_similarity = utils.compute_measure2(gt_calculate,pred_calculate)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                dice_list.append(dice)
                iou_list.append(iou)
                hausdorff_list.append(hausdorff)
                volume_similarity_list.append(volume_similarity)
                 # false_negative_list.append(false_negative)
                 # false_positive_list.append(false_positive)
            if index % 3 == 0:      # 保存测试图像
                save_path = f"{test_epoch_root}/test_result_{index}.png"
                img_show = torch.cat([slice_input, pred, slice_gt], dim=-1)
                img_show = utils.trans_to_display_window(img_show)
                cv2.imwrite(save_path, img_show)


    # dice, iou, hausdorff, volume_similarity, false_negative, false_positive

    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list, ddof=1)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    dice_mean=np.mean(dice_list)
    dice_std=np.std(dice_list, ddof=1)
    iou_mean=np.mean(iou_list)
    iou_std = np.std(iou_list)
    hausdorff_mean=np.mean(hausdorff_list)
    hausdorff_std=np.std(hausdorff_list, ddof=1)
    volume_similarity_mean=np.mean(volume_similarity_list)
    volume_similarity_std=np.std(volume_similarity_list, ddof=1)
    # false_negative_mean=np.mean(false_negative_list)
    # false_negative_std=np.std(false_negative_list, ddof=1)
    # false_positive_mean=np.mean(false_positive_list)
    # false_positive_std=np.std(false_positive_list, ddof=1)

    return (psnr_mean, psnr_std, ssim_mean,ssim_std, dice_mean, dice_std, iou_mean,iou_std, hausdorff_mean
             , hausdorff_std, volume_similarity_mean, volume_similarity_std)




    #print(f"loss:{loss_mean} psnr:{psnr_mean}-{psnr_std}   ssim:{ssim_mean}-{ssim_std}")

def test_3label(cfg,unet_1mask,unet):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device -> {device}")

    test_dataloader = DataLoader(dataset.GetTestSet(data_root=cfg.data_root, img_size=cfg.img_size),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_iterator = tqdm(test_dataloader)

    # 保存每一轮的测试图片
    test_epoch_root = f"{cfg.test_img_root}"  #/epoch{epoch}"
    if not os.path.exists(test_epoch_root):
        os.makedirs(test_epoch_root)

    # start test
    with torch.no_grad():
        # init
        #A1, A2, A3, F1_1, F1_2, F1_3, Macro_P, Macro_R, Macro_F1, Micro_P, Micro_R, Micro_F1
        psnr_list = []
        ssim_list = []
        A1_list = []
        A2_list = []
        A3_list = []
        F1_1_list = []
        F1_2_list = []
        F1_3_list = []
        Macro_P_list = []
        Macro_R_list = []
        Macro_F1_list = []
        Micro_P_list = []
        Micro_R_list = []
        Micro_F1_list = []

        for index, data in enumerate(test_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)
            gt_img = (data[1].squeeze()).to(device)
            input_img = input_img.unsqueeze(0)
            gt_img = gt_img.unsqueeze(0)
            slice_input = input_img.unsqueeze(0)
            slice_gt = gt_img.unsqueeze(0)#[1,1,h,w]

            slice_gt[slice_gt == 1] = 3
            slice_gt[slice_gt == 0.25] = 1
            slice_gt[slice_gt == 0.5] = 2
            slice_gt = slice_gt.long()

            input_mask = unet_1mask(slice_input)
            input_mask[input_mask > 0.9] = 1
            input_mask[input_mask <= 0.9] = 0
            slice_input = input_mask * slice_input

            # forward 前向传播
            pred = unet(slice_input)
            # 计算指标
            if (slice_gt.max()!=0)&(pred.max()!=0):
                pred=torch.argmax(pred, dim=1, keepdim=True)#[1,1,h,w]
                # slice_gt=utils.minmax(slice_gt)#[h,w]
                # slice_input=utils.minmax(slice_input)
                # psnr, ssim, _ = utils.compute_measure(slice_gt, pred,
                #                                       data_range=slice_gt.max() - slice_gt.min())
                (A1,A2,A3,F1_1,F1_2,F1_3,Macro_P, Macro_R, Macro_F1,Micro_P,
                    Micro_R, Micro_F1) = utils.compute_measure3_3mask(slice_gt,pred)
                # psnr_list.append(psnr)
                # ssim_list.append(ssim)
                A1_list.append(A1)
                A2_list.append(A2)
                A3_list.append(A3)
                F1_1_list.append(F1_1)
                F1_2_list.append(F1_2)
                F1_3_list.append( F1_3)
                Macro_P_list.append(Macro_P)
                Macro_R_list.append(Macro_R)
                Macro_F1_list.append(Macro_F1)
                Micro_P_list.append(Micro_P)
                Micro_R_list.append(Micro_R)
                Micro_F1_list.append(Micro_F1)

            if index % 200 == 0:      # 保存测试图像
                slice_input=utils.minmax(slice_input)
                pred=utils.minmax(pred)
                slice_gt=utils.minmax(slice_gt)
                save_path = f"{test_epoch_root}/test_result_{index}.png"
                img_show = torch.cat([slice_input, pred, slice_gt], dim=-1)
                img_show = utils.trans_to_display_window(img_show)
                cv2.imwrite(save_path, img_show)


    # dice, iou, hausdorff, volume_similarity, false_negative, false_positive

    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list, ddof=1)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    A1_mean = np.nanmean(A1_list)
    A2_mean= np.nanmean(A2_list)
    A3_mean= np.nanmean(A3_list)
    F1_1_mean= np.nanmean(F1_1_list)
    F1_2_mean= np.nanmean(F1_2_list)
    F1_3_mean= np.nanmean(F1_3_list)
    Macro_P_mean= np.nanmean(Macro_P_list)
    Macro_R_mean= np.nanmean(Macro_R_list)
    Macro_F1_mean= np.nanmean(Macro_F1_list)
    Micro_P_mean= np.nanmean(Micro_P_list)
    Micro_R_mean= np.nanmean(Micro_R_list)
    Micro_F1_mean= np.nanmean(Micro_F1_list)

    return (psnr_mean,psnr_std,ssim_mean,ssim_std,A1_mean,A2_mean,A3_mean,F1_1_mean,F1_2_mean,F1_3_mean,
                Macro_P_mean,Macro_R_mean,Macro_F1_mean,Micro_P_mean,Micro_R_mean,Micro_F1_mean)




    #print(f"loss:{loss_mean} psnr:{psnr_mean}-{psnr_std}   ssim:{ssim_mean}-{ssim_std}")

def test_1label_3d(cfg,unet):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device -> {device}")

    test_dataloader = DataLoader(dataset.GetTestSet(data_root=cfg.data_root, img_size=cfg.img_size),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_iterator = tqdm(test_dataloader)

    # 保存每一轮的测试图片
    test_epoch_root = f"{cfg.test_img_root}"  #/epoch{epoch}"
    if not os.path.exists(test_epoch_root):
        os.makedirs(test_epoch_root)

    # start test
    with torch.no_grad():
        # init

        psnr_list = []
        ssim_list = []
        dice_list = []
        iou_list = []
        hausdorff_list = []
        volume_similarity_list = []
        false_negative_list = []
        false_positive_list = []
        for index, data in enumerate(test_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)  # [3,512,512]
            gt_img = (data[1].squeeze()).to(device)  # [3,512,512]
            patch_input = input_img.unsqueeze(0).unsqueeze(0)  # [1,1,3,512,512]
            patch_gt = gt_img.unsqueeze(0).unsqueeze(0)

            patch_gt[patch_gt>0]=1
            # gt非0则测试
            if patch_gt.max()!=0:
                pred = unet(patch_input)
                # 计算指标
                for c in range(patch_gt.shape[2]):
                    gt_calculate = utils.trans_to_calculate_window_1mask(patch_gt[:, :, c, :, :])
                    pred_calculate = utils.trans_to_calculate_window_1mask(pred[:, :, c, :, :])
                    if gt_calculate.cpu().numpy().sum() != 0:
                        psnr, ssim, _ = utils.compute_measure(gt_calculate, pred_calculate,
                                                              data_range=gt_calculate.max() - gt_calculate.min())
                        dice, iou, hausdorff, volume_similarity = utils.compute_measure2(gt_calculate, pred_calculate)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        dice_list.append(dice)
                        iou_list.append(iou)
                        hausdorff_list.append(hausdorff)
                        volume_similarity_list.append(volume_similarity)
                        # false_negative_list.append(false_negative)
                        # false_positive_list.append(false_positive)
            #gt为0 pred保存为0的图像
            else:
                pred=patch_gt.clone()
            # 保存测试图像
            save_path = f"{test_epoch_root}/test_result_{index}.nii.gz"
            patch_input = patch_input.squeeze()
            patch_gt = patch_gt.squeeze()
            pred = pred.squeeze()
            # patch_show_1 = torch.cat([patch_input, pred, patch_gt], dim=1)  # [D,H,W]
            patch_show_np = pred.cpu().numpy()
            patch_show_2 = sitk.GetImageFromArray(patch_show_np)
            sitk.WriteImage(patch_show_2, save_path)

    # dice, iou, hausdorff, volume_similarity, false_negative, false_positive
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list, ddof=1)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    dice_mean=np.mean(dice_list)
    dice_std=np.std(dice_list, ddof=1)
    iou_mean=np.mean(iou_list)
    iou_std = np.std(iou_list)
    hausdorff_mean=np.mean(hausdorff_list)
    hausdorff_std=np.std(hausdorff_list, ddof=1)
    volume_similarity_mean=np.mean(volume_similarity_list)
    volume_similarity_std=np.std(volume_similarity_list, ddof=1)
    # false_negative_mean=np.mean(false_negative_list)
    # false_negative_std=np.std(false_negative_list, ddof=1)
    # false_positive_mean=np.mean(false_positive_list)
    # false_positive_std=np.std(false_positive_list, ddof=1)

    return (psnr_mean, psnr_std, ssim_mean,ssim_std, dice_mean, dice_std, iou_mean,iou_std, hausdorff_mean
             , hausdorff_std, volume_similarity_mean, volume_similarity_std)




    #print(f"loss:{loss_mean} psnr:{psnr_mean}-{psnr_std}   ssim:{ssim_mean}-{ssim_std}")

def test_3label_3d(cfg,unet_1mask,unet):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device -> {device}")

    test_dataloader = DataLoader(dataset.GetTestSet(data_root=cfg.data_root, img_size=cfg.img_size),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_iterator = tqdm(test_dataloader)

    # 保存每一轮的测试图片
    test_epoch_root = f"{cfg.test_img_root}"  #/epoch{epoch}"
    if not os.path.exists(test_epoch_root):
        os.makedirs(test_epoch_root)

    # start test
    with torch.no_grad():
        # init
        #A1, A2, A3, F1_1, F1_2, F1_3, Macro_P, Macro_R, Macro_F1, Micro_P, Micro_R, Micro_F1
        psnr_list = []
        ssim_list = []
        A1_list = []
        A2_list = []
        A3_list = []
        F1_1_list = []
        F1_2_list = []
        F1_3_list = []
        Macro_P_list = []
        Macro_R_list = []
        Macro_F1_list = []
        Micro_P_list = []
        Micro_R_list = []
        Micro_F1_list = []

        for index, data in enumerate(test_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)  # [?,128,128]
            gt_img = (data[1].squeeze()).to(device)  # [?,128,128]
            patch_input = input_img.unsqueeze(0).unsqueeze(0)  # [1,1,?,128,128]
            patch_gt = gt_img.unsqueeze(0).unsqueeze(0)

            if patch_gt.max()!=0:
                #划分面部区域
                input_mask = unet_1mask(patch_input)
                input_mask[input_mask > 0.9] = 1
                input_mask[input_mask <= 0.9] = 0
                patch_input = input_mask * patch_input

                patch_gt[patch_gt == 1] = 3
                patch_gt[patch_gt == 0.25] = 1
                patch_gt[patch_gt == 0.5] = 2
                patch_gt = patch_gt.long()
                # forward 前向传播
                pred = unet(patch_input)#[1,4,d,h,w]

                # 计算指标
                pred=torch.argmax(pred,dim=1,keepdim=True)#[1,1,d,h,w]
                # patch_gt=utils.minmax(patch_gt).unsqueeze(0).unsqueeze(0)#[1,1,d,h,w]
                for c in range(patch_gt.shape[2]):
                    if (patch_gt[:,:,c,:,:].max()!=0)&(pred[:,:,c,:,:].max()!=0):

                        # psnr, ssim, _ = utils.compute_measure(patch_gt[:,:,c,:,:], pred[:,:,c,:,:],
                        #                                       data_range=patch_gt[:,:,c,:,:].max() - patch_gt[:,:,c,:,:].min())
                        # img1 = utils.minmax(patch_input[:,:,c,:,:])
                        # img2 = utils.minmax(pred[:,:,c,:,:])
                        # img3 = utils.minmax(patch_gt[:,:,c,:,:])
                        # save_path = f"{test_epoch_root}/test_result_{index}.png"
                        # img_show = torch.cat([img1, img2, img3], dim=-1)
                        # img_show = utils.trans_to_display_window(img_show)
                        # cv2.imwrite(save_path, img_show)

                        (A1,A2,A3,F1_1,F1_2,F1_3,Macro_P, Macro_R, Macro_F1,Micro_P,
                            Micro_R, Micro_F1) = utils.compute_measure3_3mask(patch_gt[:,:,c,:,:],pred[:,:,c,:,:])
                        # psnr_list.append(psnr)
                        # ssim_list.append(ssim)
                        A1_list.append(A1)
                        A2_list.append(A2)
                        A3_list.append(A3)
                        F1_1_list.append(F1_1)
                        F1_2_list.append(F1_2)
                        F1_3_list.append( F1_3)
                        Macro_P_list.append(Macro_P)
                        Macro_R_list.append(Macro_R)
                        Macro_F1_list.append(Macro_F1)
                        Micro_P_list.append(Micro_P)
                        Micro_R_list.append(Micro_R)
                        Micro_F1_list.append(Micro_F1)
            else:
                pred=patch_gt.clone()

            # 保存测试图像
            save_path = f"{test_epoch_root}/test_result_{index}.nii.gz"
            patch_input = patch_input.squeeze()
            patch_gt = patch_gt.squeeze()
            pred = utils.minmax(pred)
            # patch_show_1 = torch.cat([patch_input, pred, patch_gt], dim=1)  # [D,H,W]
            # patch_show_np = pred.cpu().numpy()
            patch_show_2 = sitk.GetImageFromArray(pred)
            sitk.WriteImage(patch_show_2, save_path)


    # dice, iou, hausdorff, volume_similarity, false_negative, false_positive

    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list, ddof=1)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    A1_mean = np.nanmean(A1_list)
    A2_mean= np.nanmean(A2_list)
    A3_mean= np.nanmean(A3_list)
    F1_1_mean= np.nanmean(F1_1_list)
    F1_2_mean= np.nanmean(F1_2_list)
    F1_3_mean= np.nanmean(F1_3_list)
    Macro_P_mean= np.nanmean(Macro_P_list)
    Macro_R_mean= np.nanmean(Macro_R_list)
    Macro_F1_mean= np.nanmean(Macro_F1_list)
    Micro_P_mean= np.nanmean(Micro_P_list)
    Micro_R_mean= np.nanmean(Micro_R_list)
    Micro_F1_mean= np.nanmean(Micro_F1_list)

    return (psnr_mean,psnr_std,ssim_mean,ssim_std,A1_mean,A2_mean,A3_mean,F1_1_mean,F1_2_mean,F1_3_mean,
                Macro_P_mean,Macro_R_mean,Macro_F1_mean,Micro_P_mean,Micro_R_mean,Micro_F1_mean)




    #print(f"loss:{loss_mean} psnr:{psnr_mean}-{psnr_std}   ssim:{ssim_mean}-{ssim_std}")

def test_9label_3d(cfg,unet_1mask,unet_3mask,unet):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device -> {device}")

    test_dataloader = DataLoader(dataset.GetTestSet_9MASK(data_root=cfg.data_root, img_size=cfg.img_size),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_iterator = tqdm(test_dataloader)

    # 保存每一轮的测试图片
    test_epoch_root = f"{cfg.test_img_root}"  #/epoch{epoch}"
    if not os.path.exists(test_epoch_root):
        os.makedirs(test_epoch_root)

    # start test
    with torch.no_grad():
        # init
        #A1, A2, A3, F1_1, F1_2, F1_3, Macro_P, Macro_R, Macro_F1, Micro_P, Micro_R, Micro_F1
        epochs = []
        losses = []
        accuracy_list = [[] for _ in range(9)]  # 9个 Accuracy 列表
        f1_list = [[] for _ in range(9)]  # 9个 F1 列表
        macro_p_list = []
        macro_r_list = []
        macro_f1_list = []
        micro_p_list = []
        micro_r_list = []
        micro_f1_list = []
        testimg_total = 0
        for index, data in enumerate(test_iterator):
            # load imgs
            testimg_total = testimg_total+1
            input_img = (data[0].squeeze()).to(device)  # [?,128,128]
            gt_img = (data[1].squeeze()).to(device)  # [?,128,128]
            patch_input = input_img.unsqueeze(0).unsqueeze(0)  # [1,1,?,128,128]
            patch_gt = gt_img.unsqueeze(0).unsqueeze(0)#[1,1,D,H,W]

            if patch_gt.max()!=0:
                # preprocess
                input_mask = unet_1mask(patch_input)
                input_mask[input_mask > 0.7] = 1
                input_mask[input_mask <= 0.7] = 0
                patch_input = input_mask * patch_input

                input_mask2 = unet_3mask(patch_input)  # [1,4,D,H,W]
                patch_input = torch.cat((patch_input, input_mask2), dim=1)  # [1,5,D,H,W]
                patch_gt = patch_gt.long()
                # forward 前向传播
                pred = unet(patch_input)#[1,4,d,h,w]

                # 计算指标
                pred=torch.argmax(pred,dim=1,keepdim=True)#[1,1,d,h,w]
                # patch_gt=utils.minmax(patch_gt).unsqueeze(0).unsqueeze(0)#[1,1,d,h,w]
                for c in range(patch_gt.shape[2]):
                    if (patch_gt[:,:,c,:,:].max()!=0)&(pred[:,:,c,:,:].max()!=0):

                        # psnr, ssim, _ = utils.compute_measure(patch_gt[:,:,c,:,:], pred[:,:,c,:,:],
                        #                                       data_range=patch_gt[:,:,c,:,:].max() - patch_gt[:,:,c,:,:].min())
                        # img1 = utils.minmax(patch_input[:,:,c,:,:])
                        # img2 = utils.minmax(pred[:,:,c,:,:])
                        # img3 = utils.minmax(patch_gt[:,:,c,:,:])
                        # save_path = f"{test_epoch_root}/test_result_{index}.png"
                        # img_show = torch.cat([img1, img2, img3], dim=-1)
                        # img_show = utils.trans_to_display_window(img_show)
                        # cv2.imwrite(save_path, img_show)

                        metrics= utils.compute_measure9(patch_gt[:,:,c,:,:],pred[:,:,c,:,:])
                        # psnr_list.append(psnr)
                        # ssim_list.append(ssim)
                        for i in range(9):
                            accuracy_list[i].append(metrics[i])  # Accuracy1 到 Accuracy9
                            f1_list[i].append(metrics[9 + i])   # F1_1 到 F1_9


            else:
                pred=patch_gt.clone()

            if (testimg_total%20==0)&(patch_gt.max()!=0):
                # 保存测试图像
                save_path = f"{test_epoch_root}/test_result_{index}.nii.gz"
                patch_input = patch_input.squeeze()
                patch_gt = patch_gt.squeeze()
                # pred = utils.minmax(pred)
                # patch_show_1 = torch.cat([patch_input, pred, patch_gt], dim=1)  # [D,H,W]
                patch_show_np = pred.cpu().numpy()
                patch_show_2 = sitk.GetImageFromArray(patch_show_np)
                sitk.WriteImage(patch_show_2, save_path)


    # dice, iou, hausdorff, volume_similarity, false_negative, false_positive

    accuracy_means = [np.nanmean(acc) for acc in accuracy_list]
    f1_means = [np.nanmean(f1) for f1 in f1_list]
    # macro_p_mean = np.mean(macro_p_list)
    # macro_r_mean = np.mean(macro_r_list)
    # macro_f1_mean = np.mean(macro_f1_list)
    # micro_p_mean = np.mean(micro_p_list)
    # micro_r_mean = np.mean(micro_r_list)
    # micro_f1_mean = np.mean(micro_f1_list)

    mean_metrics = accuracy_means + f1_means

    return mean_metrics




    #print(f"loss:{loss_mean} psnr:{psnr_mean}-{psnr_std}   ssim:{ssim_mean}-{ssim_std}")
def get_3mask_3d_result(cfg,unet_1mask,unet_3mask):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device -> {device}")

    test_dataloader = DataLoader(dataset.GetTestSet(data_root='E:\zjy\zjy\get_result3d_3mask', img_size=cfg.img_size),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_iterator = tqdm(test_dataloader)

    # 保存每一轮的测试图片
    test_epoch_root = r'E:\zjy\zjy\get_result3d_3mask\test_result'  # /epoch{epoch}"
    if not os.path.exists(test_epoch_root):
        os.makedirs(test_epoch_root)
    with torch.no_grad():
        for index, data in enumerate(test_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)  # [z,128,128]
            gt_img = (data[1].squeeze()).to(device)  # [?,128,128]

            input_name=data[2]
            print(input_name)
            patch_input = input_img.unsqueeze(0).unsqueeze(0)  # [1,1,?,128,128]
            patch_gt = gt_img.unsqueeze(0).unsqueeze(0)

            if patch_gt.max() != 0:
                # 划分面部区域
                input_mask = unet_1mask(patch_input)
                input_mask[input_mask > 0.9] = 1
                input_mask[input_mask <= 0.9] = 0
                patch_input = input_mask * patch_input
                # forward 前向传播
                pred = unet_3mask(patch_input)
            else:
                pred=patch_gt.clone()
            #保存结果
            save_path = f'{test_epoch_root}/test_result_{input_name[0]}'
            pred = utils.minmax(torch.argmax(pred, 1, keepdim=True))* 255
            pred = pred.type(torch.int16)
            # pred = pred.squeeze()
            # patch_show_np = pred.cpu().numpy()
            patch_show_2 = sitk.GetImageFromArray(pred)
            sitk.WriteImage(patch_show_2, save_path)

def get_3mask_2d_result(cfg,unet_1mask,unet_3mask):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device -> {device}")

    test_dataloader = DataLoader(dataset.GetTestSet(data_root='E:\zjy\zjy\get_result2d_3mask', img_size=cfg.img_size),
                                 batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_iterator = tqdm(test_dataloader)

    data_root = 'E:\zjy\zjy\get_result2d_3mask'
    input_root = 'E:\zjy\zjy\____input_all\____input_interest'
    # 保存每一轮的测试图片
    test_epoch_root = r'E:\zjy\zjy\get_result2d_3mask\test_result'  # /epoch{epoch}"
    if not os.path.exists(test_epoch_root):
        os.makedirs(test_epoch_root)


    with torch.no_grad():
        for index, data in enumerate(test_iterator):
            # load imgs
            input_img = (data[0].squeeze()).to(device)
            input_name=re.findall(r'\d+',data[2][0])
            # input_origin_image_path=f"{input_root}\_{input_name[0]}_interest.nii.gz"#_1001_interest.nii.gz
            print(input_name[0])
            input_img = input_img.unsqueeze(0)
            # gt_img = gt_img.unsqueeze(0)
            slice_input = input_img.unsqueeze(0)
            input_mask = unet_1mask(slice_input)
            input_mask[input_mask > 0.9] = 1
            input_mask[input_mask <= 0.9] = 0
            slice_input = input_mask * slice_input

            # forward 前向传播
            pred = unet_3mask(slice_input)

            #保存结果
            save_path = f'{test_epoch_root}/_test_result_{input_name[0]}_slice_{input_name[2]}.dcm'
            pred = utils.minmax(torch.argmax(pred,1,keepdim=True))*255
            pred=pred.type(torch.int16)
            # patch_show_np = pred.cpu().numpy()
            patch_show_1 = sitk.GetImageFromArray(pred)
            patch_show_2=sitk.Cast(patch_show_1,sitk.sitkInt16)
            # sitk.WriteImage(patch_show_2, save_path)
            # slice_image = sitk.GetImageFromArray(patch_show_np)
            writer = sitk.ImageFileWriter()
            writer.SetFileName(save_path)

            # input_origin_image=sitk.ReadImage(input_origin_image_path)
            # patch_show_2.SetSpacing(input_origin_image.GetSpacing())
            # patch_show_2.SetOrigin(input_origin_image.GetOrigin())
            # patch_show_2.SetDirection(input_origin_image.GetDirection())
            # patch_show_2.SetMetaData("0028|0008",  str(sitk.GetArrayFromImage(input_origin_image).shape[0]))
            # patch_show_2.SetMetaData("0020|0013", str(index+1))

            # 获取图像的基础信息（如 spacing 和方向）
            # 写入 DICOM 文件
            writer.Execute(patch_show_2)
            print(f"Saved {save_path}")




if __name__ == '__main__':
    #1m 3d
    # cfg = config1.get_config_1mask_3d()
    # print(cfg)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_1mask= Unet3d.UNet().to(device)
    # weight1 = torch.load(cfg.weight_path)
    # unet_1mask.load_state_dict(weight1["unet"])
    # # test_1label_3d(cfg,unet_1mask)
    # #3m 3d
    # cfg2= config1.get_config_3mask_3d()
    # print(cfg2)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_3mask= Unet3d_3.UNet().to(device)
    # weight1 = torch.load(cfg2.weight_path)
    # unet_3mask.load_state_dict(weight1["unet"])
    # # test_1label_3d(cfg, unet_3mask)
    #
    # get_3mask_3d_result(cfg2,unet_1mask,unet_3mask)

    # # 1m 2d
    # cfg = config1.get_config_1mask_2d()
    # print(cfg)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_1mask = UNet2d.UNet().to(device)
    # weight1 = torch.load(cfg.weight_path)
    # unet_1mask.load_state_dict(weight1["unet"])
    # # test_1label_3d(cfg,unet_1mask)
    # # 3m 2d
    # cfg2 = config1.get_config_3mask_2d()
    # print(cfg2)
    # # load model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet_3mask = UNet2d_3.UNet().to(device)
    # weight1 = torch.load(cfg2.weight_path)
    # unet_3mask.load_state_dict(weight1["unet"])
    # # test_1label_3d(cfg, unet_3mask)
    #
    # get_3mask_2d_result(cfg2, unet_1mask, unet_3mask)

    cfg3 = config1.get_config_1mask_3d()
    cfg4 = config1.get_config_3mask_3d()
    cfg5 = config1.get_config_9mask_3d()
    print(cfg3)
    print(cfg4)
    print(cfg5)
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_1mask_3d = Unet3d.UNet().to(device)
    unet_3mask_3d = Unet3d_3.UNet().to(device)
    unet_9mask_3d = Unet3d_9.UNet().to(device)
    # load weight
    weight3 = torch.load(cfg3.weight_path)
    unet_1mask_3d.load_state_dict(weight3["unet"])
    weight4 = torch.load(cfg4.weight_path)
    unet_3mask_3d.load_state_dict(weight4["unet"])
    unet_9mask_3d.train()
    test_9label_3d(cfg5, unet_1mask_3d, unet_3mask_3d, unet_9mask_3d)