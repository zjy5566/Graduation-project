import numpy
import torch
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
from math import exp
from torch.autograd import Variable
import matplotlib.pyplot as plt
from math import nan
# denormalize [-1000, 1000]HU to [0, 255] for calculating PSNR, SSIM and RMSE
def trans_to_calculate_window_1mask(img, MIN_B=-1024, MAX_B=3071, cut_min=-1000, cut_max=1000):
    # img = img * (MAX_B - MIN_B) + MIN_B
    # img[img < cut_min] = cut_min
    # img[img > cut_max] = cut_max
    # img = 255 * (img - cut_min) / (cut_max - cut_min)
    img[img < 0.9] = 0
    img[img > 0.9] = 1
    return img
def trans_to_calculate_window_3mask(img,num):
    img[img<num-0.01]=0
    img[img>num+0.01] = 0
    img[(img>=(num-0.01))&(img<=num+0.01)]=1
    # img[img!=num]=0
    # img[img==num]=1

    return img

def minmax(img):
    img2 = img.squeeze().detach().cpu().numpy()
    if np.max(img2) - np.min(img2) != 0:
        img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    return torch.from_numpy(img2)
# denormalize to [-100, 200]HU for display
def trans_to_display_window(img, MIN_B=-1024, MAX_B=3071, cut_min=-100, cut_max=200):  # [1, 1, 256, 256] -> [256, 256]
    img = img.squeeze().detach().cpu().numpy()
    img=255*img
    # img = img * (MAX_B - MIN_B) + MIN_B
    # img[img < cut_min] = cut_min
    # img[img > cut_max] = cut_max
    # img = 255 * (img - cut_min) / (cut_max - cut_min)
    return img


# 控制梯度变化
def set_grad(networks, grad_mode=False):  # 默认设置：控制传入网络 不更新梯度
    # 如果传入的网络不是以list组织格式在一起，就将其转为list格式
    if not isinstance(networks, list):
        networks = [networks]
    # 设置network的梯度更新状态
    for network in networks:
        if network is not None:
            for paramter in network.parameters():
                paramter.requires_grad = grad_mode



def compute_measure(y, pred, data_range):
    psnr = compute_PSNR(pred, y, data_range)
    ssim = compute_SSIM(pred, y, data_range)
    rmse = compute_RMSE(pred, y)
    return psnr, ssim, rmse


def compute_MSE(img1, img2):
    return ((img1/1.0 - img2/1.0) ** 2).mean()


def compute_RMSE(img1, img2):
    img1 = img1 * 2000 / 255 - 1000
    img2 = img2 * 2000 / 255 - 1000
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.shape) == 2:
        h, w = img1.shape
        if type(img1) == torch.Tensor:
            img1 = img1.view(1, 1, h, w)
            img2 = img2.view(1, 1, h, w)
        else:
            img1 = torch.from_numpy(img1[np.newaxis, np.newaxis, :, :])
            img2 = torch.from_numpy(img2[np.newaxis, np.newaxis, :, :])
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    # if size_average:
    #     return ssim_map.mean().item()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1).item()
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def compute_dice(gt, pred):
    tp = np.sum(pred * gt)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    return 2 * tp / (2 * tp + fp + fn)

def compute_iou(gt, pred):
    intersection = np.sum(pred * gt)   #交集
    union = np.sum(pred) + np.sum(gt) - intersection  #并集
    return intersection / union

def compute_hausdorff(gt, pred):
    if gt.sum()==0:
        return 0
    elif pred.sum()==0:
        return 0
    else:
        img1=sitk.Cast(sitk.GetImageFromArray(gt),sitk.sitkUInt64)
        img2=sitk.Cast(sitk.GetImageFromArray(pred),sitk.sitkUInt64)
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(img1>0.5, img2>0.5)
        return hausdorff_distance_filter.GetHausdorffDistance()

def compute_volume_similarity(gt, pred):
    """计算体积相关误差"""
    v_pred = np.sum(pred)
    v_gt = np.sum(gt)
    return abs(v_pred - v_gt) / v_gt

def compute_measure2(label,seg):
    label=label.squeeze().unsqueeze(2)
    seg=seg.squeeze().unsqueeze(2)
    gt=label.cpu().numpy()
    pred=seg.cpu().numpy()

    dice=compute_dice(gt,pred)
    iou=compute_iou(gt,pred)
    hausdorff=compute_hausdorff(gt,pred)
    volume_similarity=compute_volume_similarity(gt,pred)

    return dice, iou, hausdorff,volume_similarity

def get_TP_TN_FN_FP(gt, pred):
    predict=pred.copy().astype(bool)
    label=gt.copy().astype(bool)
    TP=np.sum((predict==1)&(label==1))
    TN=np.sum((predict==0)&(label==0))
    FN=np.sum((predict==0)&(label==1))
    FP=np.sum((predict==1)&(label==0))

    return TP,TN,FN,FP

def Accuracy(gt, pred):
    TP, TN, FN, FP=get_TP_TN_FN_FP(gt, pred)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    return accuracy
def Precision(gt, pred):
    TP, TN, FN, FP = get_TP_TN_FN_FP(gt, pred)
    if (TP + FP)!=0:
        precision = TP / (TP + FP)
        return precision
    else:
        return 0

def Recall(gt, pred):
    TP, TN, FN, FP = get_TP_TN_FN_FP(gt, pred)
    if (TP + FN)!=0:
        recall = TP / (TP + FN)
        return recall
    else:
        return 0
def F1_score(gt, pred):
    precision=Precision(gt, pred)
    recall=Recall(gt, pred)
    if (precision*recall) != 0:
        F1 = (2*precision*recall) / (precision + recall)
        return F1
    else:
        return 0

def Macro(gt, pred):        #宏召回率、宏准确率、宏F1
    gt1 =gt.copy()
    pred1 =pred.copy()
    gt2 =gt.copy()
    pred2 =pred.copy()
    gt3 =gt.copy()
    pred3 =pred.copy()
    trans_to_calculate_window_3mask(gt1, 1)  # 肌肉
    trans_to_calculate_window_3mask(pred1, 1)
    trans_to_calculate_window_3mask(gt2, 2)
    trans_to_calculate_window_3mask(pred2, 2)
    trans_to_calculate_window_3mask(gt3, 3)
    trans_to_calculate_window_3mask(pred3, 3)  # 脂肪

    P1 = Precision(gt1,pred1)
    P2 = Precision(gt2, pred2)
    P3 = Precision(gt3, pred3)
    R1 = Recall(gt1,pred1)
    R2 = Recall(gt2,pred2)
    R3 = Recall(gt3,pred3)

    if (gt1.sum()!=0)&(gt2.sum()!=0)&(gt3.sum()!=0):
        Macro_P = ( P1+P2 + P3) / 3
        Macro_R = (R1+R2 + R3) / 3
        Macro_F1 = 2 * Macro_R * Macro_P / (Macro_R + Macro_P)
        return Macro_P, Macro_R, Macro_F1
    else:
        return 0,0,0

def Micro(gt, pred):
    gt1 = gt.copy()
    pred1 = pred.copy()
    gt2 = gt.copy()
    pred2 = pred.copy()
    gt3 = gt.copy()
    pred3 = pred.copy()
    trans_to_calculate_window_3mask(gt1, 1)  # 肌肉
    trans_to_calculate_window_3mask(pred1, 1)
    trans_to_calculate_window_3mask(gt2, 2)
    trans_to_calculate_window_3mask(pred2, 2)
    trans_to_calculate_window_3mask(gt3, 3)
    trans_to_calculate_window_3mask(pred3, 3)  # 脂肪

    TP1, _, FN1, FP1=get_TP_TN_FN_FP(gt1,pred1)
    TP2, _, FN2, FP2=get_TP_TN_FN_FP(gt2,pred2)
    TP3, _, FN3, FP3=get_TP_TN_FN_FP(gt3,pred3)

    TP=(TP1+TP2+TP3)/3
    FN=(FN1+FN2+FN3)/3
    FP=(FP1+FP2+FP3)/3

    if (gt1.sum()!=0)&(gt2.sum()!=0)&(gt3.sum()!=0):
        Micro_P = TP / (TP + FP)
        Micro_R=TP / (TP + FN)
        Micro_F1=2*Micro_R*Micro_P/(Micro_R+Micro_P)
        return Micro_P,Micro_R,Micro_F1
    else:
        return 0,0,0

def compute_measure3_3mask(label,seg):
    label_ = label.squeeze().unsqueeze(2)
    seg_= seg.squeeze().unsqueeze(2)
    gt = label_.cpu().numpy()
    pred = seg_.cpu().numpy()
    gt1= gt.copy()
    pred1 = pred.copy()
    gt2 =gt.copy()
    pred2 =pred.copy()
    gt3 =gt.copy()
    pred3=pred.copy()
    gt4 =gt.copy()
    pred4 =pred.copy()
    gt5 =gt.copy()
    pred5 =pred.copy()

    trans_to_calculate_window_3mask(gt1, 1)#肌肉
    trans_to_calculate_window_3mask(pred1, 1)
    trans_to_calculate_window_3mask(gt2, 2)
    trans_to_calculate_window_3mask(pred2, 2)
    trans_to_calculate_window_3mask(gt3, 3)
    trans_to_calculate_window_3mask(pred3, 3)#脂肪

    A1=Precision(gt1,pred1)
    A2= Precision(gt2, pred2)
    A3=Precision(gt3, pred3)
    C1=Recall(gt1,pred1)
    C2=Recall(gt2,pred2)
    C3=Recall(gt3,pred3)

    F1_1=F1_score(gt1,pred1)
    F1_2=F1_score(gt2,pred2)
    F1_3=F1_score(gt3,pred3)

    Macro_P, Macro_R, Macro_F1=Macro(gt4,pred4)
    Micro_P, Micro_R, Micro_F1=Micro(gt5,pred5)

    return A1,A2,A3,F1_1,F1_2,F1_3,Macro_P, Macro_R, Macro_F1,Micro_P, Micro_R, Micro_F1


def compute_measure9(label, seg):
    label_ = label.squeeze().unsqueeze(2).cpu().numpy()
    seg_ = seg.squeeze().unsqueeze(2).cpu().numpy()

    P_list = []
    F1_list = []
    for i in range(1, 10):  # 从1到9，处理9个标签
        gt = trans_to_calculate_window_3mask(label_.copy(), i)
        pred = trans_to_calculate_window_3mask(seg_.copy(), i)
        P_list.append(Precision(gt, pred))
        F1_list.append(F1_score(gt, pred))

    return P_list + F1_list

if __name__ == '__main__':
    # gt=np.array([0,1,0,0,1,1,0,1])
    # pred=np.array([0,0,0,0,1,1,1,1])
    # print(get_TP_TN_FN_FP(gt,pred))
    a=torch.zeros(3,3)
    trans_to_calculate_window_3mask(a,0)
    print(a)

