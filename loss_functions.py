import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps=eps
        return

    def forward(self, inp, target):
        diff = torch.abs(inp - target) ** 2 + self.eps ** 2
        out = torch.sqrt(diff)
        loss = torch.mean(out)

        return loss


def loss_func_vae(recon_x, x, mu, logvar):
    x = x.detach()
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)   # 重构的差异
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())   # 假设概率分布和真实概率分布之间的差异
    return BCE + KLD, BCE, KLD


def PSNR(labels, outputs):
    img1 = labels.cpu().detach().numpy()
    img2 = outputs.cpu().detach().numpy()
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr


def SSIM(img1, img2):
    '''
        calculate SSIM
        img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    t, c, h, w = img1.shape
    img1 = img1 * 255
    img2 = img2 * 255
    img1 = torch.reshape(img1, ((-1, h, w))).permute(1, 2, 0).cpu().numpy()
    img2 = torch.reshape(img2, ((-1, h, w))).permute(1, 2, 0).cpu().numpy()

    if img1.ndim == 2:
        return cal_ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(cal_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return cal_ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def cal_ssim(target, prediction):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
