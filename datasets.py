import os
import glob
import random

import numpy as np
import torch
from img_preprocess import imread, img_trans, modcrop, img_normal
import torch.nn.functional as F
from option import opt
from torch.utils import data
from torchvision import transforms
from Guassian import Guassian_downsample


class Train_Vimeo(data.Dataset):
    def __init__(self):
        self.train_data = open(opt.train_Vimeo, 'rt').read().splitlines()
        self.scale = opt.scale
        self.num_frames = opt.num_frames
        self.trans_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = sorted(glob.glob(os.path.join('./data81/sequences', self.train_data[idx], '*.png')))

        HR_all = []

        for i in range(self.num_frames):
            # HR
            img = imread(img_path[i])
            HR_all.append(img)

        h, w, c = HR_all[0].shape
        trans_idx = random.randint(-1, 4)
        h_start = random.randint(0, h - opt.crop_size_HR)
        w_start = random.randint(0, w - opt.crop_size_HR)

        HR_all = [modcrop(HR, h_start, w_start, opt.crop_size_HR) for HR in HR_all]
        HR_all = img_trans(HR_all, trans_idx)
        HR_all = [self.trans_tensor(HR) for HR in HR_all]
        HR_all = torch.stack(HR_all, dim=1)

        LR = Guassian_downsample(HR_all, self.scale)

        return LR, HR_all

    def __len__(self):
        return len(self.train_data)


