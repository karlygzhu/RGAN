import torch
import math
import torch.nn.functional as F
import numpy as np
from option import opt
from torch import nn
from torch.nn import Conv2d, Conv3d, Sequential
from utils import Res_Block2D, UPSAMPLE, PFRB, Res_Block2D_Add


class FeaExt(nn.Module):
    def __init__(self, nf, nc, act):
        super(FeaExt, self).__init__()
        self.nf = nf
        self.nc = nc
        self.act = act
        self.ref = Sequential(Conv2d(3, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                              Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                              Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                              Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act)

        self.agg = Sequential(Conv2d(3, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                              Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                              Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                              Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act)

        self.att = Conv2d(self.nf, self.nc + 1, (3, 3), stride=(1, 1), padding=(1, 1))
        self.fus = Conv2d(self.nc * 3 + self.nf * 3, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        b, t, c, h, w = inputs.shape
        idx = t // 2
        ref = inputs[:, idx, :, :, :]
        ref = self.ref(ref)

        agg = self.agg(inputs.reshape(b * t, c, h, w))
        att = self.act(self.att(agg))
        x_soft = att[:, self.nc:self.nc + 1, :, :].reshape(b, t, -1, h, w)
        x_soft = F.softmax(x_soft, dim=1)
        x_results = att[:, :self.nc, :, :].reshape(b, t, -1, h, w)
        x_att = torch.mul(x_soft, x_results).reshape(b, -1, h, w)
        agg_out = self.fus(torch.cat((x_att, agg.reshape(b, t, -1, h, w).reshape(b, -1, h, w)), dim=1))

        return ref, agg_out


class SpatialAttention(nn.Module):
    def __init__(self, nf, act):
        super(SpatialAttention, self).__init__()
        self.nf = nf
        self.act = act
        self.conv = Conv2d(2, 1, (1, 1), bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        '''
        :param inputs: Modules that need to apply the attention mechanism. Size (B, C, H, W）
        :return: Results of fusion after application of attentional mechanism. Size (B, C, H, W）
        '''
        avg_out = torch.mean(inputs, dim=1, keepdim=True)
        max_out, _ = torch.max(inputs, dim=1, keepdim=True)
        att = torch.cat((avg_out, max_out), dim=1)
        att = self.act(self.conv(att))
        att = self.sigmoid(att)
        outputs = torch.mul(att, inputs)

        return outputs


class DCN(nn.Module):
    def __init__(self, nf, nc, hf, act):
        super(DCN, self).__init__()
        self.nf = nf
        self.nc = nc
        self.hf = hf
        self.act = act
        self.conv1 = Conv2d(self.nf, self.nf, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = Conv2d(self.nf, self.nc, (3, 3), stride=(1, 1), padding=(1, 1))
        self.att1 = SpatialAttention(self.nc, self.act)
        self.conv3 = Conv2d(self.nf + self.nc, self.nf + self.nc, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv4 = Conv2d(self.nf + self.nc, self.nc, (3, 3), stride=(1, 1), padding=(1, 1))
        self.att2 = SpatialAttention(self.nc, self.act)
        self.compress = Conv2d(self.nf + 2 * self.nc, self.hf, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        x1 = self.act(self.conv1(inputs))
        x1 = self.act(self.conv2(x1))
        x1 = self.att1(x1)
        x1 = torch.cat((inputs, x1), dim=1)
        x2 = self.act(self.conv3(x1))
        x2 = self.act(self.conv4(x2))
        x2 = self.att2(x2)
        x2 = torch.cat((x1, x2), dim=1)
        outputs = self.act(self.compress(x2))
        return outputs


class VSR(nn.Module):
    def __init__(self, nf=32, nc=16, hf=64, num_inputs=3, num_b1=3, num_b2=3):
        super(VSR, self).__init__()
        self.nf = nf
        self.nc = nc
        self.hf = hf
        self.num_inputs = num_inputs
        self.num_b1 = num_b1
        self.num_b2 = num_b2
        self.scale = opt.scale
        self.idx = self.num_inputs // 2
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.fea_ext = FeaExt(self.nf, self.nc, self.act)

        # post
        self.conv_post = Conv2d(2 * self.nf + 2 * self.hf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.ht_post = Conv2d(self.hf, self.hf, (1, 1), stride=(1, 1), padding=(0, 0))
        self.out_post = Conv2d(self.hf, hf, (1, 1), stride=(1, 1), padding=(0, 0))

        self.dual = DCN(self.nf, self.nc, self.hf, self.act)
        self.res = Sequential(*[Res_Block2D(self.hf) for _ in range(self.num_b1)])

        self.conv_pre = Conv2d(2 * self.nf + 2 * self.hf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.ht_pre = Conv2d(self.hf, self.hf, (1, 1), stride=(1, 1), padding=(0, 0))
        self.out_pre = Conv2d(self.hf, hf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.fus1 = Res_Block2D_Add(self.nf + self.hf * 2, self.hf, self.num_b1, self.act)
        self.fus2 = Conv2d(self.hf, 48, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        B, C, T, H, W = inputs.shape

        ht_post = torch.zeros((B, self.hf, H, W), dtype=torch.float, device=inputs.device)
        out_post = torch.zeros((B, self.hf, H, W), dtype=torch.float, device=inputs.device)
        out_post_all = []

        for i in range(T - 1, -1, -1):
            in_group = generate_group(inputs, i, self.num_inputs, T)
            ref, agg = self.fea_ext(in_group.permute(0, 2, 1, 3, 4))
            fus_post = self.act(self.conv_post(torch.cat((ref, agg, ht_post, out_post), dim=1)))
            fus_post = self.dual(fus_post)
            fus_post = self.res(fus_post)
            ht_post = self.act(self.ht_post(fus_post))
            out_post = self.act(self.out_post(fus_post))
            out_post_all.append(out_post)

        out_post_all = out_post_all[::-1]
        ht_pre = torch.zeros((B, self.hf, H, W), dtype=torch.float, device=inputs.device)
        out_pre = torch.zeros((B, self.hf, H, W), dtype=torch.float, device=inputs.device)
        out_all = []

        for i in range(T):
            img_ref = inputs[:, :, i, :, :]
            img_ref_bic = F.interpolate(img_ref, scale_factor=self.scale, mode='bicubic', align_corners=False)
            in_group = generate_group(inputs, i, self.num_inputs, T)
            ref, agg = self.fea_ext(in_group.permute(0, 2, 1, 3, 4))
            fus_pre = self.act(self.conv_pre(torch.cat((ref, agg, ht_pre, out_pre), dim=1)))
            fus_pre = self.dual(fus_pre)
            fus_pre = self.res(fus_pre)
            ht_pre = self.act(self.ht_pre(fus_pre))
            out_pre = self.act(self.out_pre(fus_pre))
            out = torch.cat((out_post_all[i], out_pre, ref), dim=1)
            out = self.fus1(out)
            out = self.fus2(out)
            out = F.pixel_shuffle(out, self.scale) + img_ref_bic
            out_all.append(out)

        outputs = torch.stack(out_all, dim=2)
        return outputs


def generate_group(inputs, idx, num_inputs=2, t=7):
    index = np.array([idx - num_inputs // 2 + i for i in range(num_inputs)])
    index = np.clip(index, 0, t - 1).tolist()
    outputs = inputs[:, :, index]
    return outputs


def pixel_unshuffle(inputs, scale):
    B, C, H, W = inputs.shape
    x = inputs.reshape(B, C, H // scale, scale, W // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C * scale ** 2, H // scale, W // scale)

    return x


