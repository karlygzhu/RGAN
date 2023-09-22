import torch
import math
from torch import nn
from torch.nn import Conv2d, Conv3d, Sequential, ReLU
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, act):
        super(SpatialAttention, self).__init__()
        self.act = act
        self.conv1 = nn.Conv2d(2, 1, (1, 1), bias=False)
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
        att = self.act(self.conv1(att))
        att = self.sigmoid(att)
        outputs = torch.mul(att, inputs)

        return outputs


class ChannelAttention(nn.Module):
    def __init__(self, nf, act=nn.LeakyReLU(0.1, inplace=True), ratio=16):
        super(ChannelAttention, self).__init__()
        self.nf = nf
        self.ratio = ratio
        self.act = act

        self.conv_c1 = nn.Conv2d(self.nf, self.nf // self.ratio, (1, 1), bias=False)
        self.conv_c2 = nn.Conv2d(self.nf // self.ratio, self.nf, (1, 1), bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        '''
        :param inputs: Modules that need to apply the attention mechanism. Size (B, C, H, W）
        :return: Results of fusion after application of attentional mechanism. Size (B, C, H, W）
        '''
        ca = torch.nn.functional.adaptive_avg_pool2d(inputs, (1,1))
        ca = self.conv_c1(ca)                   # ca channel attention
        ca = self.act(ca)
        ca = self.conv_c2(ca)

        channel_fea = self.sigmoid(ca)
        outputs = torch.mul(channel_fea, inputs)

        return outputs


class Res_Block2D(nn.Module):
    '''Residual block w/o BN
        ---Conv-ReLU-Conv-+-
         |________________|
    '''
    def __init__(self, nf, act=ReLU()):
        super(Res_Block2D, self).__init__()
        self.nf = nf
        self.act = act
        self.conv1 = Conv2d(self.nf, self.nf, (3, 3), stride=(1,), padding=(1,))
        self.conv1.apply(initialize_weights)
        self.conv2 = Conv2d(self.nf, self.nf, (3, 3), stride=(1,), padding=(1,))
        self.conv2.apply(initialize_weights)

    def forward(self, inputs):
        identity = inputs
        x = self.act(self.conv1(inputs))
        x = self.conv2(x)
        return identity + x



class Res_Block2D_Add(nn.Module):
    '''Residual block w/o BN, Adjust the dimension and carry out residual learning
     ---Conv---Conv-ReLU-Conv-+-
             |________________|
    '''
    def __init__(self, nf, hf, n_b, act=ReLU()):
        super(Res_Block2D_Add, self).__init__()
        self.nf = nf
        self.hf = hf
        self.n_b = n_b
        self.act = act
        self.conv = Conv2d(self.nf, self.hf, (3, 3), stride=(1,), padding=(1,))
        self.res = Sequential(*[Res_Block2D(self.hf) for _ in range(self.n_b)])

    def forward(self, inputs):
        out = self.act(self.conv(inputs))
        out = self.res(out)
        return out


class PFRB(nn.Module):
    '''
    Progressive Fusion Residual Block
    Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations, ICCV 2019
    '''

    def __init__(self, num_fea=64, num_channel=3):
        super(PFRB, self).__init__()
        self.nf = num_fea
        self.nc = num_channel
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv0 = nn.Sequential(*[nn.Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)) for _ in range(num_channel)])
        self.conv1 = nn.Conv2d(self.nf * num_channel, self.nf, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Sequential(*[nn.Conv2d(self.nf * 2, self.nf, (3, 3), stride=(1, 1), padding=(1, 1)) for _ in range(num_channel)])

    def forward(self, x):
        x1 = [self.lrelu(self.conv0[i](x[i])) for i in range(self.nc)]
        merge = torch.cat(x1, 1)
        base = self.lrelu(self.conv1(merge))
        x2 = [torch.cat([base, i], 1) for i in x1]
        x2 = [self.lrelu(self.conv2[i](x2[i])) for i in range(self.nc)]

        return [torch.add(x[i], x2[i]) for i in range(self.nc)]


class UPSAMPLE(nn.Module):
    def __init__(self, nf, act):
        super(UPSAMPLE, self).__init__()
        self.nf = nf
        self.act = act
        self.upconv1 = Conv2d(self.nf, self.nf * 4, (3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2 = Conv2d(self.nf, self.nf * 4, (3, 3), stride=(1, 1), padding=(1, 1))
        self.last = Conv2d(self.nf, 3, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        x = self.act(self.upconv1(inputs))
        x = F.pixel_shuffle(x, 2)
        x = self.act(self.upconv2(x))
        x = F.pixel_shuffle(x, 2)
        x = self.act(self.last(x))
        return x


class NonLocal(nn.Module):
    def __init__(self, nf, act=nn.ReLU()):
        super(NonLocal, self).__init__()
        self.nf = nf
        self.act = act
        self.g = Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.theta = Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.phi = Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.W = Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        g_x = self.act(self.g(inputs)).reshape(B, C, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.act(self.theta(inputs)).reshape(B, C, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.act(self.phi(inputs)).reshape(B, C, -1)
        f = torch.matmul(theta_x, phi_x)
        f_soft_max = F.softmax(f, dim=-1)
        f = torch.matmul(f_soft_max, g_x)
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        f = self.act(self.W(f))
        out = f + inputs

        return out


class SelfAttention(nn.Module):
    def __init__(self, nf, act=nn.ReLU()):
        super(SelfAttention, self).__init__()
        self.nf = nf
        self.act = act

        self.q = Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.k = Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.v = Conv2d(self.nf, self.nf, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        q = self.q(inputs)
        k = self.k(inputs)
        v = self.v(inputs)
        q = F.unfold(q, kernel_size=(4, 4), padding=0, stride=4)
        k = F.unfold(k, kernel_size=(4, 4), padding=0, stride=4)
        v = F.unfold(v, kernel_size=(4, 4), padding=0, stride=4)

        assert q.size(2) == q.size(2)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.size(2))
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, v)
        out = F.fold(out, output_size=(H, W), kernel_size=(4, 4), padding=0, stride=4)

        return out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    print(x.size()[-2:])
    print(flow.size()[1:3])
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


def initialize_weights(m):
    if isinstance(m, Conv2d):
        nn.init.xavier_normal_(m.weight, gain=1.)
        nn.init.zeros_(m.bias)
    if isinstance(m, Conv3d):
        nn.init.xavier_normal_(m.weight, gain=1.)
        nn.init.zeros_(m.bias)





