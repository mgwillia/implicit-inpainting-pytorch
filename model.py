import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        print(self.kernel_size, self.stride, self.dilation, pad_h, pad_w)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class GenConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, rate=1, activation=nn.ELU()):
        super(GenConv, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2dSame(in_channels, out_channels, ksize, stride, dilation=rate)


    def forward(self, x):
        x = self.conv(x)
        if self.activation is None:
            return x
        split_size = int(x.shape[1] / 2)
        x, y = torch.split(x, split_size, dim=1)
        x = self.activation(x)
        y = self.sigmoid(y)
        x = x * y
        return x


class GenDeconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenDeconv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = GenConv(in_channels, out_channels, 3, 1)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class CoarseNet(nn.Module):
    def __init__(self, cnum):
        super(CoarseNet, self).__init__()
        self.conv1 = GenConv(4, cnum, 5, 1)
        self.conv2 = GenConv(cnum//2, 2*cnum, 3, 2)
        self.conv3 = GenConv(cnum, 2*cnum, 3, 1)
        self.conv4 = GenConv(cnum, 4*cnum, 3, 2)
        self.conv5 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv6 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv7 = GenConv(2*cnum, 4*cnum, 3, rate=2)
        self.conv8 = GenConv(2*cnum, 4*cnum, 3, rate=4)
        self.conv9 = GenConv(2*cnum, 4*cnum, 3, rate=8)
        self.conv10 = GenConv(2*cnum, 4*cnum, 3, rate=16)
        self.conv11 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv12 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.conv13 = GenDeconv(2*cnum, 2*cnum)
        self.conv14 = GenConv(cnum, 2*cnum, 3, 1)
        self.conv15 = GenDeconv(cnum, cnum)
        self.conv16 = GenConv(cnum//2, cnum//2, 3, 1)
        self.conv17 = GenConv(cnum//4, 3, 3, 1, activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        return x

class MainBranch(nn.Module):
    def __init__(self, cnum):
        super(MainBranch, self).__init__()
        self.fineconv1 = GenConv(4, cnum, 5, 1)
        self.fineconv2 = GenConv(cnum // 2, cnum, 3, 2)
        self.fineconv3 = GenConv(cnum // 2, 2*cnum, 3, 1)
        self.fineconv4 = GenConv(cnum, 2*cnum, 3, 2)
        self.fineconv5 = GenConv(cnum, 4*cnum, 3, 1)
        self.fineconv6 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.fineconv7 = GenConv(2*cnum, 4*cnum, 3, rate=2)
        self.fineconv8 = GenConv(2*cnum, 4*cnum, 3, rate=4)
        self.fineconv9 = GenConv(2*cnum, 4*cnum, 3, rate=8)
        self.fineconv10 = GenConv(2*cnum, 4*cnum, 3, rate=16)

    def forward(self, x):
        x = self.fineconv1(x)
        x = self.fineconv2(x)
        x = self.fineconv3(x)
        x = self.fineconv4(x)
        x = self.fineconv5(x)
        x = self.fineconv6(x)
        x = self.fineconv7(x)
        x = self.fineconv8(x)
        x = self.fineconv9(x)
        x = self.fineconv10(x)
        return x


class AuxBranch(nn.Module):
    def __init__(self, cnum):
        super(AuxBranch, self).__init__()
        self.fineconv2_1 = GenConv(4, cnum, 5, 1)
        self.fineconv2_2 = GenConv(cnum // 2, cnum, 3, 2)
        self.fineconv2_3 = GenConv(cnum // 2, 2*cnum, 3, 1)
        self.fineconv2_4 = GenConv(cnum, 4*cnum, 3, 2)
        self.fineconv2_5 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.fineconv2_6 = GenConv(2*cnum, 4*cnum, 3, 1, activation=nn.ReLU())
        self.fineconv2_9 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.fineconv2_10 = GenConv(2*cnum, 4*cnum, 3, 1)


    def forward(self, x):
        x = self.fineconv2_1(x)
        x = self.fineconv2_2(x)
        x = self.fineconv2_3(x)
        x = self.fineconv2_4(x)
        x = self.fineconv2_5(x)
        x = self.fineconv2_6(x)
        x = self.fineconv2_9(x)
        x = self.fineconv2_10(x)
        return x

        
class OutBranch(nn.Module):
    def __init__(self, cnum):
        super(OutBranch, self).__init__()
        self.outconv1 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.outconv2 = GenConv(2*cnum, 4*cnum, 3, 1)
        self.outconv3 = GenDeconv(2*cnum, 2*cnum)
        self.outconv4 = GenConv(cnum, 2*cnum, 3, 1)
        self.outconv5 = GenDeconv(cnum, cnum)
        self.outconv6 = GenConv(cnum//2, cnum//2, 3, 1)
        self.outconv7 = GenConv(cnum//4, 3, 3, 1, activation=None)

    def forward(self, x):
        x = self.outconv1(x)
        x = self.outconv2(x)
        x = self.outconv3(x)
        x = self.outconv4(x)
        x = self.outconv5(x)
        x = self.outconv6(x)
        x = self.outconv7(x)
        return x


class RefineModel(nn.Module):
    def __init__(self):
        super(RefineModel, self).__init__()
        cnum = 48

        self.coarse_net = CoarseNet(cnum)
        self.main_branch = MainBranch(cnum)
        self.aux_branch = AuxBranch(cnum)
        self.out_branch = OutBranch(cnum)

    def forward(self, x, mask):
        # input
        xin = x

        # stage 1
        x = self.coarse_net(x)
        x = torch.tanh(x)
        x_stage1 = x

        print(f'debugging... xin shape: {xin.shape}, x shape: {x.shape}')
        # stage 2
        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)
        x.view(list(xin[:, 0:3, :, :].shape))

        # first branch
        xnow = x
        x = self.main_branch(xnow)
        x_hallu = x

        # second branch
        x = self.aux_branch(xnow)
        pm = x

        # out branch
        x = torch.cat([x_hallu, pm], dim=1)
        x = self.out_branch(x)
        x = torch.nn.tanh(x)
        x_stage2 = x

        return x_stage1, x_stage2
        

def refine_model():
    return RefineModel()
