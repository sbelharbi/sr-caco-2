import sys
from os.path import join, dirname, abspath
import math
from abc import ABC
from math import prod
from typing import Tuple
from collections import OrderedDict

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat

# Credit: https://github.com/Paper99/SRFBN_CVPR19
# Paper: https://arxiv.org/pdf/1903.09814.pdf


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['SRFBN']

# Paper: "Feedback Network for Image Super-Resolution",
# CVPR, 2019, Zhen Li, Jinglei Yang, Zheng Liu, Xiaomin Yang, Gwanggil Jeon,
# Wei Wu.



################
# Basic blocks
################

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError(
            '[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError(
            '[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            '[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                '[ERROR] %s.sequential() does not '
                'support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1,
              bias=True, valid_padding=True, padding=0, act_type='relu',
              norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode ' \
                                     'in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False

################
# Advanced blocks
################

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channle, mid_channel, kernel_size,
                 stride=1, valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type='bn', act_type='relu', mode='CNA',
                 res_scale=1):
        super(ResBlock, self).__init__()
        conv0 = ConvBlock(in_channel, mid_channel, kernel_size, stride,
                          dilation, bias, valid_padding, padding, act_type,
                          norm_type, pad_type, mode)
        act_type = None
        norm_type = None
        conv1 = ConvBlock(mid_channel, out_channle, kernel_size, stride,
                          dilation, bias, valid_padding, padding, act_type,
                          norm_type, pad_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res

class UpprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 valid_padding=False, padding=0, bias=True, pad_type='zero',
                 norm_type=None, act_type='prelu'):
        super(UpprojBlock, self).__init__()

        self.deconv_1 = DeconvBlock(in_channel, out_channel, kernel_size,
                                    stride=stride, padding=padding,
                                    norm_type=norm_type, act_type=act_type)

        self.conv_1 = ConvBlock(out_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                valid_padding=valid_padding,
                                norm_type=norm_type, act_type=act_type)

        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size,
                                    stride=stride, padding=padding,
                                    norm_type=norm_type, act_type=act_type
                                    )

    def forward(self, x):
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_1(H_0_t)
        H_1_t = self.deconv_2(L_0_t-x)

        return H_0_t + H_1_t

class D_UpprojBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 valid_padding=False, padding=0, bias=True,
                 pad_type='zero', norm_type=None, act_type='prelu'):
        super(D_UpprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1,
                                norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size,
                                    stride=stride, padding=padding,
                                    norm_type=norm_type, act_type=act_type)
        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                valid_padding=valid_padding,
                                norm_type=norm_type, act_type=act_type)
        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size,
                                    stride=stride, padding=padding,
                                    norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_2(H_0_t)
        H_1_t = self.deconv_2(L_0_t - x)

        return H_1_t + H_0_t

class DownprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 valid_padding=True, padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='prelu',
                 mode='CNA', res_scale=1):
        super(DownprojBlock, self).__init__()

        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                valid_padding=valid_padding,
                                norm_type=norm_type, act_type=act_type)

        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size,
                                    stride=stride, padding=padding,
                                    norm_type=norm_type, act_type=act_type)

        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                valid_padding=valid_padding,
                                norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        L_0_t = self.conv_1(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_2(H_0_t - x)

        return L_0_t + L_1_t

class D_DownprojBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 valid_padding=False, padding=0, bias=True,
                 pad_type='zero', norm_type=None, act_type='prelu'):
        super(D_DownprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1,
                                norm_type=norm_type, act_type=act_type)

        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                valid_padding=valid_padding,
                                norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size,
                                    stride=stride, padding=padding,
                                    norm_type=norm_type, act_type=act_type)
        self.conv_3 = ConvBlock(out_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                valid_padding=valid_padding,
                                norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        L_0_t = self.conv_2(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_3(H_0_t - x)

        return L_1_t + L_0_t

class DensebackprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 bp_stages, stride=1, valid_padding=True,
                 padding=0, dilation=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='prelu',
                 mode='CNA', res_scale=1):
        super(DensebackprojBlock, self).__init__()

        # This is an example that I have to create nn.ModuleList() to append
        # a sequence of models instead of list()
        self.upproj = nn.ModuleList()
        self.downproj = nn.ModuleList()
        self.bp_stages = bp_stages
        self.upproj.append(UpprojBlock(in_channel, out_channel, kernel_size,
                                       stride=stride, valid_padding=False,
                                       padding=padding, norm_type=norm_type,
                                       act_type=act_type))

        for index in range(self.bp_stages - 1):
            if index < 1:
                self.upproj.append(UpprojBlock(out_channel, out_channel,
                                               kernel_size, stride=stride,
                                               valid_padding=False,
                                               padding=padding,
                                               norm_type=norm_type,
                                               act_type=act_type)
                                   )
            else:
                uc = ConvBlock(out_channel*(index+1), out_channel,
                               kernel_size=1, norm_type=norm_type,
                               act_type=act_type)
                u = UpprojBlock(out_channel, out_channel, kernel_size,
                                stride=stride, valid_padding=False,
                                padding=padding, norm_type=norm_type,
                                act_type=act_type)
                self.upproj.append(sequential(uc, u))

            if index < 1:
                self.downproj.append(DownprojBlock(out_channel, out_channel,
                                                   kernel_size, stride=stride,
                                                   valid_padding=False,
                                                   padding=padding,
                                                   norm_type=norm_type,
                                                   act_type=act_type)
                                     )
            else:
                dc = ConvBlock(out_channel*(index+1), out_channel,
                               kernel_size=1, norm_type=norm_type,
                               act_type=act_type)
                d = DownprojBlock(out_channel, out_channel, kernel_size,
                                  stride=stride, valid_padding=False,
                                  padding=padding, norm_type=norm_type,
                                  act_type=act_type)
                self.downproj.append(sequential(dc, d))

    def forward(self, x):
        low_features = []
        high_features = []

        H = self.upproj[0](x)
        high_features.append(H)

        for index in range(self.bp_stages - 1):
            if index < 1:
                L = self.downproj[index](H)
                low_features.append(L)
                H = self.upproj[index+1](L)
                high_features.append(H)
            else:
                H_concat = torch.cat(tuple(high_features), 1)
                L = self.downproj[index](H_concat)
                low_features.append(L)
                L_concat = torch.cat(tuple(low_features), 1)
                H = self.upproj[index+1](L_concat)
                high_features.append(H)

        output = torch.cat(tuple(high_features), 1)
        return output


class ResidualDenseBlock_8C(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image
     Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True,
                 pad_type='zero',norm_type=None, act_type='relu',
                 mode='CNA'):
        super(ResidualDenseBlock_8C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv5 = ConvBlock(nc+4*gc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv6 = ConvBlock(nc+5*gc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv7 = ConvBlock(nc+6*gc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        self.conv8 = ConvBlock(nc+7*gc, gc, kernel_size, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv9 = ConvBlock(nc+8*gc, nc, 1, stride, bias=bias,
                               pad_type=pad_type, norm_type=norm_type,
                               act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), 1)
        return output

################
# Upsampler
################
def UpsampleConvBlock(upscale_factor, in_channels, out_channels, kernel_size,
                      stride, valid_padding=True, padding=0, bias=True,
                      pad_type='zero', act_type='relu', norm_type=None,
                      mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = ConvBlock(in_channels, out_channels, kernel_size, stride,
                     bias=bias, valid_padding=valid_padding, padding=padding,
                     pad_type=pad_type, act_type=act_type, norm_type=norm_type)
    return sequential(upsample, conv)


def PixelShuffleBlock():
    pass


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1,
                bias=True, padding=0,
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode ' \
                                     'in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)


################
# helper funcs
################

def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding



class FeedbackBlock(nn.Module):
    def __init__(self,
                 num_features,
                 num_groups,
                 upscale_factor,
                 act_type,
                 norm_type
                 ):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2 * num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvBlock(num_features, num_features,
                                             kernel_size=kernel_size,
                                             stride=stride, padding=padding,
                                             act_type=act_type,
                                             norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size,
                                             stride=stride, padding=padding,
                                             act_type=act_type,
                                             norm_type=norm_type,
                                             valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features,
                              kernel_size=1, stride=1,
                              act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(
                    ConvBlock(num_features * (idx + 1), num_features,
                              kernel_size=1, stride=1,
                              act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups * num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def flush(self):
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features),
                             1)  # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx - 1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx - 1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]),
                           1)  # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class SRFBN(nn.Module):
    def __init__(self,
                 upscale: int = 2,
                 in_chans: int = 3,
                 num_features: int = 64,
                 num_steps: int = 4,
                 num_groups: int = 6,
                 act_type='prelu',
                 norm_type=None
                 ):
        super(SRFBN, self).__init__()

        upscale_factor = upscale
        in_channels = in_chans
        out_channels = in_chans

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6

        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7

        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8

        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        else:
            raise NotImplementedError(upscale_factor)

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4 * num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor,
                                   act_type, norm_type)

        # reconstruction block
        # uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor,
        # mode='bilinear')

        self.out = DeconvBlock(num_features, num_features,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        self.intermediate_outs = []  # holds intermediate outputs.

    def flush(self):
        self.intermediate_outs = []
        self.block.flush()

    def forward(self, x):
        self.flush()
        self._reset_state()

        # x = self.sub_mean(x)
        # uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)

        # comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x,
                                              scale_factor=self.upscale_factor,
                                              mode='bilinear',
                                              align_corners=False
                                              )

        x = self.conv_in(x)
        x = self.feat_in(x)

        for _ in range(self.num_steps):
            h = self.block(x)

            h = torch.add(inter_res, self.conv_out(self.out(h)))
            # h = self.add_mean(h)
            self.intermediate_outs.append(h)

        msg = f"{len(self.intermediate_outs)} | {self.num_steps}"
        assert len(self.intermediate_outs) == self.num_steps, msg

        # return only the final prediction one to not break code.
        # the intermediate predictions are relevant only for training.
        # the model prediction is the final output.
        return self.intermediate_outs[-1]

    def _reset_state(self):
        self.block.reset_state()


if __name__ == '__main__':

    device = torch.device('cuda:0')
    scale = 2
    c = 1
    b = 32
    x = torch.rand(b, c, 64, 64).to(device)

    # SRFBN-L
    model = SRFBN(upscale=scale,
                  in_chans=c,
                  num_features=64,
                  num_steps=4,
                  num_groups=6
                  ).to(device)

    with torch.no_grad():
        y = model(x)
    print(f'scale: {scale}', x.shape, y.shape)