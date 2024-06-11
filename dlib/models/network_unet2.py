import sys
import time
import math
from os.path import join, dirname, abspath
import itertools
from typing import List, Tuple
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.shared import count_params
from dlib.utils import constants
import dlib.dllogger as DLLogger

# ref: https://gitlab.mpi-klsb.mpg.de/jdong/dwdn

__all__ = ['UNet']


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size//2), bias=bias)

class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1,
                 padding=0, bias=True, bn=False, act=False):
        super(Conv , self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels, n_feats, kernel_size, stride,
                           padding, bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act:m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
    def forward(self, input):
        return self.body(input)

class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2,
                 padding=0, output_padding=0 , bias=True, act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size,
                                    stride=stride, padding=padding,
                                    output_padding=output_padding, bias=bias))
        if act: m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, padding = 0 ,bias=True,
                 bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, padding=padding,
                          bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x

        return res



class UNet(nn.Module):
    def __init__(self,
                 upscale: int,
                 in_channel: int = 6,
                 out_channel: int = 3,
                 outksz: int = 3,
                 inner_channel: int = 32,
                 res_blocks: int = 3,
                 use_global_residual: bool = False,
                 task: str = constants.REGRESSION,
                 color_min: int = 0,
                 color_max: int = 255
                 ):
        super(UNet, self).__init__()

        assert isinstance(upscale, int), type(upscale)
        assert upscale > 0, upscale

        assert isinstance(in_channel, int), type(in_channel)
        assert in_channel > 0, in_channel

        assert isinstance(inner_channel, int), type(inner_channel)
        assert inner_channel > 0, inner_channel

        assert isinstance(out_channel, int), type(out_channel)
        assert out_channel > 0, out_channel

        assert isinstance(outksz, int), type(outksz)
        assert outksz > 0, outksz
        assert outksz % 2 == 1, f'outksz: {outksz}. ksz % 2 = {outksz % 2}.'

        assert isinstance(res_blocks, int), type(res_blocks)
        assert res_blocks > 0, res_blocks

        assert task in [constants.REGRESSION, constants.SEGMENTATION], task
        assert isinstance(task, str), type(task)

        if task == constants.SEGMENTATION:
            assert not use_global_residual, f'tasK: {task},global residual: on.'
            assert out_channel == 1, f'task: {task}, ' \
                                     f'supported out channels: 1. ' \
                                     f'provided: {out_channel}'

        nbr_colors = len(list(range(color_min, color_max))) + 1
        assert isinstance(color_min, int), type(color_min)
        assert isinstance(color_max, int), type(color_max)
        assert color_min < color_max, f'{color_min}, {color_max}'

        self.upscale = upscale
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.outksz = outksz
        self.inner_channel = inner_channel
        self.res_blocks = res_blocks
        self.use_global_residual = use_global_residual
        self.task = task
        self.color_min = color_min
        self.color_max = color_max
        self.nbr_colors = nbr_colors

        self.global_residual: torch.Tensor = None
        self.x_interp = None
        # segmentation task
        self.expected_pred = None
        self.argmax_pred = None
        self.raw_segmentation = None

        n_resblock = res_blocks
        kernel_size = 5

        FeatureBlock = [Conv(in_channel, inner_channel, kernel_size, padding=2,
                             act=True),
                        ResBlock(Conv, inner_channel, kernel_size, padding=2),
                        ResBlock(Conv, inner_channel, kernel_size, padding=2),
                        ResBlock(Conv, inner_channel, kernel_size, padding=2)]

        InBlock1 = [Conv(inner_channel, inner_channel, kernel_size, padding=2,
                         act=True),
                   ResBlock(Conv, inner_channel, kernel_size, padding=2),
                   ResBlock(Conv, inner_channel, kernel_size, padding=2),
                   ResBlock(Conv, inner_channel, kernel_size, padding=2)]
        # InBlock2 = [Conv(2 * inner_channel, inner_channel, kernel_size,
        #                  padding=2, act=True),
        #            ResBlock(Conv, inner_channel, kernel_size, padding=2),
        #            ResBlock(Conv, inner_channel, kernel_size, padding=2),
        #            ResBlock(Conv, inner_channel, kernel_size, padding=2)]

        # encoder1
        Encoder_first= [Conv(inner_channel, inner_channel * 2, kernel_size,
                             padding=2,
                             stride=2, act=True),
                        ResBlock(Conv, inner_channel*2, kernel_size,padding=2),
                        ResBlock(Conv, inner_channel*2, kernel_size,padding=2),
                        ResBlock(Conv, inner_channel*2, kernel_size,padding=2)]
        # encoder2
        Encoder_second = [Conv(inner_channel*2, inner_channel*4, kernel_size,
                               padding=2, stride=2, act=True),
                          ResBlock(Conv, inner_channel*4, kernel_size,
                                   padding=2),
                          ResBlock(Conv, inner_channel*4, kernel_size,
                                   padding=2),
                          ResBlock(Conv, inner_channel*4, kernel_size,
                                   padding=2)]
        # decoder2
        Decoder_second = [ResBlock(Conv, inner_channel*4, kernel_size, padding=2)
                          for _ in range(n_resblock)]
        Decoder_second.append(Deconv(inner_channel*4, inner_channel*2,
                                     kernel_size=3, padding=1, output_padding=1,
                                     act=True))
        # decoder1
        Decoder_first = [ResBlock(Conv, inner_channel*2, kernel_size,
                                  padding=2) for _ in range(n_resblock)]
        Decoder_first.append(Deconv(inner_channel*2, inner_channel,
                                    kernel_size=3, padding=1,
                                    output_padding=1, act=True))

        OutBlock = [ResBlock(Conv, inner_channel, kernel_size,
                             padding=2) for _ in range(n_resblock)]
        # todo: define output kernel.
        p = int((outksz - 1) / 2)
        if task == constants.REGRESSION:
            OutBlock2 = [Conv(inner_channel, out_channel, outksz, padding=p)]

        elif task == constants.SEGMENTATION:
            assert out_channel == 1, out_channel

            OutBlock2 = [Conv(inner_channel, nbr_colors, outksz, padding=p)]

        else:
            raise NotImplementedError(f'Task {task}')

        self.FeatureBlock = nn.Sequential(*FeatureBlock)
        self.inBlock1 = nn.Sequential(*InBlock1)
        # self.inBlock2 = nn.Sequential(*InBlock2)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)
        self.outBlock2 = nn.Sequential(*OutBlock2)

    def flush(self):
        self.global_residual = None
        self.x_interp = None

        self.expected_pred = None
        self.argmax_pred = None
        self.raw_segmentation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.flush()

        assert x.ndim == 4, x.ndim

        b, c, h, w = x.shape

        assert c == self.in_channel, f'{c} {self.in_channel}'
        assert c == self.out_channel, f'{c} {self.out_channel}'

        self.x_interp = x
        feature_out = self.FeatureBlock(x)
        first_scale_inblock = self.inBlock1(feature_out)
        first_scale_encoder_first = self.encoder_first(first_scale_inblock)
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
        first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
        first_scale_decoder_first = self.decoder_first(
            first_scale_decoder_second+first_scale_encoder_first)

        if first_scale_decoder_first.shape[2:] != first_scale_inblock.shape[2:]:
            hi, wi = first_scale_decoder_first.shape[2:]
            first_scale_inblock = F.interpolate(
                input=first_scale_inblock,
                size=[hi, wi],
                mode='bilinear',
                align_corners=False
            )

        input_pre = self.outBlock(first_scale_decoder_first+first_scale_inblock)
        out = self.outBlock2(input_pre)

        if out.shape[2:] != self.x_interp.shape[2:]:
            hi, wi = self.x_interp.shape[2:]

            out = F.interpolate(
                input=out,
                size=[hi, wi],
                mode='bilinear',
                align_corners=False
            )

        if self.task == constants.REGRESSION:

            if self.use_global_residual:

                self.global_residual = out

                out = self.x_interp + out

            assert out.shape == self.x_interp.shape, f'{out.shape}, ' \
                                                     f'{self.x_interp.shape}'

        elif self.task == constants.SEGMENTATION:

            _b, _c, _h, _w = self.x_interp.shape
            assert out.shape[0] == _b, f'{out.shape[0]} {_b}'
            assert out.shape[1] == self.nbr_colors, f'{out.shape[1]} ' \
                                                    f'{self.nbr_colors}'
            assert out.shape[2] == _h, f'{out.shape[2]} {_h}'
            assert out.shape[3] == _w, f'{out.shape[3]} {_w}'

            self.raw_segmentation = out
            self.argmax_pred = torch.argmax(out, dim=1, keepdim=True)  # b,
            # 1, h, w
            # todo: support rgb.
            assert self.argmax_pred.shape == self.x_interp.shape, \
                f'{self.argmax_pred.shape}, {self.x_interp.shape}'


            out = self.compute_expected_val(out)  # b, 1, h, w.
            assert out.shape == self.x_interp.shape, f'{out.shape}, ' \
                                                     f'{self.x_interp.shape}'

            out = out / float(self.color_max)  # in [0, 1]

            self.expected_pred = out

            if not self.training:
                out = self.argmax_pred / float(self.color_max)  # in [0, 1]

        else:
            raise NotImplementedError(self.task)

        return out

    def compute_expected_val(self, x: torch.Tensor) -> torch.Tensor:
        assert self.task == constants.SEGMENTATION, self.task

        assert x.ndim == 4, x.ndim  # b, c, h, w

        a = torch.arange(self.nbr_colors).to(x.device)
        a = a.view(1, -1, 1, 1)
        z = torch.ones(x.shape, dtype=torch.float32, device=x.device,
                       requires_grad=False)
        z = z * a

        out = torch.softmax(x, dim=1) * z
        out = out.sum(dim=1, keepdim=True)  # b, 1, h, w. in [0, nbr_colors]
        # [0, 255]

        return out



def test_unet():
    import time

    device = torch.device('cuda:0')
    upscale = 1
    b = 1
    c = 1
    h = 64
    w = 64
    in_channel = c
    x = torch.rand(b, in_channel, h, w).to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    net = UNet(upscale=upscale,
               in_channel=in_channel,
               out_channel=c,
               outksz=3,
               inner_channel=32,
               res_blocks=3,
               use_global_residual=False,
               task=constants.SEGMENTATION,
               color_min=0,
               color_max=255
               ).to(device)
    print(f'Testing {net}')
    print(f'NBR params: {count_params(net)}')

    for i in range(1):
        print(f'run {i}.')
        start.record()
        t0 = time.perf_counter()
        out = net(x)
        t1 = time.perf_counter()
        end.record()
        torch.cuda.synchronize()
        print(f'Elapsed time: {start.elapsed_time(end)} (ms)')
        print(f'Elapsed time 2: {t1 - t0} (s)')

        print(x.shape, out.shape)
        if net.task == constants.SEGMENTATION:
            print(net.raw_segmentation.shape, net.argmax_pred.shape,
                  net.expected_pred.shape)
            print(net.argmax_pred.max())
            print(net.expected_pred.max())

        torch.cuda.empty_cache()
        print(80 * '*')


if __name__ == '__main__':
    test_unet()