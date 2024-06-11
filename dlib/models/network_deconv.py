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

# ref: https://gitlab.mpi-klsb.mpg.de/jdong/dwdn

__all__ = ['Pyramid']


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



class X2(nn.Module):
    def __init__(self,
                 in_channel: int = 6,
                 out_channel: int = 3,
                 outksz: int = 3,
                 inner_channel: int = 32,
                 res_blocks: int = 3,
                 use_global_residual: bool = False
                 ):
        super(X2, self).__init__()

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

        self.upscale = 2
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.outksz = outksz
        self.inner_channel = inner_channel
        self.res_blocks = res_blocks
        self.use_global_residual = use_global_residual

        self.global_residual: torch.Tensor = None
        self.x_interp = None

        z = res_blocks
        ksz = 3
        p = int((ksz - 1) / 2)

        block0 = [
            Conv(in_channel, inner_channel, ksz, padding=p, act=True),
            *[ResBlock(Conv, inner_channel, 1, padding=0) for _ in range(z)],
            #
            Conv(inner_channel, inner_channel, 1, padding=0, act=True),
            *[ResBlock(Conv, inner_channel, 1, padding=0) for _ in range(z)],
            #
            Conv(inner_channel, inner_channel*2, 1, padding=0, act=True),
            *[ResBlock(Conv, inner_channel*2, 1, padding=0) for _ in range(z)],
            #
            Conv(inner_channel*2, inner_channel*4, 1, padding=0, act=True),
            *[ResBlock(Conv, inner_channel*4, 1, padding=0) for _ in range(z)],
        ]

        block1up = [Deconv(inner_channel*4, inner_channel*2, kernel_size=3,
                   padding=1, output_padding=1, act=True)]

        block2 = [
            Conv(inner_channel*2, inner_channel*2, ksz, padding=p, act=True),
            *[ResBlock(Conv, inner_channel*2, 1, padding=0) for _ in range(z)],
            #
            Conv(inner_channel*2, inner_channel*2, 1, padding=0, act=True),
            *[ResBlock(Conv, inner_channel*2, 1, padding=0) for _ in range(z)],
            #
            Conv(inner_channel*2, inner_channel * 2, 1, padding=0, act=True),
            *[ResBlock(Conv, inner_channel*2, 1, padding=0) for _ in range(z)],
            #
            Conv(inner_channel * 2, inner_channel * 4, 1, padding=0, act=True),
            *[ResBlock(Conv, inner_channel*4, 1, padding=0) for _ in range(z)],
        ]

        # todo: define output kernel.
        p = int((outksz - 1) / 2)
        block3 = [Conv(inner_channel*4, out_channel, outksz, padding=p)]

        self.block0 = nn.Sequential(*block0)
        self.block1 = nn.Sequential(*block1up)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)

        self.model = nn.Sequential(*[
            self.block0,
            self.block1,
            self.block2,
            self.block3
        ])


    def interpolatex2(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim
        b, c, h, w = x.shape

        out = F.interpolate(
            input=x,
            size=[h * 2, w * 2],
            mode='bilinear',
            align_corners=False
        )
        return out

    def flush(self):
        self.global_residual = None
        self.x_interp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.flush()

        assert x.ndim == 4, x.ndim

        b, c, h, w = x.shape

        assert c == self.in_channel, f'{c} {self.in_channel}'
        assert c == self.out_channel, f'{c} {self.out_channel}'

        self.x_interp = self.interpolatex2(x)

        out = self.model(x)
        _b, _c, _h, _w = out.shape
        assert _b == b, f'{b}, {_b}'
        assert _c == c, f'{c}, {_c}'

        if _h != (h * 2) or _w != (w * 2):
            print('error', out.shape, x.shape)
            out = F.interpolate(
                input=out,
                size=[h * 2, w * 2],
                mode='bilinear',
                align_corners=False
            )

        if self.use_global_residual:

            self.global_residual = out

            out = self.x_interp + out

        assert out.shape == self.x_interp.shape, f'{out.shape}, ' \
                                                 f'{self.x_interp.shape}'

        return out


class Pyramid(nn.Module):
    def __init__(self,
                 upscale: int,
                 in_channel: int = 6,
                 out_channel: int = 3,
                 outksz: int = 3,
                 inner_channel: int = 32,
                 res_blocks: int = 3,
                 use_global_residual: bool = False
                 ):
        super(Pyramid, self).__init__()

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

        self.upscale = upscale
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.outksz = outksz
        self.inner_channel = inner_channel
        self.res_blocks = res_blocks
        self.use_global_residual = use_global_residual

        self.global_residual: torch.Tensor = None
        self.x_interp = None

        # todo: change to pyramid.
        self.model = X2(in_channel=in_channel,
                        out_channel=out_channel,
                        outksz=outksz,
                        inner_channel=inner_channel,
                        res_blocks=res_blocks,
                        use_global_residual=use_global_residual
                        )

    def flush(self):
        self.model.flush()
        self.global_residual = None
        self.x_interp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        self.x_interp = self.model.x_interp

        if self.use_global_residual:
            self.global_residual = self.model.global_residual


        return out


def test_x2():
    import time

    device = torch.device('cuda:0')
    b = 1
    c = 1
    h = 31
    w = 31
    in_channel = c
    x = torch.rand(b, in_channel, h, w).to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    net = X2(in_channel=in_channel,
             out_channel=c,
             outksz=3,
             inner_channel=32,
             res_blocks=3,
             use_global_residual=True
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
        torch.cuda.empty_cache()
        print(80 * '*')


def test_pyramid():
    import time

    device = torch.device('cuda:0')
    up_scale = 2
    b = 1
    c = 1
    h = 31
    w = 31
    in_channel = c
    x = torch.rand(b, in_channel, h, w).to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    net = Pyramid(upscale=up_scale,
                  in_channel=in_channel,
                  out_channel=c,
                  outksz=3,
                  inner_channel=32,
                  res_blocks=3,
                  use_global_residual=True
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
        torch.cuda.empty_cache()
        print(80 * '*')


if __name__ == '__main__':
    # test_x2()
    test_pyramid()