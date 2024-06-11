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


__all__ = ['UNet']


# Source: https://github.com/Janspiry/
# Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules
# /unet.py

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class OutBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, ksz: int = 3, groups: int = 32,
                 dropout: float = 0):
        super().__init__()

        assert ksz % 2 == 1, f'ksz: {ksz}. ksz % 2 = {ksz % 2}.'
        padding = int((ksz - 1) / 2)

        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, kernel_size=ksz, padding=padding)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x):
        x = self.res_block(x)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        upscale: int,
        in_channel: int = 6,
        out_channel: int = 3,
        outksz: int = 3,
        inner_channel: int = 32,
        norm_groups: int = 32,
        channel_mults=(1, 2, 4, 8, 8),
        res_blocks: int = 3,
        dropout: float = 0.0,
        use_global_residual: bool = False
    ):
        super().__init__()

        # the output has the same size as input. the scale is just for info.
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

        assert isinstance(dropout, float), type(dropout)
        assert dropout >= 0.0, dropout

        for c in channel_mults:
            assert isinstance(c, int), type(c)
            assert c > 0, c
            msg = f'multi {c} * {inner_channel} not div by ngroup {norm_groups}'
            assert (c * inner_channel) % norm_groups == 0, msg

        self.upscale = upscale
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.outksz = outksz
        self.channel_mults = channel_mults
        self.inner_channel = inner_channel
        self.res_blocks = res_blocks
        self.dropout = dropout
        self.use_global_residual = use_global_residual

        self.global_residual: torch.Tensor = None
        self.x_interp = None


        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = False
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult,
                    norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = False
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult,
                    norm_groups=norm_groups, dropout=dropout,
                    with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))

        self.ups = nn.ModuleList(ups)

        self.final_conv = OutBlock(
            pre_channel, default(out_channel, in_channel), ksz=self.outksz,
            groups=norm_groups)

    def flush(self):
        self.global_residual = None
        self.x_interp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.flush()

        assert x.ndim == 4, x.ndim

        b, c, h, w = x.shape


        assert c == self.in_channel, f'{c} {self.in_channel}'
        assert c == self.out_channel, f'{c} {self.out_channel}'

        self.x_interp = x

        feats = []
        for layer in self.downs:
            x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                hi, wi = feats[-1].shape[2:]
                if x.shape[2:] != feats[-1].shape[2:]:

                    x = F.interpolate(
                        input=x,
                        size=[hi, wi],
                        mode='bilinear',
                        align_corners=False
                        )

                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        out = self.final_conv(x)

        if out.shape[2:] != self.x_interp.shape[2:]:
            hi, wi = self.x_interp.shape[2:]

            out = F.interpolate(
                input=out,
                size=[hi, wi],
                mode='bilinear',
                align_corners=False
            )

        if self.use_global_residual:

            self.global_residual = out

            out = self.x_interp + out

        assert out.shape == self.x_interp.shape, f'{out.shape}, ' \
                                                 f'{self.x_interp.shape}'

        return out


def test_unet():
    import time

    device = torch.device('cuda:0')
    upscale = 1
    b = 1
    c = 1
    h = 1024
    w = 1024
    in_channel = c
    x = torch.rand(b, in_channel, h, w).to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    net = UNet(upscale=upscale,
               in_channel=in_channel,
               out_channel=c,
               outksz=3,
               inner_channel=32,
               norm_groups=16,
               channel_mults=[1, 2, 4, 8, 16, 32, 32, 32],
               res_blocks=1,
               dropout=0.0,
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
    test_unet()