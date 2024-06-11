import sys
from os.path import join, dirname, abspath
import math
from abc import ABC
from math import prod
from typing import Tuple, Union
from collections import OrderedDict

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat

# Credit: https://github.com/phoenix104104/LapSRN  -- matlab.
# https://github.com/Lornatang/LapSRN-PyTorch
# https://github.com/twtygqyy/pytorch-LapSRN

# Journal: https://arxiv.org/pdf/1710.01992.pdf


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['MSLapSRN']

# Paper: "Fast and Accurate Image Super-Resolution with Deep Laplacian
# Pyramid Networks", TPAMI, 2018, Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja,
# Ming-Hsuan Yang


# Copy from `https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py`
def get_upsample_filter(size: int) -> torch.Tensor:
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    bilinear_filter = torch.from_numpy(
        (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    ).float()

    return bilinear_filter


class ConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvLayer, self).__init__()
        self.cl = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cl(x)

        return out


class MSLapSRN(nn.Module):
    def __init__(self,
                 upscale: int = 2,
                 in_chans: int = 3
                 ) -> None:
        super(MSLapSRN, self).__init__()

        assert upscale in [2, 4, 8], upscale
        self.upscale = upscale
        self.in_chans = in_chans

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Scale 2
        laplacian_pyramid_conv1 = []
        for _ in range(10):
            laplacian_pyramid_conv1.append(ConvLayer(64))
        laplacian_pyramid_conv1.append(
            nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
        laplacian_pyramid_conv1.append(nn.LeakyReLU(0.2, True))

        self.laplacian_pyramid_conv1 = nn.Sequential(*laplacian_pyramid_conv1)
        self.laplacian_pyramid_conv2 = nn.ConvTranspose2d(
            1, 1, (4, 4), (2, 2), (1, 1))
        self.laplacian_pyramid_conv3 = nn.Conv2d(64, in_chans,
                                                 (3, 3), (1, 1), (1, 1))

        if upscale > 2:
            # Scale 4
            laplacian_pyramid_conv4 = []
            for _ in range(10):
                laplacian_pyramid_conv4.append(ConvLayer(64))
            laplacian_pyramid_conv4.append(
                nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
            laplacian_pyramid_conv4.append(nn.LeakyReLU(0.2, True))

            self.laplacian_pyramid_conv4 = nn.Sequential(
                *laplacian_pyramid_conv4)
            self.laplacian_pyramid_conv5 = nn.ConvTranspose2d(
                1, 1, (4, 4), (2, 2), (1, 1))
            self.laplacian_pyramid_conv6 = nn.Conv2d(64, in_chans,
                                                     (3, 3), (1, 1), (1, 1))

        if upscale > 4:
            # Scale 8
            laplacian_pyramid_conv7 = []
            for _ in range(10):
                laplacian_pyramid_conv7.append(ConvLayer(64))
            laplacian_pyramid_conv7.append(
                nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
            laplacian_pyramid_conv7.append(nn.LeakyReLU(0.2, True))

            self.laplacian_pyramid_conv7 = nn.Sequential(
                *laplacian_pyramid_conv7)
            self.laplacian_pyramid_conv8 = nn.ConvTranspose2d(
                1, 1, (4, 4), (2, 2), (1, 1))
            self.laplacian_pyramid_conv9 = nn.Conv2d(64, in_chans,
                                                     (3, 3), (1, 1), (1, 1))

        self.intermediate_outs = []  # hold only intermediate predictions.

    def flush(self):
        self.intermediate_outs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.intermediate_outs = []

        out = self.conv1(x)

        # todo: store intermediate predictions for training.

        # X2
        lpc1 = self.laplacian_pyramid_conv1(out)
        lpc2 = self.laplacian_pyramid_conv2(x)
        lpc3 = self.laplacian_pyramid_conv3(lpc1)
        out1 = lpc2 + lpc3

        # X4
        out2 = None
        if self.upscale in [4, 8]:
            self.intermediate_outs.append(out1)

            lpc4 = self.laplacian_pyramid_conv4(lpc1)
            lpc5 = self.laplacian_pyramid_conv5(out1)
            lpc6 = self.laplacian_pyramid_conv6(lpc4)
            out2 = lpc5 + lpc6

        # X8
        out3 = None
        if self.upscale == 8:
            self.intermediate_outs.append(out2)

            lpc7 = self.laplacian_pyramid_conv7(lpc4)
            lpc8 = self.laplacian_pyramid_conv8(out2)
            lpc9 = self.laplacian_pyramid_conv9(lpc7)
            out3 = lpc8 + lpc9

        if self.upscale == 2:
            return out1

        if self.upscale == 4:
            return out2

        if self.upscale == 8:
            return out3

        raise NotImplementedError(self.upscale)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.ConvTranspose2d):
                c1, c2, h, w = module.weight.data.size()
                weight = get_upsample_filter(h)
                module.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if module.bias is not None:
                    module.bias.data.zero_()


if __name__ == '__main__':

    device = torch.device('cuda:0')
    scale = 8
    c = 1
    b = 2
    x = torch.rand(b, c, 64, 64).to(device)

    model = MSLapSRN(upscale=scale,
                     in_chans=c
                     ).to(device)

    with torch.no_grad():
        y = model(x)
        print(len(model.intermediate_outs), f'scale: x{scale}')
        for i, z in enumerate(model.intermediate_outs):
            print(i, z.shape)
        print(f'scale: x{scale}', x.shape, y.shape)