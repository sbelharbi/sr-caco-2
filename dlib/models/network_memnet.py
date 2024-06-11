import sys
from os.path import join, dirname, abspath
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

# Credit: https://github.com/Lornatang/MemNet-PyTorch
# https://github.com/tyshiwo/MemNet
# https://github.com/Vandermode/pytorch-MemNet


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['MemNet']

# Paper: "MemNet: {A} Persistent Memory Network for Image Restoration", ICCV,
# 2017, Y. Tai, J. Yang, X. Liu, C. Xu.

class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identify = x
        out = self.residual_block(x)
        out = torch.add(out, identify)

        return out


class _MemoryBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 num_memory_blocks: int,
                 num_residual_blocks: int
                 ) -> None:
        super(_MemoryBlock, self).__init__()
        gate_channels = int((num_residual_blocks + num_memory_blocks) * channels)
        self.num_residual_blocks = num_residual_blocks

        recursive_unit = []
        for _ in range(num_residual_blocks):
            recursive_unit.append(_ResidualBlock(channels))
        self.recursive_unit = nn.Sequential(*recursive_unit)

        self.gate_unit = nn.Sequential(
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(True),
            nn.Conv2d(gate_channels, channels, (1, 1), (1, 1), (0, 0),
                      bias=False),
        )

    def forward(self, x: torch.Tensor, long_outs: list) -> torch.Tensor:
        out = x

        short_outs = []
        for _ in range(self.num_residual_blocks):
            out = self.recursive_unit(out)
            short_outs.append(out)

        gate_out = self.gate_unit(torch.cat(short_outs + long_outs, 1))
        long_outs.append(gate_out)

        return gate_out


class MemNet(nn.Module):
    def __init__(self,
                 in_chans: int,
                 upscale: int,
                 num_memory_blocks: int,
                 num_residual_blocks: int
                 ):
        super(MemNet, self).__init__()

        assert upscale > 0, upscale
        assert isinstance(upscale, int), type(upscale)
        self.upscale = upscale

        assert isinstance(in_chans, int), type(in_chans)
        assert in_chans > 0, in_chans
        self.in_chans = in_chans

        self.global_residual: torch.Tensor = None
        self.x_interp: torch.Tensor = None

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(in_chans),
            nn.ReLU(True),
            nn.Conv2d(in_chans, 64, (3, 3), (1, 1), (1, 1), bias=False)
        )

        dense_memory_blocks = []
        for i in range(num_memory_blocks):
            dense_memory_blocks.append(
                _MemoryBlock(64, i + 1, num_residual_blocks))
        self.dense_memory_blocks = nn.Sequential(*dense_memory_blocks)

        self.reconstructor = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, in_chans, (1, 1), (1, 1), (0, 0), bias=False)
        )

        # Initialize neural network weights
        self._initialize_weights()

    def flush(self):

        self.global_residual = None
        self.x_interp = None

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        _h = self.upscale * h
        _w = self.upscale * w

        out = F.interpolate(input=x,
                            size=(_h, _w),
                            mode='bicubic',
                            align_corners=False
                            )
        out = torch.clamp(out, min=0.0, max=1.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        self.flush()

        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        assert c == self.in_chans, f'c: {c}, img-nc: {self.in_chans}'

        _x = self._interpolate(x)
        self.x_interp = _x
        ident_x = _x

        out = self.feature_extractor(_x)

        long_outs = [out]
        for memory_block in self.dense_memory_blocks:
            out = memory_block(out, long_outs)

        out = self.reconstructor(out)

        # out
        self.global_residual = out  # [-1, 1]

        out = out + ident_x

        assert out.shape == _x.shape, f'{out.shape}, {_x.shape}'

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


if __name__ == '__main__':
    device = torch.device('cuda:1')
    upscale = 8
    s = 64
    b = 32
    in_chans = 3
    num_memory_blocks = 6
    num_residual_blocks = 6
    model = MemNet(in_chans=in_chans,
                   upscale=upscale,
                   num_memory_blocks=num_memory_blocks,
                   num_residual_blocks=num_residual_blocks
                   )
    model.to(device)

    x = torch.randn((b, in_chans, s, s), device=device)

    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        y = model(x)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print('time gpu: {} (ms)'.format(elapsed_time_ms))
    print(f'input {x.shape}, scale x{upscale} out {y.shape}')
    z = list(x.shape)
    z[2] = z[2] * upscale
    z[3] = z[3] * upscale
    print(f'out: {y.shape} expected shape: {z}')