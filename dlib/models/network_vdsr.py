import sys
from os.path import join, dirname, abspath
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

# Credit: https://github.com/Lornatang/VDSR-PyTorch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['VDSR']

# Paper: "Accurate Image Super-Resolution Using Very Deep Convolutional
# Networks", CVPR, 2016, Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee.


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1),
                              bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out


class VDSR(nn.Module):
    def __init__(self,
                 in_chans: int,
                 upscale: int) -> None:
        super(VDSR, self).__init__()

        assert upscale > 0, upscale
        assert isinstance(upscale, int), type(upscale)
        self.upscale = upscale


        assert isinstance(in_chans, int), type(in_chans)
        assert in_chans > 0, in_chans
        self.in_chans = in_chans

        self.global_residual: torch.Tensor = None
        self.x_interp: torch.Tensor = None

        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, in_chans, (3, 3), (1, 1), (1, 1), bias=False)

        # Initialize model weights
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

        out = self.conv1(_x)
        out = self.trunk(out)
        out = self.conv2(out)

        # out
        self.global_residual = out  # [-1, 1]

        out = out + ident_x

        assert out.shape == _x.shape, f'{out.shape}, {_x.shape}'

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(
                    0.0,
                    sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))


if __name__ == '__main__':
    device = torch.device('cuda:1')
    upscale = 2
    s = 64
    b = 32
    in_chans = 3
    model = VDSR(in_chans=in_chans, upscale=upscale)
    model.to(device)

    x = torch.randn((b, in_chans, s, s), device=device)

    # interpolate
    x_inter = F.interpolate(x,
                            size=(s * upscale, s * upscale),
                            mode='bicubic',
                            align_corners=True
                            )

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