import sys
import time
from os.path import join, dirname, abspath
import math
import itertools
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.utils.shared import count_params
from dlib.utils.tools import chunks_into_n

__all__ = ['DsrSplines']


def _full_conv(in_planes: int, out_planes: int, ksize: int) -> nn.Conv2d:
    assert ksize % 2 == 1, ksize
    padding = int((ksize - 1) / 2)

    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                     kernel_size=ksize, stride=1, padding=padding,
                     dilation=1, groups=1, bias=True, padding_mode='reflect')

class _Identity(nn.Module):
    def __init__(self):
        super(_Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Layer(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, ksze: int,
                 out_activation: str, use_local_residual: bool):
        super(_Layer, self).__init__()

        assert in_planes > 0, in_planes
        assert out_planes > 0, out_planes
        assert isinstance(in_planes, int), type(in_planes)
        assert isinstance(out_planes, int), type(out_planes)
        assert ksze % 2 == 1, ksze
        assert out_activation in constants.ACTIVATIONS, out_activation
        assert isinstance(use_local_residual, bool), type(use_local_residual)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.ksz = ksze
        self.out_activation = out_activation
        self.use_local_residual = use_local_residual

        self.relu = nn.ReLU(inplace=False)
        self.out_act = self._build_out_activation()

        self.conv = _full_conv(in_planes=in_planes, out_planes=out_planes,
                               ksize=ksze)

        if use_local_residual:
            if in_planes != out_planes:
                self.match_sz = _full_conv(
                    in_planes=in_planes, out_planes=out_planes, ksize=1)
            else:
                self.match_sz = _Identity()
        else:
            self.match_sz = None

    def _build_out_activation(self):
        if self.out_activation == constants.RELU:
            return nn.ReLU(inplace=False)

        elif self.out_activation == constants.TANH:
            return nn.Tanh()

        else:
            raise NotImplementedError(self.out_activation)

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.use_local_residual:
            out = self.relu(out)
            out = out + self.match_sz(x)
            out = self.out_act(out)
        else:
            out = self.out_act(out)

        return out

    def __str__(self):
        return f"Layer. in {self.in_planes}. out: {self.out_planes}. " \
               f"kszie: {self.ksz}. " \
               f"Use local residual: {self.use_local_residual}. " \
               f"Output activation: {self.out_activation}"


class _SplineNet(nn.Module):
    def __init__(self,
                 in_planes: int,
                 h_layers: List[int],
                 in_ksz: int,
                 knots: List[Tuple[int, int]],
                 use_global_residual: bool,
                 use_local_residual: bool,
                 color_min: int = 0,
                 color_max: int = 255
                 ):
        super(_SplineNet, self).__init__()

        assert in_planes > 0, in_planes
        assert isinstance(in_planes, int), type(in_planes)
        assert in_ksz > 0, in_ksz
        assert isinstance(in_ksz, int), type(in_ksz)
        assert in_ksz % 2 == 1, in_ksz
        assert len(h_layers) > 0, len(h_layers)

        for h in h_layers:
            assert h > 0, h
            assert isinstance(h, int), type(h)

        assert isinstance(color_min, int), type(color_min)
        assert isinstance(color_max, int), type(color_max)
        assert color_max < 2**8, color_max  # uint8 colors.
        assert color_min < color_max, f'{color_min}, {color_max}'
        assert isinstance(use_global_residual, bool), type(use_global_residual)
        assert isinstance(use_local_residual, bool), type(use_local_residual)

        assert len(knots) == in_planes, f'{len(knots)}, {in_planes}'

        for knot in knots:
            assert len(knot) == 2, len(knot)

            knot_low, knot_high = knot

            assert color_min <= knot_low <= color_max, knot_low
            assert isinstance(knot_low, int), type(knot_low)
            assert color_min <= knot_high <= color_max, knot_high
            assert isinstance(knot_high, int), type(knot_high)
            assert knot_low <= knot_high, f'low: {knot_low}. high: {knot_high}'


        self.in_planes = in_planes
        self.out_planes = in_planes
        self.h_layers = h_layers
        self.in_ksz = in_ksz
        self.color_min = color_min
        self.color_max = color_max
        self.knots = knots
        self.use_global_residual = use_global_residual
        self.use_local_residual = use_local_residual

        self.model = self._make_layers()
        self.mask = None
        self.prediction = None

    def flush(self):
        self.mask = None
        self.prediction = None

    def _make_layers(self):
        relu = constants.RELU
        layers = [_Layer(in_planes=self.in_planes, out_planes=self.h_layers[0],
                         ksze=self.in_ksz, out_activation=relu,
                         use_local_residual=self.use_local_residual)]


        for i, h in enumerate(self.h_layers):
            if i < (len(self.h_layers) - 1):
                out = self.h_layers[i + 1]
            else:
                break

            layers.append(_Layer(in_planes=h, out_planes=out, ksze=1,
                                 out_activation=relu,
                                 use_local_residual=self.use_local_residual))

        if self.use_global_residual:
            out_activ = constants.TANH
        else:
            out_activ = constants.RELU

        layers.append(_Layer(in_planes=self.h_layers[-1],
                             out_planes=self.out_planes, ksze=1,
                             out_activation=out_activ,
                             use_local_residual=self.use_local_residual))

        return nn.Sequential(*layers)

    @staticmethod
    def _get_mask_img_one_plane(x: torch.Tensor,
                                knot_low: int,
                                knot_high: int) -> torch.Tensor:
        assert x.ndim == 4, x.ndim
        b, c, h, w = x.shape
        assert c == 1, c

        # same device as x.
        upper = ((x < knot_high) | (x == knot_high)).type(torch.float32)
        lower = ((x > knot_low) | (x == knot_low)).type(torch.float32)

        out = upper * lower  # same size as x: b, 1, h, w
        assert out.shape == x.shape

        return out  # binary tensor. float32. same device as x.

    def get_mask(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim
        b, c, h, w = x.shape
        assert c == self.in_planes, f'c: {c}, img-nc: {self.in_planes}'
        assert c == len(self.knots), f'c: {c}, knots: {len(self.knots)}'
        assert c > 0, c

        # un-normalize image. input normalization has to be done simply by
        # dividing by max color so un-normalization would be consistent.

        # dont miss float values located at edge of knots.
        # todo: weak assumption: uint8. assumes max is 255.
        x_unormed = (x * self.color_max).type(torch.uint8).type(torch.float32)

        x_unormed = torch.clamp(x_unormed, min=self.color_min,
                                max=self.color_max)

        mask = None

        for i in range(c):
            _x = x_unormed[:, i, :, :].unsqueeze(1)  # b, 1, h, w.
            k_low, k_high = self.knots[i]

            _mask = self._get_mask_img_one_plane(x=_x, knot_low=k_low,
                                                 knot_high=k_high)
            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask  # and op.

        assert mask is not None

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.flush()

        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        assert c == self.in_planes, f'c: {c}, img-nc: {self.in_planes}'

        self.mask = self.get_mask(x).detach()  # b, 1, h, w.

        out = self.model(x)  # b, c, h, w
        assert out.shape == x.shape, f'{out.shape}, {x.shape}'

        out = out * self.mask
        self.prediction = out

        return out

    def __str__(self):
        l = [self.in_planes] + self.h_layers + [self.out_planes]
        l_str = [f'{l[0]}({self.in_ksz}x{self.in_ksz})']
        l_str += [f'{s}(1x1)' for s in self.h_layers + [self.out_planes]]
        l_str = '->'.join(l_str)

        knots = ''
        for k in self.knots[:-1]:
            knots += f'({k[0]}, {k[1]}) - '
        knots += f'({self.knots[-1][0]}, {self.knots[-1][1]})'

        nbr_params = count_params(self)
        h = len(self.h_layers)

        return f'Arch (h={h}): {l_str}. Knots: {knots}. ' \
               f'Use local residual: {self.use_local_residual}. ' \
               f'Use global residual: {self.use_global_residual}. ' \
               f'Color-min: {self.color_min}. Color-max: {self.color_max}. ' \
               f'N-params: {nbr_params}.'


class DsrSplines(nn.Module):
    def __init__(self,
                 upscale: int,
                 in_planes: int,
                 in_ksz: int,
                 splinenet_type: str,
                 n_splines_per_color: int,
                 use_global_residual: bool,
                 use_local_residual: bool,
                 color_min: int = 0,
                 color_max: int = 255
                 ):
        super(DsrSplines, self).__init__()

        assert upscale > 0, upscale
        assert isinstance(upscale, int), type(upscale)
        assert in_planes > 0, in_planes
        assert isinstance(in_planes, int), type(in_planes)
        assert in_ksz > 0, in_ksz
        assert isinstance(in_ksz, int), type(in_ksz)
        assert in_ksz % 2 == 1, in_ksz
        assert splinenet_type in constants.SPLINE_NET_TYPES, splinenet_type
        assert n_splines_per_color > 0, n_splines_per_color
        assert isinstance(n_splines_per_color, int), type(n_splines_per_color)

        nbr_colors = len(list(range(color_min, color_max))) + 1
        mx_spl = nbr_colors
        assert n_splines_per_color <= mx_spl, f'{n_splines_per_color}, {mx_spl}'
        assert isinstance(color_min, int), type(color_min)
        assert isinstance(color_max, int), type(color_max)
        assert color_min < color_max, f'{color_min}, {color_max}'
        assert isinstance(use_global_residual, bool), type(use_global_residual)
        assert isinstance(use_local_residual, bool), type(use_local_residual)

        self.upscale = upscale
        self.in_planes = in_planes
        self.in_ksz = in_ksz
        self.splinenet_type = splinenet_type
        self.n_splines_per_color = n_splines_per_color
        self.use_global_residual = use_global_residual
        self.use_local_residual = use_local_residual
        self.color_min = color_min
        self.color_max = color_max

        self.splines = None
        self.knots = []
        self._make_splines()
        self.n_splines = len(self.knots)

        self.global_residual: torch.Tensor = None
        self.x_interp = None

    def flush(self):
        for spline in self.splines:
            spline.flush()

        self.global_residual = None
        self.x_interp = None

    def _make_splines(self):
        colors_per_plane = list(range(self.color_min, self.color_max, 1))
        colors_per_plane += [self.color_max]
        splits = np.array_split(colors_per_plane, self.n_splines_per_color)
        splits = [list(s) for s in splits]

        bounds = [(int(min(s)), int(max(s))) for s in splits]
        knots = []

        tmp = [bounds for _ in range(self.in_planes)]
        # todo: deal with RGB. nsplines = n_splines_per_c**3.
        for combo in itertools.product(*tmp):
            knots.append(list(combo))

        self.knots = knots
        self.splines = nn.ModuleList()

        for lknots in knots:
            self.splines.append(self._make_a_spline(lknots))

    def _make_a_spline(self, knots: List[Tuple[int, int]]):
        h_layers = constants.SPLINEHIDDEN[self.splinenet_type]

        return _SplineNet(
            in_planes=self.in_planes,
            h_layers=h_layers,
            in_ksz=self.in_ksz,
            use_global_residual=self.use_global_residual,
            use_local_residual=self.use_local_residual,
            knots=knots,
            color_min=self.color_min,
            color_max=self.color_max
        )

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        _h = self.upscale * h
        _w = self.upscale * w

        out = F.interpolate(input=x,
                            size=(_h, _w),
                            mode='bicubic',
                            align_corners=False)
        out = torch.clamp(out, min=0.0, max=1.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        assert c == self.in_planes, f'c: {c}, img-nc: {self.in_planes}'
        _x = self._interpolate(x)
        self.x_interp = _x
        ident_x = _x

        for spline in self.splines:
            spline(_x)

        out = None
        for spline in self.splines:
            if out is None:
                out = spline.prediction
            else:
                out = out + spline.prediction

        if self.use_global_residual:

            self.global_residual = out  # [-1, 1]

            out = out + ident_x

        return out

    def __str__(self):
        desc_spline = f'{self.splines[0]}'
        nbr_params = count_params(self)
        return f'NBR-splines: {self.n_splines}. NBR-params: {nbr_params}. ' \
               f'Use global residuals: {self.use_global_residual}. ' \
               f'Use local residuals: {self.use_local_residual}. ' \
               f'Spline type: {self.splinenet_type}. ' \
               f'Spline-description: {desc_spline}'


def test_conv():
    device = torch.device('cuda:0')
    b = 32
    c = 3
    h = 63
    w = 63

    x = torch.rand(b, c, h, w).to(device)
    conv = _full_conv(3, 12, 1).to(device)

    out = conv(x)
    print(x.shape, out.shape)


def test_layer():
    device = torch.device('cuda:0')
    b = 32
    c = 3
    h = 63
    w = 63
    x = torch.rand(b, c, h, w).to(device)
    outc = 12
    layer = _Layer(in_planes=c, out_planes=outc, ksze=3,
                   out_activation=constants.RELU, use_local_residual=True
                   ).to(device)
    print(f'Testing layer {layer}')

    out = layer(x)
    print(x.shape, out.shape)


def test_spline_net():
    device = torch.device('cuda:0')
    b = 32
    c = 3
    h = 63
    w = 63
    x = torch.rand(b, c, h, w).to(device)
    layers = [
        [16],
        [32, 16],
        [64, 32, 16],
        [128, 64, 32, 16],
        [256, 128, 64, 32, 16],
        [512, 256, 128, 64, 32, 16]
    ]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for in_ksz in [1, 3, 5, 7, 9, 11, 13, 15]:

        for h_layers in layers:
            net = _SplineNet(in_planes=c, h_layers=h_layers, in_ksz=in_ksz,
                             knots=[(30, 50), (70, 100), (30, 250)],
                             use_local_residual=True, use_global_residual=True,
                             color_min=0, color_max=255).to(device)
            print(f'Testing {net}')

            for i in range(2):
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
        print(80 * '*')


def test_dsr_splines():
    device = torch.device('cuda:0')
    b = 1
    c = 1
    h = 64
    w = 64
    upscale = 2
    x = torch.rand(b, c, h, w).to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    splines = [
               constants.SPLINE_NET_TYPE8
               ]

    for in_ksz in [15]:
        for n_splines_per_c in [256]:
            for splinenet_type in splines:
                for run in range(1):
                    net = DsrSplines(upscale=upscale, in_planes=c,
                                     in_ksz=in_ksz,
                                     splinenet_type=splinenet_type,
                                     n_splines_per_color=n_splines_per_c,
                                     use_local_residual=True,
                                     use_global_residual=True,
                                     color_min=0, color_max=255)
                    print(f'run {run}. Net: {net}')
                    net = net.to(device)
                    start.record()
                    out = net(x)
                    end.record()
                    torch.cuda.synchronize()
                    print(f'Elapsed time: {start.elapsed_time(end)} (ms)')
                    print(x.shape, out.shape)
                    net.flush()
                    torch.cuda.empty_cache()

                print(80 * '=')
                print(80 * '=')



if __name__ == '__main__':
    test_conv()
    test_layer()
    test_spline_net()
    test_dsr_splines()
