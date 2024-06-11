import sys
import time
from os.path import join, dirname, abspath
import math
import itertools
from typing import List, Tuple, Union

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


__all__ = ['ConstrainedSupResCnn']


def _full_conv(in_planes: int,
               out_planes: int,
               ksize: int,
               ngroups: int) -> nn.Conv2d:

    assert isinstance(ngroups, int), type(ngroups)
    assert in_planes % ngroups == 0, f'in: {in_planes} not div. by {ngroups}'
    assert out_planes % ngroups == 0, f'out: {in_planes} not div. by {ngroups}'

    assert ksize % 2 == 1, ksize
    padding = int((ksize - 1) / 2)


    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                     kernel_size=ksize, stride=1, padding=padding,
                     dilation=1, groups=ngroups, bias=True,
                     padding_mode='reflect')


class _Identity(nn.Module):
    def __init__(self):
        super(_Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Layer(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 ksze: int,
                 ngroups: int,
                 out_activation: str,
                 use_local_residual: bool
                 ):
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
        self.ngroups = ngroups
        self.out_activation = out_activation
        self.use_local_residual = use_local_residual

        self.relu = nn.ReLU(inplace=False)
        self.out_act = self._build_out_activation()

        self.avg_output = None

        self.conv = _full_conv(in_planes=in_planes, out_planes=out_planes,
                               ksize=ksze, ngroups=ngroups)

        if use_local_residual:
            if in_planes != out_planes:
                self.match_sz = _full_conv(
                    in_planes=in_planes, out_planes=out_planes, ksize=1,
                    ngroups=ngroups)
            else:
                self.match_sz = _Identity()
        else:
            self.match_sz = None

    def _build_out_activation(self):
        if self.out_activation == constants.RELU:
            return nn.ReLU(inplace=False)

        elif self.out_activation == constants.TANH:
            return nn.Tanh()

        elif self.out_activation == constants.NONE_ACTIV:
            return _Identity()
        else:
            raise NotImplementedError(self.out_activation)

    def flush(self):
        self.avg_output = None

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.use_local_residual:
            out = self.relu(out)
            out = out + self.match_sz(x)
            out = self.out_act(out)
        else:
            out = self.out_act(out)

        # out: b, c, h, w.
        self.avg_output = out.mean(dim=1, keepdim=True)  # b, 1, h, w

        return out

    def __str__(self):
        return f"Layer. in {self.in_planes}. out: {self.out_planes}. " \
               f"kszie: {self.ksz}. ngroups: {self.ngroups}. " \
               f"Use local residual: {self.use_local_residual}. " \
               f"Output activation: {self.out_activation}"


class ConstrainedSupResCnn(nn.Module):
    def __init__(self,
                 upscale: int,
                 in_planes: int,
                 h_layers: Union[List[int], str],
                 in_ksz: int,
                 ngroups: int,
                 use_global_residual: bool,
                 use_local_residual: bool
                 ):
        super(ConstrainedSupResCnn, self).__init__()

        if isinstance(h_layers, str):
            h_layers = constants.NETS_CNN[h_layers]
        else:
            assert isinstance(h_layers, list), type(h_layers)
            assert len(h_layers) > 0, len(h_layers)

        assert upscale > 0, upscale
        assert isinstance(upscale, int), type(upscale)
        assert in_planes > 0, in_planes
        assert isinstance(in_planes, int), type(in_planes)
        assert in_ksz > 0, in_ksz
        assert isinstance(in_ksz, int), type(in_ksz)
        assert in_ksz % 2 == 1, in_ksz

        assert ngroups > 0, ngroups
        assert isinstance(ngroups, int), type(ngroups)

        for h in h_layers:
            assert h > 0, h
            assert isinstance(h, int), type(h)


        assert isinstance(use_global_residual, bool), type(use_global_residual)
        assert isinstance(use_local_residual, bool), type(use_local_residual)

        self.upscale = upscale
        self.in_planes = in_planes
        self.out_planes = in_planes
        self.h_layers = h_layers
        self.in_ksz = in_ksz
        self.ngroups = ngroups
        self.use_global_residual = use_global_residual
        self.use_local_residual = use_local_residual

        self.model = self._make_layers()

        self.n_layers = len(self.model)

        self.global_residual: torch.Tensor  = None
        self.x_interp = None

    def flush(self):
        for i in range(self.n_layers):
            self.model[i].flush()

        self.global_residual = None
        self.x_interp = None

    def _make_layers(self):
        relu = constants.RELU
        layers = [_Layer(in_planes=self.in_planes, out_planes=self.h_layers[0],
                         ksze=self.in_ksz, ngroups=1, out_activation=relu,
                         use_local_residual=self.use_local_residual)]


        for i, h in enumerate(self.h_layers):
            if i < (len(self.h_layers) - 1):
                out = self.h_layers[i + 1]
            else:
                break

            layers.append(_Layer(in_planes=h, out_planes=out, ksze=1,
                                 out_activation=relu, ngroups=self.ngroups,
                                 use_local_residual=self.use_local_residual))

        out_activ = constants.NONE_ACTIV

        layers.append(_Layer(in_planes=self.h_layers[-1],
                             out_planes=self.out_planes, ksze=1, ngroups=1,
                             out_activation=out_activ,
                             use_local_residual=self.use_local_residual))

        return nn.Sequential(*layers)

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
        self.flush()

        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        assert c == self.in_planes, f'c: {c}, img-nc: {self.in_planes}'

        _x = self._interpolate(x)
        self.x_interp = _x
        ident_x = _x

        out = self.model(_x)  # b, c, h, w

        if self.use_global_residual:

            self.global_residual = out  # [-1, 1]

            out = out + ident_x

        assert out.shape == _x.shape, f'{out.shape}, {_x.shape}'

        return out

    def __str__(self):
        l = [self.in_planes] + self.h_layers + [self.out_planes]
        l_str = [f'{l[0]}({self.in_ksz}x{self.in_ksz})']
        l_str += [f'{s}(1x1)' for s in self.h_layers + [self.out_planes]]
        l_str = '->'.join(l_str)

        nbr_params = count_params(self)
        h = len(self.h_layers)

        return f'upscale: {self.upscale}. Arch (h={h}): {l_str}. ' \
               f'ngroups: {self.ngroups}. ' \
               f'Use local residual: {self.use_local_residual}. ' \
               f'Use global residual: {self.use_global_residual}. ' \
               f'N-params: {nbr_params}.'


def test_conv():
    device = torch.device('cuda:0')
    b = 32
    c = 32
    h = 63
    w = 63
    outp = 64

    x = torch.rand(b, c, h, w).to(device)
    for ngroups in [1, 2, 4]:
        conv = _full_conv(c, outp, 1, ngroups).to(device)

        out = conv(x)
        print(x.shape, out.shape)
        print(f'nbr params ngroups = {ngroups}: {count_params(conv)}')


def test_layer():
    device = torch.device('cuda:0')
    b = 32
    c = 32
    h = 63
    w = 63
    x = torch.rand(b, c, h, w).to(device)
    outc = 12
    ngroups = 2
    layer = _Layer(in_planes=c, out_planes=outc, ksze=3, ngroups=ngroups,
                   out_activation=constants.RELU, use_local_residual=True
                   ).to(device)
    print(f'Testing layer {layer}')

    out = layer(x)
    print(x.shape, out.shape)


def test_constrained_sup_res_cnn():
    device = torch.device('cuda:0')
    b = 4
    c = 3
    h = 128
    w = 128
    x = torch.rand(b, c, h, w).to(device)
    layers = [
        # [16],
        # [32, 16],
        # [64, 32, 16],
        # [128, 64, 32, 16],
        # [256, 128, 64, 32, 16],
        [512, 512, 512, 512, 512, 512]
    ]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    upscale = 2

    for in_ksz in [15]:

        for h_layers in layers:
            net = ConstrainedSupResCnn(upscale=upscale, in_planes=c,
                                       h_layers=h_layers,
                                       in_ksz=in_ksz, ngroups=8,
                                       use_local_residual=True,
                                       use_global_residual=True).to(device)
            print(f'Testing {net}')

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
    # test_conv()
    # test_layer()
    test_constrained_sup_res_cnn()