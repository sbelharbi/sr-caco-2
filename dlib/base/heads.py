import sys
from os.path import dirname, abspath
from typing import Tuple

import torch
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.base.modules import Flatten, Activation
from dlib.utils import constants

__all__ = ['SegmentationHead', 'ClassificationHead', 'ReconstructionHead',
           'BboxHead']


class SegmentationHead(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 activation=None,
                 upsampling=1
                 ):
        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2
                           )
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2,
                 activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), "
                "got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class ReconstructionHead(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 activation=constants.RANGE_TANH,
                 upsampling=1
                 ):
        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2
                           )
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class BboxHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 h: int,
                 w: int):
        super(BboxHead, self).__init__()

        assert h > 0
        assert w > 0
        assert isinstance(h, int)
        assert isinstance(w, int)

        assert h == float(h)
        assert w == float(w)

        self.h = float(h)
        self.w = float(w)

        assert in_channels > 0
        assert isinstance(in_channels, int)
        self.in_channels = in_channels

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(in_channels, 4, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor]:
        z = self.avgpool(x).squeeze(-1).squeeze(-1)  # b, n
        out = self.fc(z)  # x1, y1, x2, y2.
        return out

    def __str__(self):
        return "classes h: {}. classes w: {}. dropout: {}. " \
               "in channels: {}".format(
                self.classes_h, self.classes_w, self._dropout, self.in_channels)


def test_BboxHead():
    def count_params(model: torch.nn.Module):
        return sum([p.numel() for p in model.parameters()])
    cuda = 0
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    conts = {
        'resnet50': [2024, 28, 28],
        'vgg16': [1024, 28, 28],
        'inceptionv3': [1024, 29, 29]
    }
    CL = 224
    b = 32
    for k in conts:
        in_ch, h, w = conts[k]
        bbox = BboxHead(in_channels=in_ch, w=CL, h=CL)
        bbox.to(DEVICE)
        x = torch.rand((b, in_ch, h, w), device=DEVICE, requires_grad=True)
        box = bbox(x)
        print(f'number of parameters: {count_params(bbox)}')
        print(f'x {x.shape} box {box.shape}')
        print(f'boxes: {box}')


if __name__ == "__main__":
    from dlib.utils.utils_reproducibility import set_seed

    set_seed(0)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = False

    test_BboxHead()
