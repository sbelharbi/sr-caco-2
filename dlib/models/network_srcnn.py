import sys
from os.path import join, dirname, abspath
import math

import torch
from torch import nn
import torch.nn.functional as F

# Credit: https://github.com/Lornatang/SRCNN-PyTorch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['SRCNN']

# Paper: "Learning a Deep Convolutional Network for Image Super-Resolution",
# ECCV, 2014, Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang.


class SRCNN(nn.Module):
    def __init__(self,in_chans: int) -> None:
        super(SRCNN, self).__init__()

        assert isinstance(in_chans, int), type(in_chans)
        assert in_chans > 0, in_chans
        self.in_chans = in_chans

        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 1024, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(False)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(1024, 128, (1, 1), (1, 1)),
            nn.ReLU(False)
        )

        # Rebuild the layer.
        # self.reconstruction = nn.Conv2d(32, in_chans, (5, 5), (1, 1), (2, 2))
        self.reconstruction = nn.Conv2d(128, in_chans, (1, 1), (1, 1))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


class BasicSRCNN(nn.Module):
    """
    Leads to blurred output.
    """
    def __init__(self,in_chans: int) -> None:
        super(BasicSRCNN, self).__init__()

        assert isinstance(in_chans, int), type(in_chans)
        assert in_chans > 0, in_chans
        self.in_chans = in_chans

        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, in_chans, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0,
                                math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


if __name__ == '__main__':
    device = torch.device('cuda:1')
    upscale = 1
    s = 512
    b = 2
    in_chans = 1
    model = SRCNN(in_chans=in_chans)
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

        y = model(x_inter)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print('time gpu: {} (ms)'.format(elapsed_time_ms))
    print(f'input {x.shape}, scale x{upscale} out {y.shape}')
    z = list(x.shape)
    z[2] = z[2] * upscale
    z[3] = z[3] * upscale
    print(f'out: {y.shape} expected shape: {z}')