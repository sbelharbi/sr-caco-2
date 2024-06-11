import sys
from os.path import dirname, abspath
from typing import Tuple

import torch.nn as nn
from torch.nn import functional as F
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.utils_reproducibility import set_seed


__all__ = ['PatchMoments']


class PatchMoments(nn.Module):
    """
    Split (unfold) tensor into patches.
    Compute their each mean and variance.
    Keep tensor unfolded.
    """
    def __init__(self, ksz: int, take_center_avg: bool = False):
        """

        :param ksz: int. kernel size. must be odd.
        :param take_center_avg: if true, we dont compute mean, but we return
        the value of the tensor corresponding of the center of the kernel.
        """
        super(PatchMoments, self).__init__()

        assert isinstance(ksz, int), type(ksz)
        assert ksz % 2 == 1, f'kernel size must be odd: {ksz}.'

        self.ksz: int = ksz
        self.padsz: int = int((ksz - 1) / 2)
        self.take_center_avg = take_center_avg

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 4, x.ndim  # b, c, h, w.
        assert x.shape[1] == 1, x.shape[1]
        _x = x
        if self.padsz:
            _x = F.pad(
                _x, pad=(self.padsz, self.padsz, self.padsz, self.padsz),
            mode='reflect')

        # _x: b, 1, h', w'
        z = F.unfold(_x, kernel_size=self.ksz, dilation=1, padding=0, stride=1)
        # z: b, 1 x ksz x ksz, hxw.
        # permute
        z = torch.permute(z, (0, 2, 1))  # b, hxw, 1 x ksz x ksz
        vari, avg = torch.var_mean(z, dim=-1, unbiased=True, keepdim=False)
        if self.take_center_avg:
            c = (self.ksz ** 2) // 2
            avg = z[:, :, c]  # b, hxw.
        assert vari.shape == avg.shape, f'{vari.shape}, {avg.shape}'
        # shape: b, hxw. each.

        return avg, vari


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 4, x.ndim  # b, c, h, w.
        assert x.shape[1] == 1, x.shape[1]  # todo: support only one dim. (grey)
        # loop if c > 1.
        avg, vari = self._forward(x)

        return avg, vari


def test_patch_moments():
    cuda = '0'
    device = torch.device(f'cuda:{cuda}')
    b = 1
    c = 1
    h = 12
    w = 12


    x = torch.rand(b, c, h, w)
    for ksz in [3]:
        op = PatchMoments(ksz=ksz, take_center_avg=True).to(device)
        print(x)
        avg, vari = op(x)
        print(x.shape, avg.shape, vari.shape)
        print(avg[0, 0], vari[0, 0])



if __name__ == '__main__':
    test_patch_moments()