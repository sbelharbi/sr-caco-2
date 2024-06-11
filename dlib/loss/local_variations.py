import sys
from os.path import dirname, abspath
from typing import Tuple

import torch.nn as nn
from torch.nn import functional as F
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.utils_reproducibility import set_seed


__all__ = ['ImageGradient', 'LaplacianFilter', 'LocalVariation']


class ImageGradient(nn.Module):
    """
    First order derivative of image.
    """
    def __init__(self):
        super(ImageGradient, self).__init__()

        device = torch.device(f'cuda:{torch.cuda.current_device()}')

        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

        kernel = torch.cat((kernel_h, kernel_v), dim=0).to(device)
        self.register_buffer("weights2d", kernel)

        self.img_grad = F.conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        assert c == 1, c  # supports only grey.

        # full conv.
        _x = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
        out = self.img_grad(_x, weight=self.weights2d, bias=None, stride=1,
                            padding=0, dilation=1, groups=1)

        _b, _c, _h, _w = out.shape
        assert _c == 2, _c
        assert [_b, _h, _w] == [b, h, w], f'{[_b, _h, _w]} {[b, h, w]}'

        return out


class LaplacianFilter(nn.Module):
    """
    Second order derivative of image.
    """
    def __init__(self):
        super(LaplacianFilter, self).__init__()

        device = torch.device(f'cuda:{torch.cuda.current_device()}')

        kernel = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)

        kernel = kernel.to(device)
        self.register_buffer("weights2d", kernel)

        self.laplace = F.conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        assert c == 1, c  # supports only grey.

        # full conv.
        _x = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
        out = self.laplace(_x, weight=self.weights2d, bias=None, stride=1,
                           padding=0, dilation=1, groups=1)

        _b, _c, _h, _w = out.shape
        assert _c == 1, _c
        assert [_b, _h, _w] == [b, h, w], f'{[_b, _h, _w]} {[b, h, w]}'

        return out


class LocalVariation(nn.Module):
    def __init__(self, ksz: int):
        super(LocalVariation, self).__init__()
        assert isinstance(ksz, int), type(ksz)
        assert ksz > 2, ksz
        assert ksz % 2 == 1, f'{ksz % 2}'

        self.ksz = ksz

        c = ksz // 2
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        kernels = []
        for i in range(ksz):
            for j in range(ksz):
                if i == j == c:
                    continue

                kernel = torch.zeros(size=(ksz, ksz), dtype=torch.float32,
                                     requires_grad=False, device=device)
                kernel[c, c] = 1
                kernel[i, j] = -1
                kernels.append(kernel.unsqueeze(0).unsqueeze(0))


        kernel = torch.cat(kernels, dim=0)
        self.nbr_p = ksz * ksz - 1
        assert kernel.shape == (self.nbr_p, 1, ksz, ksz), kernel.shape

        self.register_buffer("weights2d", kernel)

        self.conv = F.conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w
        b, c, h, w = x.shape
        assert c == 1, c  # supports only grey.

        # full conv.
        p = int((self.ksz - 1) / 2)
        _x = F.pad(x, pad=(p, p, p, p), mode='replicate')
        out = self.conv(_x, weight=self.weights2d, bias=None, stride=1,
                        padding=0, dilation=1, groups=1)

        _b, _c, _h, _w = out.shape
        assert _c == self.nbr_p, _c
        assert [_b, _h, _w] == [b, h, w], f'{[_b, _h, _w]} {[b, h, w]}'

        return out



def test_conv2():
    conv = nn.Conv2d(1, 2, kernel_size=3,
                     stride=1, padding=1, dilation=1, groups=1,
                     bias=False, padding_mode='reflect')
    print(conv.weight.shape)
    kernel_v = [[0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]]
    kernel_h = [[0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0]]
    kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
    kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

    kernel = torch.cat((kernel_h, kernel_v), dim=0)
    print(kernel.shape)
    print(kernel)


def test_img_grad():
    device = torch.device('cuda:0')
    set_seed(0)
    x = torch.rand(1, 1, 7, 2).to(device)
    img_grad = ImageGradient().to(device)
    out = img_grad(x)
    print(x.shape, out.shape)
    print(x)
    print(out)


def test_laplacian_filter():
    device = torch.device('cuda:0')
    set_seed(0)
    x = torch.rand(1, 1, 2, 2).to(device)
    laplace = LaplacianFilter().to(device)
    out = laplace(x)
    print(x.shape, out.shape)
    print(x)
    print(out)


def test_local_variation():
    device = torch.device('cuda:0')
    set_seed(0)
    x = torch.rand(1, 1, 200, 19).to(device)
    laplace = LocalVariation(ksz=7).to(device)
    out = laplace(x)
    print(x.shape, out.shape)
    print(x)
    print(out)



if __name__ == "__main__":
    # test_conv2()
    # test_img_grad()
    # test_laplacian_filter()
    test_local_variation()