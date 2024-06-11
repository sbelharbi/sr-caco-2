import sys
from os.path import dirname, abspath
from typing import Tuple
import math

import torch.nn as nn
from torch.nn import functional as F
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.utils_reproducibility import set_seed


__all__ = ['SoftHistogram', 'GaussianKDE']


class SoftHistogram(nn.Module):
    """
    Computes a (soft)-histogram of `torch.histc`.
    Histograms are not differentiable.
    We use an approximation based on binning.

    NOTE:
    OPERATES ON BATCHES (EXPECTED THE FIRST DIM TO BE THE BATCH SIZE).

    Ref:
    https://discuss.pytorch.org/t/differentiable-torch-histc/25865/9
    """
    def __init__(self, bins=256, min=0., max=1., sigma=1e5):
        """
        Init. function.
        :param bins: int. number of bins.
        :param min: int or float. minimum possible value.
        :param max: int or float. maximum possible value.
        :param sigma: float. the heat of the sigmoid. the higher the value,
        the sharper the sigmoid (~ the better the approximation).
        """
        super(SoftHistogram, self).__init__()

        assert bins != 0, "'bins' can not be 0."

        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = float(sigma)
        self.delta = float(max - min) / float(bins)
        self.register_buffer(
            "centers",
            float(min) + self.delta * (torch.arange(bins).float() + 0.5))

    def forward(self, x, mask=None):
        """
        Forward function.
        Computes histogram.
        :param x: vector, pytorch tensor. of shape (batch_size, n).
        :param mask: vector, pytorch tensor or None. same size as x. can be
        used to exclude some components from x. ideally, x should be in {0,
        1}. continuous values in [0, 1] are accepted.
        :return: pytorch tensor of shape (batch_size, bins).
        """
        assert x.ndim == 2, x.ndim

        x_shape = x.shape
        x = x.unsqueeze(1) - self.centers.unsqueeze(1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2.)) - torch.sigmoid(
            self.sigma * (x - self.delta/2.))
        # x shape: batch_size, bins, number of components in x.
        if mask is not None:
            msg = "shape mismatch x={}, mask={}.".format(x.shape, mask.shape)
            assert x_shape == mask.shape, msg
            # unsqueeze for the batch size.
            x = x * mask.unsqueeze(1)

        x = x.sum(dim=-1)
        return x


class GaussianKDE(nn.Module):
    """
    KDE for RGB images. Computes a KDE per plan.
    """
    def __init__(self,
                 kde_bw: float,
                 nbin: int = 128,
                 max_color: float = 255,
                 ndim: int = 3
                 ):
        super(GaussianKDE, self).__init__()

        assert isinstance(kde_bw, float)
        assert kde_bw > 0

        assert isinstance(ndim, int)
        assert ndim > 0

        assert isinstance(nbin, int)
        assert nbin > 0, nbin

        self.nbin = nbin
        self.max_color = max_color

        self.kde_bw = kde_bw
        self.ndim = ndim

        self._device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.color_space = self._get_color_space()

        self.const1 = torch.tensor((2. * math.pi * kde_bw)**(-1 / 2.),
                                   dtype=torch.float32, device=self._device,
                                   requires_grad=False)
        self.const2 = torch.tensor(2. * kde_bw, dtype=torch.float32,
                                   device=self._device, requires_grad=False)

    def _get_color_space(self) -> torch.Tensor:
        x = torch.linspace(start=0., end=self.max_color, steps=self.nbin,
                           device=self._device, dtype=torch.float32,
                           requires_grad=False)
        return x.view(-1, 1)  # nbin, 1

    def _get_px_one_img(self,
                        img: torch.Tensor,
                        mask: torch.Tensor = None) -> torch.Tensor:
        assert img.ndim == 3
        assert img.shape[0] == self.ndim

        if mask is not None:
            assert mask.ndim == 2
            assert mask.shape == img.shape[1:]

        dim, h, w = img.shape
        x: torch.Tensor = img.contiguous().view(dim, 1, h * w)

        ki = (x - self.color_space)**2  # ndim, nbin, h*w
        ki = ki / self.const2  # ndim, nbin, h*w
        ki = self.const1 * torch.exp(-ki)  # ndim, nbin, h*w

        if mask is not None:
            roi: torch.Tensor = mask.contiguous().view(1, 1, -1)  # 1, 1, h*w
            assert roi.shape[-1] == x.shape[-1]
            p = (roi * ki)  # ndim, nbin, h*w
            p = p.sum(dim=-1)  # ndim, nbin

            if roi.sum() != 0.:
                p = p / roi.sum()  # ndim, nbin

        else:
            p = ki  # ndim, nbin, h*w
            p = p.mean(dim=-1)  # ndim, nbin.

        assert p.shape == (self.ndim, self.nbin)

        return p

    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor = None) -> torch.Tensor:

        assert images.ndim == 4
        assert images.shape[1] == self.ndim

        if masks is not None:
            assert masks.ndim == 4
            assert masks.shape[1] == 1
            assert masks.shape[2:] == images.shape[2:]

        b, c, h, w = images.shape

        _p_ = None
        for i in range(b):
            if masks is not None:
                p_ = self._get_px_one_img(
                    img=images[i],
                    mask=masks[i].squeeze(0)
                )

            else:
                p_ = self._get_px_one_img(img=images[i], mask=None)

            if _p_ is None:
                _p_ = p_.unsqueeze(0)
            else:
                _p_ = torch.vstack((_p_, p_.unsqueeze(0)))

        assert _p_.shape == (b, self.ndim, self.nbin)

        return _p_

    def extra_repr(self) -> str:
        return f'kde_bw: {self.kde_bw}, nbin: {self.nbin}, ' \
               f'max_color:{self.max_color}, ndim: {self.ndim}'


def test_soft_histogram():
    """
    Test function: SoftHistogram().
    """
    set_seed(0)
    cuda = "0"
    print("cuda:{}".format(cuda))
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    bins = 256
    min = 0.
    max = 1.
    sigma = 1000000.  # 1e5
    m = SoftHistogram(bins=bins, min=min, max=max, sigma=sigma)
    m.to(DEVICE)

    batch_sz = 3
    x = torch.rand((batch_sz, 5)).to(DEVICE)

    mask = torch.ones((batch_sz, 5)).to(DEVICE)
    output = m(x, mask=mask)


    better_hist = None
    for i in range(batch_sz):
        if i == 0:
            better_hist = torch.histc(x[i, :], bins=bins, min=min,
                                      max=max).view(1, -1)
        else:
            better_hist = torch.cat(
                (better_hist,
                 torch.histc(x[i, :], bins=bins, min=min, max=max).view(1, -1)),
                dim=0
            )

    print(f"error {sigma}: {torch.abs(output - better_hist).sum(dim=1)}")


def test_gausian_kde():
    """
    Test function: GaussianKDE().
    """
    set_seed(0)
    cuda = 0
    torch.cuda.set_device(cuda)
    print("cuda:{}".format(cuda))
    device = torch.device(f'cuda:{torch.cuda.current_device()}')


    kde_bw = 10. / (255 ** 2)
    nbin = 256
    min = 0.
    ndim = 1
    max_color = 255 / 255.
    h = w = 64
    m = GaussianKDE(kde_bw=kde_bw,
                    nbin=nbin,
                    max_color=max_color,
                    ndim=ndim).to(device)

    batch_sz = 3
    x = torch.rand((batch_sz, ndim, h, w)).to(device)

    output = m(x)
    print(x.shape, output.shape)


if __name__ == "__main__":
    # test_soft_histogram()
    test_gausian_kde()