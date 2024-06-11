import sys
from os.path import dirname, abspath
from typing import Union, List

import torch.nn as nn
import torch
import numpy as np


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.loss.core import ElementaryLoss
from dlib.loss.ssim import SSIMLoss
from dlib.losses.elb import ELB
from dlib.utils import constants
from dlib.loss.local_terms import PatchMoments
from dlib.loss.local_variations import ImageGradient
from dlib.loss.local_variations import LaplacianFilter
from dlib.loss.local_variations import LocalVariation
from dlib.loss.global_terms import SoftHistogram
from dlib.loss.global_terms import GaussianKDE


__all__ = ['L1',
           'L2',
           'L2Sum',
           'Charbonnier',
           'NegativeSsim',
           'BoundedPrediction',
           'LocalMoments',
           'ImageGradientLoss',
           'LaplacianFilterLoss',
           'LocalVariationLoss',
           'NormImageGradientLoss',
           'NormLaplacianFilterLoss',
           'NormLocalVariationLoss',
           'HistogramMatch',
           'KDEMatch',
           'CrossEntropyL',
           'WeightsSparsityLoss'
           ]


class L1(ElementaryLoss):
    def __init__(self, **kwargs):
        super(L1, self).__init__(**kwargs)

        self.loss = nn.L1Loss(reduction='none').to(self._device)

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(L1, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        if trg_per_pixel_weight is None:
            return self.lambda_ * self.loss(_y_pred, _y_target).mean()

        else:

            self.sanity_check_trg_per_pixel_weight(trg_per_pixel_weight,
                                                   _y_target)

            loss = self.loss(_y_pred, _y_target)
            loss = loss * trg_per_pixel_weight

            return self.lambda_ * loss.mean()


class L2(ElementaryLoss):
    def __init__(self, **kwargs):
        super(L2, self).__init__(**kwargs)

        self.loss = nn.MSELoss(reduction='none').to(self._device)

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(L2, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        return self.lambda_ * self.loss(_y_pred, _y_target).mean()


class L2Sum(ElementaryLoss):
    def __init__(self, **kwargs):
        super(L2Sum, self).__init__(**kwargs)

        self.loss = nn.MSELoss(reduction='sum').to(self._device)

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(L2Sum, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        return self.lambda_ * self.loss(_y_pred, _y_target).mean()


class Charbonnier(ElementaryLoss):
    def __init__(self, **kwargs):
        super(Charbonnier, self).__init__(**kwargs)

        self.eps = 1e-9

    def set_eps(self, eps):
        assert eps > 0
        self.eps = eps

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(Charbonnier, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        diff = _y_target - _y_pred

        return self.lambda_ * torch.sqrt((diff * diff) + self.eps).mean()


class NegativeSsim(ElementaryLoss):
    def __init__(self, **kwargs):
        super(NegativeSsim, self).__init__(**kwargs)

        self.window_size = 11

        self.loss = SSIMLoss(window_size=self.window_size,
                             size_average=False).to(self._device)

    def set_window_size(self, window_size):
        assert isinstance(window_size, int)
        assert window_size > 0

        self.window_size = window_size

        self.loss = SSIMLoss(window_size=self.window_size,
                             size_average=False).to(self._device)

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(NegativeSsim, self).forward(epoch=epoch)

        assert not self.use_residuals

        if not self.is_on():
            return self._zero

        return - self.lambda_ * self.loss(y_pred, y_target).mean()


class BoundedPrediction(ElementaryLoss):
    def __init__(self, **kwargs):
        super(BoundedPrediction, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

        self.eps = 0.0
        self.eps_already_set = False

    def set_eps(self, eps: float):
        assert eps >= 0, eps
        assert isinstance(eps, float), type(eps)
        self.eps = eps

        self.eps_already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(BoundedPrediction, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        b, c, h, w = _y_pred.shape
        y_hat = _y_pred.contiguous().view(b, -1)
        y = _y_target.contiguous().view(b, -1)

        if self.restore_range:
            y_hat = y_hat * self.color_max
            y = y * self.color_max

        # y - e <= yhat <= y + e.
        diff_right = y_hat - (y + self.eps)
        diff_left = y - self.eps - y_hat
        loss = self.elb(diff_right.view(-1, ))
        loss = loss + self.elb(diff_left.view(-1, ))
        loss = loss / 2.

        return self.lambda_ * loss


class LocalMoments(ElementaryLoss):
    def __init__(self, **kwargs):
        super(LocalMoments, self).__init__(**kwargs)

        self.ksz: List[int] = [3]
        self.ksz_already_set = False

        self.patch_m = [
            PatchMoments(ksz=k, take_center_avg=False) for k in self.ksz]
        self.patch_m = [p.to(self._device) for p in self.patch_m]

        self.eps = 1.

    def set_ksz(self, ksz: List[int]):
        assert len(ksz) > 0, len(ksz)

        for k in ksz:
            assert k > 1, k
            assert isinstance(k, int), type(k)

        ksz.sort(reverse=False)  # ascendant.
        self.ksz = ksz

        self.ksz_already_set = True

    def kl_2_gaussians(self,
                       src_m: torch.Tensor,
                       src_v: torch.Tensor,
                       trg_m: torch.Tensor,
                       trg_v: torch.Tensor
                       ) -> torch.Tensor:
        # kl(p(sigma1, mu1): trg, q(sigma2, mu2): src)
        # b, hxw.
        assert src_m.ndim == 2, src_m.ndim

        assert src_v.shape == src_m.shape, f'{src_v.shape}, {src_m.shape}'
        assert trg_m.shape == src_m.shape, f'{trg_m.shape}, {src_m.shape}'
        assert trg_v.shape == src_m.shape, f'{trg_v.shape}, {src_m.shape}'

        _src_v = src_v + self.eps
        _trg_v = trg_v + self.eps

        src_std = torch.sqrt(_src_v)
        trg_std = torch.sqrt(_trg_v)

        kl = torch.log(src_std / trg_std)
        kl = kl + (_trg_v + (trg_m - src_m)**2) / (2 * _src_v)
        kl = kl - 0.5

        return kl

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(LocalMoments, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        loss = 0
        filter_smooth = None
        # discard patches with null variance using the smallest kernel sz.
        for i, m in enumerate(self.patch_m):

            src_m, src_v = m(_y_pred)
            trg_m, trg_v = m(_y_target)
            if i == 0:
                filter_smooth = (trg_v == 0).float()

            tmp = self.kl_2_gaussians(src_m=src_m,
                                      src_v=src_v,
                                      trg_m=trg_m,
                                      trg_v=trg_v)
            tmp = tmp * filter_smooth
            loss = loss + tmp.mean()

        return self.lambda_ * loss


class ImageGradientLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ImageGradientLoss, self).__init__(**kwargs)

        self.norm_str = constants.NORM2
        self.already_set = False

        self.op = ImageGradient().to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, norm_str: str):
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [constants.NORM2, constants.NORM1], norm_str

        self.norm_str = norm_str

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(ImageGradientLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        with torch.no_grad():
            trg = self.op(_y_target)

        pred = self.op(_y_pred)
        loss = self.norm(pred, trg).mean()

        return self.lambda_ * loss


class LaplacianFilterLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(LaplacianFilterLoss, self).__init__(**kwargs)

        self.norm_str = constants.NORM2
        self.already_set = False

        self.op = LaplacianFilter().to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, norm_str: str):
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [constants.NORM2, constants.NORM1], norm_str

        self.norm_str = norm_str

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(LaplacianFilterLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        with torch.no_grad():
            trg = self.op(_y_target)

        pred = self.op(_y_pred)
        loss = self.norm(pred, trg).mean()

        return self.lambda_ * loss


class LocalVariationLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(LocalVariationLoss, self).__init__(**kwargs)

        self.ksz: int = 3
        self.norm_str = constants.NORM2
        self.already_set = False

        self.op = LocalVariation(ksz=self.ksz).to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, ksz: int, norm_str: str):
        assert isinstance(ksz, int), type(ksz)
        assert ksz % 2 == 1, ksz
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [constants.NORM2, constants.NORM1], norm_str

        self.ksz = ksz
        self.norm_str = norm_str

        self.op = LocalVariation(ksz=self.ksz).to(self._device)

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(LocalVariationLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        with torch.no_grad():
            trg = self.op(_y_target)

        pred = self.op(_y_pred)
        loss = self.norm(pred, trg).mean()

        return self.lambda_ * loss



class NormImageGradientLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(NormImageGradientLoss, self).__init__(**kwargs)

        self.norm_str = constants.NORM2
        self.already_set = False

        self.op = ImageGradient().to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, norm_str: str):
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [constants.NORM2, constants.NORM1], norm_str

        self.norm_str = norm_str

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(NormImageGradientLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        with torch.no_grad():
            trg = self.op(_y_target)
            trg = trg.norm(p=2, dim=1, keepdim=True)  # l2 norm.

        pred = self.op(_y_pred)
        pred = pred.norm(p=2, dim=1, keepdim=True)  # l2 norm.
        loss = self.norm(pred, trg).mean()

        return self.lambda_ * loss


class NormLaplacianFilterLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(NormLaplacianFilterLoss, self).__init__(**kwargs)

        self.norm_str = constants.NORM2
        self.already_set = False

        self.op = LaplacianFilter().to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, norm_str: str):
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [constants.NORM2, constants.NORM1], norm_str

        self.norm_str = norm_str

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(NormLaplacianFilterLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        with torch.no_grad():
            trg = self.op(_y_target)
            trg = trg.norm(p=2, dim=1, keepdim=True)  # l2 norm.

        pred = self.op(_y_pred)
        pred = pred.norm(p=2, dim=1, keepdim=True)  # l2 norm.
        loss = self.norm(pred, trg).mean()

        return self.lambda_ * loss


class NormLocalVariationLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(NormLocalVariationLoss, self).__init__(**kwargs)

        self.ksz: int = 3
        self.norm_str = constants.NORM2
        self.already_set = False

        self.op = LocalVariation(ksz=self.ksz).to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, ksz: int, norm_str: str):
        assert isinstance(ksz, int), type(ksz)
        assert ksz % 2 == 1, ksz
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [constants.NORM2, constants.NORM1], norm_str

        self.ksz = ksz
        self.norm_str = norm_str

        self.op = LocalVariation(ksz=self.ksz).to(self._device)

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(NormLocalVariationLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        _y_pred, _y_target = self.switch_to_residuals(model, y_target, y_pred)

        assert _y_target.shape == _y_pred.shape, f'{_y_target.shape}, ' \
                                                 f'{_y_pred.shape}'

        with torch.no_grad():
            trg = self.op(_y_target)
            trg = trg.norm(p=2, dim=1, keepdim=True)  # l2 norm.

        pred = self.op(_y_pred)
        pred = pred.norm(p=2, dim=1, keepdim=True)  # l2 norm.
        loss = self.norm(pred, trg).mean()

        return self.lambda_ * loss


class Bhattacharyya(nn.Module):
    def __init__(self):
        super(Bhattacharyya, self).__init__()

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        assert p.ndim == 2, p.ndim  # b, d
        assert p.shape == q.shape, f'{p.shape} {q.shape}'

        out = torch.sqrt(p * q).sum(dim=1).view(-1, )

        return out


class HistogramMatch(ElementaryLoss):
    def __init__(self, **kwargs):
        super(HistogramMatch, self).__init__(**kwargs)

        self.norm_str: str = constants.NORM2
        self.sigma: float = 1e5
        self.already_set = False

        self.nbins = len(list(range(self.color_min, self.color_max))) + 1

        self.op = SoftHistogram(bins=self.nbins, min=0.0, max=1.,
                                sigma=self.sigma).to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, norm_str: str, sigma: float):
        assert isinstance(sigma, float), type(sigma)
        assert sigma > 0., sigma
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [
            constants.NORM2, constants.NORM1, constants.KL, constants.BH
        ], norm_str

        self.sigma = sigma
        self.norm_str = norm_str

        self.nbins = len(list(range(self.color_min, self.color_max))) + 1

        self.op = SoftHistogram(bins=self.nbins, min=0.0, max=1.,
                                sigma=self.sigma).to(self._device)

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        elif norm_str == constants.KL:
            self.norm = nn.KLDivLoss(reduction='batchmean',
                                     log_target=False).to(self._device)

        elif norm_str == constants.BH:
            assert isinstance(self.elb, ELB)

            self.norm = Bhattacharyya().to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(HistogramMatch, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert y_target.shape == y_pred.shape, f'{y_target.shape}, ' \
                                               f'{y_pred.shape}'

        b, _, _, _ = y_target.shape
        with torch.no_grad():

            trg = self.op(y_target.contiguous().view(b, -1))  # b, d
            assert trg.ndim == 2, trg.ndim

            trg = trg + 1.
            trg = trg / trg.sum(dim=-1).view(-1, 1)

        pred = self.op(y_pred.contiguous().view(b, -1))
        assert pred.ndim == 2, pred.ndim  # b, d
        pred = pred + 1.
        pred = pred / pred.sum(dim=-1).view(-1, 1)

        if self.norm_str == constants.KL:
            # todo: issue: generate nan pred after few iterations.
            loss = self.norm(pred.log(), trg).mean()

        elif self.norm_str == constants.BH:
            norm = self.norm(pred, trg)  # b,
            loss = self.elb(-norm)
        else:
            loss = self.norm(pred, trg).mean()


        return self.lambda_ * loss


class KDEMatch(ElementaryLoss):
    def __init__(self, **kwargs):
        super(KDEMatch, self).__init__(**kwargs)

        self.kde_bw = 1. / (255.**2)
        self.ndim = 1
        self.norm_str: str = constants.NORM2
        self.already_set = False

        assert self.color_min == 0, self.color_min
        assert self.color_max == 1, self.color_max

        self.nbins = len(np.arange(
            self.color_min, self.color_max, 1./255.).tolist()) + 1

        self.op = GaussianKDE(kde_bw=self.kde_bw, nbin=self.nbins,
                              max_color=self.color_max, ndim=self.ndim
                              ).to(self._device)

        self.norm = nn.MSELoss(reduction='none').to(self._device)

    def set_it(self, norm_str: str, kde_bw: float, ndim: int, nbins: int):
        assert isinstance(kde_bw, float), type(kde_bw)
        assert kde_bw > 0., kde_bw
        assert isinstance(norm_str, str), norm_str
        assert norm_str in [
            constants.NORM2, constants.NORM1, constants.BH
        ], norm_str

        assert isinstance(ndim, int), type(ndim)
        assert ndim > 0, ndim
        assert ndim == 1, ndim  # color images adds more computations.

        assert isinstance(nbins, int), type(nbins)
        assert nbins > 0, nbins

        self.kde_bw = kde_bw
        self.norm_str = norm_str
        self.ndim = ndim

        self.nbins = len(np.arange(
            self.color_min, self.color_max, 1. / 255.).tolist()) + 1

        self.op = GaussianKDE(kde_bw=self.kde_bw, nbin=self.nbins,
                              max_color=self.color_max, ndim=self.ndim
                              ).to(self._device)

        if norm_str == constants.NORM1:
           self.norm = nn.L1Loss(reduction='none').to(self._device)

        elif norm_str == constants.NORM2:
            self.norm = nn.MSELoss(reduction='none').to(self._device)

        elif norm_str == constants.BH:
            assert isinstance(self.elb, ELB)

            self.norm = Bhattacharyya().to(self._device)

        else:
            raise NotImplementedError(norm_str)


        self.already_set = True

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(KDEMatch, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert y_target.shape == y_pred.shape, f'{y_target.shape}, ' \
                                               f'{y_pred.shape}'

        b, _, _, _ = y_target.shape
        eps = 1e-4

        with torch.no_grad():

            trg = self.op(y_target)  # b, ndim, d
            assert trg.shape[1] == self.ndim, f'{trg.shape[1]} {self.ndim}'
            assert trg.shape[2] == self.nbins, f'{trg.shape[2]} {self.nbins}'
            assert trg.shape[0] == b, f'{trg.shape[0]} {b}'
            assert trg.shape[1] == 1, trg.shape[1]

            trg = trg.squeeze(1)  # b, d
            assert trg.ndim == 2, trg.ndim

            trg = trg + eps

        pred = self.op(y_pred)  # b, ndim, d
        assert pred.shape[1] == self.ndim, f'{pred.shape[1]} {self.ndim}'
        assert pred.shape[2] == self.nbins, f'{pred.shape[2]} {self.nbins}'
        assert pred.shape[0] == b, f'{pred.shape[0]} {b}'
        assert pred.shape[1] == 1, pred.shape[1]

        pred = pred.squeeze(1)  # b, d
        assert pred.ndim == 2, pred.ndim

        pred = pred + eps

        if self.norm_str == constants.BH:
            norm = self.norm(pred, trg)  # b,
            loss = self.elb(-norm)
        else:
            loss = self.norm(pred, trg).mean() / float(pred.shape[1])  #
            # avoid nan by averaging.

        return self.lambda_ * loss


class CrossEntropyL(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CrossEntropyL, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self._device)

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(CrossEntropyL, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero


        _y_pred = model.raw_segmentation
        assert not self.use_residuals, f'not expected to use residuals.'

        assert y_target.ndim ==4, y_target.ndim  # b, c, h, w
        assert _y_pred.ndim == 4, y_target.ndim  # b, nbr_colors, h, w

        # todo: support rgb.
        assert y_target.shape[1] == 1, f'{y_target.shape[1]} must be 1.'

        _y_target = (y_target * self.color_max).long()
        _y_target = _y_target.squeeze(1)  # b, h, w.


        loss = self.loss(_y_pred, _y_target)

        return self.lambda_ * loss.mean()


class WeightsSparsityLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(WeightsSparsityLoss, self).__init__(**kwargs)

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        super(WeightsSparsityLoss, self).forward(epoch=epoch)

        assert model is not None

        if not self.is_on():
            return self._zero

        l1loss = self._zero
        for w in model.parameters():
            l1loss = l1loss + torch.linalg.norm(w.view(-1), ord=1)

        return l1loss * self.lambda_