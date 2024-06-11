import sys
from os.path import dirname, abspath

import torch
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.kde import GaussianKDE
from dlib.kde import IterativeGaussianKDE

from dlib.utils import constants

__all__ = ['ColorDistDisentangle', 'MaskColorKDE', 'BhattacharyyaCoeffs',
           'KDE4Loss']


class ColorDistDisentangle(nn.Module):
    def __init__(self,
                 device,
                 kde_bw: float,
                 nbin: int = 128,
                 max_color: int = 255,
                 ndim: int = 3,
                 blocksz: int = 64,
                 itera: bool = True,
                 norm: int = constants.NORM2,
                 alpha: float = 5.):
        super(ColorDistDisentangle, self).__init__()

        self._device = device

        assert isinstance(itera, bool)

        self.itera = itera
        self.kde = {
            True: IterativeGaussianKDE,
            False: GaussianKDE
        }[itera](device=device,
                 kde_bw=kde_bw,
                 nbin=nbin,
                 max_color=max_color,
                 ndim=ndim,
                 blocksz=blocksz
                 ).to(device)

        self.metric = BhattacharyyaCoeffs(iter=itera, ndim=ndim).to(device)

        assert norm in constants.NORMS
        self.norm_type = norm
        self.fg_norm = None
        self.bg_norm = None

        assert isinstance(alpha, float)
        assert alpha > 0

        self.alpha = alpha

    def forward(self,
                images: torch.Tensor,
                masks_fg: torch.Tensor,
                masks_bg: torch.Tensor) -> torch.Tensor:

        _p_fg, _p_bg = self.kde(images=images,
                                masks_fg=masks_fg,
                                masks_bg=masks_bg)
        # iter: # b, nbin**ndim each
        # non-iter: b, ndim, nbin.

        # self.fg_norm = self._get_norm(_p_fg)
        # self.bg_norm = self._get_norm(_p_bg)

        return self.metric(p_fg=_p_fg, p_bg=_p_bg)

    def _get_norm(self, v: torch.Tensor):
        if self.itera:
            assert v.ndim == 2
            # b, nbin ** ndim
            b, z = v.shape
        else:
            # b, ndim, nbin.
            assert v.ndim == 3
            b, _, z = v.shape

        if self.norm_type == constants.NORM2:
            norm = (v ** 2).sum(dim=-1).sqrt() / float(z)
        elif self.norm_type == constants.NORM1:
            norm = v.abs().sum(dim=-1) / float(z)
        elif self.norm_type == constants.NORM0EXP:
            norm = (1. - torch.exp(-self.alpha * v.abs())
                    ).sum(dim=-1) / float(z)

        if self.itera:
            assert norm.ndim == 1
            norm = norm.contiguous().view(-1, 1)  # b, 1
        else:
            assert norm.ndim == 2  # b, ndim

        return norm


class MaskColorKDE(nn.Module):
    """
    Compute color KDE over an image using a mask.
    """
    def __init__(self,
                 device,
                 kde_bw: float,
                 nbin: int = 128,
                 max_color: int = 255,
                 ndim: int = 3,
                 blocksz: int = 64,
                 itera: bool = True):
        super(MaskColorKDE, self).__init__()

        self._device = device

        assert isinstance(itera, bool)
        print(ndim)

        self.itera = itera
        self.kde = {
            True: IterativeGaussianKDE,
            False: GaussianKDE
        }[itera](device=device,
                 kde_bw=kde_bw,
                 nbin=nbin,
                 max_color=max_color,
                 ndim=ndim,
                 blocksz=blocksz
                 ).to(device)

    def forward(self,
                images: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:

        p = self.kde(images=images, masks=masks)
        # itera: b, nbin**ndim
        # not-ietra: b, ndim, nbin
        return p


class KDE4Loss(nn.Module):
    def forward(self, images: torch.Tensor,
                masks: torch.Tensor, kde: MaskColorKDE) -> list:
        if kde.itera:
            return [
                kde(images=self._build_pair(images, 0, 1), masks=masks),
                kde(images=self._build_pair(images, 0, 2), masks=masks),
                kde(images=self._build_pair(images, 1, 2), masks=masks)
            ]
        else:
            return [kde(images=images, masks=masks)]

    def _build_pair(self, img: torch.Tensor, i: int, j: int) -> torch.Tensor:
        assert img.ndim == 4
        return torch.cat((img[:, i, :, :].unsqueeze(1),
                          img[:, j, :, :].unsqueeze(1)), dim=1)


class BhattacharyyaCoeffs(nn.Module):
    def __init__(self, itera: bool = True, ndim: int = 3):
        super(BhattacharyyaCoeffs, self).__init__()

        self.itera = itera
        self.ndim = ndim

    def forward(self, p_fg: torch.Tensor, p_bg: torch.Tensor) -> torch.Tensor:
        if self.itera:
            # batchs, nbin
            self._assert_iter(p_fg, p_bg)

            out = (p_fg * p_bg).sqrt().sum(dim=-1).view(-1, 1)  # batchs, 1
            return out

        else:
            # batchs, ndim==n color plans, nbin
            self._assert_non_iter(p_fg, p_bg)

            out = (p_fg * p_bg).sqrt().sum(dim=-1)  # batchs, ndim
            assert out.shape == p_fg.shape[:2]

        return out

    def _assert_iter(self, p_fg: torch.Tensor, p_bg: torch.Tensor):
        assert self.itera
        # batchs, nbin
        assert p_fg.ndim == 2, p_fg.ndim
        assert p_bg.ndim == 2, p_bg.ndim
        assert p_fg.shape == p_bg.shape

    def _assert_non_iter(self, p_fg: torch.Tensor, p_bg: torch.Tensor):
        assert not self.itera
        # batchs, ndim==n color plans, nbin
        assert p_fg.ndim == self.ndim, p_fg.ndim
        assert p_bg.ndim == self.ndim, p_bg.ndim
        assert p_fg.shape == p_bg.shape



def test_ColorDistDisentangle():
    seed = 0
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    cuda = "0"
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    kde_bw = 1.
    nbin = 128
    max_color = 255
    ndim = 2
    h = int(224 / 2)
    w = int(224 / 2)
    b = 32

    d = 20

    path_imng = join(root_dir, 'data/debug/input',
                     'Black_Footed_Albatross_0002_55.jpg')
    img = Image.open(path_imng, 'r').convert('RGB').resize(
        (w, h), resample=Image.BICUBIC)
    image = np.array(img, dtype=np.float32)  # h, w, 3
    image = image.transpose(2, 0, 1)  # 3, h, w
    image = torch.tensor(image, dtype=torch.float32)  # 3, h, w
    image = image[0:2, :, :]
    assert image.shape[0] == ndim

    images = image.repeat(b, 1, 1, 1).to(device)
    mask_fg = torch.zeros((h, w), dtype=torch.float32, device=device,
                          requires_grad=True) * 0.
    mask_fg[int(h/2.) - d: int(h/2.) + d, int(w/2.) - d: int(w/2.) + d] = 1.
    mask_bg = 1. - mask_fg

    masks_fg = mask_fg.repeat(b, 1, 1, 1)
    masks_bg = mask_bg.repeat(b, 1, 1, 1)

    disen = ColorDistDisentangle(device=device,
                                 kde_bw=kde_bw,
                                 nbin=nbin,
                                 max_color=max_color,
                                 ndim=ndim,
                                 itera=True)

    announce_msg("testing {}".format(disen))
    set_seed(seed=seed)

    t0 = dt.datetime.now()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with autocast(enabled=False):
        t0 = dt.datetime.now()
        dist = disen(images=images, masks_fg=masks_fg, masks_bg=masks_bg)
        t1 = dt.datetime.now()


    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    t1 = dt.datetime.now()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'time op: {elapsed_time_ms} (batchsize: {b}, h*w: {h}*{w})')
    print(f'time: {t1 - t0}')
    print(dist)
    print(dist.shape)


if __name__ == '__main__':
    from dlib.utils.shared import announce_msg
    from dlib.utils.utils_reproducibility import set_seed

    from os.path import join
    import datetime as dt

    import numpy as np
    from PIL import Image
    from torch.cuda.amp import autocast

    test_ColorDistDisentangle()
