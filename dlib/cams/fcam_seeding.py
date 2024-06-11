import math
import operator
import sys
import os
from os.path import dirname, abspath
import time
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from skimage.util.dtype import dtype_range

from kornia.morphology import dilation
from kornia.morphology import erosion
from skimage.filters import threshold_otsu
from skimage import filters

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.cams.core_seeding import STOtsu

__all__ = ['GetFastSeederSLFCAMS',
           'MBSeederSLFCAMS',
           'SeederCBOX'
           ]


class _STFG(nn.Module):
    def __init__(self, max_: int):
        super(_STFG, self).__init__()
        self.max_ = max_

    def forward(self, roi: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        # roi: h,w
        idx_fg = torch.nonzero(roi, as_tuple=True)  # (idx, idy)
        n_fg = idx_fg[0].numel()
        if (n_fg > 0) and (self.max_ > 0):
            probs = torch.ones(n_fg, dtype=torch.float)
            selected = probs.multinomial(
                num_samples=min(self.max_, n_fg), replacement=False)
            fg[idx_fg[0][selected], idx_fg[1][selected]] = 1

        return fg


class _STBG(nn.Module):
    def __init__(self, nbr_bg, min_):
        super(_STBG, self).__init__()

        self.nbr_bg = nbr_bg
        self.min_ = min_

    def forward(self, cam: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2
        # cam: h, w
        h, w = cam.shape
        val, idx_bg_ = torch.sort(cam.view(h * w), dim=0, descending=False)

        tmp = torch.zeros_like(bg)
        if self.nbr_bg > 0:
            tmp = tmp.view(h * w)
            tmp[idx_bg_[:self.nbr_bg]] = 1
            tmp = tmp.view(h, w)

            idx_bg = torch.nonzero(tmp, as_tuple=True)  #
            # (idx, idy)
            n_bg = idx_bg[0].numel()
            if (n_bg > 0) and (self.min_ > 0):
                probs = torch.ones(n_bg, dtype=torch.float)
                selected = probs.multinomial(
                    num_samples=min(self.min_, n_bg),
                    replacement=False)
                bg[idx_bg[0][selected], idx_bg[1][selected]] = 1

        return bg


class _STOneSample(nn.Module):
    def __init__(self, min_, max_, nbr_bg):
        super(_STOneSample, self).__init__()

        self.min_ = min_
        self.max_ = max_

        self.otsu = STOtsu()
        self.fg_capture = _STFG(max_=max_)
        self.bg_capture = _STBG(nbr_bg=nbr_bg, min_=min_)

    def forward(self, cam: torch.Tensor,
                erode: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)
        bg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        # otsu
        cam_ = torch.floor(cam * 255)
        th = self.otsu(x=cam_)

        if th == 0:
            th = 1.
        if th == 255:
            th = 254.

        if self.otsu.bad_egg:
            return fg, bg

        # ROI
        roi = (cam_ > th).long()
        roi = erode(roi.unsqueeze(0).unsqueeze(0)).squeeze()

        fg = self.fg_capture(roi=roi, fg=fg)
        bg = self.bg_capture(cam=cam, bg=bg)
        return fg, bg


class _CBOXOneSample(nn.Module):
    def __init__(self, n: int, bg_low_z: float, bg_up_z: float):
        super(_CBOXOneSample, self).__init__()

        assert isinstance(n, int)
        assert n > 0

        assert isinstance(bg_low_z, float)
        assert isinstance(bg_up_z, float)
        assert 0. <= bg_low_z <= 1.
        assert 0. <= bg_up_z <= 1.
        assert bg_low_z <= bg_up_z

        self.bg_low_z = bg_low_z
        self.bg_up_z = bg_up_z

        self.n = n

        self.otsu = STOtsu()

    def forward(self,
                cam: torch.Tensor,
                erode: Callable[[torch.Tensor], torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        bg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        # otsu
        cam_ = torch.floor(cam * 255)
        th = self.otsu(x=cam_)

        if self.otsu.bad_egg:
            th = torch.median(cam * 255)

        if th == 0:
            th = 1.
        if th == 255:
            th = 254.

        # ROI
        roi = (cam_ > th).long()
        roi = erode(roi.unsqueeze(0).unsqueeze(0)).squeeze()  # h, w

        fg = self.random_fg(roi=roi, fg=fg)
        bg = self.random_bg(cam=cam, bg=bg)
        return fg, bg

    def random_fg(self, roi: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        # roi: h,w
        idx_fg = torch.nonzero(roi, as_tuple=True)  # (idx, idy)
        n_fg = idx_fg[0].numel()

        if self.n > 0:
            probs = torch.ones(n_fg, dtype=torch.float)
            selected = probs.multinomial(
                num_samples=min(self.n, n_fg), replacement=False)
            fg[idx_fg[0][selected], idx_fg[1][selected]] = 1

        return fg

    def random_bg(self, cam: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2
        # cam: h, w
        h, w = cam.shape
        val, idx_bg_ = torch.sort(cam.view(h * w), dim=0, descending=False)

        tmp = torch.zeros_like(bg)
        z = self.random_z()
        nbr_bg: int = min(math.ceil(z * h * w), h * w)

        if self.n > 0:
            tmp = tmp.view(h * w)
            tmp[idx_bg_[:nbr_bg]] = 1
            tmp = tmp.view(h, w)

            idx_bg = torch.nonzero(tmp, as_tuple=True)  #
            # (idx, idy)
            probs = torch.ones(nbr_bg, dtype=torch.float)
            selected = probs.multinomial(
                num_samples=min(self.n, nbr_bg),
                replacement=False)
            bg[idx_bg[0][selected], idx_bg[1][selected]] = 1

        return bg

    def random_z(self):
        return np.random.uniform(low=self.bg_low_z,
                                 high=self.bg_up_z, size=(1, )).item()


class MBSeederSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int,
                 max_: int,
                 min_p: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int
                 ):
        super(MBSeederSLFCAMS, self).__init__()

        assert not multi_label_flag

        self._device = torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

    def mb_erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = x

        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def mb_dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.mb_erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        nbr_bg = int(self.min_p * h * w)

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _STOneSample(min_=self.min_, max_=self.max_, nbr_bg=nbr_bg)

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze(), erode=erode)

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, min_p: {}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'support_background={},' \
               'multi_label_flag={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.min_p, self.fg_erode_k,
                self.fg_erode_iter,
                self.support_background,
                self.multi_label_flag, self.ignore_idx)


class SeederCBOX(nn.Module):
    def __init__(self,
                 n: int,
                 bg_low_z: float,
                 bg_up_z: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 seg_ignore_idx: int,
                 device
                 ):
        super(SeederCBOX, self).__init__()

        self._device = device

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(n, int)
        assert n > 1
        self.n = n

        # bg
        assert isinstance(bg_low_z, float)
        assert isinstance(bg_up_z, float)
        assert 0. <= bg_low_z <= 1.
        assert 0. <= bg_up_z <= 1.
        assert bg_low_z <= bg_up_z
        self.bg_low_z = bg_low_z
        self.bg_up_z = bg_up_z

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.ignore_idx = seg_ignore_idx

    def mb_erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = x

        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def mb_dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.mb_erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        opx = _CBOXOneSample(n=self.n, bg_low_z=self.bg_low_z,
                             bg_up_z=self.bg_up_z)

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze(), erode=erode)

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)  # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)  # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # fg/bg
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()  # b, h, w.

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return f'fg_erode_k: {self.fg_erode_k}, ' \
               f'fg_erode_iter: {self.fg_erode_iter}, ' \
               f'pfg: {self.pfg}, pbg: {self.pbg}, ' \
               f'bg_low_z: {self.bg_low_z}, ' \
               f'bg_up_z: {self.bg_up_z}, ' \
               f'ignore index: {self.ignore_idx}'


class GetFastSeederSLFCAMS(nn.Module):
    def __init__(self,
                 min_: int,
                 max_: int,
                 min_p: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int
                 ):
        super(GetFastSeederSLFCAMS, self).__init__()
        assert not multi_label_flag

        self._device = torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

    def erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = x
        tmp = x
        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)
            if out.sum() == 0:
                out = tmp
                break
            else:
                tmp = out

        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[0] == 1
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)
        assert 0 <= torch.min(out) <= 1
        assert 0 <= torch.max(out) <= 1
        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.
        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device)

        nbr_bg = int(self.min_p * h * w)

        for i in range(b):
            cam = x[i].squeeze()  # h, w
            t0 = time.perf_counter()
            # cant send to cpu. too expensive.*******************************
            cam_img = (cam.cpu().detach().numpy() * 255).astype(np.uint8)
            # print('time to cpu {}'.format(time.perf_counter() - t0))
            _bad_egg = False

            fg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)

            bg = torch.zeros((h, w), dtype=torch.long, device=self._device,
                             requires_grad=False)

            if cam_img.min() == cam_img.max():
                _bad_egg = True

            if not _bad_egg:
                import datetime as dt
                t0 = dt.datetime.now()
                # convert to gpu + batch. ******************************
                otsu_thresh = threshold_otsu(cam_img)
                # print('otsu {}'.format(dt.datetime.now() - t0))

                if otsu_thresh == 0:
                    otsu_thresh = 1
                if otsu_thresh == 255:
                    otsu_thresh = 254

                # GPU + BATCH *************************************************
                ROI = torch.from_numpy(cam_img > otsu_thresh).to(self._device)
                # GPU + BATCH *************************************************
                ROI = erode(ROI.unsqueeze(0).unsqueeze(0) * 1).squeeze()

                # fg
                idx_fg = torch.nonzero(ROI, as_tuple=True)  # (idx, idy)
                n_fg = idx_fg[0].numel()
                if n_fg > 0:
                    if self.max_ > 0:
                        probs = torch.ones(n_fg, dtype=torch.float)
                        selected = probs.multinomial(
                            num_samples=min(self.max_, n_fg), replacement=False)
                        fg[idx_fg[0][selected], idx_fg[1][selected]] = 1
                        # xxxxxxxxxxxxxxxxxxxx
                        fg = dilate(fg.view(1, 1, h, w)).squeeze()

                # bg
                val, idx_bg_ = torch.sort(cam.view(h * w), dim=0,
                                          descending=False)
                tmp = bg * 1.
                if nbr_bg > 0:
                    tmp = tmp.view(h * w)
                    tmp[idx_bg_[:nbr_bg]] = 1
                    tmp = tmp.view(h, w)

                    idx_bg = torch.nonzero(tmp, as_tuple=True)  #
                    # (idx, idy)
                    n_bg = idx_bg[0].numel()
                    if n_bg >= 0:
                        if self.min_ > 0:
                            probs = torch.ones(n_bg, dtype=torch.float)
                            selected = probs.multinomial(
                                num_samples=min(self.min_, n_bg),
                                replacement=False)
                            bg[idx_bg[0][selected], idx_bg[1][selected]] = 1
                            # xxxxxxxxxxxxxxxx
                            bg = dilate(bg.view(1, 1, h, w)).squeeze()

            # all this is gpu batchable.
            # sanity
            outer = fg + bg
            fg[outer == 2] = 0
            bg[outer == 2] = 0

            seeds = torch.zeros((h, w), dtype=torch.long, device=self._device,
                                requires_grad=False) + self.ignore_idx

            seeds[fg == 1] = 1
            seeds[bg == 1] = 0

            out[i] = seeds.detach().clone()

        assert out.dtype == torch.long
        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, min_p: {}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'support_background={},' \
               'multi_label_flag={}, seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.min_p, self.fg_erode_k,
                self.fg_erode_iter,
                self.support_background,
                self.multi_label_flag, self.ignore_idx)


def test_Linear_vs_Conc_SeederSLFCAMS():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 10
    max_ = 10
    min_p = .2
    fg_erode_k = 11
    fg_erode_iter = 1

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda'), requires_grad=False)

    cam = cam * 0
    for i in range(batchs):
        cam[i, 0, 100:150, 100:150] = 1
    limgs_lin = [(cam_2Img(cam), 'CAM')]
    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for ksz in [3]:
        set_seed(seed)
        module_linear = GetFastSeederSLFCAMS(
            min_=min_,
            max_=max_,
            min_p=min_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255)
        announce_msg('Testing {}'.format(module_linear))

        # cProfile.runctx('module_linear(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_linear(cam)
        print('time LINEAR: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_linear.ignore_idx] = 0

        if batchs == 1:
            limgs_lin.append((out.squeeze().cpu().numpy().astype(np.uint8),
                              'pseudo_ksz_{}_linear'.format(ksz)))

    for ksz in [3]:
        set_seed(seed)
        module_conc = MBSeederSLFCAMS(
            min_=min_,
            max_=max_,
            min_p=min_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255)
        announce_msg('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam)

        print('time CONC: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'pseudo_ksz_{}_conc'.format(ksz)))

    if batchs == 1:
        plot_limgs(limgs_lin, 'LINEAR')
        plot_limgs(limgs_conc, 'CONCURRENT')


def test_SeederCBOX():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = "0"
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    seed = 0
    pfg = .001
    pbg = .009
    bg_low_z = .5
    bg_up_z = .6
    ksz = 1
    fg_erode_k = 11
    fg_erode_iter = 1
    seg_ignore_idx = -255

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=device, requires_grad=False)

    cam = cam * 0
    for i in range(batchs):
        cam[i, 0, 100:150, 100:150] = 1
    limgs = [(cam_2Img(cam), 'CAM')]

    set_seed(seed)
    fg_seeder = SeederCBOX(pfg=pfg,
                           pbg=pbg,
                           bg_low_z=bg_low_z,
                           bg_up_z=bg_up_z,
                           fg_erode_k=fg_erode_k,
                           fg_erode_iter=fg_erode_iter,
                           ksz=ksz,
                           device=device,
                           seg_ignore_idx=seg_ignore_idx)
    announce_msg('Testing {}'.format(fg_seeder))

    t0 = time.perf_counter()
    out = fg_seeder(cam)
    print('time fg: {}s'.format(time.perf_counter() - t0))

    out[out == 1] = 255
    out[out == 0] = int(255 / 2)
    out[out == seg_ignore_idx] = 0

    if batchs == 1:
        limgs.append((out.squeeze().cpu().numpy().astype(np.uint8), 'FG/BG'))

    if batchs == 1:
        plot_limgs(limgs, 'FG/BG')


if __name__ == "__main__":
    import datetime as dt

    from dlib.utils.shared import announce_msg
    from dlib.utils.utils_reproducibility import set_seed

    # set_seed(0)
    # test_Linear_vs_Conc_SeederSLFCAMS()

    set_seed(0)
    test_SeederCBOX()

