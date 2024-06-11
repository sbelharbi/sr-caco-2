import sys
from os.path import dirname, abspath
from typing import Callable, Tuple

import torch
import torch.nn as nn
import numpy as np

from kornia.morphology import dilation
from kornia.morphology import erosion

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.cams.core_seeding import STOtsu

from dlib.utils import constants

__all__ = ['TCAMSeeder']


class TCAMSeeder(nn.Module):
    def __init__(self,
                 seed_tech: str,
                 min_: int,
                 max_: int,
                 max_p: float,
                 min_p: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int,
                 cuda_id: int
                 ):
        super(TCAMSeeder, self).__init__()

        assert seed_tech in constants.SEED_TECHS, seed_tech
        self.seed_tech = seed_tech

        assert not multi_label_flag

        assert isinstance(cuda_id, int)
        assert cuda_id >= 0, cuda_id
        self._device = torch.device(cuda_id)

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

        assert isinstance(max_p, float)
        assert 0. <= max_p <= 1.
        self.max_p = max_p

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

    def forward(self, x: torch.Tensor,
                roi_thresh: torch.Tensor = None) -> torch.Tensor:

        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        if roi_thresh is not None:
            assert isinstance(roi_thresh, torch.Tensor)
            assert roi_thresh.ndim == 1
            assert roi_thresh.shape[0] == x.shape[0]

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
        assert d == 1  # todo multilabel.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        nbr_bg = int(self.min_p * h * w)

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _OneSample(min_p=self.min_p,
                         max_p=self.max_p,
                         min_=self.min_,
                         max_=self.max_,
                         seed_tech=self.seed_tech
                         )

        for i in range(b):
            thresh = None
            if roi_thresh is not None:
                thresh = roi_thresh[i]
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze(), erode=erode,
                                       thresh=thresh)

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
        return f'min_={self.min_}, max_={self.max_}, min_p={self.min_p},' \
               f'max_p={self.max_p}, ksz={self.ksz}, fg_erode_k: ' \
               f'{self.fg_erode_k}, fg_erode_iter: {self.fg_erode_iter}' \
               f'support_background={self.support_background},' \
               f'multi_label_flag={self.multi_label_flag}, ' \
               f'seg_ignore_idx={self.ignore_idx}, seed_tech={self.seed_tech}'


class _OneSample(nn.Module):
    def __init__(self,
                 min_p: float,
                 max_p: float,
                 min_: int,
                 max_: int,
                 seed_tech: str):
        super(_OneSample, self).__init__()

        self.otsu = STOtsu()
        self.fg_capture = _SFG(max_p=max_p, max_=max_, seed_tech=seed_tech)
        self.bg_capture = _SBG(min_p=min_p, min_=min_, seed_tech=seed_tech)

    def forward(self,
                cam: torch.Tensor,
                erode: Callable[[torch.Tensor], torch.Tensor],
                thresh: float = None
                ) -> Tuple[
        torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)
        bg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        cam_ = torch.floor(cam * 255.)

        th = thresh
        if th is None:
            th = self.otsu(x=cam_)

        if th == 0:
            th = 1.
        if th == 255:
            th = 254.

        if cam.min() == cam.max():
            return fg, bg

        # ROI
        roi = (cam_ > th).long()
        roi = erode(roi.unsqueeze(0).unsqueeze(0)).squeeze()

        fg = self.fg_capture(cam=cam, roi=roi, fg=fg)
        bg = self.bg_capture(cam=cam, bg=bg)
        return fg, bg


class _SFG(nn.Module):
    def __init__(self, max_p: float, max_: int, seed_tech: str):
        super(_SFG, self).__init__()

        self.max_ = max_
        self.max_p = max_p
        self.seed_tech = seed_tech

    def forward(self,
                cam: torch.Tensor,
                roi: torch.Tensor,
                fg: torch.Tensor) -> torch.Tensor:

        assert cam.shape == roi.shape
        h, w = cam.shape

        n = roi.sum()

        n = int(self.max_p * n)
        _cam = cam * roi
        _cam = _cam + 1e-8

        _cam_flatten = _cam.view(h * w)

        val, idx_ = torch.sort(_cam_flatten, dim=0, descending=True,
                               stable=True)

        if (n > 0) and (self.max_ > 0):

            tmp = _cam_flatten * 0.
            tmp[idx_[:n]] = 1
            tmp = tmp.view(h, w)

            _idx = torch.nonzero(tmp, as_tuple=True)  # (idx, idy)

            if self.seed_tech == constants.SEED_UNIFORM:
                probs = torch.ones(n, dtype=torch.float, device=cam.device)

            elif self.seed_tech == constants.SEED_WEIGHTED:
                probs = _cam[_idx[0], _idx[1]]  # 1d array. numel: n
                assert probs.numel() == n

            else:
                raise NotImplementedError(self.seed_tech)

            selected = probs.multinomial(
                num_samples=min(self.max_, n), replacement=False)

            fg[_idx[0][selected], _idx[1][selected]] = 1

        return fg


class _SBG(nn.Module):
    def __init__(self, min_p: float, min_: int, seed_tech: str):
        super(_SBG, self).__init__()

        self.min_ = min_
        self.min_p = min_p
        self.seed_tech = seed_tech

    def forward(self, cam: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape

        n = int(self.min_p * h * w)

        _cam = cam + 1e-8
        _cam_flatten = _cam.view(h * w)
        val, idx_ = torch.sort(_cam_flatten, dim=0, descending=False,
                               stable=True)

        if (n > 0) and (self.min_ > 0):

            tmp = _cam_flatten * 0.
            tmp[idx_[:n]] = 1
            tmp = tmp.view(h, w)

            _idx = torch.nonzero(tmp, as_tuple=True)  # (idx, idy)

            if self.seed_tech == constants.SEED_UNIFORM:
                probs = torch.ones(n, dtype=torch.float, device=cam.device)

            elif self.seed_tech == constants.SEED_WEIGHTED:
                probs = 1. - _cam[_idx[0], _idx[1]]  # 1d array. numel: n
                probs = torch.relu(probs) + 1e-8
                assert probs.numel() == n

            else:
                raise NotImplementedError(self.seed_tech)

            selected = probs.multinomial(
                num_samples=min(self.min_, n),
                replacement=False)
            bg[_idx[0][selected], _idx[1][selected]] = 1

        return bg


def test_TCAMSeeder():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import time
    from os.path import join

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'y'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return (_cam.squeeze().cpu().numpy() * 255).astype(np.uint8)

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
            if tag == 'CAM':
                axes[0, i].imshow(im, cmap='jet')
            else:
                axes[0, i].imshow(im, cmap=get_cm())

            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    device = torch.device(f'cuda:{cuda}')
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 1000
    max_ = 1000
    min_p = 0.1
    max_p = .4
    fg_erode_k = 11
    fg_erode_iter = 0
    ksz = 3

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda'), requires_grad=False)
    cam = cam * 0

    path_cam = join(root_dir,
                    'data/debug/input/train_n02229544_n02229544_2648.JPEG.npy')
    cam = np.load(path_cam).squeeze()
    cam: torch.Tensor = torch.from_numpy(cam).to(device).float()
    cam.requires_grad = False
    cam = cam.view(1, 224, 224).repeat(batchs, 1, 1, 1)

    # for i in range(batchs):
    #     cam[i, 0, 100:150, 100:150] = 1

    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for seed_tech in [constants.SEED_UNIFORM, constants.SEED_WEIGHTED]:
        set_seed(seed)
        module_conc = TCAMSeeder(
            seed_tech=seed_tech,
            min_=min_,
            max_=max_,
            min_p=min_p,
            max_p=max_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255,
            cuda_id=cuda)
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
                               'S_{}'.format(seed_tech)))

    if batchs == 1:
        plot_limgs(limgs_conc, 'Seeding techs')


if __name__ == '__main__':
    from dlib.utils.shared import announce_msg
    from dlib.utils.utils_reproducibility import set_seed

    set_seed(0)
    test_TCAMSeeder()
