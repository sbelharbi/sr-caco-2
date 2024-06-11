import copy
import os
import random
import sys
from os.path import dirname, abspath, join
from typing import Tuple, List, Union
import math

import numpy as np
import cv2
import torch.utils.data as data
from tqdm import tqdm
import torch
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mlt_patches
import matplotlib.colors as mcolors
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt as edt_fn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.utils.utils_image as util
from dlib.utils import constants
from dlib.diagnosis.stats_numpy import unnormed_histogram
from dlib.utils.utils_image import get_cell_type
from dlib.loss.local_terms import PatchMoments
from dlib.utils.utils_image import get_cell_type
from dlib.utils.shared import reformat_id

from dlib.utils.utils_reproducibility import set_seed


__all__ = ['DatasetDPSR']

# path_l[key] = {abs_path: val, low_path_key: val}


def _build_img(x: np.ndarray, cell_type: str) -> Image.Image:
    assert isinstance(x, np.ndarray), type(x)
    assert x.ndim == 2, x.ndim  # HW
    img = np.expand_dims(x, axis=2)  # HxWx1
    img = np.repeat(img, 3, axis=2)  # HW3: RGB

    raise ValueError

    img = Image.fromarray(img, mode='RGB')

    return img


def _debug_save_img(img: np.ndarray, cell_type: str, file_path: str):
    img = _build_img(img, cell_type)
    img.save(file_path)


class Augment(object):
    def __init__(self,
                 upscale: int,
                 nbr_steps: int,
                 use_roi: bool,
                 color_min: int,
                 color_max: int
                 ):
        super(Augment, self).__init__()

        assert isinstance(upscale, int), type(upscale)
        assert upscale > 0, upscale

        assert isinstance(nbr_steps, int), type(nbr_steps)
        assert nbr_steps > 0, nbr_steps

        assert isinstance(use_roi, bool), type(use_roi)

        assert isinstance(color_min, int), type(color_min)
        assert isinstance(color_max, int), type(color_max)
        assert color_min < color_max, f'{color_min} {color_max}'

        self.upscale = upscale
        self.nbr_steps = nbr_steps
        self.use_roi = use_roi

        self.color_min = color_min
        self.color_max = color_max

        if nbr_steps == 1:
            self.scales = [1.]
        else:
            self.scales: list = np.arange(
                1., nbr_steps, upscale / nbr_steps).tolist()

            self.scales.append(float(upscale))

    def generate_noise(self,
                       x: torch.Tensor,
                       avg: torch.Tensor,
                       vari: torch.Tensor,
                       v: torch.Tensor
                       ) -> Tuple[Union[torch.Tensor, None],
                                  Union[torch.Tensor, None]]:
        assert x.ndim == 2, x.ndim
        assert avg.ndim == 1, avg.ndim
        assert vari.ndim == 1, vari.ndim

        v_var = vari[avg == v]
        v_var, _ = torch.sort(v_var, dim=0, descending=False)

        if v_var.numel() == 0:  # if avg has no matching values in x.
            return None, None

        v_var, var_count = torch.unique(v_var, sorted=False,
                                        return_inverse=False,
                                        return_counts=True, dim=None)

        selector = x == v
        n = selector.sum()
        # probs = torch.ones(n, dtype=torch.float)
        var_count = var_count + 1.
        probs = (var_count / var_count.sum()).float()
        selected = probs.multinomial(num_samples=n, replacement=True)
        selected_vars = v_var[selected]
        centers = selected_vars * 0
        noise = torch.normal(mean=centers, std=torch.sqrt(selected_vars))

        return noise, selector

    def round_it(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.round(x)
        out = torch.clamp(out, min=self.color_min, max=self.color_max)
        return out

    def add_noise(self,
                  x_src: torch.Tensor,
                  x_trg: torch.Tensor,
                  roi: torch.Tensor = None
                  ) -> torch.Tensor:
        """
        Add random noise to x_srx.
        :param x_src: low res hxw. float. torch.Tensor.
        :param x_trg: high res hxw. float. torch.Tensor.
        :param roi: None or torch.Tensor hxw. computed over low res after being
        interpolated to high res. has same shape as high res. binary.
        :return: torch.tensor. hxw. float. == x_src + random_noise (over
        only roi if roi is not none, else everywhere)
        """

        assert x_src.dtype == torch.float32, x_src.dtype
        assert x_trg.dtype == torch.float32, x_trg.dtype

        assert x_src.shape == x_trg.shape, f'{x_src.shape} {x_trg.shape}'

        assert x_src.ndim == 2, x_src.ndim
        assert x_trg.ndim == 2, x_trg.ndim

        if roi is not None:
            assert roi.ndim == 2, roi.ndim
            assert roi.shape == x_src.shape, f'{roi.shape} {x_src.shape}'
            assert roi.dtype == torch.float32, roi.dtype

        # device = torch.device(f'cuda:{torch.cuda.current_device()}')
        device = torch.device('cpu')
        patcher = PatchMoments(ksz=3, take_center_avg=True).to(device)
        avg, vari = patcher(x_trg.unsqueeze(0).unsqueeze(0))  # 1, h*w; each.
        avg = avg.squeeze()  # h*w.
        vari = vari.squeeze()  # h*w.

        x_src_uint8 = self.round_it(x_src)
        unique_src = torch.unique(x_src_uint8, sorted=False,
                                  return_inverse=False,
                                  return_counts=False, dim=None)
        out = x_src * 1.


        for v in unique_src:

            noise, s = self.generate_noise(x=x_src_uint8, avg=avg, vari=vari,
                                           v=v)

            if noise is not None:
                if roi is None:
                    out[s] = out[s] + noise
                else:
                    out[s] = out[s] + noise * roi[s]
            else:
                pass
                # print(f'failed to get noise for value {v} ')

        return out

    def perturbate(self,
                   img_low: np.ndarray,
                   img_high: np.ndarray,
                   roi: np.ndarray = None) -> np.ndarray:
        """
        Takes x_src then performs a series of perturbations as different scales.
        starting from low resolution, we add noise, then upscale to the next
        scale. .... This processes is repeated several steps (scales) until
        reaching the high scale. This simulates adding random information to
        upscale an image from low resolution to high resolution.

        :param img_low: low res h`xw`. uint8. np.ndarray.
        :param img_high: high res hxw. uint8. np.ndarray.
        :param roi: None or np.ndarray hxw. computed over low res after being
        interpolated to high res. has same shape as high res. binary.
        :returns: x_src upscaled to high resolution. hxw. uint8. np.ndarray.
        """

        assert isinstance(img_low, np.ndarray), type(img_low)
        assert isinstance(img_high, np.ndarray), type(img_high)
        if roi is not None:
            assert isinstance(roi, np.ndarray), type(roi)
            assert roi.shape == img_high.shape
            assert roi.ndim == 2, roi.ndim

        assert img_low.ndim == 2, img_low.ndim
        assert img_high.ndim == 2, img_high.ndim

        # device = torch.device(f'cuda:{torch.cuda.current_device()}')
        device = torch.device('cpu')
        x_l = torch.asarray(img_low).to(device)
        x_h = torch.asarray(img_high).to(device)
        if roi is not None:
            roi = torch.asarray(roi).to(device)
            roi = roi.type(torch.float32)

        x_l = self.round_it(x_l.float())
        x_h = self.round_it(x_h.float())

        h_l, w_l = x_l.shape

        for scale in self.scales:
            _h = int(h_l * scale)
            _w = int(w_l * scale)

            x_h_2_l = F.interpolate(
                        input=x_h.unsqueeze(0).unsqueeze(0),
                        size=[_h, _w],
                        mode='bilinear',
                        align_corners=False
                        ).squeeze()

            if scale != 1:
                x_l = F.interpolate(
                        input=x_l.unsqueeze(0).unsqueeze(0),
                        size=[_h, _w],
                        mode='bilinear',
                        align_corners=False
                        ).squeeze()

            _roi = None
            if roi is not None:
                _roi = F.interpolate(
                        input=roi.unsqueeze(0).unsqueeze(0),
                        size=[_h, _w],
                        mode='neasrest',
                        align_corners=False
                        ).squeeze()
                _roi = _roi.type(torch.float32)

            x_l = self.add_noise(x_src=x_l, x_trg=x_h_2_l, roi=_roi)
        assert x_l.shape == x_h.shape, f'{x_l.shape} {x_h.shape}'
        x_l = self.round_it(x_l)
        x_l = x_l.cpu().squeeze().numpy().astype(np.uint8)  # h, w. uint8. [0,
        # 255].
        return x_l

    def __call__(self,
                 img_low: np.ndarray,
                 img_high: np.ndarray,
                 roi: np.ndarray = None) -> np.ndarray:
        """
        Takes x_src then performs a series of perturbations as different scales.
        starting from low resolution, we add noise, then upscale to the next
        scale. .... This processes is repeated several steps (scales) until
        reaching the high scale. This simulates adding random information to
        upscale an image from low resolution to high resolution.

        :param img_low: low res h`xw`. uint8. np.ndarray.
        :param img_high: high res hxw. uint8. np.ndarray.
        :param roi: None or np.ndarray hxw. computed over low res after being
        interpolated to high res. has same shape as high res. binary.
        :returns: x_src upscaled to high resolution. hxw. uint8. np.ndarray.
        """

        return self.perturbate(img_low, img_high, roi)



class PatchSampler(object):
    def __init__(self,
                 sample_type: str,
                 psize: int,
                 nbr_colors: int,
                 threshold_style: str,
                 threshold: float
                 ):
        super(PatchSampler, self).__init__()

        assert sample_type in constants.SAMPLE_PATCHES, sample_type
        assert isinstance(psize, int), type(psize)
        assert psize > 0, psize
        assert isinstance(nbr_colors, int), type(nbr_colors)
        assert nbr_colors > 0, nbr_colors

        self.psize = psize
        self.sample_type = sample_type
        self.nbr_colors = nbr_colors

        msg = f"{threshold_style} not in {constants.ROI_STYLE_TH}"

        assert threshold_style in constants.ROI_STYLE_TH, msg
        self.threshold_style = threshold_style
        self.threshold = threshold

    def _uniform(self, img: np.ndarray) -> Tuple[int, int]:
        assert self.sample_type == constants.SAMPLE_UNIF, self.sample_type

        assert img.ndim == 2, img.ndim  # hxw.
        h, w = img.shape

        rnd_h = random.randint(0, max(0, h - self.psize))
        rnd_w = random.randint(0, max(0, w - self.psize))

        return rnd_h, rnd_w

    def _roi(self, img: np.ndarray, roi: np.ndarray) -> Tuple[int, int]:
        assert self.sample_type == constants.SAMPLE_ROI, self.sample_type

        assert img.ndim == 2, img.ndim  # hxw.
        assert roi is not None, roi
        assert roi.shape == img.shape, f'{roi.shape} {img.shape}'
        assert roi.dtype == np.float64, roi.dtype

        h, w = img.shape

        lhalf = int(self.psize / 2)
        rhalf = math.ceil(self.psize / 2)

        # th = threshold_otsu(image=img, nbins=self.nbr_colors)
        # roi = (img > th).astype(np.float64)
        cropped_roi = roi[lhalf:h - rhalf, lhalf:w - rhalf]
        tmp = np.exp(cropped_roi * 5.)
        cropped_roi_p = (tmp.flatten() + 1.) / (tmp + 1.).sum()

        s = np.random.multinomial(1, cropped_roi_p,
                                  size=1).reshape(cropped_roi.shape)
        assert s.sum() == 1, s.sum()
        nnzero = s.nonzero()

        patch_c = [lhalf + nnzero[0][0], lhalf + nnzero[1][0]]
        x0 = patch_c[0] - lhalf
        y0 = patch_c[1] - lhalf

        x1 = x0 + self.psize
        y1 = y0 + self.psize

        assert 0 <= x0 <= h - self.psize, x0
        assert 0 <= y0 <= w - self.psize, y0

        assert self.psize <= x1 <= h, x1
        assert self.psize <= y1 <= w, y1

        r = [(x0, y0), self.psize, self.psize]

        return x0, y0

    def _edt(self, img: np.ndarray, roi: np.ndarray) -> Tuple[int, int]:
        assert self.sample_type == constants.SAMPLE_EDT, self.sample_type

        assert img.ndim == 2, img.ndim  # hxw.
        assert roi is not None, roi
        assert roi.shape == img.shape, f'{roi.shape} {img.shape}'
        assert roi.dtype == np.float64, roi.dtype

        h, w = img.shape
        lhalf = int(self.psize / 2)
        rhalf = math.ceil(self.psize / 2)

        # th = threshold_otsu(image=img, nbins=256)
        # roi = (img > th).astype(np.float64)
        edt = edt_fn(input=roi, return_distances=True, return_indices=False)
        cropped_edt = edt[lhalf:h - rhalf, lhalf:w - rhalf]

        # tmp = np.exp(cropped_edt)  # overheat
        tmp = cropped_edt
        cropped_edt_p = (tmp.flatten() + 1.) / (tmp + 1.).sum()

        s = np.random.multinomial(1, cropped_edt_p,
                                  size=1).reshape(cropped_edt.shape)
        assert s.sum() == 1, s.sum()
        nnzero = s.nonzero()
        patch_c = [lhalf + nnzero[0][0], lhalf + nnzero[1][0]]
        x0 = patch_c[0] - lhalf
        y0 = patch_c[1] - lhalf

        x1 = x0 + self.psize
        y1 = y0 + self.psize

        assert 0 <= x0 <= h - self.psize, x0
        assert 0 <= y0 <= w - self.psize, y0

        assert self.psize <= x1 <= h, x1
        assert self.psize <= y1 <= w, y1

        r = [(x0, y0), self.psize, self.psize]

        return x0, y0

    def _edt_roi(self, img: np.ndarray, roi: np.ndarray) -> Tuple[int, int]:
        assert self.sample_type == constants.SAMPLE_EDTXROI, self.sample_type

        assert img.ndim == 2, img.ndim  # hxw.
        assert roi is not None, roi
        assert roi.shape == img.shape, f'{roi.shape} {img.shape}'
        assert roi.dtype == np.float64, roi.dtype

        h, w = img.shape
        lhalf = int(self.psize / 2)
        rhalf = math.ceil(self.psize / 2)

        # th = threshold_otsu(image=img, nbins=256)
        # roi = (img > th).astype(np.float64)
        edt = edt_fn(input=roi, return_distances=True, return_indices=False)
        cropped_edt = edt[lhalf:h - rhalf, lhalf:w - rhalf]
        cropped_roi = roi[lhalf:h - rhalf, lhalf:w - rhalf]

        tmp = np.exp(cropped_roi * 5.)
        cropped_roi_p = (tmp.flatten() + 1.) / (tmp + 1.).sum()

        tmp = np.exp(cropped_edt)
        cropped_edt_p = (tmp.flatten() + 1.) / (tmp + 1.).sum()

        prob = cropped_roi_p * cropped_edt_p
        prob = prob / prob.sum()

        s = np.random.multinomial(1, prob, size=1).reshape(cropped_edt.shape)
        assert s.sum() == 1, s.sum()
        nnzero = s.nonzero()
        patch_c = [lhalf + nnzero[0][0], lhalf + nnzero[1][0]]
        x0 = patch_c[0] - lhalf
        y0 = patch_c[1] - lhalf

        x1 = x0 + self.psize
        y1 = y0 + self.psize

        assert 0 <= x0 <= h - self.psize, x0
        assert 0 <= y0 <= w - self.psize, y0

        assert self.psize <= x1 <= h, x1
        assert self.psize <= y1 <= w, y1

        r = [(x0, y0), self.psize, self.psize]

        return x0, y0

    def __call__(self,
                 img: np.ndarray,
                 return_roi: bool
                 ) -> Tuple[int, int, Union[None, np.ndarray]]:

        roi_uint8 = None
        roi = None
        if return_roi or self.sample_type in [constants.SAMPLE_ROI,
                                              constants.SAMPLE_EDT,
                                              constants.SAMPLE_EDTXROI]:

            if self.threshold_style == constants.TH_AUTO:
                th = threshold_otsu(image=img, nbins=self.nbr_colors)
            elif self.threshold_style == constants.TH_FIX:
                th = self.threshold
            else:
                raise NotImplementedError(self.threshold_style)

            _roi = img >= th
            roi = _roi.astype(np.float64)

            if return_roi:
                roi_uint8 = _roi.astype(np.uint8)


        if self.sample_type == constants.SAMPLE_UNIF:
            x0, y0 = self._uniform(img)

            return x0, y0, roi_uint8

        elif self.sample_type == constants.SAMPLE_ROI:
            x0, y0 = self._roi(img, roi)

            return x0, y0, roi_uint8

        elif self.sample_type == constants.SAMPLE_EDT:
            x0, y0 = self._edt(img, roi)

            return x0, y0, roi_uint8

        elif self.sample_type == constants.SAMPLE_EDTXROI:
            x0, y0 = self._edt_roi(img, roi)

            return x0, y0, roi_uint8

        else:
            raise NotImplementedError(self.sample_type)



class DatasetDPSR(data.Dataset):
    def __init__(self,
                 args: object,
                 phase: str,
                 pairs_h: dict,
                 pairs_l: dict = None
                 ):
        self.args = args
        assert args.task in constants.TASKS, args.task
        self.task = args.task

        # reconstruction. ------------------------------------------------------
        assert args.reconstruct_type in constants.RECONSTRUCT_TYPE, \
            args.reconstruct_type
        self.reconstruct_type = args.reconstruct_type
        assert args.reconstruct_input in constants.RECON_INPUTS, \
            args.reconstruct_input

        if phase == constants.TRAIN_PHASE and args.task == constants.RECONSTRUCT:
            if args.reconstruct_type == constants.LOW_RES:
                assert args.reconstruct_input == constants.RECON_IN_FAKE, \
                    args.reconstruct_input
            elif args.reconstruct_type == constants.HIGH_RES:
                assert args.reconstruct_input == constants.RECON_IN_HR, \
                    args.reconstruct_input

            else:
                raise NotImplementedError(args.reconstruct_type)

        self.reconstruct_input = args.reconstruct_input
        # reconstruction -------------------------------------------------------

        self.n_channels = args.n_channels
        self.sf = args.scale
        self.patch_size = args.h_size
        self.l_size = self.patch_size // self.sf

        self.use_interpolated_low = args.use_interpolated_low

        self.pairs_h = pairs_h
        self.pairs_l = pairs_l

        assert phase in constants.PHASES
        self.phase = phase

        self.im_h_ids: list = list(pairs_h.keys())

        assert self.pairs_h

        self.im_h_ids_to_float: dict = dict()
        self.float_to_im_h_ids: dict = dict()

        self.per_color_weight: np.ndarray = None
        self._build_per_color_weight()

        self.nbr_colors = len(list(range(args.color_min, args.color_max))) + 1

        self.patch_sampler = None
        if self.phase == constants.TRAIN_PHASE:
            self.patch_sampler = PatchSampler(sample_type=args.sample_tr_patch,
                                              psize=self.patch_size,
                                              nbr_colors=self.nbr_colors,
                                              threshold_style=args.sample_tr_patch_th_style,
                                              threshold=args.sample_tr_patch_th
                                              )
        self.augmenter = None
        if args.method == constants.CSRCNN_MTH and args.augment:
            self.augmenter = Augment(upscale=args.scale,
                                     nbr_steps=args.augment_nbr_steps,
                                     use_roi=args.augment_use_roi,
                                     color_min=args.color_min,
                                     color_max=args.color_max)

    def _ids_to_float_and_reverse(self):
        for i, k in enumerate(self.im_h_ids):
            f = float(i)
            assert f not in self.float_to_im_h_ids
            assert k not in self.im_h_ids_to_float

            self.float_to_im_h_ids[f] = k
            self.im_h_ids_to_float[k] = f

    def _build_per_color_weight(self):

        cnd = (self.phase == constants.TRAIN_PHASE)

        h_id = self.im_h_ids[0]
        h_path = self.pairs_h[h_id]['abs_path']

        cnd &= (util._is_biosr(h_path)
                or
                util._is_caco2(h_path))
        cnd &= (self.n_channels == 1)  # todo: add support to rgb.

        if (self.phase == constants.TRAIN_PHASE) and not cnd and self.args.ppiw:
            raise NotImplementedError

        cnd &= self.args.ppiw

        if not cnd:
            return 0

        print(f'Computing per-color weights for {self.phase}...')

        full_histo = 1.
        color_min = self.args.color_min
        color_max = self.args.color_max
        nbr_colors = len(list(range(color_min, color_max))) + 1

        for i in tqdm(range(len(self)), ncols=80, total=len(self)):
            h_id = self.im_h_ids[i]
            h_path = self.pairs_h[h_id]['abs_path']
            img_h = util.imread_uint(h_path, self.n_channels).squeeze()  # hxw

            histo = unnormed_histogram(
                img_h, nbr_colors, range=(color_min, color_max))[0]

            full_histo = histo + full_histo

        full_histo = nbr_colors * full_histo / float(full_histo.sum())
        weights = 1. / full_histo
        weights = weights / weights.sum()  # probabs to avoid very large weights
        # renormalize to avid tiny weights (e.g. 10-9.)
        a = self.args.ppiw_min_per_col_w
        b = 1.
        weights += 1e-8
        _min = weights.min()
        _max = weights.max()
        w = (b - a) * (weights - _min) / (_max - _min) + a
        weights = w

        self.per_color_weight = weights.flatten()  # (nbrcolorss,)

        return 0

    def _reconstruct_blure_true_lr(self, im: np.ndarray) -> np.ndarray:

        # loose the details: upscale2, down2, upscale2, down2.
        # simulate blurring without blurring hyper-parameters.

        x = util.imresize_np(im, 2, True)  # up
        x = util.imresize_np(x, 1 / 2, True)  # down

        x = util.imresize_np(x, 2, True)  # up
        x = util.imresize_np(x, 1 / 2, True)  # down

        assert x.shape == im.shape, f"{x.shape} | {im.shape}"

        return x

    def _resize_low_to_scale(self, img: np.ndarray, size: tuple):
        """
        upscale low resolution to some size.
        :param img: np.ndarray.
        :param size: [height, width].
        :return: np.ndarray.
        """
        new_img = cv2.resize(src=img, dsize=size, interpolation=cv2.INTER_CUBIC)
        # note: cv2.resize (h, w, 1) --> (h', w')

        if img.ndim == 3:
            if img.shape[2] == 1:
                assert new_img.ndim == 2, new_img.ndim
                new_img = np.expand_dims(new_img, axis=2)  # HxWx1

                assert new_img.ndim == 3, new_img.ndim

            elif img.shape[2] > 1:
                assert new_img.ndim == img.ndim, f'{new_img.ndim}, {img.ndim}'
            else:
                raise NotImplementedError

        return new_img

    @staticmethod
    def interpolate_torch(x: np.ndarray,
                          scale: int,
                          mode: str = constants.INTER_BICUBIC,
                          min_v: float = 0.0,
                          max_v: float = 255.) -> np.ndarray:
        assert isinstance(x, np.ndarray), type(x)
        assert x.dtype == np.uint8, x.dtype

        assert x.ndim == 3, x.ndim  # hwc
        assert x.shape[-1] in [1, 3], x.shape[-1]  # hwc.
        v = torch.from_numpy(x).float()
        v = torch.permute(v, (2, 0, 1))  # c, h, w
        c, h, w = v.shape
        _new_h = int(h * scale)
        _new_w = int(w * scale)

        out = F.interpolate(v.unsqueeze(0),
                            size=(_new_h, _new_w),
                            mode=mode
                            ).squeeze(0)
        out = torch.clamp(out, min=min_v, max=max_v)

        out = torch.permute(out, (1, 2, 0))  # h,w,c
        out = out.numpy().astype(x.dtype)

        return out

    @staticmethod
    def simulate_low_res(x: np.ndarray,
                         seed: int,
                         th: float,
                         sigma: float,
                         min_v: float = 0.0,
                         max_v: float = 255.
                         ) -> np.ndarray:

        assert sigma >= 0.0, sigma
        assert isinstance(sigma, float), type(sigma)
        assert th >= 0, th
        assert isinstance(th, float), type(th)

        assert isinstance(x, np.ndarray), type(x)
        assert x.ndim == 3, x.ndim  # hwc
        assert x.shape[-1] in [1, 3], x.shape[-1]  # hwc.
        v = torch.from_numpy(x).float()

        roi_h_to_l = (v >= th).float()

        set_seed(seed=seed, verbose=False)  # todo: DEBUG. deterministic

        new_low = torch.normal(mean=v, std=sigma)

        new_low = torch.clamp(new_low, 0.0, 255.)
        new_low = new_low * roi_h_to_l + (1 - roi_h_to_l) * v

        new_low = torch.clamp(new_low, min=min_v, max=max_v)

        out = new_low.numpy().astype(x.dtype)

        return out

    def __getitem__(self, index: int):
        h_id = self.im_h_ids[index]
        l_id = self.pairs_h[h_id]['low_path_key']

        h_path = self.pairs_h[h_id]['abs_path']
        img_h = util.imread_uint(h_path, self.n_channels)

        color_min = self.args.color_min
        color_max = self.args.color_max
        img_h_to_l = self.interpolate_torch(img_h,
                                            scale=(1. / self.sf),
                                            mode=self.args.basic_interpolation,
                                            min_v=color_min,
                                            max_v=color_max
                                            )
        # todo: debug
        # zz = Image.fromarray(img_h_to_l.squeeze()).convert('RGB')
        # zz.save('xx1.png')

        # img_h_to_l = util.imresize_np(img_h, 1 / self.sf, True)


        img_h_to_l = np.clip(img_h_to_l, a_min=color_min, a_max=color_max)
        # [0, 255.]

        _img_h_2_l_sim = np.copy(img_h_to_l)

        img_h = util.uint2single(img_h)  # h, w, d, [0., 1.]
        # reconstruct
        img_h_to_l = util.uint2single(img_h_to_l)  # [0., 1.], h, w, d.

        img_h: np.ndarray = util.modcrop(img_h, self.sf)  # todo: deal with.
        # make sure the rest is consistent.

        h_per_pixel_weight = None
        synthesize = True

        if self.pairs_l:
            l_path = self.pairs_l[l_id]['abs_path']
            synthesize = not os.path.isfile(l_path)
            synthesize |= self.use_interpolated_low

        if synthesize and util._is_caco2(h_path):
            _img_h_2_l_sim = self.simulate_low_res(
                x=_img_h_2_l_sim,
                seed=index,  # generate the same.,
                th=self.args.inter_low_th,
                sigma=self.args.inter_low_sigma
                )
            _img_h_2_l_sim = util.uint2single(_img_h_2_l_sim)  # [0, 1], h,w,d


        if synthesize:

            if util._is_caco2(h_path):
                img_l = np.copy(_img_h_2_l_sim)

                # todo: debug
                # zz = (img_l * 255.).astype(np.uint8)
                # zz = Image.fromarray(zz.squeeze()).convert('RGB')
                # zz.save('xx.png')
                # sys.exit()

            else:
                img_l = util.imresize_np(img_h, 1 / self.sf, True)

            img_low_blurred = img_l
            l_path = h_path

            _h = img_h.shape[0]  // self.sf
            _w = img_h.shape[1] // self.sf
            img_l_interp_to_h = (img_l * self.args.color_max).astype(np.uint8)
            # downscale, then upscale.
            img_l_interp_to_h = self._resize_low_to_scale(img_l_interp_to_h,
                                                          (_w, _h))
            _zh, _zw, _ = img_h.shape
            img_l_interp_to_h = self._resize_low_to_scale(img_l_interp_to_h,
                                                          (_zw, _zh))
            img_l_interp_to_h = util.uint2single(img_l_interp_to_h)  # [0., 1.]
            # h, w, d.
        else:
            img_l = util.imread_uint(l_path, self.n_channels)  # h, w, d

            img_low_blurred = self._reconstruct_blure_true_lr(img_l)
            img_low_blurred = util.uint2single(img_low_blurred)  # [0., 1.]
            # h, w, d

            _zh, _zw, _ = img_h.shape
            img_l_interp_to_h = self._resize_low_to_scale(img_l, (_zw, _zh))

            img_l = util.uint2single(img_l)  # [0, 1.]

            img_l_interp_to_h = util.uint2single(img_l_interp_to_h)  # [0., 1.]
            # h, w, d

        if self.phase == constants.TRAIN_PHASE:
            h, w, _ = img_h.shape

            aug_img_l_interp_to_h = img_l_interp_to_h

            # sample from high dim.
            if util._is_caco2(l_path):
                # todo: weak conversion. assumes 255 is max color.

                # sample from roi.
                _img = img_l_interp_to_h
                _img = (_img * self.args.color_max).astype(np.uint8).squeeze()
                return_roi = self.augmenter is not None
                return_roi &= self.args.augment_use_roi

                rnd_h_h, rnd_w_h, roi = self.patch_sampler(_img, return_roi)

                if self.augmenter is not None:  # allowed only for caco2.
                    # dropped for now.
                    pass

            else:
                spatch = self.args.sample_tr_patch
                assert spatch == constants.SAMPLE_UNIF, spatch

                rnd_h_h = random.randint(0, max(0, h - self.patch_size))
                rnd_w_h = random.randint(0, max(0, w - self.patch_size))


            rnd_h_l = rnd_h_h // self.sf
            rnd_w_l = rnd_w_h // self.sf
            img_l = img_l[rnd_h_l : rnd_h_l + self.l_size,
                          rnd_w_l : rnd_w_l + self.l_size, :]

            img_h = img_h[rnd_h_h: rnd_h_h + self.patch_size,
                          rnd_w_h: rnd_w_h + self.patch_size, :]

            img_l_to_h = img_l_interp_to_h[rnd_h_h: rnd_h_h + self.patch_size,
                          rnd_w_h: rnd_w_h + self.patch_size, :]

            aug_img_l_to_h = aug_img_l_interp_to_h[
                             rnd_h_h: rnd_h_h + self.patch_size,
                             rnd_w_h: rnd_w_h + self.patch_size, :]

            # reconstruction.
            img_low_blurred = img_low_blurred[rnd_h_l : rnd_h_l + self.l_size,
                              rnd_w_l : rnd_w_l + self.l_size, :]
            # not used for training, but for consistency, we cropped.
            img_h_to_l = img_h_to_l[rnd_h_l : rnd_h_l + self.l_size,
                         rnd_w_l : rnd_w_l + self.l_size, :]

            mode = random.randint(0, 7)
            img_l = util.augment_img(img_l, mode)
            img_h = util.augment_img(img_h, mode)  # h, w, c. in [0, 1]

            img_low_blurred = util.augment_img(img_low_blurred, mode)
            img_h_to_l = util.augment_img(img_h_to_l, mode)

            # LR only data augmentation. ---
            img_l = self._apply_da_blur(img_l)
            img_h = np.clip(img_h, 0., 1.)
            img_l = self._apply_da_dot_bin_noise(img_l)
            img_h = np.clip(img_h, 0., 1.)
            img_l = self._apply_da_add_gaus_noise(img_l)
            img_h = np.clip(img_h, 0., 1.)
            # ------------------------------

            _h, _w, _ = img_h.shape

            img_l_to_h = self._resize_low_to_scale(img_l, (_w, _h))
            img_l_to_h = np.clip(img_l_to_h, a_min=0., a_max=1.)
            aug_img_l_to_h = img_l_to_h * 1.  # todo: change later.

            img_h = util.single2tensor3(img_h)  # c, h,w. [0, 1]
            img_l = util.single2tensor3(img_l)  # c, h`,w`. [0, 1]
            img_l_to_h = util.single2tensor3(img_l_to_h)  # c, h,w. [0, 1]
            aug_img_l_to_h = util.single2tensor3(aug_img_l_to_h)  # c, h,
            # w. [0, 1]

            img_low_blurred = util.single2tensor3(img_low_blurred)  # c, h,w. [0, 1]
            img_h_to_l = util.single2tensor3(img_h_to_l)  # c, h,w. [0, 1]

            if self.per_color_weight is not None:
                _x = (img_h * 255.).type(torch.uint8)
                h_per_pixel_weight = self._get_per_pixel_weight(x=_x)
                # weights: 1xhxw.
                # todo: supported hxwx1 only.
                assert img_h.shape == h_per_pixel_weight.shape

        elif self.phase == constants.EVAL_PHASE:

            aug_img_l_interp_to_h = img_l_interp_to_h

            if self.augmenter is not None:  # todo: allowed only for cacox.
                # todo: passed.
                pass


            img_h = util.single2tensor3(img_h)
            img_l = util.single2tensor3(img_l)
            img_l_to_h = util.single2tensor3(img_l_interp_to_h)
            aug_img_l_to_h = util.single2tensor3(aug_img_l_interp_to_h)

            img_low_blurred = util.single2tensor3(img_low_blurred)
            img_h_to_l = util.single2tensor3(img_h_to_l)


        else:
            raise NotImplementedError(f'Unknown phase: {self.phase}.')

        # reconstruct task:
        in_reconstruct = None
        trg_reconstruct = None

        if self.reconstruct_type == constants.HIGH_RES:
            if self.reconstruct_input == constants.HIGH_RES:
                in_reconstruct = img_h
                trg_reconstruct = img_h

            elif self.reconstruct_input == constants.RECON_IN_L_TO_HR:
                in_reconstruct = img_l_to_h
                trg_reconstruct = img_h

        elif self.reconstruct_type == constants.LOW_RES:
            if self.reconstruct_input == constants.RECON_IN_FAKE:
                in_reconstruct = img_low_blurred
                trg_reconstruct = img_l

            elif self.reconstruct_input == constants.RECON_IN_REAL:
                assert self.phase == constants.EVAL_PHASE, self.phase
                in_reconstruct = img_h_to_l
                trg_reconstruct = img_h_to_l

            else:
                raise NotImplementedError(self.reconstruct_input)

        else:
            raise NotImplementedError(self.reconstruct_type)


        if h_per_pixel_weight is None:
            return {'l_im': img_l,
                    'l_id': l_id,
                    'l_path': l_path,
                    'l_to_h_img': img_l_to_h,
                    'l_to_h_img_aug': aug_img_l_to_h,
                    'h_im': img_h,
                    'h_id': h_id,
                    'h_path': h_path,
                    'in_reconstruct': in_reconstruct,
                    'trg_reconstruct': trg_reconstruct
                    }
        else:
            return {'l_im': img_l,
                    'l_id': l_id,
                    'l_path': l_path,
                    'l_to_h_img': img_l_to_h,
                    'l_to_h_img_aug': aug_img_l_to_h,
                    'h_im': img_h,
                    'h_id': h_id,
                    'h_path': h_path,
                    'h_per_pixel_weight': h_per_pixel_weight,
                    'in_reconstruct': in_reconstruct,
                    'trg_reconstruct': trg_reconstruct
                    }

    def _apply_da_blur(self, img: np.ndarray) -> np.ndarray:
        if self.args.da_blur:
            return np_blur(img=img,
                           prob=self.args.da_blur_prob,
                           area=self.args.da_blur_area,
                           sigma=self.args.da_blur_sigma
                           )

        return img

    def _apply_da_dot_bin_noise(self, img: np.ndarray) -> np.ndarray:
        if self.args.da_dot_bin_noise:
            return np_prod_binary_noise(img=img,
                                        prob=self.args.da_dot_bin_noise_prob,
                                        area=self.args.da_dot_bin_noise_area,
                                        p=self.args.da_dot_bin_noise_p
                                        )
        return img



    def _apply_da_add_gaus_noise(self, img: np.ndarray) -> np.ndarray:
        if self.args.da_add_gaus_noise:
            return np_add_gaussian_noise(img=img,
                                         prob=self.args.da_add_gaus_noise_prob,
                                         area=self.args.da_add_gaus_noise_area,
                                         std=self.args.da_add_gaus_noise_std
                                         )
        return img

    def _get_per_pixel_weight(self, x: torch.Tensor) -> torch.Tensor:
        assert self.per_color_weight is not None

        assert x.ndim == 3, x.ndim
        # x: 1xhxw.
        assert x.shape[0] == 1, x.shape[2]  # todo: support rgb.
        color_min = self.args.color_min
        color_max = self.args.color_max
        nbr_colors = len(list(range(color_min, color_max))) + 1

        out: torch.Tensor = torch.zeros_like(x, dtype=torch.float32,
                                     requires_grad=False)

        for i in range(nbr_colors):
            out[x == i] = self.per_color_weight[i]

        return out

    def __len__(self):
        return len(self.im_h_ids)

# numpy data augmentation.

def get_random_coordinates_block(h: int,
                                 w: int,
                                 area: float
                                 ) -> Tuple[int, int, int, int]:
    assert 0. <= area <= 1., area
    ratio = np.random.randn() * 0.01 + area
    bh = np.int64(h * ratio)
    bw = np.int64(w * ratio)

    ch = np.random.randint(0, h - bh + 1)
    cw = np.random.randint(0, w - bw + 1)

    return ch, cw, bh, bw


def np_blur(img: np.ndarray,
            prob: float,
            area: float,
            sigma: float
            ) -> np.ndarray:
    """
    Blur a random block of the image.
    :param img: np.ndarray image h, w, c.
    :param prob: probability to apply this operation.
    :param area: area of the block.
    :param sigma: Sigma for the Gaussian kernel.
    :return: image after operation if applied. otherwise, re return the same
    input image.
    Note: the output needs to be clipped later according to its allowed margin.
    """
    assert isinstance(img, np.ndarray), type(img)
    assert img.ndim == 3, img.ndim  # h, w, c.

    assert 0. <= prob <= 1., prob
    assert 0. <= area <= 1., area

    if area == 0 or np.random.rand(1) >= prob:
        return img

    h, w, c = img.shape
    ch, cw, bh, bw = get_random_coordinates_block(h, w, area)

    blurred_img = gaussian_filter(input=img, sigma=sigma)

    # apply inside the block area or outside.

    if np.random.rand(1) >= 0.98:
        img[ch:ch + bh, cw:cw + bw, :] = blurred_img[ch:ch + bh, cw:cw + bw, :]
    else:
        im = np.copy(blurred_img)
        im[ch:ch + bh, cw:cw + bw, :] = img[ch:ch + bh, cw:cw + bw, :]
        img = im

    return img

def np_prod_binary_noise(img: np.ndarray,
                         prob: float,
                         area: float,
                         p: float
                         ) -> np.ndarray:
    """
    Multiply a random block of the image with a random binary mask sampled
    from a Bernoulli distribution with probability 1 - p (a pixel has a
    probability of p to be set to zero).
    :param img: np.ndarray image h, w, c.
    :param prob: probability to apply this operation.
    :param area: area of the block.
    :param p: probability for a pixel to be set to zero. 1 - p is the
    Bernoulli dist to take value 1.
    :return: image after operation if applied. otherwise, re return the same
    input image.
    Note: the output needs to be clipped later according to its allowed margin.
    """
    assert isinstance(img, np.ndarray), type(img)
    assert img.ndim == 3, img.ndim  # h, w, c.

    assert 0. <= prob <= 1., prob
    assert 0. <= area <= 1., area
    assert 0. <= p <= 1., p

    if area == 0 or np.random.rand(1) >= prob:
        return img

    h, w, c = img.shape
    ch, cw, bh, bw = get_random_coordinates_block(h, w, area)

    mask = np.random.binomial(
        n=1, p=1. - p, size=(bh, bw, 1)).astype(np.float32)

    # apply only inside the block area.
    img[ch:ch + bh, cw:cw + bw, :] = img[ch:ch + bh, cw:cw + bw, :] * mask

    return img

def np_add_gaussian_noise(img: np.ndarray,
                         prob: float,
                         area: float,
                         std: float
                         ) -> np.ndarray:
    """
    Add to a random block of the image random Gaussian noise centered at zero
    and has std asn a standard deviation.
    :param img: np.ndarray image h, w, c.
    :param prob: probability to apply this operation.
    :param area: area of the block.
    :param std: Standard deviation of the Gaussian.
    :return: image after operation if applied. otherwise, re return the same
    input image.
    Note: the output needs to be clipped later according to its allowed margin.
    """
    assert isinstance(img, np.ndarray), type(img)
    assert img.ndim == 3, img.ndim  # h, w, c.

    assert 0. <= prob <= 1., prob
    assert 0. <= area <= 1., area
    assert std >= 0., std

    if area == 0 or np.random.rand(1) >= prob:
        return img

    h, w, c = img.shape
    ch, cw, bh, bw = get_random_coordinates_block(h, w, area)

    noise = np.random.normal(loc=0.0, scale=std, size=(bh, bw, c))

    # apply only inside the block area.
    img[ch:ch + bh, cw:cw + bw, :] = img[ch:ch + bh, cw:cw + bw, :] + noise

    return img


def _load_cell(path, resize: Tuple[int, int] = None) -> np.ndarray:
    im = Image.open(path, 'r').convert('RGB')
    if resize is not None:
        im = im.resize(size=resize, resample=PIL.Image.BICUBIC)

    im = np.array(im)  # h, w, 3

    cell_name = get_cell_type(path)
    assert cell_name is not None, path

    im = im[:, :, constants.PLAN_IMG[cell_name]]  # h, w.

    return im


def _build_img(x: np.ndarray, cell_type: str) -> Image.Image:

    assert isinstance(x, np.ndarray), type(x)
    assert x.ndim == 2, x.ndim  # HW
    img = np.expand_dims(x, axis=2)  # HxWx1
    img = np.repeat(img, 3, axis=2)  # HW3: RGB

    raise ValueError

    img = Image.fromarray(img, mode='RGB')

    return img


def tagax(ax, text, xy: list, alpha_: float = 0.0,
          facecolor: str = 'white'):
    ax.text(xy[0], xy[1],
            text, bbox={'facecolor': facecolor, 'pad': 1, 'alpha': alpha_},
            color='red', fontsize=3
            )


def _show_img_and_tag(ax, img, tag: str, patches: List[List[int]] = None):

    ax.imshow(img)
    top_tag_xy = [1, 70]
    tagax(ax, tag, top_tag_xy)


    if patches is not None:
        color = mcolors.CSS4_COLORS['orange']
        for p in patches:
            anchor, height, width = p
            anchor = [anchor[1], anchor[0]]  # switch

            rect_gt_mbo = mlt_patches.Rectangle(anchor, width, height,
                                                linewidth=.2,
                                                edgecolor=color,
                                                facecolor='none')
            ax.add_patch(rect_gt_mbo)


def _clean_axes_fig(fig):

    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())


def _closing(fig, outf):
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0,
    #                     wspace=0)
    fig.savefig(outf, pad_inches=0, bbox_inches='tight', dpi=250,
                optimize=True)
    plt.close(fig)


def _plot_images(fout: str,
                 images: dict,
                 patches: dict,
                 sz: int):
    n = len(list(images.keys()))
    n_p = len(list(patches.keys()))

    nrows = 1
    ncols = n + n_p

    him, wim = 400, 400
    r = him / float(wim)
    fw = 30
    r_prime = r * (nrows / float(ncols))
    fh = r_prime * fw

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    for i, k in enumerate(images):
        _show_img_and_tag(axes[0, i], images[k], f'{sz}. {k}')

    for i, k in enumerate(patches):
        print(k)
        _show_img_and_tag(axes[0, i + n], images[k],
                          f'{sz}. Patches using {k}', patches=patches[k]
                          )

    # _show_img_and_tag(axes[0, 1], images['interp'], f'INTER.{l}')
    # _show_img_and_tag(axes[0, 2], images['interp_updated'],
    #                   f'INTER-UPDATED.{l}')

    _clean_axes_fig(fig)
    _closing(fig, fout)


def test_sampling(fd: str, sz: int, patch_sz: int,
                  resize: Tuple[int, int] = None, nbr_patches=10):
    set_seed(0)

    print(fd)

    from os.path import join
    _CELLS = []  # fixit

    outd = join(root_dir, 'data/debug/input/roi')
    os.makedirs(outd, exist_ok=True)


    for cell in _CELLS:
        print(cell)

        img_path = join(fd, f'{cell}.tif')


        assert os.path.isfile(img_path), img_path
        img = _load_cell(img_path, resize=resize)  # hxw
        th = threshold_otsu(image=img, nbins=256)
        roi = (img > th).astype(np.float64)
        edt = edt_fn(input=roi, return_distances=True, return_indices=False)
        h, w = edt.shape
        lhalf = int(patch_sz / 2)
        rhalf = math.ceil(patch_sz / 2)

        cropped_edt = edt[lhalf:h - rhalf, lhalf:w - rhalf]
        cropped_roi = roi[lhalf:h - rhalf, lhalf:w - rhalf]

        tmp = np.exp(cropped_roi * 5.)
        cropped_roi_p = (tmp.flatten() + 1.) / (tmp + 1.).sum()
        tmp = np.exp(cropped_edt)
        cropped_edt_p = (tmp.flatten() + 1.) / (tmp + 1.).sum()


        patches = {
            'roi': [],
            'edt': [],
            'roi * edt': [],
            'uniform': []
        }

        # generate patches
        for i in range(nbr_patches):
            # using roi
            s = np.random.multinomial(1, cropped_roi_p,
                                      size=1).reshape(cropped_roi.shape)
            assert s.sum() == 1, s.sum()
            nnzero = s.nonzero()

            patch_c = [lhalf + nnzero[0][0], lhalf + nnzero[1][0]]
            x0 = patch_c[0] - lhalf
            y0 = patch_c[1] - lhalf

            x1 = x0 + patch_sz
            y1 = y0 + patch_sz

            assert 0 <= x0 <= h - patch_sz, x0
            assert 0 <= y0 <= w - patch_sz, y0

            assert patch_sz <= x1 <= h, x1
            assert patch_sz <= y1 <= w, y1

            r = [(x0, y0), patch_sz, patch_sz]

            patches['roi'].append(r)

            # using edt

            s = np.random.multinomial(1, cropped_edt_p,
                                      size=1).reshape(cropped_edt.shape)
            assert s.sum() == 1, s.sum()
            nnzero = s.nonzero()
            patch_c = [lhalf + nnzero[0][0], lhalf + nnzero[1][0]]
            x0 = patch_c[0] - lhalf
            y0 = patch_c[1] - lhalf

            x1 = x0 + patch_sz
            y1 = y0 + patch_sz

            assert 0 <= x0 <= h - patch_sz, x0
            assert 0 <= y0 <= w - patch_sz, y0

            assert patch_sz <= x1 <= h, x1
            assert patch_sz <= y1 <= w, y1

            r = [(x0, y0), patch_sz, patch_sz]

            patches['edt'].append(r)

            # using roi * edt
            prob = cropped_roi_p * cropped_edt_p
            prob = prob / prob.sum()

            s = np.random.multinomial(1, prob,
                                      size=1).reshape(cropped_edt.shape)
            assert s.sum() == 1, s.sum()
            nnzero = s.nonzero()
            patch_c = [lhalf + nnzero[0][0], lhalf + nnzero[1][0]]
            x0 = patch_c[0] - lhalf
            y0 = patch_c[1] - lhalf

            x1 = x0 + patch_sz
            y1 = y0 + patch_sz

            assert 0 <= x0 <= h - patch_sz, x0
            assert 0 <= y0 <= w - patch_sz, y0

            assert patch_sz <= x1 <= h, x1
            assert patch_sz <= y1 <= w, y1

            r = [(x0, y0), patch_sz, patch_sz]

            patches['roi * edt'].append(r)

            # using uniform
            prob = cropped_roi_p * 0. + 1.
            prob = prob / prob.sum()

            s = np.random.multinomial(1, prob,
                                      size=1).reshape(cropped_edt.shape)
            assert s.sum() == 1, s.sum()
            nnzero = s.nonzero()
            patch_c = [lhalf + nnzero[0][0], lhalf + nnzero[1][0]]
            x0 = patch_c[0] - lhalf
            y0 = patch_c[1] - lhalf

            x1 = x0 + patch_sz
            y1 = y0 + patch_sz

            assert 0 <= x0 <= h - patch_sz, x0
            assert 0 <= y0 <= w - patch_sz, y0

            assert patch_sz <= x1 <= h, x1
            assert patch_sz <= y1 <= w, y1

            r = [(x0, y0), patch_sz, patch_sz]

            patches['uniform'].append(r)


        images = {
            'img': _build_img(img, cell),
            'roi': roi,
            'edt': edt,
            'roi * edt': roi * edt,
            'uniform': _build_img(img, cell)
        }

        _outd = join(outd, f'{sz}')
        os.makedirs(_outd, exist_ok=True)
        fout = join(_outd, f'{cell}.png')
        _plot_images(fout=fout, images=images, patches=patches, sz=sz)


def test_np_blur():

    imp = join(root_dir, 'data/debug/input/data_to_roi/'
                         'tile_HighRes1024-1_0_0_512_1792_2304_CELL2.tif')
    img = util.imread_uint(imp, 1)
    plt.imshow(img)
    plt.show()
    img = util.uint2single(img)  # h, w, 1
    img = np_blur(img, prob=0.9, area=0.7, sigma=1)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    print(img.shape)


def test_np_prod_binary_noise():

    imp = join(root_dir, 'data/debug/input/data_to_roi/'
                         'tile_HighRes1024-1_0_0_512_1792_2304_CELL2.tif')
    img = util.imread_uint(imp, 1)
    plt.imshow(img)
    plt.show()
    img = util.uint2single(img)  # h, w, 1
    img = np_prod_binary_noise(img, prob=0.9, area=0.7, p=0.5)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    print(img.shape)


def test_np_add_gaussian_noise():

    imp = join(root_dir, 'data/debug/input/data_to_roi/'
                         'tile_HighRes1024-1_0_0_512_1792_2304_CELL2.tif')
    img = util.imread_uint(imp, 1)
    plt.imshow(img)
    plt.show()
    img = util.uint2single(img)  # h, w, 1
    img = np_add_gaussian_noise(img, prob=0.9, area=0.7, std=0.03)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    print(img.shape)


def test_1():
    from dlib.utils.utils_reproducibility import set_seed

    nbr_patches = 100

    test_sampling(fd=join(root_dir, 'data/debug/input/128/Image-1'), sz=128,
                  patch_sz=32, resize=(512, 512), nbr_patches=nbr_patches)
    test_sampling(fd=join(root_dir, 'data/debug/input/1024/Image-1'), sz=1024,
                  patch_sz=64, resize=None, nbr_patches=nbr_patches)

if __name__ == '__main__':
    # test_1()
    # test_np_blur()
    # test_np_prod_binary_noise()
    test_np_add_gaussian_noise()
