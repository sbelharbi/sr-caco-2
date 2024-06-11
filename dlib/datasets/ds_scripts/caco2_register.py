import copy
import fnmatch
import os
import random
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import math
import pprint
from typing import List, Tuple
from datetime import date

import PIL.Image
import matplotlib.pyplot as plt
import pylab as pl
import tqdm
import yaml
import munch
import numpy as np
import torch
import cv2
from PIL import Image
import rasterio
import tifffile
import skimage.io
from skimage.transform import resize


# scikit-image: optical flow
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle
from skimage.registration import optical_flow_ilk
from skimage.registration import optical_flow_tvl1
from skimage.registration import phase_cross_correlation
from skimage.transform import warp
from PIL.Image import NEAREST


root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)

from dlib.utils.utils_config import get_root_datasets
from dlib.utils import constants
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils.shared import fmsg
from dlib.datasets.ds_scripts.patch_sampler import SamplePatchesFromTile


_SEED = 0
_OPTICAL_FLOW_TVL1 = 'optical_flow_tvl1'


def scikit_img_optical_flow_tvl1(ref: np.ndarray,
                                 img: np.ndarray,
                                 global_shfit: bool = False
                                 ) -> Tuple[np.ndarray, float, float]:

    v, u = optical_flow_tvl1(ref, img)
    mean_u = u.mean()
    mean_v = v.mean()

    print(mean_u, mean_v)
    # mean_v = np.sign(mean_v) * math.floor(np.abs(mean_v))
    # mean_u = np.sign(mean_u) * math.floor(np.abs(mean_u))

    mean_u = round(mean_u)
    mean_v = round(mean_v)

    print(mean_u, mean_v)

    if global_shfit:
        v = (v * 0 + mean_v).astype(v.dtype)
        u = (u * 0 + mean_u).astype(u.dtype)


    nr, nc = ref.shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')

    img_warp = warp(image=img,
                    inverse_map=np.array([row_coords + v, col_coords + u]),
                    mode='constant',
                    preserve_range=True).astype(np.uint8)

    return img_warp, mean_u, mean_v

def global_shift_img(im: np.ndarray, u: float, v: float) -> np.ndarray:
    assert im.ndim == 2, im.ndim

    nr, nc = im.shape
    u = im * 0 + u
    v = im * 0 + v


    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')

    img_warp = warp(image=im,
                    inverse_map=np.array([row_coords + v, col_coords + u]),
                    mode='constant',
                    preserve_range=True).astype(np.uint8)

    return img_warp


def prepare_data(p_hr: str, p_lr: str) -> Tuple[np.ndarray, np.ndarray]:
    hr = cv2.imread(p_hr, 0)
    lr = cv2.imread(p_lr, 0)

    h, w = hr.shape
    print(h, w, lr.shape)

    im = Image.fromarray(lr).resize(size=(w, h),
                                    resample=NEAREST
                                    )
    im = np.array(im)

    # check: revers back
    # im2 = Image.fromarray(im).resize(size=(int(w/4), int(h/4)),
    #                                 resample=NEAREST
    #                                 )
    # print(np.abs(im2 - lr).sum())  # 0.
    return hr, im

if __name__ == "__main__":
    ALIGN_TYPE = _OPTICAL_FLOW_TVL1
    GLOCAL_SHIFT = True
    print(f'Align type: {ALIGN_TYPE} \nGlobal shift: {GLOCAL_SHIFT}')

    local_dir = dirname(abspath(__file__))
    ref_path =  join(local_dir, 'data/div1.tif')
    scale2_path = join(local_dir, 'data/div2.tif')
    scale4_path = join(local_dir, 'data/div4.tif')
    scale8_path = join(local_dir, 'data/div8.tif')

    # scale 2
    scales = {
        2: scale2_path,
        4: scale4_path,
        8: scale8_path
    }

    for scale in [2]:
        path = scales[scale]

        ref, img = prepare_data(ref_path, path)
        if ALIGN_TYPE == _OPTICAL_FLOW_TVL1:
            new_im, mean_u, mean_v = scikit_img_optical_flow_tvl1(
                ref, img, GLOCAL_SHIFT)

        else:
            raise NotImplementedError

        lr = cv2.imread(path, 0)
        lr_shifted = global_shift_img(lr, mean_u / scale, mean_v / scale)
        lr_shifted = Image.fromarray(lr_shifted)
        lr_shifted.save(join(local_dir, f'data/div{scale}_lr_shifted.tif'))


        img_aligned = Image.fromarray(new_im)
        img_aligned.save(join(local_dir, f'data/div{scale}_aligned.tif'))

        img = Image.fromarray(img)
        img.save(join(local_dir, f'data/div{scale}_upscaled.tif'))

        # downscale
        h, w = new_im.shape

        im2 = img_aligned.resize(size=(int(w/scale), int(h/scale)),
                                 resample=NEAREST
                                 )

        im2.save(join(local_dir, f'data/div{scale}_aligned_scaled_down.tif'))

        plt.imshow(Image.fromarray(new_im))
        plt.show()
        plt.imshow(Image.fromarray(ref))
        plt.show()
        plt.imshow(img)
        plt.show()
        plt.imshow(im2)
        plt.show()
        plt.imshow(lr_shifted)
        plt.show()


