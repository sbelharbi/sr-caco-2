import copy
import fnmatch
import os
import random
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import math
import pprint
from typing import List, Tuple

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

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)

from dlib.utils.utils_config import get_root_datasets
from dlib.utils import constants
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils.shared import fmsg
from skimage.filters import threshold_otsu

# SAMPLE PATCHES FROM A TILE ACROSS DIFFERENT SCALES, AND CELL TYPES FOR
# CACO2 DATASET.


_SEED = 0


class SamplePatchesFromTile(object):
    def __init__(self,
                 psize: int,
                 shift: int,
                 out_fd: str,
                 min_area: float,
                 real_psize: int,
                 threshold: int = None
                 ):

        assert isinstance(psize, int), type(psize)
        assert psize > 0, psize
        assert psize % 2 == 0, f'psize ({psize}) should be even.'
        assert psize % 4 == 0, f'psize ({psize}) should be divisible by 4.'
        assert psize % 8 == 0, f'psize ({psize}) should be divisible by 8.'

        assert isinstance(real_psize, int), type(real_psize)
        assert real_psize > 0, real_psize
        assert real_psize % 2 == 0, f'psize ({real_psize}) should be even.'
        assert real_psize % 4 == 0, f'psize ({real_psize}) should be divisible by 4.'
        assert real_psize % 8 == 0, f'psize ({real_psize}) should be divisible by 8.'

        assert isinstance(shift, int), type(shift)
        assert shift > 0, shift

        assert isinstance(out_fd, str), type(out_fd)

        assert isinstance(min_area, float), type(min_area)
        assert 0 < min_area <=1, min_area

        self.psize = psize
        self.real_psize = real_psize  # for image registration, psize is
        # augmented.
        self.shift = shift
        self.out_fd = out_fd
        os.makedirs(out_fd, exist_ok=True)
        self.min_area = min_area

        self.threshold = threshold

        if threshold is not None:
            assert isinstance(threshold, int), type(threshold)
            assert threshold >= 0, threshold

        # per tile stats.
        self.nbr = 0
        self.reject = 0
        self.th = 0
        self.patch_id = 0

    def reset(self):
        self.nbr = 0
        self.reject = 0
        self.th = 0

    def store_path(self,
                   path: str,
                   tag: str,
                   scale: int,
                   i: int,
                   ii: int,
                   j: int,
                   jj: int):

        mtx = tifffile.imread(path)  # 3, h, w
        assert isinstance(mtx, np.ndarray), np.ndarray
        assert mtx.ndim == 3, mtx.ndim  # 3, h, w.
        out_fd = join(self.out_fd, tag)
        os.makedirs(out_fd, exist_ok=True)

        b = basename(path).split('.')[0]
        # loop over cells: 0, 1, 2.
        p = self.psize // scale

        for c in constants.CACO2_CELL_INDEX:
            ci = constants.CACO2_CELL_INDEX[c]
            assert ii - i == p, f"i {i}, ii {ii}, ii-i {ii-i} ps {p}"

            assert jj - j == p, f"j {j}, jj {jj}, jj-j {jj - j} ps {p}"
            x = mtx[ci, i: ii, j: jj]
            k = self.patch_id
            path_c = join(out_fd, f"tile_{b}_{k}_{i}_{ii}_{j}_{jj}_{c}.tif")
            tifffile.imwrite(path_c, x)

    def is_ok_patch(self,
                    mtx: np.ndarray,
                    scale: int,
                    i: int,
                    ii: int,
                    j: int,
                    jj: int):

        p = self.psize // scale

        c = constants.CELL2  # use the most bright cell type.
        ci = constants.CACO2_CELL_INDEX[c]

        assert ii - i == p, f"i {i}, ii {ii}, ii-i {ii - i} ps {p}"
        assert jj - j == p, f"j {j}, jj {jj}, jj-j {jj - j} ps {p}"

        if self.psize == self.real_psize:
            x = mtx[ci, i: ii, j: jj]

        else:
            x = mtx[ci, i: ii, j: jj]
            h, w = x.shape
            z = self.real_psize // 2
            x = x[z: h - z, z: w - z]

        cnd = (x >= self.th).sum() / float(x.size)

        return cnd >= self.min_area

    @staticmethod
    def threshold_tile(mtx: np.ndarray) -> float:

        assert isinstance(mtx, np.ndarray), np.ndarray
        assert mtx.ndim == 3, mtx.ndim  # 3, h, w.

        c = constants.CELL2  # use the most bright cell type.
        ci = constants.CACO2_CELL_INDEX[c]

        x = mtx[ci]
        th = threshold_otsu(x)
        return th


    def sample(self,
               path_1024: str,
               path_512: str,
               path_256: str,
               path_128: str
               ):

        self.patch_id = 0

        x = tifffile.imread(path_1024)  # 3, h, w
        d, h, w = x.shape

        psize = self.psize
        shift = self.shift

        if self.threshold is None:
            self.th = self.threshold_tile(x)
        else:
            self.th = self.threshold

        for i in range(0, h, shift):
            ii = i + psize
            if ii >= h:
                break


            for j in range(0, w, shift):
                jj = j + psize

                if jj >= w:
                    break

                # acceptance test.
                if not self.is_ok_patch(x, scale=1, i=i, ii=ii, j=j, jj=jj):
                    self.reject += 1
                    continue

                self.nbr += 1

                # store 1024 resolution
                self.store_path(path_1024, tag='hr_div_1', scale=1,
                                i=i, ii=ii, j=j, jj=jj)

                # store 512 resolution
                self.store_path(path_512, tag='hr_div_2', scale=2,
                                i=i//2, ii=ii//2, j=j//2, jj=jj//2)

                # store 256 resolution
                self.store_path(path_256, tag='hr_div_4', scale=4,
                                i=i//4, ii=ii//4, j=j//4, jj=jj//4)

                # store 128 resolution
                self.store_path(path_128, tag='hr_div_8', scale=8,
                                i=i//8, ii=ii//8, j=j//8, jj=jj//8)

                self.patch_id += 1

    def summary(self) -> str:
        msg = f"sampled {self.nbr} patches of size {self.psize}x" \
              f"{self.psize}. Rejected: {self.reject}. " \
              f"Thresh: {self.th}. shift: {self.shift}"
        return msg

    def __str__(self):
        return f"{self.__class__.__name__}(" \
               f"psize: {self.psize}, " \
               f"shift: {self.shift}, " \
               f"min_area: {self.min_area}, " \
               f"threshold: {self.threshold})."


if __name__ == "__main__":
    out_fd = join(get_root_datasets(constants.SUPER_RES), 'caco2-debug')
    if os.path.isdir(out_fd):
        cmd = f"rm -r {out_fd}"
        print(f"Running {cmd} ...")
        os.system(cmd)
    os.makedirs(out_fd)
    sampler = SamplePatchesFromTile(psize=512,
                                    shift=128,
                                    out_fd=out_fd,
                                    min_area=0.2,
                                    threshold=4
                                    )
    path = join(get_root_datasets(constants.SUPER_RES),
                'caco2-tiles/all-tiles')
    fds = ['HighRes1024', 'LowRes512', 'LowRes256', 'LowRes128']

    path_1024 = join(path, fds[0], f"{fds[0]}-1.tif")
    path_512 = join(path, fds[1], f"{fds[1]}-1.tif")
    path_256 = join(path, fds[2], f"{fds[2]}-1.tif")
    path_128 = join(path, fds[3], f"{fds[3]}-1.tif")

    print(path_1024)
    print(path_512)
    print(path_256)
    print(path_128)

    assert os.path.isfile(path_1024), path_1024
    assert os.path.isfile(path_512), path_512
    assert os.path.isfile(path_256), path_256
    assert os.path.isfile(path_128), path_128

    sampler.sample(path_1024,
                   path_512,
                   path_256,
                   path_128)

    print(sampler.summary())
    print(sampler)

























