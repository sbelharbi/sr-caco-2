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
import torch.nn as nn
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mlt_patches
import matplotlib.colors as mcolors
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt as edt_fn
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.utils.utils_image as util
from dlib.utils import constants
from dlib.diagnosis.stats_numpy import unnormed_histogram
from dlib.utils.utils_image import get_cell_type
from dlib.loss.local_terms import PatchMoments
from dlib.utils.utils_image import get_cell_type
from dlib.utils.shared import reformat_id



def rv1d(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 1
    return torch.flip(t, dims=(0, ))


class STOtsu(nn.Module):
    def __init__(self):
        super(STOtsu, self).__init__()

        self.bad_egg = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.bad_egg = False

        min_x = x.min()
        max_x = x.max()

        if min_x == max_x:
            self.bad_egg = True
            return torch.tensor(min_x)

        # max_x = 255
        # min_x = 0
        bins = int(max_x - min_x + 1)
        bin_centers = torch.arange(min_x, max_x + 1, 1, dtype=torch.float32,
                                   device=x.device)

        hist = torch.histc(x, bins=bins)
        weight1 = torch.cumsum(hist, dim=0)
        _weight2 = torch.cumsum(rv1d(hist), dim=0)
        weight2_r = _weight2
        weight2 = rv1d(_weight2)
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / weight1
        mean2 = rv1d(torch.cumsum(rv1d(hist * bin_centers), dim=0) / weight2_r)
        diff_avg_sq = torch.pow(mean1[:-1] - mean2[1:], 2)
        variance12 = weight1[:-1] * weight2[1:] * diff_avg_sq

        idx = torch.argmax(variance12)
        threshold = bin_centers[:-1][idx]

        return threshold


def tagax(ax, text, xy: list, alpha_: float = 0.0, facecolor: str = 'white'):
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
                 th: int):
    n = len(list(images.keys()))

    nrows = 1
    ncols = n

    him, wim = 8, 8
    r = him / float(wim)
    fw = 4
    r_prime = r * (nrows / float(ncols))
    fh = r_prime * fw

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    for i, k in enumerate(images):
        _show_img_and_tag(axes[0, i], images[k], f'th: {th}. {k}')


    # _show_img_and_tag(axes[0, 1], images['interp'], f'INTER.{l}')
    # _show_img_and_tag(axes[0, 2], images['interp_updated'],
    #                   f'INTER-UPDATED.{l}')

    _clean_axes_fig(fig)
    _closing(fig, fout)

def test_STOtsu(th: float):
    from skimage.filters import threshold_otsu
    from tqdm import tqdm

    from dlib.utils.utils_reproducibility import set_seed
    from dlib.utils.shared import find_files_pattern

    set_seed(0)

    device = torch.device('cuda:0')

    fdin_imgs = join(root_dir, 'data/debug/input/data_to_roi')

    fdout = join(join(root_dir, f'data/debug/input/img-roi-th-{th}'))
    os.makedirs(fdout, exist_ok=True)

    l_imgs = find_files_pattern(fdin_imgs, '*.tif')

    otsu = STOtsu().to(device)


    for i, ipath in tqdm(enumerate(l_imgs), total=len(l_imgs), ncols=80):
        im = cv2.imread(ipath, 0)  # cv2.IMREAD_GRAYSCALE
        im = np.array(im)  # h, w
        # im = torch.from_numpy(im).to(device)
        # th = otsu(im)
        # print(type(th))
        # print(f" th: {th} {ipath}")
        # # th = 4.
        # roi = (im > th).float()
        #
        # # numpy
        # im = im.cpu().numpy()
        # roi = roi.cpu().numpy()
        # th = th.item()

        # th = threshold_otsu(im, 256)
        roi = (im >= th) * 1
        # print(f" th: {th} {ipath}")
        images = {
            'img': im,
            'roi': roi
        }

        _plot_images(join(fdout, f"{i}.png"), images, th)


if __name__ == "__main__":
    import argparse

    from dlib.utils.shared import announce_msg

    parser = argparse.ArgumentParser()
    parser.add_argument("--th", type=float,
                        default=8.,
                        help="Threshold value.")

    parsedargs = parser.parse_args()
    th = parsedargs.th
    announce_msg(f'Going to threshold ROI via {th}.')
    test_STOtsu(th=th)