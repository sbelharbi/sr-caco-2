import os
import random
import sys
from os.path import dirname, abspath
from typing import Tuple

import numpy as np
import cv2
import torch.utils.data as data
from tqdm import tqdm
import PIL
from PIL import Image
import matplotlib.pyplot as plt

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.utils import constants
from dlib.utils.utils_image import get_cell_type


def is_image(x):
    """
    Is x an image?
    i.e. numpy array of 2 or 3 dimensions.
    :param x: Input.
    :return: True/False.
    """
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim not in [2, 3]:
        return False
    return True


def is_uint8_image(x):
    """
    Is x a uint8 image?
    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    if x.dtype != np.uint8:
        return False
    return True


class ReinhardColorTransfer(object):
    """
    Normalize a patch color to the target image using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and Prgb_l. Shirley,
    'Color transfer between images',
    https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    """
    def __init__(self):
        self.trg_means = None
        self.trg_stds = None

    def fit_trg(self, target: np.ndarray):
        """
        Fit to a target image.
        :param target: Image RGB target.
        :return:
        """
        means, stds = self.get_mean_std(target)
        self.trg_means = means
        self.trg_stds = stds

    def transfer(self, I: np.ndarray) -> np.ndarray:
        """
        Transform an image using the fitted statistics.
        :param I: Image RGB uint8.
        :return:
        """
        I1, I2, I3 = self.lab_split(I)
        means, stds = self.get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.trg_stds[0] / stds[0])) + \
                self.trg_means[0]
        norm2 = ((I2 - means[1]) * (self.trg_stds[1] / stds[1])) + \
                self.trg_means[1]
        norm3 = ((I3 - means[2]) * (self.trg_stds[2] / stds[2])) + \
                self.trg_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(I: np.ndarray):
        """
        Convert from RGB uint8 into LAB and split into channels.
        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I), "Should be an RGB uint8 image."
        I = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        I_float = I.astype(np.float32)
        I1, I2, I3 = cv2.split(I_float)
        I1 /= 2.55  # should now be in range [0,100]
        I2 -= 128.0  # should now be in range [-127,127]
        I3 -= 128.0  # should now be in range [-127,127]
        return I1, I2, I3

    @staticmethod
    def merge_back(I1, I2, I3):
        """
        Take separate LAB channels and merge back to give RGB uint8.
        :param I1: L.
        :param I2: A.
        :param I3: B.
        :return: Image RGB uint8.
        """
        I1 *= 2.55  # should now be in range [0,255]
        I2 += 128.0  # should now be in range [0,255]
        I3 += 128.0  # should now be in range [0,255]
        I = np.clip(cv2.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(I, cv2.COLOR_LAB2RGB)

    def get_mean_std(self, I: np.ndarray):
        """
        Get mean and standard deviation of each channel.
        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I), "Should be an RGB uint8 image."
        I1, I2, I3 = self.lab_split(I)
        m1, sd1 = cv2.meanStdDev(I1)
        m2, sd2 = cv2.meanStdDev(I2)
        m3, sd3 = cv2.meanStdDev(I3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds


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


def build_rgb_from_grey(x: np.ndarray) -> np.ndarray:
    assert isinstance(x, np.ndarray), type(x)
    assert x.ndim == 2, x.ndim  # HW
    img = np.expand_dims(x, axis=2)  # HxWx1
    img = np.repeat(img, 3, axis=2)  # HW3: RGB

    return img


def tagax(ax, text, xy: list, alpha_: float = 0.0,
          facecolor: str = 'white'):
    ax.text(xy[0], xy[1],
            text, bbox={'facecolor': facecolor, 'pad': 1, 'alpha': alpha_},
            color='red'
            )


def _show_img_and_tag(ax, img, tag: str):


    ax.imshow(img)
    top_tag_xy = [1, 70]
    tagax(ax, tag, top_tag_xy)


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
                 l: int,
                 h: int):
    nrows = 1
    ncols = 3  # super-res, pred, interpolated.

    him, wim = 400, 400
    r = him / float(wim)
    fw = 10
    r_prime = r * (nrows / float(ncols))
    fh = r_prime * fw

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    _show_img_and_tag(axes[0, 0], images['high_res'], f'HR.{h}')
    _show_img_and_tag(axes[0, 1], images['interp'], f'INTER.{l}')
    _show_img_and_tag(axes[0, 2], images['interp_updated'],
                      f'INTER-UPDATED.{l}')

    _clean_axes_fig(fig)
    _closing(fig, fout)


def test_reinhard_rolor_transfer():
    lsz = 128
    hsz = 1024

    from os.path import join
    _CELLS = []  # fixit

    low_res_fd = join(root_dir, 'data/debug/input/128/Image-1')
    high_res_fd = join(root_dir, 'data/debug/input/1024/Image-1')
    outd = join(root_dir, 'data/debug/input/color-transfer-super-res')
    os.makedirs(outd, exist_ok=True)

    op = ReinhardColorTransfer()


    for cell in _CELLS:
        l = join(low_res_fd, f'{cell}.tif')
        h = join(high_res_fd, f'{cell}.tif')

        assert os.path.isfile(l), l
        assert os.path.isfile(h), h

        low_img = _load_cell(l, resize=(hsz, hsz))  # hxw
        high_img = _load_cell(h)  # hxw.

        images = {
            'interp': _build_img(low_img, cell),
            'high_res': _build_img(high_img, cell)
        }
        rgb_l = build_rgb_from_grey(low_img)
        rgb_h = build_rgb_from_grey(high_img)

        op.fit_trg(rgb_h)

        rgb_l_colored = op.transfer(rgb_l)  # rgb
        l_updated = cv2.cvtColor(rgb_l_colored, cv2.COLOR_RGB2GRAY)
        images['interp_updated'] = _build_img(l_updated, cell)
        _outd = join(outd, 'Reinhard')
        os.makedirs(_outd, exist_ok=True)
        fout = join(_outd, f'{cell}.png')
        _plot_images(fout=fout, images=images, l=lsz, h=hsz)


if __name__ == '__main__':
    test_reinhard_rolor_transfer()
