import copy
import math
import os
import sys
from os.path import dirname, abspath, join, basename, splitext
from typing import List, Tuple
import fnmatch
import argparse
import pprint
import time

import cv2
import torch
import yaml
from tqdm import tqdm
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from torch.nn import functional as F
from pykeops.torch import LazyTensor
from PIL import Image, ImageDraw, ImageFont

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.utils.utils_config import get_root_datasets
from dlib.utils.shared import find_files_pattern
from dlib.utils.utils_image import get_cell_type
from dlib.diagnosis.stats_numpy import unnormed_histogram
from dlib.diagnosis.stats_numpy import normed_histogram
from dlib.loss.global_terms import SoftHistogram
from dlib.loss.global_terms import GaussianKDE
from dlib.utils.utils_image import calculate_psnr
from dlib.utils.utils_image import calculate_mse
from dlib.utils.utils_image import _is_biosr
from dlib.utils.utils_dataloaders import get_pairs
from dlib.utils import utils_config

_CELLS = []  # fixit


def _is_cell(path_f: str):
    return splitext(basename(path_f))[0] in _CELLS


def _get_all_files_size(path_ds: str, size: str, ext: str):
    out = []
    for well in [1, 2, 3, 4]:
        files = find_files_pattern(join(path_ds, f'well{well}', size),
                                   f'*.{ext}')
        files = [f for f in files if _is_cell(f)]
        out.extend(files)

    return out


def _load_cell(path) -> np.ndarray:

    if _is_biosr(path):
        im = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE, h, w

    else:
        raise NotImplementedError

    return im


def _check_min_max(files: list):
    _min = None
    _max = None

    for f in tqdm(files, total=len(files), ncols=80):
        im = _load_cell(f)

        mn = im.min().item()
        mx = im.max().item()

        if _min is None:
            _min = mn
        else:
            _min = min(_min, mn)

        if _max is None:
            _max = mx
        else:
            _max = max(_max, mx)

    return _min, _max


def _check_range_all(path_ds: str, sizes: dict, ext: str):
    out = dict()
    for sz in sizes:
        print(f'Processing size: {sz}')

        files = _get_all_files_size(path_ds, sizes[sz], ext)
        stats = _check_min_max(files)
        out[sz] = {'min': stats[0], 'max': stats[1]}

    # {'1024': {'max': 254, 'min': 0},
    #  '128': {'max': 254, 'min': 0},
    #  '256': {'max': 254, 'min': 0},
    #  '512': {'max': 254, 'min': 0}}
    return out


def _files_to_dict(files: list) -> dict:
    out = dict()

    for f in files:
        k = '_'.join(f.split(os.sep)[-4:])
        assert k not in out, k
        out[k] = f

    return out


def _files_exp_to_dict(files: list) -> dict:
    out = dict()

    for f in files:
        k = splitext(basename(f))[0]
        assert k not in out, k
        out[k] = f

    return out


def _get_basics(path_ds: str, sizes: dict) -> dict:
    # super res.
    out = {}
    for size in sizes:
        files = _get_all_files_size(path_ds, sizes[size], 'tif')
        out[size] = _files_to_dict(files)

    return out


def _get_low_to_high_dataset(path_fold: str, ds: str, data_root: str) -> dict:
    assert os.path.isfile(path_fold), path_fold

    pairs = get_pairs(path_fold)
    out = dict()
    for kl in pairs:
        new_key = '_'.join(kl.split(os.sep)[-4:])
        l = join(data_root, constants.DS_DIR[ds], kl)
        h = join(data_root, constants.DS_DIR[ds], pairs[kl])

        if _is_biosr(l):
            l = l.split(constants.CODE_IDENTIFIER)[0]
            h = h.split(constants.CODE_IDENTIFIER)[0]

        out[new_key] = {
            'l': l,
            'h': h,
        }
        assert os.path.isfile(out[new_key]['l']), out[new_key]['l']
        assert os.path.isfile(out[new_key]['h']), out[new_key]['h']

    return out


def _build_img(x: np.ndarray, cell_type: str) -> Image.Image:
    assert isinstance(x, np.ndarray), type(x)
    assert x.ndim == 2, x.ndim  # HxW
    img = np.expand_dims(x, axis=2)  # HxWx1
    img = np.repeat(img, 3, axis=2)  # HW3: RGB

    if cell_type is None:  # BIOSR
        pass
    else:
        raise ValueError

    img = Image.fromarray(img, mode='RGB')

    return img


def tagax(ax, text, xy: list, alpha_: float = 0.0,
          facecolor: str = 'white'):
    ax.text(xy[0], xy[1],
            text, bbox={'facecolor': facecolor, 'pad': 1, 'alpha': alpha_},
            color='red', size=5
            )


def _show_img_and_tag(ax, img, tag: str):
    ax.imshow(img)
    top_tag_xy = [1, 70]
    tagax(ax, tag, top_tag_xy)


def _add_curve(ax, vals: np.ndarray, label: str, color: str, x_label: str,
               y_label: str, title: str):
    x = list(range(vals.size))

    ax.plot(x, vals, label=label, color=color, alpha=0.7)

    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=4)
    ax.set_ylabel(y_label, fontsize=4)
    ax.grid(True)
    ax.legend(loc='best', prop={'size': 4})
    ax.set_title(title, fontsize=4)


def _add_scatter(ax, vals: np.ndarray, x: np.ndarray, label: str, color: str,
                 x_label: str, y_label: str, title: str):
    ax.scatter(x, vals, s=0.01, label=label, color=color, alpha=0.7)

    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)
    # ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=4)
    ax.set_ylabel(y_label, fontsize=4)
    ax.grid(True)
    ax.legend(loc='best', prop={'size': 4})
    ax.set_title(title, fontsize=4)


def _add_histo_residuals(ax, vals: np.ndarray, x: np.ndarray, label: str,
                         color: str, x_label: str, y_label: str, title: str):
    x = list(range(-255, 256, 1))

    histo = unnormed_histogram(vals, len(x), range=(-255, 255))[0]
    ax.plot(x, histo, label=label, color=color, alpha=0.7)

    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=4)
    ax.set_ylabel(y_label, fontsize=4)
    ax.grid(True)
    ax.legend(loc='best', prop={'size': 4})
    ax.set_title(title, fontsize=4)


def _add_hexbin(ax, vals: np.ndarray, x: np.ndarray,
                x_label: str, y_label: str, title: str, grid_size: Tuple):
    ax.hexbin(x, vals, gridsize=grid_size, bins='log', cmap='inferno')

    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)
    # ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=4)
    ax.set_ylabel(y_label, fontsize=4)
    # ax.grid(True)
    # ax.legend(loc='best', prop={'size': 4})
    ax.set_title(title, fontsize=4)


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
                 images: list,
                 l_scales: list):
    n = len(images)
    ncols = 4
    nrows = math.ceil(n / ncols)

    him, wim = 400, 400
    r = him / float(wim)
    fw = 10
    r_prime = r * (nrows / float(ncols))
    fh = r_prime * fw

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if k >= n:
                axes[i, j].set_visible(False)
                k += 1
                continue

            _show_img_and_tag(axes[i, j], images[k], f'x{l_scales[k]}')
            k += 1

    _clean_axes_fig(fig)
    _closing(fig, fout)


def _plot_one_image(fout: str,
                    image,
                    scale):
    ncols = 1
    nrows = 1

    him, wim = 400, 400
    r = him / float(wim)
    fw = 10
    r_prime = r * (nrows / float(ncols))
    fh = r_prime * fw

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    _show_img_and_tag(axes[0, 0], image, f'x{scale}')

    _clean_axes_fig(fig)
    _closing(fig, fout)


def _get_figure_h_w(nrows: int, ncols: int, him: int,
                    wim: int) -> Tuple[int, int]:
    r = him / float(wim)
    fw = 40
    r_prime = r * (nrows / float(ncols))
    fh = r_prime * fw

    return fh, fw


def _plot_histogram(super_res: np.ndarray,
                    pred: np.ndarray,
                    inter: np.ndarray,
                    fout: str,
                    l: int,
                    h: int):
    nrows = 1
    ncols = 4  # super-res, pred, interpolated.

    fh, fw = _get_figure_h_w(nrows=1, ncols=4, him=400, wim=400)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    histo_ss = unnormed_histogram(super_res, 256, range=(0, 255))[0]
    histo_pred = unnormed_histogram(pred, 256, range=(0, 255))[0]
    histo_interp = unnormed_histogram(inter, 256, range=(0, 255))[0]

    _add_curve(axes[0, 0], histo_ss,
               label=f'HR.{h}', color='red', x_label='Pixel Intensity',
               y_label='% Image', title=f'Log histogram: {l} -> {h}')

    _add_curve(axes[0, 0], histo_pred,
               label=f'PRED.{h}', color='blue', x_label='Pixel Intensity',
               y_label='% Image', title=f'Log histogram: {l} -> {h}')

    _add_curve(axes[0, 0], histo_interp,
               label=f'INTER.{l}', color='green', x_label='Pixel Intensity',
               y_label='% Image', title=f'Log histogram: {l} -> {h}')

    _add_curve(axes[0, 1], histo_ss,
               label=f'HR.{h}', color='red', x_label='Pixel Intensity',
               y_label='% Image', title=f'Log histogram: {l} -> {h}')

    _add_curve(axes[0, 2], histo_pred,
               label=f'PRED.{h}', color='blue', x_label='Pixel Intensity',
               y_label='% Image', title=f'Log histogram: {l} -> {h}')

    _add_curve(axes[0, 3], histo_interp,
               label=f'INTER.{l}', color='green', x_label='Pixel Intensity',
               y_label='% Image', title=f'Log histogram: {l} -> {h}')

    _closing(fig, fout)


def _get_vals(l: list) -> list:
    return [e[1] for e in l]


def _plot_x_y(super_res: np.ndarray,
              pred: np.ndarray,
              inter: np.ndarray,
              fout: str,
              l: int,
              h: int):
    nrows = 1
    ncols = 4  # super-res, pred, interpolated.

    fh, fw = _get_figure_h_w(nrows=1, ncols=4, him=400, wim=400)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    flat_ss = super_res.flatten()
    flat_pred = pred.flatten()
    flat_interp = inter.flatten()

    l_ss = []
    l_pred = []

    error_pred = []
    error_inter = []

    for i in range(flat_interp.size):
        l_ss.append([flat_interp[i].item(), flat_ss[i].item()])

        l_pred.append([flat_interp[i].item(), flat_pred[i].item()])

        error_pred.append([flat_interp[i],
                           abs(flat_pred[i].item() - flat_ss[i].item())])

        error_inter.append([flat_interp[i],
                            abs(flat_interp[i].item() - flat_ss[i].item())])

    # x -> y

    _add_scatter(axes[0, 0], vals=np.asarray(_get_vals(l_ss)), x=flat_interp,
                 label='LR(X) -> HR(Y)', color='red',
                 x_label='X: Pixel Intensity', y_label='Y',
                 title=f'X --> Y. {l} -> {h}')

    _add_scatter(axes[0, 1], vals=np.asarray(_get_vals(l_pred)),
                 x=flat_interp, label='LR(X) -> PRED(Y)', color='blue',
                 x_label='X: Pixel Intensity', y_label='Y',
                 title=f'X --> Y. {l} -> {h}')

    # Error

    _add_scatter(axes[0, 2], vals=np.asarray(_get_vals(error_inter)),
                 x=flat_interp, label='|INTER - HR|', color='green',
                 x_label='X: Pixel Intensity', y_label='Y',
                 title=f'Error. {l} -> {h}')

    _add_scatter(axes[0, 3], vals=np.asarray(_get_vals(error_pred)),
                 x=flat_interp, label='|PRED - HR|', color='blue',
                 x_label='X: Pixel Intensity', y_label='Y',
                 title=f'Error. {l} -> {h}')

    _closing(fig, fout)

    # Residuals

    fh, fw = _get_figure_h_w(nrows=1, ncols=3, him=400, wim=400)

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    x = list(range(-255, 256, 1))
    vals_inter = np.asarray(_get_vals(error_inter))
    res_inter = unnormed_histogram(vals_inter, len(x), range=(-255, 255))[0]

    vals_pred = np.asarray(_get_vals(error_pred))
    res_pred = unnormed_histogram(vals_pred, len(x), range=(-255, 255))[0]

    _add_histo_residuals(axes[0, 0], vals=res_inter,
                         x=x, label='HR - INTER', color='green',
                         x_label='X: Error', y_label='Count',
                         title=f'Log(Residuals). {l} -> {h}')

    _add_histo_residuals(axes[0, 0], vals=res_pred,
                         x=x, label='HR - PRED', color='blue',
                         x_label='X: Error', y_label='Count',
                         title=f'Log(Residuals). {l} -> {h}')

    _add_histo_residuals(axes[0, 1], vals=res_inter,
                         x=x, label='HR - INTER', color='green',
                         x_label='X: Error', y_label='Count',
                         title=f'Log(Residuals). {l} -> {h}')

    _add_histo_residuals(axes[0, 2], vals=res_pred,
                         x=x, label='HR - PRED', color='blue',
                         x_label='X: Error', y_label='Count',
                         title=f'Log(Residuals). {l} -> {h}')

    tmp = splitext(basename(fout))[0] + f'--residuals' \
                                        f'{splitext(basename(fout))[1]}'
    _closing(fig, join(dirname(fout), tmp))

    # hexbin

    fh, fw = _get_figure_h_w(nrows=1, ncols=4, him=400, wim=400)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False, figsize=(fw, fh))

    # x -> y

    _add_hexbin(axes[0, 0], vals=np.asarray(_get_vals(l_ss)), x=flat_interp,
                x_label='X: Pixel Intensity', y_label='Y',
                title=f'LR(X) -> HR(Y). {l} -> {h}', grid_size=(50, 50))

    _add_hexbin(axes[0, 1], vals=np.asarray(_get_vals(l_pred)),
                x=flat_interp,
                x_label='X: Pixel Intensity', y_label='Y',
                title=f'LR(X) -> PRED(Y). {l} -> {h}', grid_size=(50, 50))

    # Error

    _add_hexbin(axes[0, 2], vals=np.asarray(_get_vals(error_inter)),
                x=flat_interp,
                x_label='X: Pixel Intensity', y_label='Y',
                title=f'Error: |INTER - HR|. {l} -> {h}', grid_size=(50, 50))

    _add_hexbin(axes[0, 3], vals=np.asarray(_get_vals(error_pred)),
                x=flat_interp,
                x_label='X: Pixel Intensity', y_label='Y',
                title=f'Error: |PRED - HR|. {l} -> {h}', grid_size=(50, 50))

    tmp = splitext(basename(fout))[0] + f'--hexbin{splitext(basename(fout))[1]}'
    _closing(fig, join(dirname(fout), tmp))


def min_max(path_ds: str, sizes: dict, ext: str):
    stats = _check_range_all(path_ds, sizes, ext)
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(stats)


def compare(super_res: dict, pred: dict, interp: dict, fdout: str, l: int,
            h: int):
    os.makedirs(fdout, exist_ok=True)
    n = len(list(super_res.keys()))

    for k in tqdm(super_res, ncols=80, total=n):
        cell_type = get_cell_type(super_res[k])

        sres_p = super_res[k]
        p_p = pred[k]
        int_p = interp[k]

        ss = _load_cell(sres_p)
        p = _load_cell(p_p)
        int_p = _load_cell(int_p)

        assert ss.shape == p.shape, f'{ss.shape}, {p.shape}'
        assert ss.shape == int_p.shape, f'{ss.shape}, {int_p.shape}'

        # images
        images = {
            'super_res': _build_img(ss, cell_type),
            'pred': _build_img(p, cell_type),
            'interp': _build_img(int_p, cell_type)
        }

        # out folder per file
        outd_file = join(fdout, k)
        os.makedirs(outd_file, exist_ok=True)

        # 1. Images
        _plot_images(ss, p, int_p, join(outd_file, 'images.png'), images, l, h)

        # 2. histogram
        _plot_histogram(ss, p, int_p, join(outd_file, 'histo.png'), l, h)

        # 3. x -> y.
        _plot_x_y(ss, p, int_p, join(outd_file, 'x-y-error.png'), l, h)


def add_noise_global_var(m: np.ndarray, var_g: float) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=np.sqrt(var_g), size=m.shape)

    return m + noise


def hash_tensor(x: torch.Tensor) -> str:
    assert isinstance(x, torch.Tensor), type(x)
    assert x.dtype == torch.long, x.dtype
    assert x.ndim == 1, x.ndim

    out = x.tolist()
    out = [str(v) for v in out]
    out = '_'.join(out)
    return out


def build_mapping(limg: torch.Tensor, himg: torch.Tensor, sz: int,
                  silent: bool = True) -> Tuple[dict, torch.Tensor, List]:
    assert isinstance(limg, torch.Tensor), type(limg)
    assert limg.dtype == torch.float, limg.dtype
    assert limg.ndim == 4, limg.ndim  # 1, 1, h', w'
    assert limg.shape[0] == 1, limg.shape[0]
    assert limg.shape[1] == 1, limg.shape[1]

    assert isinstance(himg, torch.Tensor), type(himg)
    assert himg.dtype == torch.float, limg.dtype
    assert himg.ndim == 4, himg.ndim  # 1, 1, h, w
    assert himg.shape[0] == 1, himg.shape[0]
    assert himg.shape[1] == 1, himg.shape[1]

    l_patches = F.unfold(limg, kernel_size=sz, dilation=1, padding=0,
                         stride=1).transpose(1, 2)  # 1, nbr-patches, szxsz
    l_patches = l_patches.squeeze(0)  # n1, sz*sz
    n1 = l_patches.shape[0]
    l_patches = l_patches.contiguous().view(n1, sz, sz)  # n1, sz, sz.

    h_patches = F.unfold(himg, kernel_size=2, dilation=1, padding=0,
                         stride=2).transpose(1, 2)  # 1, nbr-patches, 2x2
    h_patches = h_patches.squeeze(0)  # n2, 2*2
    n2 = h_patches.shape[0]
    h_patches = h_patches.contiguous().view(n2, 2, 2)  # n2, 2, 2.
    assert n1 == n2, f'n1 {n1} n2 {n2}'

    flat_l = l_patches.view(n1, -1).long()
    flat_h = h_patches.view(n2, -1).long()

    dict_atoms_h = dict()
    dict_atoms_l = dict()
    v_atoms_l = []
    v_atoms_h = []

    for i in tqdm(range(n1), total=n1, ncols=80, disable=silent):
        atom_l_flat = flat_l[i]
        atom_h_flat = flat_h[i]

        atom_h = h_patches[i]  # 2, 2
        atom_l = l_patches[i]  # sz, sz

        hash_atom_l = hash_tensor(atom_l_flat)
        hash_atom_h = hash_tensor(atom_h_flat)

        if hash_atom_l in dict_atoms_l:
            if hash_atom_h in dict_atoms_l[hash_atom_l]['high']:
                dict_atoms_l[hash_atom_l]['high'][hash_atom_h]['freq'] += 1

            else:
                dict_atoms_l[hash_atom_l]['high'][hash_atom_h] = {
                    'val': atom_h,  # 2, 2
                    'freq': 1
                }

                v_atoms_h.append(atom_h)

        else:
            dict_atoms_l[hash_atom_l] = {
                'val': atom_l,  # sz, sz
                'high': {
                    hash_atom_h: {
                        'val': atom_h,  # 2, 2
                        'freq': 1
                    }
                }
            }

            v_atoms_l.append(atom_l)

    dict_atoms_l_to_high = dict()
    atoms_l_out = []
    atoms_l_hash_out = []

    avg_n_atoms = 0.0

    for hash_atom_l in dict_atoms_l:
        dict_atoms_l_to_high[hash_atom_l] = {
            'val': dict_atoms_l[hash_atom_l]['val'],
            'high': get_stats(dict_atoms_l[hash_atom_l]['high'])
        }
        atoms_l_out.append(dict_atoms_l[hash_atom_l]['val'])
        atoms_l_hash_out.append(hash_atom_l)

        avg_n_atoms += len(dict_atoms_l_to_high[hash_atom_l]['high']['freqs'])

    atoms_l_out = torch.stack(atoms_l_out, dim=0)  # nbr, sz, sz

    sz_dict = len(list(dict_atoms_l.keys()))
    avg_n_atoms = avg_n_atoms / float(sz_dict)

    # if not silent:
    print(f'sz: {sz}x{sz}. nbr-patches: {n1}. '
          f'size dict: {sz_dict}. avg: {avg_n_atoms}')

    return dict_atoms_l_to_high, atoms_l_out, atoms_l_hash_out


def unfold_input(limg: torch.Tensor, sz: int) -> torch.Tensor:
    assert isinstance(limg, torch.Tensor), type(limg)
    assert limg.dtype == torch.float, limg.dtype
    assert limg.ndim == 4, limg.ndim  # 1, 1, h', w'
    assert limg.shape[0] == 1, limg.shape[0]
    assert limg.shape[1] == 1, limg.shape[1]

    l_patches = F.unfold(limg, kernel_size=sz, dilation=1, padding=0,
                         stride=1).transpose(1, 2)  # 1, nbr-patches, szxsz
    l_patches = l_patches.squeeze(0)  # npatches, sz*sz

    return l_patches


def get_stats(x: dict) -> dict:
    out = {
        'vals_hash': [],
        'vals': [],
        'freqs': [],
        'probs': [],
        'n': 0
    }
    for k in x:
        out['vals_hash'].append(k)
        out['vals'].append(x[k]['val'])
        out['freqs'].append(x[k]['freq'])

    freqs = copy.deepcopy(out['freqs'])

    # remove super-rare freqs
    # min_frq = 1
    # for i in range(len(freqs)):
    #     if freqs[i] <= min_frq:
    #         freqs[i] = 0
    #
    # if sum(freqs) == 0:
    #     freqs[0] = 1

    # print(freqs)
    # input('o')
    n = len(freqs)
    s = sum(freqs) * 1.
    probs = [z / s for z in freqs]
    # probs = np.asarray(probs)
    # probs = np.exp(10 * probs).tolist()
    out['probs'] = probs
    out['n'] = n

    return out


def exact_l_to_h(l_patches: torch.Tensor, dict_atoms_l_to_high: dict, h: int,
                 w: int, sz: int, silent: bool = True) -> torch.Tensor:
    assert h % 2 == 0, f'{h} % {2} = {h % 2}'
    assert w % 2 == 0, f'{w} % {2} = {w % 2}'

    assert l_patches.ndim == 2, l_patches.ndim  # npatches, sz*sz
    assert l_patches.shape[1] == sz * sz, f'{l_patches.shape[1]} {sz * sz}'

    npatches = l_patches.shape[0]

    out = []
    if not silent:
        print('Mapping low patches to high patches ...')

    for i in tqdm(range(l_patches.shape[0]), total=npatches, ncols=80,
                  disable=silent):
        hash_atom_l = hash_tensor(l_patches[i])

        # todo: use knn, k=1 to find the closest in case it is not in dict.

        high_vals_patches = dict_atoms_l_to_high[hash_atom_l]['high']['vals']
        probs = dict_atoms_l_to_high[hash_atom_l]['high']['probs']
        n = dict_atoms_l_to_high[hash_atom_l]['high']['n']

        out.append(high_vals_patches[
                       np.random.choice(a=n, size=1, replace=True,
                                        p=probs).item()
                   ]
                   )

    out = torch.stack(out, dim=0)  # npatches, 2, 2
    assert out.shape == (npatches, 2, 2)
    out = out.contiguous().view(npatches, 2 * 2)
    out = out.transpose(1, 0)  # 2*2, npatches
    out = out.unsqueeze(0)  # 1, 2*2, npatches
    out = F.fold(out, output_size=(h, w), kernel_size=2, dilation=1, padding=0,
                 stride=2)
    # 1, 1, h, w
    assert out.shape == (1, 1, h, w), f'{out.shape} (1, 1, {h}, {w})'

    return out


def low_to_h_mixed(l_patches: torch.Tensor, dict_atoms_l_to_high: dict,
                   ordered_keys_l: list, t_patches_l: torch.Tensor,
                   h: int, w: int, sz: int, kn: int, silent: bool = True):
    t_patches_l = t_patches_l.double()
    npatches = l_patches.shape[0]

    ring = get_ring(int(math.sqrt(l_patches.shape[1])), c=300.).to(
        l_patches.device)  #
    # sz, sz
    ring = ring.view(1, 1, -1).contiguous().double()
    # Using lazy tensors
    lazy_l_patches = LazyTensor(l_patches.unsqueeze(1).double().contiguous())
    lazy_t_patches_l = LazyTensor(t_patches_l.unsqueeze(0).contiguous())
    lt_ring = LazyTensor(ring)
    lazy_distance = (((lazy_l_patches - lazy_t_patches_l).abs()) * lt_ring).sum(
        dim=2)  # npatches, m
    # min_dis = lazy_distance.min(dim=1).squeeze(1)  # npacthes
    # zeros_dist = (min_dis == 0).double().mean()
    # print(f'zeros distance {zeros_dist * 100}  %')
    idx_patch_l = lazy_distance.argKmin(kn, dim=1)  # npatches, kn
    # idx_patch_l = idx_patch_l.squeeze(1)  # npatches.

    # idx_patch_l = idx_patch_l.tolist()

    out = []

    for i in tqdm(range(l_patches.shape[0]), total=npatches, ncols=80,
                  disable=silent):

        patch = None
        for zz in idx_patch_l[i]:
            hash_atom_l = ordered_keys_l[zz.item()]

            high_vals_patches = dict_atoms_l_to_high[hash_atom_l]['high']['vals']
            probs = dict_atoms_l_to_high[hash_atom_l]['high']['probs']
            n = dict_atoms_l_to_high[hash_atom_l]['high']['n']

            r = high_vals_patches[
                           np.random.choice(a=n, size=1, replace=True,
                                            p=probs).item()
                       ]
            if patch is None:
                patch = r

            else:
                patch = patch + r

        out.append(patch / float(kn))

    out = torch.stack(out, dim=0)  # npatches, 2, 2
    assert out.shape == (npatches, 2, 2)
    out = out.contiguous().view(npatches, 2 * 2)
    out = out.transpose(1, 0)  # 2*2, npatches
    out = out.unsqueeze(0)  # 1, 2*2, npatches
    out = F.fold(out, output_size=(h, w), kernel_size=2, dilation=1, padding=0,
                 stride=2)
    # 1, 1, h, w
    assert out.shape == (1, 1, h, w), f'{out.shape} (1, 1, {h}, {w})'

    return out


def get_keys_closest_patch(patch: torch.Tensor,
                           dict_atoms_l_to_high: dict) -> str:
    ring = get_ring(int(math.sqrt(patch.numel())), c=300.).view(-1).contiguous()

    out = None
    c_d = 0
    for k in dict_atoms_l_to_high:
        v = dict_atoms_l_to_high[k]['val'].contiguous().view(-1)

        # distance
        d = ((v - patch).abs() * ring).sum()

        if d == 0:
            return k

        elif out is None or d < c_d:
            out = k
            c_d = d

    return out


def low_to_h_iterative(l_patches: torch.Tensor, dict_atoms_l_to_high: dict,
                       t_patches_l: torch.Tensor,
                       h: int, w: int, sz: int, silent: bool = True):
    t_patches_l = t_patches_l.float()
    npatches = l_patches.shape[0]

    ring = get_ring(int(math.sqrt(l_patches.shape[1])), c=300.).to(
        l_patches.device)  #
    # sz, sz
    ring = ring.view(1, 1, -1).contiguous()
    # Using lazy tensors
    lazy_l_patches = LazyTensor(l_patches.unsqueeze(1).float().contiguous())
    lazy_t_patches_l = LazyTensor(t_patches_l.unsqueeze(0).contiguous())
    lt_ring = LazyTensor(ring)
    lazy_distance = (((lazy_l_patches - lazy_t_patches_l).abs()) * lt_ring).sum(
        dim=2)  # npatches, m
    min_dis = lazy_distance.min(dim=1).squeeze(1)  # npacthes
    zeros_dist = (min_dis == 0).float().mean()
    print(f'zeros distance {zeros_dist * 100}  %')
    idx_patch_l = lazy_distance.argKmin(1, dim=1)  # npatches, 1
    idx_patch_l = idx_patch_l.squeeze(1)  # npatches.

    idx_patch_l = idx_patch_l.tolist()

    out = []

    for i in tqdm(range(l_patches.shape[0]), total=npatches, ncols=80,
                  disable=silent):
        hash_atom_l = get_keys_closest_patch(l_patches[i],
                                             dict_atoms_l_to_high)

        high_vals_patches = dict_atoms_l_to_high[hash_atom_l]['high']['vals']
        probs = dict_atoms_l_to_high[hash_atom_l]['high']['probs']
        n = dict_atoms_l_to_high[hash_atom_l]['high']['n']

        out.append(high_vals_patches[
                       np.random.choice(a=n, size=1, replace=True,
                                        p=probs).item()
                   ]
                   )

    out = torch.stack(out, dim=0)  # npatches, 2, 2
    assert out.shape == (npatches, 2, 2)
    out = out.contiguous().view(npatches, 2 * 2)
    out = out.transpose(1, 0)  # 2*2, npatches
    out = out.unsqueeze(0)  # 1, 2*2, npatches
    out = F.fold(out, output_size=(h, w), kernel_size=2, dilation=1, padding=0,
                 stride=2)
    # 1, 1, h, w
    assert out.shape == (1, 1, h, w), f'{out.shape} (1, 1, {h}, {w})'

    return out


def get_ring(sz: int, c: float = 1):
    assert sz % 2 == 1, sz

    if sz == 1:
        return torch.ones(1, 1)

    p = []

    z = math.ceil(sz / 2)
    for i in range(z):
        if i == z - 1:
            p.append((i + 1) * c)
        else:
            p.append((i + 1))

    p = p[::-1]
    a = torch.ones(1, 1) * p[0]

    for v in p[1:]:
        a = F.pad(a, (1, 1, 1, 1), mode='constant', value=v)

    assert a.shape[0] == sz, f'{a.shape[0]} {sz}'
    assert a.shape[1] == sz, f'{a.shape[1]} {sz}'

    return a


def vectorized_l_to_h(l_patches: torch.Tensor,
                      t_patches_l: torch.Tensor,
                      t_probs: torch.Tensor,
                      t_n: torch.Tensor,
                      atoms_h: list,
                      atoms_h_tensor: torch.Tensor,
                      h: int,
                      w: int,
                      chunk_sz: int = 1024) -> torch.Tensor:
    """
    :param l_patches: npatches, sz*sz
    :param t_patches_l: m, sz*sz
    :param t_probs: m, r
    :param t_n: m
    :param atoms_h: m
    :return:
    """
    t_patches_l = t_patches_l.float()
    npatches = l_patches.shape[0]

    # memory hungry...
    idx_patch_l = None
    print(f'NBR patches: {npatches}')

    # does not scale.

    # for chunk in torch.split(l_patches, chunk_sz):
    #     distance = torch.cdist(chunk.unsqueeze(0).float(),
    #                            t_patches_l.unsqueeze(0),
    #                            p=2.0)  # 1, chunksz, m
    #     distance = distance.squeeze(0)  # chunksz, m
    #     if idx_patch_l is None:
    #         idx_patch_l = distance.argmin(dim=1)  # chunksz
    #
    #     else:
    #         idx_patch_l = torch.cat((idx_patch_l,
    #                                  distance.argmin(dim=1)),
    #                                 dim=0)

    ring = get_ring(int(math.sqrt(l_patches.shape[1])), c=300.).to(
        l_patches.device)  #
    # sz, sz
    ring = ring.view(1, 1, -1).contiguous()
    # Using lazy tensors
    lazy_l_patches = LazyTensor(l_patches.unsqueeze(1).float().contiguous())
    lazy_t_patches_l = LazyTensor(t_patches_l.unsqueeze(0).contiguous())
    lt_ring = LazyTensor(ring)
    lazy_distance = (((lazy_l_patches - lazy_t_patches_l).abs()) * lt_ring).sum(
        dim=2)  # npatches, m
    # min_dis = lazy_distance.min(dim=1).squeeze(1)  # npacthes
    # zeros_dist = (min_dis == 0).float().mean()
    # print(f'zeros distance {zeros_dist * 100}  %')
    idx_patch_l = lazy_distance.argKmin(1, dim=1)  # npatches, 1
    idx_patch_l = idx_patch_l.squeeze(1)  # npatches.

    assert idx_patch_l.shape[0] == npatches, f'{idx_patch_l.shape[0]} ' \
                                             f'{npatches}'

    # one step: memory-expensive.
    # distance = torch.cdist(l_patches.unsqueeze(0).float(),
    #                        t_patches_l.unsqueeze(0).float(),
    #                        p=2.0)  # 1, npatches, m
    #
    # distance = distance.squeeze(0)  # npatches, m
    # idx_patch_l = distance.argmin(dim=1)  # npatches

    probs = t_probs[idx_patch_l]  # npatches, r
    first_idx_h = torch.multinomial(probs, 1)  # npatches, 1
    first_idx_h = first_idx_h.squeeze(1)  # npatches
    # make sure we didnt sample larger than n
    z = (first_idx_h >= t_n[idx_patch_l]).float().sum()
    assert z == 0, z
    out = atoms_h_tensor[idx_patch_l, first_idx_h]  # npacthes, 2, 2

    assert out.shape == (npatches, 2, 2)
    out = out.contiguous().view(npatches, 2 * 2)
    out = out.transpose(1, 0)  # 2*2, npatches
    out = out.unsqueeze(0)  # 1, 2*2, npatches
    out = F.fold(out.float(), output_size=(h, w), kernel_size=2, dilation=1,
                 padding=0, stride=2)
    # 1, 1, h, w
    assert out.shape == (1, 1, h, w), f'{out.shape} (1, 1, {h}, {w})'

    return out


def dict_to_tensor(dict_atoms_l_to_high: dict):
    keys = list(dict_atoms_l_to_high.keys())
    n = len(keys)

    s = dict_atoms_l_to_high[keys[0]]['val'].numel()
    cpu = torch.device("cpu")

    t_patches_l = torch.zeros((n, s), dtype=torch.long, requires_grad=False,
                              device=cpu)
    atoms_h = []

    n_a = -1
    # lowpatches, natoms, atoms high.
    for i, k in enumerate(keys):
        t_patches_l[i] = dict_atoms_l_to_high[k]['val'].contiguous().view(-1)
        n_a = max(n_a, dict_atoms_l_to_high[k]['high']['n'])
        vals = dict_atoms_l_to_high[k]['high']['vals']
        atoms_h.append(vals)

    # probs, n
    t_n = torch.zeros((n,), dtype=torch.long, requires_grad=False, device=cpu)
    t_probs = torch.zeros((n, n_a), dtype=torch.float, requires_grad=False,
                          device=cpu)

    atoms_h_tensor = torch.zeros((n, n_a, 2, 2), dtype=torch.long,
                                 requires_grad=False, device=cpu)

    for i, k in enumerate(keys):
        t_n[i] = dict_atoms_l_to_high[k]['high']['n']
        t_probs[i] = torch.zeros((n_a,), dtype=torch.float,
                                 requires_grad=False, device=cpu)
        _n = t_n[i]
        p = dict_atoms_l_to_high[k]['high']['probs']
        p = torch.tensor(p, dtype=torch.float, device=cpu, requires_grad=False)
        p = p.view(-1)
        t_probs[i, :_n] = p

        vals = dict_atoms_l_to_high[k]['high']['vals']
        for j, atom in enumerate(vals):
            atoms_h_tensor[i, j] = atom

    return t_patches_l, t_probs, t_n, atoms_h, atoms_h_tensor


def vectorize_l_patches(dict_atoms_l_to_high: dict):
    keys = list(dict_atoms_l_to_high.keys())
    n = len(keys)

    s = dict_atoms_l_to_high[keys[0]]['val'].numel()
    cpu = torch.device("cpu")

    t_patches_l = torch.zeros((n, s), dtype=torch.long, requires_grad=False,
                              device=cpu)

    ordered_keys_l: list = []
    for i, k in enumerate(keys):
        t_patches_l[i] = dict_atoms_l_to_high[k]['val'].contiguous().view(-1)
        ordered_keys_l.append(k)

    return t_patches_l, ordered_keys_l


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
# https://pytorch.org/docs/stable/generated/torch.cdist.html


def build_low_to_high_map_single_img(
        dict_atoms_l: dict, limg: torch.Tensor, himg: torch.Tensor, sz: int):
    assert isinstance(limg, torch.Tensor), type(limg)
    assert limg.dtype == torch.float, limg.dtype
    assert limg.ndim == 4, limg.ndim  # 1, 1, h', w'
    assert limg.shape[0] == 1, limg.shape[0]
    assert limg.shape[1] == 1, limg.shape[1]

    assert isinstance(himg, torch.Tensor), type(himg)
    assert himg.dtype == torch.float, limg.dtype
    assert himg.ndim == 4, himg.ndim  # 1, 1, h, w
    assert himg.shape[0] == 1, himg.shape[0]
    assert himg.shape[1] == 1, himg.shape[1]

    l_patches = F.unfold(limg, kernel_size=sz, dilation=1, padding=0,
                         stride=1).transpose(1, 2)  # 1, nbr-patches, szxsz
    l_patches = l_patches.squeeze(0)  # n1, sz*sz
    n1 = l_patches.shape[0]
    l_patches = l_patches.contiguous().view(n1, sz, sz)  # n1, sz, sz.

    h_patches = F.unfold(himg, kernel_size=2, dilation=1, padding=0,
                         stride=2).transpose(1, 2)  # 1, nbr-patches, 2x2
    h_patches = h_patches.squeeze(0)  # n2, 2*2
    n2 = h_patches.shape[0]
    h_patches = h_patches.contiguous().view(n2, 2, 2)  # n2, 2, 2.
    assert n1 == n2, f'n1 {n1} n2 {n2}'

    flat_l = l_patches.view(n1, -1).long()
    flat_h = h_patches.view(n2, -1).long()

    for i in tqdm(range(n1), total=n1, ncols=80, disable=True):
        atom_l_flat = flat_l[i]
        atom_h_flat = flat_h[i]

        atom_h = h_patches[i]  # 2, 2
        atom_l = l_patches[i]  # sz, sz

        hash_atom_l = hash_tensor(atom_l_flat)
        hash_atom_h = hash_tensor(atom_h_flat)

        if hash_atom_l in dict_atoms_l:
            if hash_atom_h in dict_atoms_l[hash_atom_l]['high']:
                dict_atoms_l[hash_atom_l]['high'][hash_atom_h]['freq'] += 1

            else:
                dict_atoms_l[hash_atom_l]['high'][hash_atom_h] = {
                    'val': atom_h,  # 2, 2
                    'freq': 1
                }

        else:
            dict_atoms_l[hash_atom_l] = {
                'val': atom_l,  # sz, sz
                'high': {
                    hash_atom_h: {
                        'val': atom_h,  # 2, 2
                        'freq': 1
                    }
                }
            }

    return dict_atoms_l


def build_train_dict(paths: dict, sz: str, super_sz: str, outdir: str,
                     sz_l: int, logfile):
    print('Create train dict...')

    assert sz_l % 2 == 1, sz_l
    _SILENT = True

    os.makedirs(outdir, exist_ok=True)

    n = len(list(paths.keys()))

    dict_atoms_l = dict()

    zz = 0

    for k in tqdm(paths, ncols=80, total=n):
        low_path = paths[k]['l']
        high_path = paths[k]['h']

        if _is_biosr(low_path):
            low_path = low_path.split(constants.CODE_IDENTIFIER)[0]
            high_path = high_path.split(constants.CODE_IDENTIFIER)[0]

        path_src = low_path
        img_lr_np: np.ndarray = _load_cell(path_src)  # [0, 255]

        path_hr = high_path
        img_hr_np: np.ndarray = _load_cell(path_hr)  # [0, 255]

        img_l = torch.from_numpy(img_lr_np).long()
        img_l = img_l.unsqueeze(0).unsqueeze(0)  # 1, 1, h', w'
        padding = int((sz_l - 1) / 2)
        img_l = F.pad(img_l.float(), (padding, padding, padding, padding),
                      mode='reflect').long()
        img_h = torch.from_numpy(img_hr_np).long()
        img_h = img_h.unsqueeze(0).unsqueeze(0)  # 1, 1, h, w

        dict_atoms_l = build_low_to_high_map_single_img(
            dict_atoms_l, img_l.float(), img_h.float(), sz_l)

        zz += 1

        if zz == 10:
            break

    dict_atoms_l_to_high = dict()
    atoms_l_out = []
    atoms_l_hash_out = []

    avg_n_atoms = 0.0
    min_n_atoms = float('inf')
    max_n_atoms = 0

    for hash_atom_l in dict_atoms_l:
        high_stats = get_stats(dict_atoms_l[hash_atom_l]['high'])

        dict_atoms_l_to_high[hash_atom_l] = {
            'val': dict_atoms_l[hash_atom_l]['val'],
            'high': high_stats
        }
        atoms_l_out.append(dict_atoms_l[hash_atom_l]['val'])
        atoms_l_hash_out.append(hash_atom_l)

        z = len(dict_atoms_l_to_high[hash_atom_l]['high']['freqs'])
        avg_n_atoms += z
        min_n_atoms = min(min_n_atoms, z)
        max_n_atoms = max(max_n_atoms, z)

    atoms_l_out = torch.stack(atoms_l_out, dim=0)  # nbr, sz, sz

    sz_dict = len(list(dict_atoms_l.keys()))
    avg_n_atoms = avg_n_atoms / float(sz_dict)

    # if not silent:
    info = f'{sz_l} -> {super_sz}: \n' \
           f'Low patch size: {sz}x{sz} \n' \
           f'Number of low res. patches (size dict.): {sz_dict} \n' \
           f'High res. atoms: {2}x{2}: \n' \
           f'--> min. nbr. atoms: {min_n_atoms} \n' \
           f'--> max. nbr. atoms: {max_n_atoms} \n' \
           f'--> avg. nbr. atoms: {avg_n_atoms}. \n'
    print(info)

    logfile.write(info)

    return dict_atoms_l_to_high, atoms_l_out, atoms_l_hash_out


def evaluate_vectorized(paths: dict, sz: str, super_sz: str, outdir: str,
                        sz_l: int, logfile,
                        t_patches_l: torch.Tensor,
                        t_probs: torch.Tensor,
                        t_n: torch.Tensor,
                        atoms_h_tensor: torch.Tensor,
                        chunk_sz: int = 1024):
    print('Evaluating...')
    assert sz_l % 2 == 1, sz_l
    _SILENT = True

    outdir = join(outdir, sz)
    os.makedirs(outdir, exist_ok=True)

    n = len(list(paths.keys()))

    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    perf = {
        'patch': {
            'MSE': [],
            'PSNR': [],
        },
        'bicubic': {
            'MSE': [],
            'PSNR': []
        }
    }

    process_time = 0

    for k in tqdm(paths, ncols=80, total=n):
        low_path = paths[k]['l']
        high_path = paths[k]['h']

        cell_type = get_cell_type(low_path)

        path_src = low_path
        img_lr_np: np.ndarray = _load_cell(path_src)  # [0, 255]

        path_hr = high_path
        img_hr_np: np.ndarray = _load_cell(path_hr)  # [0, 255]

        img_l = torch.from_numpy(img_lr_np).long()
        img_l = img_l.unsqueeze(0).unsqueeze(0)  # 1, 1, h', w'
        padding = int((sz_l - 1) / 2)
        img_l = F.pad(img_l.float(), (padding, padding, padding, padding),
                      mode='reflect').long()

        l_patches = unfold_input(img_l.float(), sz_l)  # n, sz*sz

        l_patches = l_patches.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        with torch.no_grad():
            sr_tensor_v = vectorized_l_to_h(l_patches=l_patches,
                                            t_patches_l=t_patches_l,
                                            t_probs=t_probs,
                                            t_n=t_n,
                                            atoms_h=atoms_h,
                                            atoms_h_tensor=atoms_h_tensor,
                                            h=int(super_sz),
                                            w=int(super_sz),
                                            chunk_sz=chunk_sz).cpu()  # 1, 1,
            # h, w.

        end.record()
        torch.cuda.synchronize()
        c_time = start.elapsed_time(end)  # float
        process_time += c_time

        # zzz = 100
        # for kk in range(zzz):
        #     torch.cuda.empty_cache()
        #     print(kk)
        #     with torch.no_grad():
        #         sr_tensor_v = sr_tensor_v + vectorized_l_to_h(
        #             l_patches=l_patches,
        #                                     t_patches_l=t_patches_l,
        #                                     t_probs=t_probs,
        #                                     t_n=t_n,
        #                                     atoms_h=atoms_h,
        #                                     atoms_h_tensor=atoms_h_tensor,
        #                                     h=int(super_sz),
        #                                     w=int(super_sz),
        #                                     chunk_sz=chunk_sz).cpu()  # 1, 1,
        #         # h, w.
        #
        # sr_tensor_v = sr_tensor_v / (zzz + 1)

        # print(f'GPU time: {c_time} (ms)')
        sr_tensor_v = sr_tensor_v.squeeze().cpu().numpy().astype(
            np.uint8)  #
        # h, w
        # ----------------------------------------------------------------------

        # interpolated
        img = img_lr_np.copy()
        h_s, w_s = int(super_sz), int(super_sz)
        img_l_2_h = Image.fromarray(img).resize((w_s, h_s),
                                                resample=PIL.Image.BICUBIC)
        img_l_2_h = np.array(img_l_2_h)

        # psnr
        psnr_patch_tensor = calculate_psnr(sr_tensor_v, img_hr_np)
        psnr_bicubic = calculate_psnr(img_l_2_h, img_hr_np)

        perf['patch']['PSNR'].append(psnr_patch_tensor)
        perf['bicubic']['PSNR'].append(psnr_bicubic)

        # mse
        mse_patch_tensor = calculate_mse(sr_tensor_v, img_hr_np).item()
        mse_bicubic = calculate_mse(img_l_2_h, img_hr_np).item()

        perf['patch']['MSE'].append(mse_patch_tensor)
        perf['bicubic']['MSE'].append(mse_bicubic)

        # plot

        limgs = [
            _build_img(img_l_2_h, cell_type),
            _build_img(sr_tensor_v, cell_type),
            _build_img(img_hr_np, cell_type)
        ]
        ltags = [
            f'Interpolated: {sz} -> {super_sz}. '
            f'MSE: {mse_bicubic:.3f}. PSNR: {psnr_bicubic:.3f}',
            f'Patch: {sz} -> {super_sz}. '
            f'MSE: {mse_patch_tensor:.3f}. PSNR: {psnr_patch_tensor:.3f}',
            f'High res.: {super_sz}.'
        ]
        fout = join(outdir, f'{k}.png')
        gather_images(limgs, ltags, fout)

    process_time /= n
    logfile.write(f'Average process time (GPU): {process_time} / Image (ms).\n')
    # log perf.
    mse_bicubic_avg = sum(perf['bicubic']['MSE']) / float(n)
    mse_patch_avg = sum(perf['patch']['MSE']) / float(n)

    # psnr
    psnr_bicubic_avg = sum(perf['bicubic']['PSNR']) / float(n)

    psnr_real = 0.0
    nbr_psnr_real = 0.
    nbr_psnr_inf = 0.
    for vl in perf['patch']['PSNR']:

        if vl == float('inf'):
            nbr_psnr_inf += 1

        else:
            psnr_real += vl
            nbr_psnr_real += 1.

    psnr_patch_avg = psnr_real / float(nbr_psnr_real)

    msg = f'Performance:\n' \
          f'NBR samples: {n} \n' \
          f'AVG PSNR BICUBIC: {psnr_bicubic_avg} \n' \
          f'AVG MSE BICUBIC: {mse_bicubic_avg} \n' \
          f'AVG MSE PATCH: {mse_patch_avg} \n' \
          f'AVG PSNR PATCH: {psnr_patch_avg} ({nbr_psnr_real}).' \
          f' INF: {nbr_psnr_inf} \n' \
          f'AVG MSE PATCH: {mse_patch_avg} \n'

    logfile.write(msg)

    with open(join(fdout, f'perf-{sz}-{super_sz}-{sz_l}-2.yml'), 'w') as fx:
        yaml.dump({
            'low': sz,
            'high': super_sz,
            'low_patch_sz': sz_l,
            "high_atom_size": 2,
            'avg_psnr_bicubic': psnr_bicubic_avg,
            'avg_mse_bicubic': mse_bicubic_avg,
            "mse_patch_avg": mse_patch_avg,
            'avg_psnr_patch_real': psnr_patch_avg,
            'nbr_psnr_real_patch': nbr_psnr_real,
            'nbr_psnr_inf_patch': nbr_psnr_inf,
            'nbr_samples': n
        }, fx)


def evaluate_mixed(paths: dict, sz: str, super_sz: str, outdir: str,
                   sz_l: int, logfile, t_patches_l: torch.Tensor,
                   ordered_keys_l: list, kn: int):
    print('Evaluating...')
    assert sz_l % 2 == 1, sz_l
    _SILENT = True

    os.makedirs(outdir, exist_ok=True)

    n = len(list(paths.keys()))

    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    perf = {
        'patch': {
            'MSE': [],
            'PSNR': [],
        },
        'bicubic': {
            'MSE': [],
            'PSNR': []
        }
    }

    process_time = 0

    for k in tqdm(paths, ncols=80, total=n):
        low_path = paths[k]['l']
        high_path = paths[k]['h']

        if _is_biosr(low_path):
            low_path = low_path.split(constants.CODE_IDENTIFIER)[0]
            high_path = high_path.split(constants.CODE_IDENTIFIER)[0]

        cell_type = None

        path_src = low_path
        img_lr_np: np.ndarray = _load_cell(path_src)  # [0, 255]

        path_hr = high_path
        img_hr_np: np.ndarray = _load_cell(path_hr)  # [0, 255]

        img_l = torch.from_numpy(img_lr_np).long()
        img_l = img_l.unsqueeze(0).unsqueeze(0)  # 1, 1, h', w'
        padding = int((sz_l - 1) / 2)
        img_l = F.pad(img_l.float(), (padding, padding, padding, padding),
                      mode='reflect').long()

        l_patches = unfold_input(img_l.float(), sz_l)  # n, sz*sz

        l_patches = l_patches.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        sr_tensor_v = low_to_h_mixed(l_patches, dict_atoms_l_to_high,
                                     ordered_keys_l, t_patches_l,
                                     h=int(super_sz), w=int(super_sz), sz=sz,
                                     kn=kn, silent=True)  # 1, 1, h, w.

        end.record()
        torch.cuda.synchronize()
        c_time = start.elapsed_time(end)  # float
        process_time += c_time

        # print(f'GPU time: {c_time} (ms)')
        sr_tensor_v = sr_tensor_v.squeeze().cpu().numpy().astype(
            np.uint8)  #
        # h, w
        # ----------------------------------------------------------------------

        # interpolated
        img = img_lr_np.copy()
        h_s, w_s = int(super_sz), int(super_sz)
        img_l_2_h = Image.fromarray(img).resize((w_s, h_s),
                                                resample=PIL.Image.BICUBIC)
        img_l_2_h = np.array(img_l_2_h)

        # psnr
        psnr_patch_tensor = calculate_psnr(sr_tensor_v, img_hr_np)
        psnr_bicubic = calculate_psnr(img_l_2_h, img_hr_np)

        perf['patch']['PSNR'].append(psnr_patch_tensor)
        perf['bicubic']['PSNR'].append(psnr_bicubic)

        # mse
        mse_patch_tensor = calculate_mse(sr_tensor_v, img_hr_np).item()
        mse_bicubic = calculate_mse(img_l_2_h, img_hr_np).item()

        perf['patch']['MSE'].append(mse_patch_tensor)
        perf['bicubic']['MSE'].append(mse_bicubic)

        # plot

        limgs = [
            _build_img(img_l_2_h, cell_type),
            _build_img(sr_tensor_v, cell_type),
            _build_img(img_hr_np, cell_type)
        ]
        ltags = [
            f'Interpolated: {sz} -> {super_sz}. '
            f'MSE: {mse_bicubic:.3f}. PSNR: {psnr_bicubic:.3f}',
            f'Patch: {sz} -> {super_sz}. '
            f'MSE: {mse_patch_tensor:.3f}. PSNR: {psnr_patch_tensor:.3f}',
            f'High res.: {super_sz}.'
        ]
        fout = join(outdir, f'{k}.png')
        gather_images(limgs, ltags, fout)

    process_time /= n
    logfile.write(f'Average process time (GPU): {process_time} / Image (ms).\n')
    # log perf.
    mse_bicubic_avg = sum(perf['bicubic']['MSE']) / float(n)
    mse_patch_avg = sum(perf['patch']['MSE']) / float(n)

    # psnr
    psnr_bicubic_avg = sum(perf['bicubic']['PSNR']) / float(n)

    psnr_real = 0.0
    nbr_psnr_real = 0.
    nbr_psnr_inf = 0.
    for vl in perf['patch']['PSNR']:

        if vl == float('inf'):
            nbr_psnr_inf += 1

        else:
            psnr_real += vl
            nbr_psnr_real += 1.

    psnr_patch_avg = psnr_real / float(nbr_psnr_real)

    msg = f'Performance:\n' \
          f'NBR samples: {n} \n' \
          f'AVG PSNR BICUBIC: {psnr_bicubic_avg} \n' \
          f'AVG MSE BICUBIC: {mse_bicubic_avg} \n' \
          f'AVG MSE PATCH: {mse_patch_avg} \n' \
          f'AVG PSNR PATCH: {psnr_patch_avg} ({nbr_psnr_real}).' \
          f' INF: {nbr_psnr_inf} \n' \
          f'AVG MSE PATCH: {mse_patch_avg} \n'

    logfile.write(msg)

    with open(join(fdout, f'perf-{sz}-{super_sz}-{sz_l}-2.yml'), 'w') as fx:
        yaml.dump({
            'low': sz,
            'high': super_sz,
            'low_patch_sz': sz_l,
            "high_atom_size": 2,
            'avg_psnr_bicubic': psnr_bicubic_avg,
            'avg_mse_bicubic': mse_bicubic_avg,
            "mse_patch_avg": mse_patch_avg,
            'avg_psnr_patch_real': psnr_patch_avg,
            'nbr_psnr_real_patch': nbr_psnr_real,
            'nbr_psnr_inf_patch': nbr_psnr_inf,
            'nbr_samples': n
        }, fx)


def evaluate_iterative(paths: dict, sz: str, super_sz: str, outdir: str,
                       sz_l: int, logfile, t_patches_l: torch.Tensor):
    print('Evaluating...')
    assert sz_l % 2 == 1, sz_l
    _SILENT = True

    outdir = join(outdir, sz)
    os.makedirs(outdir, exist_ok=True)

    n = len(list(paths.keys()))

    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    perf = {
        'patch': {
            'MSE': [],
            'PSNR': [],
        },
        'bicubic': {
            'MSE': [],
            'PSNR': []
        }
    }

    process_time = 0

    for k in tqdm(paths, ncols=80, total=n):
        low_path = paths[k]['l']
        high_path = paths[k]['h']

        cell_type = get_cell_type(low_path)

        path_src = low_path
        img_lr_np: np.ndarray = _load_cell(path_src)  # [0, 255]

        path_hr = high_path
        img_hr_np: np.ndarray = _load_cell(path_hr)  # [0, 255]

        img_l = torch.from_numpy(img_lr_np).long()
        img_l = img_l.unsqueeze(0).unsqueeze(0)  # 1, 1, h', w'
        padding = int((sz_l - 1) / 2)
        img_l = F.pad(img_l.float(), (padding, padding, padding, padding),
                      mode='reflect').long()

        l_patches = unfold_input(img_l.float(), sz_l)  # n, sz*sz

        l_patches = l_patches.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        sr_tensor_v = low_to_h_iterative(
            l_patches, dict_atoms_l_to_high,
            t_patches_l, h=int(super_sz), w=int(super_sz), sz=sz,
            silent=True)  # 1, 1, h, w.

        end.record()
        torch.cuda.synchronize()
        c_time = start.elapsed_time(end)  # float
        process_time += c_time

        # print(f'GPU time: {c_time} (ms)')
        sr_tensor_v = sr_tensor_v.squeeze().cpu().numpy().astype(
            np.uint8)  #
        # h, w
        # ----------------------------------------------------------------------

        # interpolated
        img = img_lr_np.copy()
        h_s, w_s = int(super_sz), int(super_sz)
        img_l_2_h = Image.fromarray(img).resize((w_s, h_s),
                                                resample=PIL.Image.BICUBIC)
        img_l_2_h = np.array(img_l_2_h)

        # psnr
        psnr_patch_tensor = calculate_psnr(sr_tensor_v, img_hr_np)
        psnr_bicubic = calculate_psnr(img_l_2_h, img_hr_np)

        perf['patch']['PSNR'].append(psnr_patch_tensor)
        perf['bicubic']['PSNR'].append(psnr_bicubic)

        # mse
        mse_patch_tensor = calculate_mse(sr_tensor_v, img_hr_np).item()
        mse_bicubic = calculate_mse(img_l_2_h, img_hr_np).item()

        perf['patch']['MSE'].append(mse_patch_tensor)
        perf['bicubic']['MSE'].append(mse_bicubic)

        # plot

        limgs = [
            _build_img(img_l_2_h, cell_type),
            _build_img(sr_tensor_v, cell_type),
            _build_img(img_hr_np, cell_type)
        ]
        ltags = [
            f'Interpolated: {sz} -> {super_sz}. '
            f'MSE: {mse_bicubic:.3f}. PSNR: {psnr_bicubic:.3f}',
            f'Patch: {sz} -> {super_sz}. '
            f'MSE: {mse_patch_tensor:.3f}. PSNR: {psnr_patch_tensor:.3f}',
            f'High res.: {super_sz}.'
        ]
        fout = join(outdir, f'{k}.png')
        gather_images(limgs, ltags, fout)

    process_time /= n
    logfile.write(f'Average process time (GPU): {process_time} / Image (ms).\n')
    # log perf.
    mse_bicubic_avg = sum(perf['bicubic']['MSE']) / float(n)
    mse_patch_avg = sum(perf['patch']['MSE']) / float(n)

    # psnr
    psnr_bicubic_avg = sum(perf['bicubic']['PSNR']) / float(n)

    psnr_real = 0.0
    nbr_psnr_real = 0.
    nbr_psnr_inf = 0.
    for vl in perf['patch']['PSNR']:

        if vl == float('inf'):
            nbr_psnr_inf += 1

        else:
            psnr_real += vl
            nbr_psnr_real += 1.

    psnr_patch_avg = psnr_real / float(nbr_psnr_real)

    msg = f'Performance:\n' \
          f'NBR samples: {n} \n' \
          f'AVG PSNR BICUBIC: {psnr_bicubic_avg} \n' \
          f'AVG MSE BICUBIC: {mse_bicubic_avg} \n' \
          f'AVG MSE PATCH: {mse_patch_avg} \n' \
          f'AVG PSNR PATCH: {psnr_patch_avg} ({nbr_psnr_real}).' \
          f' INF: {nbr_psnr_inf} \n' \
          f'AVG MSE PATCH: {mse_patch_avg} \n'

    logfile.write(msg)

    with open(join(fdout, f'perf-{sz}-{super_sz}-{sz_l}-2.yml'), 'w') as fx:
        yaml.dump({
            'low': sz,
            'high': super_sz,
            'low_patch_sz': sz_l,
            "high_atom_size": 2,
            'avg_psnr_bicubic': psnr_bicubic_avg,
            'avg_mse_bicubic': mse_bicubic_avg,
            "mse_patch_avg": mse_patch_avg,
            'avg_psnr_patch_real': psnr_patch_avg,
            'nbr_psnr_real_patch': nbr_psnr_real,
            'nbr_psnr_inf_patch': nbr_psnr_inf,
            'nbr_samples': n
        }, fx)


def drawonit(draw, x, y, label, fill, font, dx):
    """
    Draw text on an ImageDraw.new() object.

    :param draw: object, ImageDraw.new()
    :param x: int, x position of top left corner.
    :param y: int, y position of top left corner.
    :param label: str, text message to draw.
    :param fill: color to use.
    :param font: font to use.
    :param dx: int, how much space to use between LABELS (not word).
    Useful to compute the position of the next
    LABEL. (future)
    :return:
        . ImageDraw object with the text drawn on it as requested.
        . The next position x.
    """
    draw.text((x, y), label, fill=fill, font=font)
    x += font.getsize(label)[0] + dx

    return draw, x


def gather_images(limgs: list, ltags: list, fout: str):
    sz = 15
    base = join(root_dir, "dlib/visualization/fonts/Inconsolata")
    font_regular = ImageFont.truetype(join(base, 'Inconsolata-Regular.ttf'), sz)
    font_bold = ImageFont.truetype(join(base, 'Inconsolata-Bold.ttf'), sz)

    wt, ht = limgs[0].size
    for im in limgs:
        w, h = im.size
        assert w == wt, f'{w} {wt}'
        assert h == ht, f'{h} {ht}'

    assert len(limgs) == len(ltags), f'{len(limgs)} {len(ltags)}'
    n = len(limgs)
    margin = 10
    txt_left_margin = 10
    y = 10
    dx = 10
    orange = "rgb(255,165,0)"
    red = "rgb(255, 0, 0)"

    out = Image.new("RGB", (wt * n + (n - 1) * margin, ht), orange)

    for i in range(n):
        img = limgs[i]
        tag = ltags[i]
        draw = ImageDraw.Draw(img)

        drawonit(draw, txt_left_margin, y, tag, red, font_bold, dx)

        out.paste(img, (wt * i + margin * i, 0), None)

    out.save(fout, optimize=False, quality=100)


if __name__ == '__main__':

    _VECTORIZED = 'vectorized'  # all is tensors. requires memory. GPU.
    _MIXED = 'mixed'  # only low patches are vectorized. fast: argmin. GPU/CPU
    _ITERATIVE = 'iterative'  # iterative. all cpu. no tensors.

    COMPUTATION = _MIXED

    master_dataset = ''  # fixit

    if master_dataset == '':
        INSZ = 512
        OUTSZ = 1024

        cell_names = {
            '0': '',
            '1': '',
            '2': '',
            '3': '',
            'ALL': 'ALL'
        }

        DATASETS = {
            '': {
                constants.TRAINSET: '',
                constants.VALIDSET: '',
                constants.TESTSET: ''
            },
            'ALL': {
                constants.TRAINSET: '',
                constants.VALIDSET: '',
                constants.TESTSET: ''
            }
        }

    elif master_dataset == constants.BIOSR:
        INSZ = 256
        OUTSZ = 512

        cell_names = {
            '0': constants.CCPS,
            '1': constants.ER,
            '2': constants.F_ACTIN,
            '3': constants.MICROTUBULES
        }

        DATASETS = {
            constants.CCPS: {
                constants.TRAINSET: constants.BIOSRV1_CCPS_TRAIN_X2,
                constants.VALIDSET: constants.BIOSRV1_CCPS_VALID_X2,
                constants.TESTSET: constants.BIOSRV1_CCPS_TEST_X2,
            },
            constants.ER: {
                constants.TRAINSET: constants.BIOSRV1_ER_TRAIN_X2,
                constants.VALIDSET: constants.BIOSRV1_ER_VALID_X2,
                constants.TESTSET: constants.BIOSRV1_ER_TEST_X2,
            },
            constants.F_ACTIN: {
                constants.TRAINSET: constants.BIOSRV1_F_ACTIN_TRAIN_X2,
                constants.VALIDSET: constants.BIOSRV1_F_ACTIN_VALID_X2,
                constants.TESTSET: constants.BIOSRV1_F_ACTIN_TEST_X2,
            },
            constants.MICROTUBULES: {
                constants.TRAINSET: constants.BIOSRV1_MICROTUBULES_TRAIN_X2,
                constants.VALIDSET: constants.BIOSRV1_MICROTUBULES_VALID_X2,
                constants.TESTSET: constants.BIOSRV1_MICROTUBULES_TEST_X2,
            }
        }
    else:
        raise NotImplementedError(master_dataset)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default='0', help="cell name.")
    parser.add_argument("--sz_l", type=int, default='3',
                        help="low res. patche size [int, odd].")
    parser.add_argument("--cudaid", type=int, default=0, help="CUDA id.")
    parser.add_argument("--kn", type=int, default=1, help="k for knn.")

    # high res atom size: 2x2.
    config = parser.parse_args()
    cell = config.cell
    sz_l = config.sz_l
    cudaid = config.cudaid
    kn = config.kn
    assert kn > 0, kn

    torch.cuda.set_device(cudaid)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    assert sz_l % 2 == 1, sz_l
    assert cell in cell_names, cell
    cell_name = cell_names[cell]

    task = constants.SUPER_RES
    fd = get_root_datasets(task)

    # merge train and valid.
    data_root = utils_config.get_root_datasets(task=task)
    splits_root = join(root_dir, constants.RELATIVE_META_ROOT, task)

    ds = DATASETS[cell_name][constants.TRAINSET]
    train_ = _get_low_to_high_dataset(join(splits_root, ds, 'l_h.txt'), ds,
                                      data_root)

    ds = DATASETS[cell_name][constants.VALIDSET]
    valid_ = _get_low_to_high_dataset(join(splits_root, ds, 'l_h.txt'), ds,
                                      data_root)

    train_.update(valid_)
    train_paths = train_

    ds = DATASETS[cell_name][constants.TESTSET]
    test_paths = _get_low_to_high_dataset(join(splits_root, ds, 'l_h.txt'), ds,
                                          data_root)

    fdout = join(root_dir,
                 f'data/debug/patch/demo/{master_dataset}/cell-{cell_name}/'
                 f'{INSZ}--to--{OUTSZ}-low_res_pacth-'
                 f'{sz_l}x{sz_l}-knn-{kn}')
    os.makedirs(fdout, exist_ok=True)

    super_sz = str(OUTSZ)

    os.makedirs(fdout, exist_ok=True)
    assert os.path.isdir(fdout), fdout
    logfile = open(join(fdout, 'log.txt'), 'w')

    # 1. create dict from train.
    t0 = time.perf_counter()
    dict_atoms_l_to_high, atoms_l_out, atoms_l_hash_out = build_train_dict(
        train_paths, sz='512', super_sz=super_sz, outdir=fdout,
        sz_l=sz_l, logfile=logfile)
    t1 = time.perf_counter()
    print(f'Time to build dict: {t1 - t0} (s) \n'
          f'Number of images used to build dict: '
          f'{len(list(train_paths.keys()))} \n')

    logfile.write(f'time to build dict: {t1 - t0} (s) \n')
    logfile.close()

    if COMPUTATION == _VECTORIZED:
        t_patches_l, t_probs, t_n, atoms_h, atoms_h_tensor = dict_to_tensor(
            dict_atoms_l_to_high)

        # transfer to device
        t_patches_l = t_patches_l.to(device)
        t_probs = t_probs.to(device)
        t_n = t_n.to(device)
        atoms_h_tensor = atoms_h_tensor.to(device)

        # 2. evaluate
        evaluate_vectorized(
            test_paths, sz=str(INSZ), super_sz=super_sz, outdir=fdout,
            sz_l=sz_l, logfile=logfile,
            t_patches_l=t_patches_l, t_probs=t_probs, t_n=t_n,
            atoms_h_tensor=atoms_h_tensor)

        logfile.close()

    elif COMPUTATION == _MIXED:
        t_patches_l, ordered_keys_l = vectorize_l_patches(dict_atoms_l_to_high)
        t_patches_l = t_patches_l.to(device)

        # eval train:
        # outd_tr = join(fdout, 'train')
        # os.makedirs(outd_tr, exist_ok=True)
        # logfile_tr = open(join(outd_tr, 'log.txt'), 'w')
        #
        # evaluate_mixed(
        #     train_paths, sz='512', super_sz=super_sz, outdir=outd_tr, sz_l=sz_l,
        #     logfile=logfile_tr, t_patches_l=t_patches_l,
        #     ordered_keys_l=ordered_keys_l)
        #
        # logfile_tr.close()

        # eval test
        outd_tst = join(fdout, 'test')
        os.makedirs(outd_tst, exist_ok=True)
        logfile_tst = open(join(outd_tst, 'log.txt'), 'w')

        evaluate_mixed(
            test_paths, sz=str(INSZ), super_sz=super_sz, outdir=outd_tst,
            sz_l=sz_l,
            logfile=logfile_tst, t_patches_l=t_patches_l,
            ordered_keys_l=ordered_keys_l, kn=kn)

        logfile_tst.close()

    elif COMPUTATION == _ITERATIVE:

        t_patches_l, ordered_keys_l = vectorize_l_patches(dict_atoms_l_to_high)
        t_patches_l = t_patches_l.to(device)

        evaluate_iterative(test_paths, sz=str(INSZ), super_sz=super_sz,
                           outdir=fdout, sz_l=sz_l, logfile=logfile,
                           t_patches_l=t_patches_l)

    else:
        raise NotImplementedError(COMPUTATION)