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
    im = Image.open(path, 'r').convert('RGB')
    im = np.array(im)  # h, w, 3

    cell_name = get_cell_type(path)
    assert cell_name is not None, path

    im = im[:, :, constants.PLAN_IMG[cell_name]]  # h, w.

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


def _build_img(x: np.ndarray, cell_type: str) -> Image.Image:

        assert isinstance(x, np.ndarray), type(x)
        assert x.ndim == 2, x.ndim  # HxW
        img = np.expand_dims(x, axis=2)  # HxWx1
        img = np.repeat(img, 3, axis=2)  # HW3: RGB

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

def _add_curve(ax, vals: np.ndarray, label:str, color: str, x_label: str,
               y_label: str, title: str):

    x =list(range(vals.size))

    ax.plot(x, vals, label=label, color=color, alpha=0.7)

    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=4)
    ax.set_ylabel(y_label, fontsize=4)
    ax.grid(True)
    ax.legend(loc='best',prop={'size': 4})
    ax.set_title(title, fontsize=4)


def _add_scatter(ax, vals: np.ndarray, x: np.ndarray, label:str, color: str,
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


def _add_histo_residuals(ax, vals: np.ndarray, x: np.ndarray, label:str,
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
                title=f'LR(X) -> PRED(Y). {l} -> {h}',grid_size=(50, 50))

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

    freqs = out['freqs']
    n = len(freqs)
    s = sum(freqs) * 1.
    probs = [z / s for z in freqs]
    out['probs'] = probs
    out['n'] = n

    return out


def exact_l_to_h(l_patches: torch.Tensor, dict_atoms_l_to_high: dict, h: int,
                 w: int, sz: int, silent: bool = True) -> torch.Tensor:
    assert h % 2 == 0, f'{h} % {2} = {h % 2}'
    assert w % 2 == 0, f'{w} % {2} = {w % 2}'

    assert l_patches.ndim == 2, l_patches.ndim  # npatches, sz*sz
    assert l_patches.shape[1] == sz*sz, f'{l_patches.shape[1]} {sz*sz}'

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
            np.random.choice(a=n, size=1, replace=True, p=probs).item()
        ]
                   )

    out = torch.stack(out, dim=0)  # npatches, 2, 2
    assert out.shape == (npatches, 2, 2)
    out = out.contiguous().view(npatches, 2*2)
    out = out.transpose(1, 0)  # 2*2, npatches
    out = out.unsqueeze(0)  # 1, 2*2, npatches
    out = F.fold(out, output_size=(h, w), kernel_size=2, dilation=1, padding=0,
    stride=2)
    # 1, 1, h, w
    assert out.shape == (1, 1, h, w), f'{out.shape} (1, 1, {h}, {w})'

    return out

def vectorized_l_to_h(l_patches: torch.Tensor,
                      t_patches_l: torch.Tensor,
                      t_probs: torch.Tensor,
                      t_n: torch.Tensor,
                      atoms_h: list,
                      atoms_h_tensor: torch.Tensor,
                      h: int,
                      w: int) -> torch.Tensor:
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
    chunk_sz = 1024
    idx_patch_l = None
    print(f'NBR patches: {npatches}')

    # still too expensive. does not scale when dict is very large.
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


    # Using lazy tensors
    lazy_l_patches = LazyTensor(l_patches.unsqueeze(1).float().contiguous())
    lazy_t_patches_l =LazyTensor(t_patches_l.unsqueeze(0).contiguous())
    lazy_distance = ((lazy_l_patches - lazy_t_patches_l)**2).sum(dim=2)  #
    # npatches, m
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
        t_probs[i] = torch.zeros((n_a, ), dtype=torch.float,
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

# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
# https://pytorch.org/docs/stable/generated/torch.cdist.html


def process_scale(paths: dict, sz: str, super_sz: str, outdir: str,
                  sizes_code: dict, sz_l: int):
    assert sz_l % 2 == 1, sz_l
    _SILENT = True

    outdir = join(outdir, sz)
    os.makedirs(outdir, exist_ok=True)

    n = len(list(paths[sz].keys()))

    cuda = 0
    torch.cuda.set_device(cuda)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    psnr = {
        'patch': 0.,
        'bicubic': 0.
    }

    r = 0
    for k in tqdm(paths[sz], ncols=80, total=n):
        cell_type = get_cell_type(paths[sz][k])

        path_src = paths[sz][k]
        img_lr_np: np.ndarray = _load_cell(path_src)  # [0, 255]

        k_hr = k.replace(sizes_code[sz], sizes_code[super_sz])
        if k_hr not in paths[super_sz]:
            continue

        path_hr = paths[super_sz][k_hr]
        img_hr_np: np.ndarray = _load_cell(path_hr)  # [0, 255]

        img_l = torch.from_numpy(img_lr_np).long()
        img_l = img_l.unsqueeze(0).unsqueeze(0)  # 1, 1, h', w'
        padding = int((sz_l - 1) / 2)
        img_l = F.pad(img_l.float(), (padding, padding, padding, padding),
                      mode='reflect').long()
        img_h = torch.from_numpy(img_hr_np).long()
        img_h = img_h.unsqueeze(0).unsqueeze(0)  # 1, 1, h, w

        t0 = time.perf_counter()

        dict_atoms_l_to_high, atoms_l_out, atoms_l_hash_out = build_mapping(
            img_l.float(), img_h.float(), sz_l, _SILENT)

        t1 = time.perf_counter()
        # if not _SILENT:
        print(f'time to build dict: {t1 - t0} (s)')

        l_patches = unfold_input(img_l.float(), sz_l)  # n, sz*sz

        t0 = time.perf_counter()

        sr = exact_l_to_h(l_patches.long(), dict_atoms_l_to_high,
                         h=int(super_sz), w=int(super_sz), sz=sz_l,
                         silent=_SILENT).long()

        t1 = time.perf_counter()
        # if not _SILENT:
        print(f'time to reconstruct: {t1 - t0} (s)')
        # tensor version -------------------------------------------------------
        t_patches_l, t_probs, t_n, atoms_h, atoms_h_tensor = dict_to_tensor(
            dict_atoms_l_to_high)
        # transfer to device
        t_patches_l = t_patches_l.to(device)
        t_probs = t_probs.to(device)
        t_n = t_n.to(device)
        atoms_h_tensor = atoms_h_tensor.to(device)

        l_patches = l_patches.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        sr_tensor_v = vectorized_l_to_h(l_patches=l_patches,
                                        t_patches_l=t_patches_l,
                                        t_probs=t_probs,
                                        t_n=t_n,
                                        atoms_h=atoms_h,
                                        atoms_h_tensor=atoms_h_tensor,
                                        h=int(super_sz),
                                        w=int(super_sz))  # 1, 1, h, w.

        end.record()
        torch.cuda.synchronize()
        print(f'GPU time: {start.elapsed_time(end)} (ms)')
        print(type(start.elapsed_time(end)))
        input('o')
        sr_tensor_v = sr_tensor_v.squeeze().cpu().numpy().astype(np.uint8)  #
        # h, w
        # ----------------------------------------------------------------------

        sr = sr.squeeze().numpy().astype(np.uint8)  # h, w

        # interpolated
        img = img_lr_np.copy()
        h_s, w_s = int(super_sz), int(super_sz)
        img_l_2_h = Image.fromarray(img).resize((w_s, h_s),
                                                resample=PIL.Image.BICUBIC)
        img_l_2_h = np.array(img_l_2_h)

        # psnr
        psnr_patch = calculate_psnr(sr, img_hr_np)
        psnr_patch_tensor = calculate_psnr(sr_tensor_v, img_hr_np)
        psnr_bicubic = calculate_psnr(img_l_2_h, img_hr_np)

        # mse
        mse_patch = calculate_mse(sr, img_hr_np)
        mse_patch_tensor = calculate_mse(sr_tensor_v, img_hr_np)
        mse_bicubic = calculate_mse(img_l_2_h, img_hr_np)

        # plot
        nrows = 1
        ncols = 4
        fh, fw = _get_figure_h_w(nrows=nrows, ncols=ncols, him=400, wim=400)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        _show_img_and_tag(axes[0, 0], _build_img(img_l_2_h, cell_type),
                          f'Low interpolated: {sz} -> {super_sz}'
                          f' (MSE: {mse_bicubic}. PSNR: {psnr_bicubic})')
        _show_img_and_tag(axes[0, 1], _build_img(sr, cell_type),
                          f'Patch (CPU): {sz} -> {super_sz} '
                          f'(MSE: {mse_patch}. PSNR: {psnr_patch})')
        _show_img_and_tag(axes[0, 2], _build_img(sr_tensor_v, cell_type),
                          f'Patch (GPU): {sz} -> {super_sz} '
                          f'(MSE: {mse_patch_tensor}. '
                          f'PSNR: {psnr_patch_tensor})')
        _show_img_and_tag(axes[0, 3], _build_img(img_hr_np, cell_type),
                          f'High: {super_sz}')
        fout = join(outdir, f'{k}.png')
        _closing(fig, fout)



        if not _SILENT:
            print(f'PSNR-PATCH {k} (KS: {sz_l}x{sz_l}): {psnr_patch}')
            print(f'PSNR-BICUBIC {k}: {psnr_bicubic}')

        psnr['patch'] += psnr_patch
        psnr['bicubic'] += psnr_bicubic

        sys.exit()

        r += 1
        if r == 5:
            break

    psnr["patch"] = psnr["patch"] / float(r)
    psnr["bicubic"] = psnr["bicubic"] / float(r)

    print(f'Total PSNR-PATCH (KS: {sz_l}x{sz_l}): {psnr["patch"]}')
    print(f'Total PSNR-BICUBIC: {psnr["bicubic"]}')

    fout = join(outdir, f'stats-{sz_l}x{sz_l}.yml')
    with open(fout, 'w') as f:
        yaml.dump(psnr, f)





if __name__ == '__main__':

    task = constants.SUPER_RES
    fd = get_root_datasets(task)
    path_ds = join(fd, constants.DS_DIR[''])  # fixit
    sz_l = 5
    fdout = join(root_dir, f'data/debug/patch/in_{sz_l}x{sz_l}')
    os.makedirs(fdout, exist_ok=True)

    sizes = {
        '128': 'Lowest_res_128x128',
        '256': 'Low_res_256x256',
        '512': 'Moderate_res_512x512',
        '1024': 'High_res_1024x1024'
    }

    paths = _get_basics(path_ds, sizes)
    super_sz = '1024'

    os.makedirs(fdout, exist_ok=True)

    process_scale(paths, sz='512', super_sz=super_sz, outdir=fdout,
                  sizes_code=sizes, sz_l=sz_l)