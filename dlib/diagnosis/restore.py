import math
import os
import sys
from os.path import dirname, abspath, join, basename, splitext
from typing import List, Tuple
import fnmatch
import argparse
import pprint

import cv2
from tqdm import tqdm
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.utils.utils_config import get_root_datasets
from dlib.utils.shared import find_files_pattern
from dlib.utils.utils_image import get_cell_type
from dlib.diagnosis.stats_numpy import unnormed_histogram
from dlib.diagnosis.stats_numpy import normed_histogram

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
            color='red'
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
    fw = 10
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

def process_scale(paths: dict, sz: str, super_sz: str, nbr_s_interm: int,
                  var_type: str, var_global: float, outdir: str,
                  sizes_code: dict):
    outdir = join(outdir, sz)
    os.makedirs(outdir, exist_ok=True)

    n = len(list(paths[sz].keys()))

    for k in tqdm(paths[sz], ncols=80, total=n):
        cell_type = get_cell_type(paths[sz][k])

        path_src = paths[sz][k]
        img_src_np: np.ndarray = _load_cell(path_src)  # [0, 255]
        img = img_src_np.copy()
        h, w = img.shape
        h_s, w_s = int(super_sz), int(super_sz)
        img_l_2_h = Image.fromarray(img).resize((w_s, h_s),
                                                resample=PIL.Image.BICUBIC)
        img_l_2_h = np.array(img_l_2_h)
        psf = np.ones((5, 5)) / 25
        images = [_build_img(img_l_2_h, cell_type)]
        l_balance = ['Interpolated']
        for balance in [0.001, .01, 0.1, 1.]:

            img_r = restoration.wiener(img_l_2_h / 255., psf, balance)
            img_r = (img_r * 255.).astype(np.uint8)

            images.append(_build_img(img_r, cell_type))
            l_balance.append(balance)

        k_hr = k.replace(sizes_code[sz], sizes_code[super_sz])
        path_hr = paths[super_sz][k_hr]
        img_hr_np: np.ndarray = _load_cell(path_hr)  # [0, 255]
        images.append(_build_img(img_hr_np, cell_type))
        l_balance.append('HR')

        for i in range(len(images)):
            zimg = images[i]
            tag = l_balance[i]
            fileout = join(outdir, k)
            os.makedirs(fileout, exist_ok=True)
            fileout = join(fileout, f'{tag}.png')
            _plot_one_image(fout=fileout, image=zimg, scale=tag)


def wiener():
    from skimage import color, data, restoration
    img = color.rgb2gray(data.astronaut())
    from scipy.signal import convolve2d
    psf = np.ones((10, 10)) / 100
    img = convolve2d(img, psf, 'same')
    rng = np.random.default_rng()
    img += 0.1 * img.std() * rng.standard_normal(img.shape)
    deconvolved_img = restoration.wiener(img, psf, 1.)
    plt.figure()
    plt.title('img')
    plt.imshow(img)
    plt.figure()
    plt.title('weiner img')
    plt.imshow(deconvolved_img)
    plt.show()


if __name__ == '__main__':
    # wiener()

    task = constants.SUPER_RES
    fd = get_root_datasets(task)
    path_ds = join(fd, constants.DS_DIR[''])  # fixit
    fdout = join(root_dir, 'data/debug/wiener')
    os.makedirs(fdout, exist_ok=True)


    sizes = {
        '128': 'Lowest_res_128x128',
        '256': 'Low_res_256x256',
        '512': 'Moderate_res_512x512',
        '1024': 'High_res_1024x1024'
    }

    paths = _get_basics(path_ds, sizes)
    super_sz = '1024'


    out_fd = join(fdout, 'global_var')
    os.makedirs(out_fd, exist_ok=True)

    process_scale(paths, sz='128', super_sz=super_sz, nbr_s_interm=8,
                  var_type=constants.VAR_GLOBAL, var_global=3., outdir=out_fd,
                  sizes_code=sizes)

    print(paths)



