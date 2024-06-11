import copy
import fnmatch
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import random
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import math
import pprint
from typing import List, Tuple
from datetime import date
import datetime as dt
import argparse
import more_itertools as mit

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


from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
from PIL.Image import NEAREST


root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)

from dlib.utils.utils_config import get_root_datasets
from dlib.utils import constants
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils.shared import fmsg
from dlib.datasets.ds_scripts.patch_sampler import SamplePatchesFromTile

from dlib.utils.shared import is_tay
from dlib.utils.shared import is_gsys
import dlib.utils.utils_image as util

_SEED = 0

if is_gsys():
    virenv = "\nCONDA_BASE=$(conda info --base) \n" \
             "source $CONDA_BASE/etc/profile.d/conda.sh\n" \
             "conda activate {}\n".format(constants._ENV_NAME)

elif is_tay():
    virenv = f"\nsource /projets/AP92990/venvs" \
             f"/{constants._ENV_NAME}/bin/activate\n"
else:  # ???
    virenv = "\nCONDA_BASE=$(conda info --base) \n" \
             "source $CONDA_BASE/etc/profile.d/conda.sh\n" \
             "conda activate {}\n".format(constants._ENV_NAME)


def find_files_pattern(fd_in_, pattern_):
    assert os.path.exists(fd_in_), "Folder {} does not exist.".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def write_k_to_file(l: list, path_file):
    with open(path_file, 'w') as fx:
        for k, v in l:
            fx.write(f'{k},{v}\n')


def rename_tiles_caco2_nov(path: str):
    assert 'caco2_source_nov' in path, path
    print(f'processing {path}')

    fs = find_files_pattern(path, '*.tif')
    tiles = []

    for f in fs:
        # rename
        t = basename(f).split('_')[1]  # Tilex
        i = t.split('Tile')[1]  # x
        i = int(i)
        if i <= 5:
            i += 6  # + oct.
            new_tile = f'Tile{i}'
            new_f = f.replace(t, new_tile)
            print(f'renaming {f} -> {new_f}')
            os.rename(f, new_f)

            # rewrite in correct order: 3, 2, 1
            mtx = tifffile.imread(f)  # 3, h, w
            mtx_copy = mtx.copy()
            mtx[0, :, :] = mtx_copy[2, :, :]
            mtx[2, :, :] = mtx_copy[0, :, :]
            tifffile.imwrite(f, mtx)


def keep_only_cell0(l: list) -> list:
    out = []
    for s in l:
        if constants.CELL0 in basename(s):
            out.append(s)

    return out

def get_info_patch(path_p: str) -> dict:
    b = basename(path_p).split('.')[0]
    o = b.split('_')
    tiles_sz = o[1].split('-')[0]
    tile_nbr = o[1].split('-')[1]
    patch_id = o[2]
    i = o[3]
    ii = o[4]
    j = o[5]
    jj = o[6]
    cell = o[7]

    return {
        'tile_sz': tiles_sz,
        'tile_nbr': tile_nbr,
        'patch_id': patch_id,
        'i': int(i),
        'ii': int(ii),
        'j': int(j),
        'jj': int(jj),
        'cell': cell
    }


def remove_init_os_sep(p: str) -> str:
    if p.startswith(os.sep):
        return p[1:]
    return p

def build_scale_to_hr(hr: list, cell: str, scale: int, parent: str) -> list:
    holder = {
        1: 'HighRes1024',
        2: 'LowRes512',
        4: 'LowRes256',
        8: 'LowRes128'
    }

    low_high_res = []
    for ph in hr:
        # change ph from CELL0 (default) to current cell
        ph = ph.replace(f"_{constants.CELL0}.tif", f"_{cell}.tif")
        assert os.path.isfile(ph), ph

        info = get_info_patch(ph)
        assert info['tile_sz'] == holder[1], f"{info['tile_sz']}, {holder[1]}"

        # build low res. patch path.
        p_low = ph.replace('hr_div_1', f'hr_div_{scale}')
        d = dirname(p_low)
        b = f"{holder[scale]}-{info['tile_nbr']}"
        k = info['patch_id']
        i = info['i'] // scale
        ii = info['ii'] // scale
        j = info['j'] // scale
        jj = info['jj'] // scale
        c = cell
        p_low = join(d, f"tile_{b}_{k}_{i}_{ii}_{j}_{jj}_{c}.tif")
        assert os.path.isfile(p_low), p_low
        ph = ph.replace(parent, '')
        p_low = p_low.replace(parent, '')

        ph = remove_init_os_sep(ph)
        p_low = remove_init_os_sep(p_low)

        low_high_res.append((p_low, ph))

    return low_high_res


def process_caco2_2_x2_4_8(path: str, path_tiles: str):
    set_seed(_SEED)

    name = 'caco2'  # train, valid, test. valid != test.
    print(fmsg(f'Creating folds of {name}. X2/4/8: High res. is: 1024.'))

    hr = join(path, 'hr_div_1')
    patches_hr = find_files_pattern(hr, '*.tif')

    # keep only cell0
    patches_hr = keep_only_cell0(patches_hr)
    # shuffle
    print('shuffling...')
    for xx in range(10000):
        random.shuffle(patches_hr)

    tiles_hr = find_files_pattern(join(path_tiles, 'HighRes1024'), '*.tif')
    _l_tiles_n = []
    for f in tiles_hr:
        bn = basename(f)
        n = bn.split('-')[-1].split('.')[0]
        n = int(n)
        assert n not in _l_tiles_n, n
        _l_tiles_n.append(n)

    _l_tiles_n = sorted(_l_tiles_n)
    _l_tiles_n = [str(x) for x in _l_tiles_n]

    for i in range(1000):
        random.shuffle(_l_tiles_n)

    _N_TST = 4
    _N_VL = 3

    tl_ts = _l_tiles_n[:_N_TST]
    tl_vl = _l_tiles_n[_N_TST: _N_TST + _N_VL]
    tl_tr = _l_tiles_n[_N_TST + _N_VL: ]

    print('Tiles split:')
    print(f"{constants.TRAINSET}: {tl_tr}")
    print(f"{constants.VALIDSET}: {tl_vl}")
    print(f"{constants.TESTSET}: {tl_ts}")

    hr_tr_patches = []
    hr_vl_patches = []
    hr_tst_patches = []

    for p in patches_hr:
        info = get_info_patch(p)

        if info['tile_nbr'] in tl_tr:
            hr_tr_patches.append(p)

        elif info['tile_nbr'] in tl_vl:
            hr_vl_patches.append(p)

        elif info['tile_nbr'] in tl_ts:
            hr_tst_patches.append(p)

        else:
            raise ValueError(f"patch {p}: unknown tile subset.")


    sets_ = {
        constants.TRAINSET: hr_tr_patches,
        constants.VALIDSET: hr_vl_patches,
        constants.TESTSET: hr_tst_patches
    }
    sizes = {
        1: 512,
        2: 256,
        4: 128,
        8: 64,
    }

    task = constants.SUPER_RES
    outd = join(root_dir, constants.RELATIVE_META_ROOT, task)

    log = open(join(path, 'log_folds.txt'), 'w')
    log.write("Tiles split:\n")
    log.write(f"{constants.TRAINSET}: [{', '.join(tl_tr)}] ({len(tl_tr)} "
              f"tiles).\n")
    log.write(f"{constants.VALIDSET}: [{', '.join(tl_vl)}] ({len(tl_vl)} "
              f"tiles).\n")
    log.write(f"{constants.TESTSET}: [{', '.join(tl_ts)}] ({len(tl_ts)} "
              f"tiles).\n")

    for cell in [constants.CELL0, constants.CELL1, constants.CELL2]:

        for scale in [2, 4, 8]:
            for split in sets_:
                msg = f">> cell {cell} scale {scale} split {split}"
                print(fmsg(msg))

                l_h = build_scale_to_hr(hr=sets_[split],
                                        cell=cell,
                                        scale=scale,
                                        parent=path
                                        )
                # patches are only in CELL0.

                size_in = sizes[scale]
                size_out = sizes[1]
                ds_name = f'{name}_{split}_X_{scale}_in_{size_in}_out_' \
                          f'{size_out}_cell_{cell}'

                _outd = join(outd, ds_name)
                os.makedirs(_outd, exist_ok=True)

                h_l = [l[::-1] for l in l_h]
                assert len(l_h) == len(h_l)
                assert len(set([v[0] for v in h_l])) == len(l_h)
                write_k_to_file(l_h, join(_outd, 'l_h.txt'))
                write_k_to_file(h_l, join(_outd, 'h_l.txt'))
                print(fmsg(f"{msg}. {len(h_l)} patches. <<"))

                log.write(f"{msg}: {len(h_l)} patches.\n")

    log.close()

    print(fmsg(f'Done creating folds of {name}.'))

def move_rename_tiles(path_in:str, fd_name: str, fd_up_name: str,
                      x_before: int):

    fs = find_files_pattern(path_in, '*.tif')

    master_dest = path_in.replace(fd_name, fd_up_name)
    if os.path.isdir(master_dest):
        os.system(f'rm -r {master_dest}')

    for f in tqdm.tqdm(fs, total=len(fs), ncols=80):
        a = tifffile.imread(f)  # 3, h, w

        new_path = f.replace(fd_name, fd_up_name)
        os.makedirs(dirname(new_path), exist_ok=True)

        bn = basename(new_path)
        n = bn.split('-')[-1].split('.')[0]
        e = bn.split('-')[-1].split('.')[1]
        n = int(n)
        # if n <= 5:
        new_bn = f'{bn.split("-")[0]}-{n + x_before}.{e}'
        new_path = join(dirname(new_path), new_bn)

        print(f'writing in {new_path}')

        tifffile.imwrite(new_path, a, photometric='minisblack')


def fix_tiles(path_in:str, fd_name: str, fd_up_name: str, x_before: int):

    fs = find_files_pattern(path_in, '*.tif')

    master_dest = path_in.replace(fd_name, fd_up_name)
    if os.path.isdir(master_dest):
        os.system(f'rm -r {master_dest}')

    for f in tqdm.tqdm(fs, total=len(fs), ncols=80):
        a = tifffile.imread(f)  # 3, h, w
        a_ = a.copy()
        a[0, :, :] = a_[2, :, :]
        a[2, :, :] = a_[0, :, :]
        new_path = f.replace(fd_name, fd_up_name)
        os.makedirs(dirname(new_path), exist_ok=True)

        bn = basename(new_path)
        n = bn.split('-')[-1].split('.')[0]
        e = bn.split('-')[-1].split('.')[1]
        n = int(n)
        # if n <= 5:
        new_bn = f'{bn.split("-")[0]}-{n + x_before}.{e}'
        new_path = join(dirname(new_path), new_bn)

        print(f'writing in {new_path}')

        tifffile.imwrite(new_path, a, photometric='minisblack')


def fix_tiles_jul_aug_21(path_in:str, fd_name: str, fd_up_name: str,
                         x_before: int):
    # tiles-jul-aug21 (4 tiles): delete 3rd map. otherwise, the order is
    # fine: cell0 (dimmer), cell1 (less dim), cell2 bright.

    # cell 0 --> map4
    # cell 1 --> map2
    # cell 2 --> map1
    # delete map3.

    fs = find_files_pattern(path_in, '*.tif')

    master_dest = path_in.replace(fd_name, fd_up_name)
    if os.path.isdir(master_dest):
        os.system(f'rm -r {master_dest}')

    for f in tqdm.tqdm(fs, total=len(fs), ncols=80):
        a = tifffile.imread(f)  # 4, h, w
        x = a[:3].copy()  # 3, h, w
        x[0, :, :] = a[3, :, :]
        x[1, :, :] = a[1, :, :]
        x[2, :, :] = a[0, :, :]
        new_path = f.replace(fd_name, fd_up_name)
        os.makedirs(dirname(new_path), exist_ok=True)

        bn = basename(new_path)
        n = bn.split('-')[-1].split('.')[0]
        e = bn.split('-')[-1].split('.')[1]
        n = int(n)
        # if n <= 5:
        new_bn = f'{bn.split("-")[0]}-{n + x_before}.{e}'
        new_path = join(dirname(new_path), new_bn)

        print(f'writing in {new_path}')

        tifffile.imwrite(new_path, x, photometric='minisblack')


def fix_tiles_jul21(path_in:str, fd_name: str, fd_up_name: str,
                    x_before: int):
    # cell 0 --> map4
    # cell 1 --> map1
    # cell 2 --> map3

    # delete map2.
    fs = find_files_pattern(path_in, '*.tif')

    master_dest = path_in.replace(fd_name, fd_up_name)
    if os.path.isdir(master_dest):
        os.system(f'rm -r {master_dest}')


    for f in tqdm.tqdm(fs, total=len(fs), ncols=80):
        a = tifffile.imread(f)  # 4, h, w
        x = a[:3].copy()  # 3, h, w
        x[0, :, :] = a[3, :, :]
        x[1, :, :] = a[0, :, :]
        x[2, :, :] = a[2, :, :]
        new_path = f.replace(fd_name, fd_up_name)
        os.makedirs(dirname(new_path), exist_ok=True)

        bn = basename(new_path)
        n = bn.split('-')[-1].split('.')[0]
        e = bn.split('-')[-1].split('.')[1]
        n = int(n)
        # if n <= 5:
        new_bn = f'{bn.split("-")[0]}-{n + x_before}.{e}'
        new_path = join(dirname(new_path), new_bn)

        print(f'writing in {new_path}')

        tifffile.imwrite(new_path, x, photometric='minisblack')


def _register_(ref: np.ndarray,
               img: np.ndarray,
               global_shift: bool = False) -> np.ndarray:

    v, u = optical_flow_tvl1(ref, img)
    mean_u = u.mean()
    mean_v = v.mean()

    mean_u = round(mean_u)
    mean_v = round(mean_v)

    if global_shift:
        v = (v * 0 + mean_v).astype(v.dtype)
        u = (u * 0 + mean_u).astype(u.dtype)

    nr, nc = ref.shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')

    img_warp = warp(image=img,
                    inverse_map=np.array([row_coords + v, col_coords + u]),
                    mode='constant',
                    preserve_range=True).astype(np.uint8)

    return img_warp


def register_im(ref: np.ndarray,
                img: np.ndarray,
                scale: int,
                del_border: int,
                global_shift: bool = False
                ) -> Tuple[np.ndarray, np.ndarray]:

    # ref, img: c, h, w
    assert ref.ndim == 3, ref.ndim
    assert img.ndim == 3, img.ndim
    assert ref.shape[0] == 3, ref.shape[0]
    assert img.shape[0] == 3, img.shape[0]
    assert img.dtype == np.uint8, img.dtype
    assert ref.dtype == np.uint8, ref.dtype

    assert isinstance(del_border, int), type(del_border)
    assert del_border >= 0, del_border

    # transpose
    ref = ref.transpose((1, 2, 0))  # h, w, c
    img = img.transpose((1, 2, 0))  # h, w, c.
    h, w, c = ref.shape

    im_scaled = Image.fromarray(img).resize(size=(w, h),
                                            resample=NEAREST
                                            )
    im_scaled = np.array(im_scaled).astype(np.uint8)

    out = (ref * 0).astype(np.uint8)

    for i in range(c):
        ref_i = ref[:, :, i]
        im_i = im_scaled[:, :, i]
        out[:, :, i] = _register_(ref_i, im_i, global_shift)

    # remove edges because of registration reflect effect.
    out = crop_border(out, del_border)

    # do the same for ref.
    ref = crop_border(ref, del_border)

    h, w, c = ref.shape


    out_down = Image.fromarray(out).resize(size=(int(w / scale),
                                                 int(h / scale)),
                                            resample=NEAREST
                                            )
    out_down = np.array(out_down).astype(np.uint8)

    # swap axes to c, h, w from h, w, c.
    ref = ref.transpose((2, 0, 1))  # c, h, w
    out_down = out_down.transpose((2, 0, 1))  # c, h, w

    return out_down, ref


def _register_im_single_plan(ref: np.ndarray,
                             img: np.ndarray,
                             scale: int,
                             del_border: int,
                             global_shift: bool = False
                             ) -> Tuple[np.ndarray, np.ndarray]:

    # ref, img: c=1, h, w
    assert ref.ndim == 2, ref.ndim
    assert img.ndim == 2, img.ndim
    # assert ref.shape[0] == 1, ref.shape[0]
    # assert img.shape[0] == 1, img.shape[0]
    assert img.dtype == np.uint8, img.dtype
    assert ref.dtype == np.uint8, ref.dtype

    assert isinstance(del_border, int), type(del_border)
    assert del_border >= 0, del_border

    # transpose
    # ref = ref.transpose((1, 2, 0))  # h, w, c
    # img = img.transpose((1, 2, 0))  # h, w, c.
    h, w = ref.shape

    im_scaled = Image.fromarray(img).resize(size=(w, h),
                                            resample=NEAREST
                                            )
    im_scaled = np.array(im_scaled).astype(np.uint8)

    out = _register_(ref, im_scaled, global_shift)

    # remove edges because of registration reflect effect.
    out = crop_border_only(out, del_border)

    # do the same for ref.
    cropped_ref = crop_border_only(ref, del_border)

    h, w = cropped_ref.shape


    out_down = Image.fromarray(out).resize(size=(int(w / scale),
                                                 int(h / scale)),
                                            resample=NEAREST
                                            )
    out_down = np.array(out_down).astype(np.uint8)

    return out_down, cropped_ref


def crop_border(im: np.ndarray, b: int) -> np.ndarray:
    assert im.ndim == 3, im.ndim
    assert im.shape[2] == 3, im.shape[2]

    h, w, c = im.shape

    new_im = im[b: h - b, b: w - b, :]
    h, w, _ = new_im.shape

    b_h = h % 8
    b_w = w % 8

    new_im = im[: h - b_h, : w - b_w, :]
    h, w, _ = new_im.shape

    assert h % 8 == 0, h % 8
    assert w % 8 == 0, w % 8

    return new_im


def crop_border_only(im: np.ndarray, b: int) -> np.ndarray:
    assert im.ndim == 2, im.ndim


    h, w = im.shape

    new_im = im[b: h - b, b: w - b]

    return new_im


if __name__ == '__main__':
    # caco2:
    def caco2_fix_tiles():
        path = join(get_root_datasets(constants.SUPER_RES), 'caco2-tiles')
        # fix tiles nov.


        # tiles-jul-21 (1 tile):
        # map4 --> cell0. map3 --> cell 2. map 1 --> cell1.
        # delete map2.

        fd_name = 'tiles-jul-21'
        path_in = join(path, fd_name)
        fd_up_name = 'tiles-jul-21-updated'
        x_before = 0

        fix_tiles_jul21(path_in, fd_name, fd_up_name, x_before)
        x_before = 1

        # tiles-jul-aug-21 (4 tiles): delete 3rd map. otherwise, the order is
        # fine: cell0 (dimmer), cell1 (less dim), cell2 bright.
        # map2 -> cell 1. map4 -> cell0. map1 -> cell2
        # delete map3.

        fd_name = 'tiles-jul-aug-21'
        path_in = join(path, fd_name)
        fd_up_name = 'tiles-jul-aug-21-updated'
        fix_tiles_jul_aug_21(path_in, fd_name, fd_up_name, x_before)
        x_before = 4 + 1

        fd_name = 'tiles-may-22'
        path_nov = join(path, fd_name)
        fd_up_name = f'{fd_name}-updated'
        move_rename_tiles(path_nov, fd_name, fd_up_name, x_before)
        x_before = 4 + 1 + 6

        fd_name = 'tiles-nov-22'
        path_nov = join(path, fd_name)
        fd_up_name = f'{fd_name}-updated'
        fix_tiles(path_nov, fd_name, fd_up_name, x_before)
        x_before = 4 + 1 + 6 + 5

        fd_name = 'tiles-oct-22'
        path_nov = join(path, fd_name)
        fd_up_name = f'{fd_name}-updated'
        fix_tiles(path_nov, fd_name, fd_up_name, x_before)
        x_before = 4 + 1 + 6 + 5 + 6

        move_to_all_tiles(['tiles-jul-21-updated',
                           'tiles-jul-aug-21-updated',
                           'tiles-may-22-updated',
                           'tiles-nov-22-updated',
                           'tiles-oct-22-updated'])


    def move_to_all_tiles(fd_names: list):
        path = join(get_root_datasets(constants.SUPER_RES), 'caco2-tiles')
        dest = join(path, 'all-tiles')
        if os.path.isdir(dest):
            os.system(f'rm -r {dest}')

        os.makedirs(dest)
        for d in fd_names:
            s = join(path, d)
            assert os.path.isdir(s), s
            cmd = f'cp -r {s}/* {dest}'
            print(f'running: {cmd}')
            os.system(cmd)


    def cmds_parallel_register_caco2():
        tag_old = 'all-tiles'
        tag_new = 'all-tiles-registered'
        path = join(get_root_datasets(constants.SUPER_RES),
                    f'caco2-tiles/{tag_old}')

        fds = ['HighRes1024', 'LowRes512', 'LowRes256', 'LowRes128']
        out = dict()

        hr = 'HighRes1024'
        hr2 = 'LowRes512'
        hr4 = 'LowRes256'
        hr8 = 'LowRes128'
        files_hr = find_files_pattern(join(path, hr), '*.tif')
        scale_map = {
            2: hr2,
            4: hr4,
            8: hr8
        }

        del_border = 200  # how much to crop from the border after registration.
        cmd_file = join(root_dir, 'register_caco2_parallel.sh')
        fx = open(cmd_file, 'w')
        fx.write(f"#!/usr/bin/env bash \n"
                 f"{virenv}"
                 )

        list_pairs = []
        for f in files_hr:
            p = [f]
            for scale in [2, 4, 8]:
                fs = f.replace(hr, scale_map[scale])
                assert os.path.isfile(fs), fs
                p.append(fs)

            list_pairs.append(p)

        for i, pair in enumerate(list_pairs):
            cmd = f'python dlib/datasets/ds_scripts/caco2_resample.py ' \
                  f'--indx {i} & \n'
            fx.write(cmd)
            print(cmd)

        fx.close()
        os.system(f"chmod +x {cmd_file}")
        print(f'Registration cmds parallel are ready. {len(list_pairs)} cases.')


    def create_cmds_parallel_patch_register_caco2():
        cmd_file = join(root_dir, 'patch_register_caco2_parallel.sh')
        fx = open(cmd_file, 'w')
        fx.write(f"#!/usr/bin/env bash \n"
                 f"{virenv}"
                 )

        cells = [constants.CELL0, constants.CELL1, constants.CELL2]
        splits = [constants.TRAINSET, constants.VALIDSET, constants.TESTSET]
        n_blocks = 10
        blocks = list(range(0, n_blocks, 1))
        l_cmds = []

        for cell in cells:
            for split in splits:
                for blockidx in blocks:
                    cmd = f'python dlib/datasets/ds_scripts/caco2_resample.py ' \
                          f'--cell {cell} ' \
                          f'--split {split} ' \
                          f'--blockidx {blockidx} ' \
                          f'--n_blocks {n_blocks} & \n'
                    fx.write(cmd)
                    print(cmd)
                    l_cmds.append(cmd)

        fx.close()
        os.system(f"chmod +x {cmd_file}")
        print(f'Registration cmds parallel are ready. '
              f'{len(l_cmds)} cases. N_Blocks: {n_blocks}.')


    def register_caco2(global_shift: bool = False):
        print(f'global_shift: {global_shift}')
        parser = argparse.ArgumentParser()
        parser.add_argument("--indx", type=int, default=-1,
                            help="index pair to process. set it to -1 to "
                                 "process all.")

        parsedargs = parser.parse_args()
        indx = parsedargs.indx

        tag_old = 'all-tiles'
        tag_new = 'all-tiles-registered'
        path = join(get_root_datasets(constants.SUPER_RES),
                    f'caco2-tiles/{tag_old}')

        fds = ['HighRes1024', 'LowRes512', 'LowRes256', 'LowRes128']
        out = dict()

        hr = 'HighRes1024'
        hr2 = 'LowRes512'
        hr4 = 'LowRes256'
        hr8 = 'LowRes128'
        files_hr = find_files_pattern(join(path, hr), '*.tif')
        scale_map = {
            2: hr2,
            4: hr4,
            8: hr8
        }

        del_border = 200  # how much to crop from the border after registration.

        list_pairs = []
        for f in files_hr:
            p = [f]
            for scale in [2, 4, 8]:
                fs = f.replace(hr, scale_map[scale])
                assert os.path.isfile(fs), fs
                p.append(fs)

            list_pairs.append(p)

        for i, pair in enumerate(list_pairs):
            if (indx != -1) and (i != indx):
                    continue

            print(f"Processing pair {i} ...")
            x_p, x2_p, x4_p, x8_p = pair

            x = tifffile.imread(x_p)  # 3, h, w. np.ndarray
            x2 = tifffile.imread(x2_p)
            x4 = tifffile.imread(x4_p)
            x8 = tifffile.imread(x8_p)

            print('Register x2 ...')
            t0 = dt.datetime.now()
            x2_, x_ = register_im(ref=x, img=x2, scale=2, del_border=del_border,
                                  global_shift=global_shift)
            print(f'Register x2 took: {dt.datetime.now()- t0}')

            print('Register x4 ...')
            t0 = dt.datetime.now()
            x4_, _ = register_im(ref=x, img=x4, scale=4, del_border=del_border,
                                 global_shift=global_shift)
            print(f'Register x4 took: {dt.datetime.now() - t0}')

            print('Register x8 ...')
            t0 = dt.datetime.now()
            x8_, _ = register_im(ref=x, img=x8, scale=8, del_border=del_border,
                                 global_shift=global_shift)
            print(f'Register x8 took: {dt.datetime.now() - t0}')


            print(x.shape, x.dtype, x_.shape, x_.dtype)
            print(x2.shape, x2.dtype, x2_.shape, x2_.dtype)
            print(x4.shape, x4.dtype, x4_.shape, x4_.dtype)
            print(x8.shape, x8.dtype, x8_.shape, x8_.dtype)

            # write
            for px in zip([x_, x2_, x4_, x8_],
                          [x_p, x2_p, x4_p, x8_p]):
                mtx, x_path = px

                x_path_new = x_path.replace(tag_old, tag_new)
                os.makedirs(dirname(x_path_new), exist_ok=True)
                tifffile.imwrite(x_path_new, mtx, photometric='minisblack')


        print('Done registration.')

    def caco2_info_tiles():
        tag_old = 'all-tiles'
        tag_new = 'all-tiles-registered'
        tag = tag_old
        path = join(get_root_datasets(constants.SUPER_RES),
                    f'caco2-tiles/{tag}')
        fds = ['HighRes1024', 'LowRes512', 'LowRes256', 'LowRes128']
        out = dict()

        for fd in fds:
            p = join(path, fd)
            fs = find_files_pattern(p, '*.tif')
            fs = sorted(fs)

            for f in fs:
                a = tifffile.imread(f)  # 3, h, w
                b = basename(f)
                b = b.split('-')[1]
                if b in out:
                    out[b].append(a.shape[1:])

                else:
                    out[b] = [a.shape[1:]]
        print('\t'.join(fds))
        for k in out:
            print(k, out[k])

    def sample_patches(use_registered_tiles: bool,
                       psize: int,
                       register_margin: int):
        assert isinstance(register_margin, int), type(register_margin)
        assert register_margin >=0, register_margin


        tag_old = 'all-tiles'
        tag_new = 'all-tiles-registered'
        tag = tag_old

        if use_registered_tiles:
            assert register_margin == 0, register_margin
            tag = tag_new
            print('*** We are going to use registered tiles ****')

        else:
            print('*** We are NOT going to use registered tiles ****')

        path = join(get_root_datasets(constants.SUPER_RES),
                    f'caco2-tiles/{tag}')

        out_fd = join(get_root_datasets(constants.SUPER_RES), 'caco2')
        if os.path.isdir(out_fd):
            cmd = f"rm -r {out_fd}"
            print(f"Running {cmd} ...")
            os.system(cmd)

        os.makedirs(out_fd)

        psize = psize
        real_psize = psize

        if not use_registered_tiles:
            psize = psize + 2 * register_margin

        shift = int(real_psize / 4) * 3 - 2 * register_margin  # 25% overlap.
        min_area = 0.2
        threshold = 4
        args = {
            'psize': psize,
            'real_psize': real_psize,
            'shift': shift,
            'min_area': min_area,
            'threshold': threshold,
            'register_margin': register_margin
        }
        print(f"Config: {args}")

        with open(join(out_fd, 'config.yaml'), 'w') as fargs:
            yaml.dump(args, fargs)

        sampler = SamplePatchesFromTile(psize=psize,
                                        shift=shift,
                                        out_fd=out_fd,
                                        min_area=min_area,
                                        threshold=threshold,
                                        real_psize=real_psize
                                        )
        with open(join(out_fd, 'config.txt'), 'w') as fout:
            fout.write(str(sampler))

        hr = 'HighRes1024'
        hr2 = 'LowRes512'
        hr4 = 'LowRes256'
        hr8 = 'LowRes128'
        fs = find_files_pattern(join(path, hr), '*.tif')
        stats = {
            'nbr': 0,
            'reject': 0
        }
        log =  open(join(out_fd, 'log.txt'), 'w')
        log.write(f"Date: {date.today().strftime('%B %d, %Y')}.\n")
        log.write(f"Configuration:\n"
                  f"Patch size: {psize}\n"
                  f"Shift: {shift}\n"
                  f"Min area: {min_area}\n"
                  f"Threshold: {threshold}\n"
                  )

        for f in tqdm.tqdm(fs, ncols=80, total=len(fs)):
            b = basename(f)

            # /2
            _hr2 = join(path, hr2, b.replace(hr, hr2))
            assert os.path.isfile(_hr2), _hr2

            # /4
            _hr4 = join(path, hr4, b.replace(hr, hr4))
            assert os.path.isfile(_hr4), _hr4

            # /8
            _hr8 = join(path, hr8, b.replace(hr, hr8))
            assert os.path.isfile(_hr8), _hr8

            sampler.sample(path_1024=f,
                           path_512=_hr2,
                           path_256=_hr4,
                           path_128=_hr8
                           )

            msg = f"{80 * '='}\n" \
                  f"Sampled from tile: {basename(f)}\n" \
                  f"{sampler.summary()} \n" \
                  f"{80 * '='}\n"
            print(f"\n {msg}")
            log.write(msg)
            stats['nbr'] = stats['nbr'] + sampler.nbr
            stats['reject'] = stats['reject'] + sampler.reject
            sampler.reset()

        print(stats)
        log.write(f"\n{80 * '*'} \n")

        log.write(f"Stats: \n"
                  f"Nbr tiles: {len(fs)}. \n"
                  f"patches {sampler.psize}x{sampler.psize}. \n"
                  f"Nbr sampled patches: {stats['nbr']}. \n"
                  f"Nbr rejected patches: {stats['reject']}."
                  )
        log.close()


    def check_if_new():
        tag_old = 'tiles-nov-22-updated'
        tag_new = 'tiles-oct-22-updated'

        path_old = join(get_root_datasets(constants.SUPER_RES),
                    f'caco2-tiles/{tag_old}')
        path_new = join(get_root_datasets(constants.SUPER_RES),
                        f'caco2-tiles/{tag_new}')

        hr = 'HighRes1024'
        fs_old = find_files_pattern(join(path_old, hr), '*.tif')
        fs_old = sorted(fs_old)
        fs_new = find_files_pattern(join(path_new, hr), '*.tif')
        fs_new = sorted(fs_new)

        for f_new in fs_new:
            x_new = tifffile.imread(f_new)  # 3, h, w. numpy.ndarray
            c = 0
            for f_old in fs_old:

                if f_old == f_new:
                    continue

                x_old = tifffile.imread(f_old)  # 3, h, w. numpy.ndarray

                if x_old.shape != x_new.shape:
                    # print(f"{basename(f_new)} {x_new.shape} "
                    #       f"vs {basename(f_old)} {x_old.shape} "
                    #       f"[SKIPPED]")
                    continue
                else:
                    pass
                    # print(f"{basename(f_new)} vs {basename(f_old)}")

                s = np.abs(x_new - x_old).sum()
                if s == 0:
                    print(f"Difference = {s}. "
                          f"old: "
                          f""
                          f"{basename(dirname(dirname(f_old)))}/{basename(f_old)} "
                          f"and "
                          f"new: "
                          f""
                          f"{basename(dirname(dirname(f_new)))}/{basename(f_new)} "
                          f"are "
                          f"the "
                          f"same.")

                    c += 1

            if c == 0:
                print(f"didnt find anything similar to this new file "
                      f"{basename(f_new)}")


    def split_caco2():
        path = join(get_root_datasets(constants.SUPER_RES), 'caco2')
        path_tiles = join(get_root_datasets(constants.SUPER_RES),
                          f'caco2-tiles/all-tiles')
        process_caco2_2_x2_4_8(path, path_tiles)

    def area_fg_bg_subset(l: list, subset: str, log: str) -> str:
        log += f'Subset: {subset} \n'

        for f in l:
            x = tifffile.imread(f)  # 3, h, w
            d, h, w = x.shape
            bs = basename(f)
            threshs: list = constants.ROI_THRESH
            threshs = [4]
            n_threshs = float(len(threshs))

            total_area = h * w

            for cell in range(d):
                mtx = x[cell]

                lfg = []
                lbg = []

                for th in threshs:
                    fg = 100. * (mtx >= th).mean()
                    bg = 100. - fg

                    lfg.append(fg)
                    lbg.append(bg)
                fg = sum(lfg) / n_threshs
                bg = sum(lbg) / n_threshs

                log += f"{bs}: CELL{cell}: hxw: ({h}x{w}). " \
                       f"FG: {fg} %. BG: {bg}. \n"
            log += '\n\n'


        return log

    def measure_area_fg_bg_tiles():
        print('Measuring area FG/BG on high resolution tiles.')

        tl_ts = ['9', '10', '14', '20']
        tl_vl = ['7', '11', '19']
        tl_tr = ['1', '2', '3', '4', '5', '6', '8', '12', '13', '15', '16',
                 '17', '18', '21', '22']

        tag_old = 'all-tiles'
        tag_new = 'all-tiles-registered'
        tag_new = tag_old

        path = join(get_root_datasets(constants.SUPER_RES),
                    f'caco2-tiles/{tag_new}/HighRes1024')

        tag = 'HighRes1024-'
        tl_ts = [join(path, f"{tag}{x}.tif") for x in tl_ts]
        tl_tr = [join(path, f"{tag}{x}.tif") for x in tl_tr]
        tl_vl = [join(path, f"{tag}{x}.tif") for x in tl_vl]

        log = ''
        log = area_fg_bg_subset(tl_ts, constants.TESTSET, log)
        log = area_fg_bg_subset(tl_vl, constants.VALIDSET, log)
        log = area_fg_bg_subset(tl_tr, constants.TRAINSET, log)

        print(log)
        out = join(get_root_datasets(constants.SUPER_RES),
                    'caco2-tiles/log-areas.txt')
        with open(out, 'w') as fx:
            fx.write(log)

    def _register_content(holder: dict,
                          parent_dir: str,
                          register_margin: int,
                          global_shift: bool,
                          blockidx: int,
                          n_blocks: int
                          ):
        assert isinstance(register_margin, int), type(register_margin)
        assert register_margin >= 0, register_margin
        assert isinstance(global_shift, bool), type(global_shift)

        all_data = dict()
        scales = list(holder.keys())

        datum = {
            scale: '' for scale in scales
        }

        for scale in holder:
            content  = holder[scale]
            for line in content:
                hr, lr = line.strip('\n').split(',')
                hr = join(parent_dir, hr)
                lr = join(parent_dir, lr)
                assert os.path.isfile(lr), lr
                assert os.path.isfile(hr), hr

                if hr in all_data:
                    all_data[hr][scale] = lr
                else:
                    all_data[hr] = copy.deepcopy(datum)
                    all_data[hr][scale] = lr

        all_keys = list(all_data.keys())
        if blockidx != -1:
            assert 0 <= blockidx < n_blocks, f"{blockidx} | {n_blocks}"

            z =  [list(c) for c in mit.divide(n_blocks, all_keys)]

            operation_keys = z[blockidx]

        else:
            operation_keys = all_keys


        for hr in tqdm.tqdm(operation_keys, total=len(operation_keys)):

            x = tifffile.imread(hr)  # h, w. np.ndarray
            # x = util.imread_uint(hr, 1).squeeze()  # h, w
            x_ = None

            for scale in scales:
                # x_scale = util.imread_uint(all_data[hr][scale], 1).squeeze()
                x_scale = tifffile.imread(all_data[hr][scale])  # h, w. np.ndarray


                x_scaled_reg, x_ = _register_im_single_plan(
                    ref=x,
                    img=x_scale,
                    scale=scale,
                    del_border=register_margin,
                    global_shift=global_shift)
                tifffile.imwrite(all_data[hr][scale], x_scaled_reg)

            tifffile.imwrite(hr, x_)


    def register_patches(register_margin: int, global_shift: bool):
        path = join(get_root_datasets(constants.SUPER_RES), 'caco2')
        name = 'caco2'  # train, valid, test. valid != test.

        task = constants.SUPER_RES
        outd = join(root_dir, constants.RELATIVE_META_ROOT, task)

        sets_ = [constants.TRAINSET, constants.VALIDSET, constants.TESTSET]
        sizes = {
            1: 512,
            2: 256,
            4: 128,
            8: 64,
        }

        parent = join(get_root_datasets(constants.SUPER_RES), name)
        assert os.path.isdir(parent), parent


        # run in parallel.
        parser = argparse.ArgumentParser()
        parser.add_argument("--split", type=str, default=-1, help="Split.")
        parser.add_argument("--cell", type=str, default=-1, help="Cell.")
        parser.add_argument("--n_blocks", type=int, default=-1,
                            help="Number of blocks into which the data is "
                                 "split.")
        parser.add_argument("--blockidx", type=int, default=-1,
                            help="The data is split into n_blocks blocks of "
                                 ". This index indicates "
                                 "which block this process will operate on.")


        parsedargs = parser.parse_args()
        splitidx = parsedargs.split
        cellidx = parsedargs.cell
        n_blocks = parsedargs.n_blocks
        blockidx = parsedargs.blockidx

        assert isinstance(n_blocks, int), type(n_blocks)
        assert n_blocks > 0, n_blocks

        assert splitidx in sets_, splitidx
        cells = [constants.CELL0, constants.CELL1, constants.CELL2]
        assert cellidx in cells, cellidx

        # size_block = 20
        # n_blocks = 100 // size_block
        # assert n_blocks * size_block == 100, f"{n_blocks} | {size_block}"

        blocks = list(range(0, n_blocks, 1))
        assert blockidx in blocks, blockidx

        for cell in cells:

            if cellidx != -1 and cellidx != cell:
                continue

            for split in sets_:

                if splitidx != -1 and splitidx != split:
                    continue

                holder = dict()
                for scale in [2, 4, 8]:

                    size_in = sizes[scale]
                    size_out = sizes[1]
                    ds_name = f'{name}_{split}_X_{scale}_in_{size_in}_out_' \
                              f'{size_out}_cell_{cell}'

                    _outd = join(outd, ds_name)
                    assert os.path.isdir(_outd), _outd
                    fx = join(_outd, 'h_l.txt')

                    with open(fx, 'r') as f:
                        content = f.readlines()

                    assert scale not in holder
                    holder[scale] = content

                # register samples.
                print(fmsg(f"Patch registration >> cell {cell} split {split}"))

                _register_content(
                    holder,
                    parent,
                    register_margin,
                    global_shift,
                    blockidx,
                    n_blocks
                )

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # caco2_fix_tiles()

    # --------------------------------------------------------------------------
    # registering tiles using global shift does not give good results.
    # --------------------------------------------------------------------------

    # cmds_parallel_register_caco2()
    # register_caco2(global_shift=True)

    # stats caco2 tiles:
    # caco2_info_tiles()
    measure_area_fg_bg_tiles()
    sys.exit()


    # tmp - delete later. check if new data is new. ----------------------------
    # check_if_new()
    #  -------------------------------------------------------------------------

    register_margin = 64
    # sample_patches(use_registered_tiles=False,
    #                psize=512,
    #                register_margin=register_margin
    #                )

    # split_caco2()  # create folds.

    # create_cmds_parallel_patch_register_caco2()
    #
    register_patches(register_margin=register_margin, global_shift=True)

