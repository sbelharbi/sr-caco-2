import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import math

import yaml
import munch
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.dataset_dpsr import DatasetDPSR
from dlib.utils import constants

import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


__all__ = ['get_train_loader', 'get_all_eval_loaders', 'get_pairs']


def get_pairs(path_file: str) -> dict:
    """
    l_h.txt or h_l.txt has the structure

    <path_h>,<path_l>
    or
    <path_l>,<path_h>
    where <path_*> is a relative path used as unique key.
    example for l_h.txt:
    path/to/image1_l.jpg, path/to/image1_h.jpg
    path/to/image2_l.jpg, path/to/image2_h.jpg
    path/to/image3_l.jpg, path/to/image3_h.jpg

    if low resolution images are meant to be created on the fly as by bicubic
    method, low resolution path key must have the form:
    'None_<something_unique>'.
    ...
    """
    assert os.path.isfile(path_file), path_file

    pairs = {}
    with open(path_file, 'r') as f:
        for line in f.readlines():
            id_1, id_2 = line.strip('\n').split(',')
            assert id_1 not in pairs, id_1
            pairs[id_1] = id_2
    return pairs


def get_train_loader(args, debug_n: int = -1):
    """

    :param args:
    :param debug_n: int. number of train samples to consider. useful for fast
    debug. set to -1 for all samples.
    :return:
    """
    assert isinstance(debug_n, int), type(debug_n)
    assert (debug_n == -1) or (debug_n > 0), debug_n

    datasets_names = args.train_dsets.split(constants.SEP)
    datasets_names = [x for x in datasets_names if x != '']

    assert len(datasets_names) > 0, 'no train sets'
    assert all([f'X_{args.scale}' in ds for ds in datasets_names])
    assert all([ds in constants.datasets for ds in datasets_names]), list(zip(
        [ds in constants.datasets for ds in datasets_names], datasets_names))

    folds_path = join(root_dir, args.splits_root)
    assert os.path.isdir(folds_path)
    pairs_h = dict()
    pairs_l = dict()

    assert isinstance(args.train_n, float), type(args.train_n)
    assert 0 < args.train_n <= 1., args.train_n

    for ds in datasets_names:
        assert f'X_{args.scale}' in ds

        path_l_h = join(args.splits_root, ds, 'l_h.txt')
        pairs_l_h = get_pairs(path_l_h)
        _n = debug_n

        if (debug_n == -1) and (args.train_n != 1.):
            z = len(list(pairs_l_h.keys()))
            _n = int(args.train_n * z)
            _n = min(max(_n, 1), z)

        if _n != -1:
            pairs_l_h = keep_n_keys_dict(pairs_l_h, _n)

        path_h_l = join(args.splits_root, ds, 'h_l.txt')
        pairs_h_l = get_pairs(path_h_l)

        if _n != -1:
            pairs_h_l = keep_keys_with_matched_values(pairs_h_l,
                                                      list(pairs_l_h.keys()))

        for k in pairs_h_l:
            assert k not in pairs_h, k

            _k = k.split(constants.CODE_IDENTIFIER)[0]
            high_abs_path = join(args.data_root, constants.DS_DIR[ds], _k)
            pairs_h[k] = {
                'low_path_key': pairs_h_l[k],
                'abs_path': high_abs_path
            }

        for k in pairs_l_h:
            assert k not in pairs_l, k

            if k.startswith('None_'):
                low_abs_path = k
            else:
                _k = k.split(constants.CODE_IDENTIFIER)[0]
                low_abs_path = join(args.data_root, constants.DS_DIR[ds], _k)

            pairs_l[k] = {
                'high_path_key': pairs_l_h[k],
                'abs_path': low_abs_path
            }

    train_set = DatasetDPSR(args=args, phase=constants.TRAIN_PHASE,
                            pairs_h=pairs_h, pairs_l=pairs_l)

    train_size = int(math.ceil(
        len(train_set) / (args.batch_size * args.num_gpus)))

    if args.distributed:
        assert args.batch_size >= args.num_gpus, f'bsize: {args.batch_size}' \
                                                 f'n-gpus: {args.num_gpus}'
        train_sampler = DistributedSampler(train_set, shuffle=True,
                                           seed=args.myseed, drop_last=True)

        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  sampler=train_sampler
                                  )

    else:
        train_sampler = None
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=True
                                  )

    DLLogger.log(fmsg(f'Train loader: {args.train_dsets} \n NBR IMs: '
                      f'{len(train_set)}, NBR ITRs: {train_size}'))

    return train_loader, train_sampler


def keep_n_keys_dict(x: dict, n: int) -> dict:
    """
    Keep only n keys.
    :param x: dict.
    :param n: -1 means all.
    :return:
    """
    assert isinstance(n, int), type(n)
    assert (n == -1) or (n > 0), n

    out = dict()
    if n == -1:
        e = list(x.keys())
    else:
        e = list(x.keys())[:n]

    for k in e:
        out[k] = x[k]

    return out


def keep_keys_with_matched_values(x: dict, values: list) -> dict:
    """
    Keep only keys that
    """
    out = dict()

    for k in x:
        if x[k] in values:
            out[k] = x[k]

    return out



def get_eval_loader(args: object, ds_name: str, n: int = -1):
    """

    :param args:
    :param ds_name:
    :param n: int. how many samples to consider. could be helpful when the
    validation set is large and you want to debug. -1 means all subset.
    :return:
    """
    assert isinstance(n, int), type(n)
    assert (n == -1) or (n > 0), n

    folds_path = join(root_dir, args.splits_root)
    assert os.path.isdir(folds_path)
    pairs_h = dict()
    pairs_l = dict()

    ds = ds_name
    assert f'X_{args.scale}' in ds
    assert ds in constants.datasets

    path_l_h = join(args.splits_root, ds, 'l_h.txt')
    pairs_l_h = get_pairs(path_l_h)

    if n != -1:
        pairs_l_h = keep_n_keys_dict(pairs_l_h, n)

    path_h_l = join(args.splits_root, ds, 'h_l.txt')
    pairs_h_l = get_pairs(path_h_l)
    if n != -1:
        pairs_h_l = keep_keys_with_matched_values(pairs_h_l,
                                                  list(pairs_l_h.keys()))

    for k in pairs_h_l:
        assert k not in pairs_h, k

        _k = k.split(constants.CODE_IDENTIFIER)[0]
        pairs_h[k] = {
            'low_path_key': pairs_h_l[k],
            'abs_path': join(args.data_root, constants.DS_DIR[ds], _k)
        }

    for k in pairs_l_h:
        assert k not in pairs_l, k

        if k.startswith('None_'):
            low_path_key = k
        else:
            _k = k.split(constants.CODE_IDENTIFIER)[0]
            low_path_key = join(args.data_root, constants.DS_DIR[ds], _k)

        pairs_l[k] = {
            'high_path_key': pairs_l_h[k],
            'abs_path': low_path_key
        }

    eval_set = DatasetDPSR(args=args, phase=constants.EVAL_PHASE,
                           pairs_h=pairs_h, pairs_l=pairs_l)

    eval_set._ids_to_float_and_reverse()  # specific to eval sets.

    multi_gpu = False
    if args.distributed and (args.eval_bsize > 1):
        assert args.eval_bsize >= args.num_gpus, f'bsize: {args.eval_bsize}' \
                                                 f'n-gpus: {args.num_gpus}'

        eval_sampler = DistributedSampler(eval_set, shuffle=False,
                                          seed=args.myseed, drop_last=False)

        eval_loader = DataLoader(eval_set,
                                 batch_size=args.eval_bsize,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False,
                                 pin_memory=True,
                                 sampler=eval_sampler)

        multi_gpu = True
    else:
        eval_loader = DataLoader(eval_set, batch_size=args.eval_bsize,
                                 shuffle=False, num_workers=args.num_workers,
                                 drop_last=False, pin_memory=True)

    DLLogger.log(fmsg(f'Eval loader [multi-gpu: {multi_gpu}]: {ds_name} \n '
                      f'NBR IMs: {len(eval_set)}, '
                      f'NBR ITRs: {len(eval_loader)}'))

    return eval_loader


def get_all_eval_loaders(args: object, ds_names: str, n: int = -1) -> dict:
    datasets_names = ds_names.split(constants.SEP)
    datasets_names = [x for x in datasets_names if x != '']

    assert len(datasets_names) > 0, f'no eval sets in: {ds_names}'
    assert all([f'X_{args.scale}' in ds for ds in datasets_names])

    assert all([ds in constants.datasets for ds in datasets_names]), list(zip(
        [ds in constants.datasets for ds in datasets_names], datasets_names))

    return {
        ds: get_eval_loader(args, ds, n) for ds in datasets_names
    }
