import copy
import fnmatch
import os
import random
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import math
import pprint
from typing import List, Tuple

import yaml
import munch
import numpy as np
import torch

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)

from dlib.utils.utils_config import get_root_datasets
from dlib.utils import constants
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils.shared import fmsg


_SEED = 0


def find_name_files(direc: str, pattern: str) -> list:
    files = []

    for r, d, f in os.walk(direc):
        for file in f:
            if fnmatch.fnmatch(file, pattern):
                files.append(os.path.join(r, file))
    return files


def pair_samples(master_path: str, path_hr: str, path_lr) -> dict:
    low_res = find_name_files(path_lr, '*.tif')
    high_res = find_name_files(path_hr, '*.tif')
    l_h = dict()
    h_l = dict()
    cont = dict()

    if not master_path.endswith(os.sep):
        _master_path = master_path + os.sep
    else:
        _master_path = master_path

    for f in low_res:
        part = f.replace(_master_path, '')
        part_l = part
        l = join(master_path, part_l)
        assert os.path.isfile(l), l

        # option 1
        part_h = part.replace('wf', 'gt')
        assert 'wf' in l, l
        h = join(master_path, part_h)

        if not os.path.isfile(h):
            part_h = None
            # option 2
            b = os.sep + basename(l)
            for p in high_res:
                if p.endswith(b):
                    part_h = p.replace(_master_path, '')
            assert part_h is not None, part_h

        if part_h not in cont:
            cont[part_h] = 0
        else:
            cont[part_h] += 1

        part_h = f'{part_h}{constants.CODE_IDENTIFIER}{cont[part_h]}'
        assert part_h not in h_l, part_h
        h_l[part_h] = part_l
        l_h[part_l] = part_h

    return l_h, h_l


def write_k_to_file(l: list, path_file):
    with open(path_file, 'w') as fx:
        for k, v in l:
            fx.write(f'{k},{v}\n')


def get_specimen_files(path: str, speciemen: str, name: str, scale: int,
                       debug: bool):

    if debug:
        _train_n = 531
        _valid_n = 527
        _test_n = 528
    else:
        _train_n = -1
        _valid_n = -1
        _test_n = -1

    REPEAT = 1000
    task = constants.SUPER_RES
    outd = join(root_dir, constants.RELATIVE_META_ROOT, task)

    # train
    l_h, h_l = pair_samples(path,
                            join(path, 'train', speciemen, 'training_gt'),
                            join(path, 'train', speciemen, 'training_wf'))

    ds_name = f'{name}-{speciemen.lower()}-{constants.TRAINSET}-X-{scale}'
    _outd = join(outd, ds_name)
    os.makedirs(_outd, exist_ok=True)
    keys = list(l_h.keys())

    for i in range(REPEAT):
        random.shuffle(keys)

    keys = keys[:_train_n]
    pairs = [[k, l_h[k]] for k in keys]

    write_k_to_file(pairs, join(_outd, 'l_h.txt'))
    pairs = [[k[1], k[0]] for k in pairs]
    write_k_to_file(pairs, join(_outd, 'h_l.txt'))

    # valid
    l_h, h_l = pair_samples(path,
                            join(path, 'train', speciemen, 'validate_gt'),
                            join(path, 'train', speciemen, 'validate_wf'))

    ds_name = f'{name}-{speciemen.lower()}-{constants.VALIDSET}-X-{scale}'
    _outd = join(outd, ds_name)
    os.makedirs(_outd, exist_ok=True)
    keys = list(l_h.keys())

    for i in range(REPEAT):
        random.shuffle(keys)

    keys = keys[:_valid_n]
    pairs = [[k, l_h[k]] for k in keys]

    write_k_to_file(pairs, join(_outd, 'l_h.txt'))
    pairs = [[k[1], k[0]] for k in pairs]
    write_k_to_file(pairs, join(_outd, 'h_l.txt'))

    # test
    l_h, h_l = pair_samples(path,
                            join(path, 'test', speciemen, 'test_gt'),
                            join(path, 'test', speciemen, 'test_wf'))

    ds_name = f'{name}-{speciemen.lower()}-{constants.TESTSET}-X-{scale}'
    _outd = join(outd, ds_name)
    os.makedirs(_outd, exist_ok=True)

    keys = list(l_h.keys())

    for i in range(REPEAT):
        random.shuffle(keys)

    keys = keys[:_test_n]
    pairs = [[k, l_h[k]] for k in keys]

    write_k_to_file(pairs, join(_outd, 'l_h.txt'))
    pairs = [[k[1], k[0]] for k in pairs]
    write_k_to_file(pairs, join(_outd, 'h_l.txt'))


def process_biosr_v1(path: str, write: bool):
    set_seed(_SEED)

    name = 'biosrv1'

    for specimen in [constants.CCPS, constants.ER, constants.F_ACTIN,
              constants.MICROTUBULES]:
        print(fmsg(f'Creating folds of {name}: {specimen}.'))

        get_specimen_files(path, specimen, name, scale=2, debug=True)


if __name__ == '__main__':
    path = join(get_root_datasets(constants.SUPER_RES), 'biosr')
    process_biosr_v1(path, write=False)
