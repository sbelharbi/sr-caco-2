import os
import sys
from os.path import join, dirname, abspath
from typing import Union, List
import subprocess


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.shared import is_cc
from dlib.utils import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


__all__ = ['move_datasets_scrach_to_node']


def write_cmds_scratch(l_cmds: list, filename: str) -> str:
    assert is_cc()

    os.makedirs(join(os.environ["SCRATCH"], constants.SCRATCH_COMM),
                exist_ok=True)
    path_file = join(os.environ["SCRATCH"],
                     constants.SCRATCH_COMM, filename)
    with open(path_file, 'w') as f:
        for cmd in l_cmds:
            f.write(cmd + '\n')

    return path_file


def move_datasets_scrach_to_node(args: object):
    assert args.is_node_master, 'not node master!!'
    assert is_cc()

    task = args.task
    greetings = 'echo "Start moving datasets into the current node...." \n'
    mk_ds = f"mkdir -p $SLURM_TMPDIR/datasets/{task} \n"

    root_data = f'{os.environ["SCRATCH"]}/datasets/{constants.SCRATCH_FOLDER}'
    destin = f"$SLURM_TMPDIR/datasets/{task}/"
    DLLogger.log(fmsg(f'Transfering data from {root_data} to {destin}'))

    l_comds = [greetings, mk_ds]

    all_ds = []
    for ds in [args.train_dsets, args.valid_dsets, args.test_dsets]:
        all_ds += ds.split(constants.SEP)

    for i, ds in enumerate(all_ds):
        set_folder = constants.DS_DIR[ds]
        src = join(root_data, f'{set_folder}.tar.gz')
        extr = f'tar -xf {src} --use-compress-program=zstd  -C {destin}'

        l_comds.append(extr)

    path_cmds = write_cmds_scratch(l_comds, f'transfer-data-{os.getpid()}.sh')

    cmd = f'bash {path_cmds}'
    try:
        p = subprocess.Popen(cmd, shell=True)
        e_ = None
    except subprocess.SubprocessError as e:
        DLLogger.log("Failed to run: {}. Error: {}".format(cmd, e))
        p = None
        e_ = e

    if isinstance(p, subprocess.Popen):
        p.wait()

    if e_ is not None:
        return -1, f'Failed subprocess.Popen: {e_}'

    DLLogger.log(f'data transfer: process pid {os.getpid()} has succeeded. '
                 f'ls {destin}:')
    os.system(f"ls {destin} ")

    return 0, 'Success'
