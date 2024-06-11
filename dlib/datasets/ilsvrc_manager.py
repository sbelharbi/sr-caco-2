import os
import sys
from os.path import join, dirname, abspath
import random
from typing import Tuple
from tqdm import tqdm
import subprocess

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants, utils_config
from dlib.datasets.wsol_loader import configure_metadata
from dlib.datasets import wsol_loader
from dlib.utils.tools import chunks_into_n
from dlib.utils.tools import chunk_it
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils.shared import is_cc


def check_files_exist():
    assert not constants.DEBUG
    args = utils_config.get_config(constants.ILSVRC)

    # os.env['TMP'] hold a tmp path to ilsvrc.
    # tod: remove this line for publication.
    args['data_root'] = os.environ['TMP']

    for split in wsol_loader._SPLITS:
        print(f'Inspecting split: {split}')
        path = os.path.normpath(join(root_dir, args['metadata_root'], split))
        meta = configure_metadata(path)
        ids = wsol_loader.get_image_ids(metadata=meta, proxy=False)

        missed = []
        for id in tqdm(ids, ncols=80, total=len(ids)):
            pathimg = join(args['data_root'], constants.ILSVRC, id)

            if not os.path.isfile(pathimg):
                missed.append(pathimg)

        print(f'Split: {split}.  Found {len(missed)} missed images')


def chunk_trainset(debug: bool):
    assert debug == constants.DEBUG
    print(f'DEBUG: {debug}')

    set_seed(seed=0)

    split = constants.TRAINSET
    args = config.get_config(constants.ILSVRC)
    path = os.path.normpath(join(root_dir, args['metadata_root'], split))
    meta = configure_metadata(path)
    ids = wsol_loader.get_image_ids(metadata=meta, proxy=False)

    nbr_chunks = constants.NBR_CHUNKS_TR[args['dataset']]
    print(f'split {split} nbr_chunks {nbr_chunks} path {path}')

    import time
    t0 = time.perf_counter()
    for i in range(1000):
        random.shuffle(ids)

    print('time {}'.format(time.perf_counter() - t0))

    for i, chunk in enumerate(chunks_into_n(ids, nbr_chunks)):
        with open(join(path, f'train_chunk_{i}.txt'), 'w') as fout:
            for sample in chunk:
                fout.write(f'{sample}\n')


def compress_chunks():
    USE_BASH_CRIPT = True

    print(f'DEBUG: {constants.DEBUG}')

    split = constants.TRAINSET
    args = utils_config.get_config(constants.ILSVRC)
    path = os.path.normpath(join(root_dir, args['metadata_root'], split))
    meta = configure_metadata(path)
    ids = wsol_loader.get_image_ids(metadata=meta, proxy=False)

    args['data_root'] = os.environ['TMP']

    nbr_chunks = constants.NBR_CHUNKS_TR[args['dataset']]

    fdtmp = join(root_dir, 'tmp')
    if not os.path.isdir(fdtmp):
        os.makedirs(fdtmp, exist_ok=True)

    for i in range(nbr_chunks):
        print(f'Processing chunck {i}/{nbr_chunks}')
        lsamples = []
        with open(join(path, f'train_chunk_{i}.txt'), 'r') as f:
            for sid in f.readlines():
                lsamples.append(sid)
                # lsamples.append(join(args['data_root'], constants.ILSVRC, sid))

        with open(join(fdtmp, f'train_chunk_{i}.txt'), 'w') as fout:
            for sample in lsamples:
                fout.write(f'{sample}')

    # compression
    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = args['dataset']
    if dsname.startswith(pre):
        dsname = dsname.replace('{}_'.format(pre), '')

    indir = join(args['data_root'], dsname)
    outdir = join(args['data_root'], f"compressed_{args['dataset']}")
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    lcmds = []
    lcmdpath = join(root_dir, 'compress.sh')
    fcmds = open(lcmdpath, 'w')
    fcmds.write('#!/usr/bin/env bash  \n')

    for i in range(nbr_chunks):
        out = join(outdir, f'train_chunk_{i}.tar.zst')
        cnt = join(fdtmp, f'train_chunk_{i}.txt')
        cmd = f'cd {indir} && '
        cmd += f"tar -cf  {out} --use-compress-program=zstd -T {cnt} "

        lcmds.append(cmd)
        fcmds.write(f'echo "{i}/{nbr_chunks}" \n'
                    f'time ( {cmd} ) \n')

    fcmds.write('I am done with compressing train chunks.')
    os.system(f'chmod +x {lcmdpath}')
    fcmds.close()

    for cm in tqdm(lcmds, ncols=80, total=len(lcmds)):
        try:
            if USE_BASH_CRIPT:
                p = None
            else:
                p = subprocess.Popen(cm, shell=True)
            # for some reason, p will get slower throw lcmds.
            # will use bash directly.
        except subprocess.SubprocessError as e:
            print("Failed to run: {}. Error: {}".format(cm, e))
            p = None
        if isinstance(p, subprocess.Popen):
            p.wait()

    if USE_BASH_CRIPT:
        print('DELEGATED TO REAL BASH!')
    else:
        print('i am not done compressing all train chunks.')


def compress_vld_tst():

    split = constants.TRAINSET
    args = utils_config.get_config(constants.ILSVRC)
    args['data_root'] = os.environ['TMP']

    outdir = join(args['data_root'], f"compressed_{args['dataset']}")
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = args['dataset']
    if dsname.startswith(pre):
        dsname = dsname.replace('{}_'.format(pre), '')

    for f in [join(outdir, 'val.tar.xz'), join(outdir, 'val2.tar.xz')]:
        if os.path.isfile(f):
            os.system(f'rm {f}')

    indir = join(args['data_root'], dsname)
    lcmds = [
        f"tar -cf {join(outdir, 'val.tar.zst')} "
        f"--use-compress-program=zstd val",
        f"tar -cf {join(outdir, 'val2.tar.zst')} "
        f"--use-compress-program=zstd val2"
    ]

    cmdx = f'cd {indir} && ' + ' { ' + " & ".join(lcmds) + ' & }'
    print(f'CMD: {cmdx}')

    lpro = []

    for cmd in lcmds:
        try:
            p = subprocess.Popen(f'cd {indir} && {cmd} ', shell=True)
            print(f'launched {cmd}')
        except subprocess.SubprocessError as e:
            print("Failed to run: {}. Error: {}".format(cmdx, e))
            p = None

        lpro.append(p)

    for i, p in enumerate(lpro):
        if isinstance(p, subprocess.Popen):
            print(f'waiting process {i}')
            p.wait()
    print('i am not done compressing vl tst.')


def _uncompress_chunks_sanity_check():

    print(f'DEBUG: {constants.DEBUG}')

    split = constants.TRAINSET
    args = utils_config.get_config(constants.ILSVRC)
    path = os.path.normpath(join(root_dir, args['metadata_root'], split))
    meta = configure_metadata(path)
    ids = wsol_loader.get_image_ids(metadata=meta, proxy=False)

    args['data_root'] = os.environ['TMP']

    nbr_chunks = constants.NBR_CHUNKS_TR[args['dataset']]

    fdtmp = join(root_dir, 'tmp')
    if not os.path.isdir(fdtmp):
        os.makedirs(fdtmp, exist_ok=True)

    for i in range(nbr_chunks):
        print(f'Processing chunck {i}/{nbr_chunks}')
        lsamples = []
        with open(join(path, f'train_chunk_{i}.txt'), 'r') as f:
            for sid in f.readlines():
                lsamples.append(sid)
                # lsamples.append(join(args['data_root'], constants.ILSVRC, sid))

        with open(join(fdtmp, f'train_chunk_{i}.txt'), 'w') as fout:
            for sample in lsamples:
                fout.write(f'{sample}')

    # compression
    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = args['dataset']
    if dsname.startswith(pre):
        dsname = dsname.replace('{}_'.format(pre), '')

    indir = join(args['data_root'], dsname)
    outdir = join(args['data_root'], f"compressed_{args['dataset']}")
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    missed = []
    lproc = []

    for i in range(nbr_chunks):
        out = f'train_chunk_{i}.tar.zst'
        cmd = f'cd {outdir} && '
        cmd += f"tar -xf  {out} --use-compress-program=zstd "
        print(cmd)

        try:
            p = subprocess.Popen(cmd, shell=True)
        except subprocess.SubprocessError as e:
            print("Failed to run: {}. Error: {}".format(cmd, e))
            p = None

        lproc.append(p)

        if True or (i == (nbr_chunks - 1)):
            for p in lproc:
                if isinstance(p, subprocess.Popen):
                    p.wait()
            lproc = []

    for i in range(nbr_chunks):
        cnt = join(fdtmp, f'train_chunk_{i}.txt')
        with open(cnt, 'r') as fout:
            for id in fout.readlines():
                pathimg = join(outdir, id.strip('\n'))
                print(pathimg)
                if not os.path.isfile(pathimg):
                    missed.append(pathimg)

    print(f'missed samples {len(missed)}.')


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


def prepare_vl_tst_sets(dataset: str) -> Tuple[int, str]:
    assert is_cc()
    assert dataset == constants.ILSVRC
    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = dataset
    if dataset.startswith(pre):
        dsname = dataset.replace('{}_'.format(pre), '')

    mk_ds = f"mkdir -p $SLURM_TMPDIR/datasets/wsol-done-right/{dsname} "

    root_data = f"{os.environ['SCRATCH']}/datasets/wsol-done-right/" \
                f"compressed_{dataset}"
    dest = f"$SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/ "

    px = 'srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --exclusive -c 1 '

    jobs = []
    errors = []
    l_comds = []

    for i, name in enumerate(['val.tar.zst', 'val2.tar.zst']):
        c = join(root_data, name)
        c_p = f'cp {c} {dest}  '
        c_d = f'cd {dest}  '
        extr = f'tar -xf {c} --use-compress-program=zstd  -C {dest}'
        rm = f'rm {name} '

        if not l_comds:
            l_comds.extend([mk_ds, extr])
        else:
            l_comds.append(extr)

    path_cmds = write_cmds_scratch(l_comds, f'eval-{os.getpid()}.sh')

    cmd = f'bash {path_cmds}'
    try:
        p = subprocess.Popen(cmd, shell=True)
        e_ = None
    except subprocess.SubprocessError as e:
        print("Failed to run: {}. Error: {}".format(cmd, e))
        p = None
        e_ = e

    if isinstance(p, subprocess.Popen):
        p.wait()

    # for i, p in enumerate(jobs):
    #     if isinstance(p, subprocess.Popen):
    #         p.wait()
            # output, error = p.communicate()
            # print('ERROR', error)
            # if error:
            #     return -1, error.strip().decode("utf-8")
    if e_ is not None:
        return -1, f'Failed subprocess.Popen: {e_}'

    print(f'vl tst: process pid {os.getpid()} has succeeded. ls:')
    os.system(f"ls $SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/ ")
    return 0, 'Success'


def prepare_next_bucket(bucket: int, dataset: str) -> Tuple[int, str]:
    assert is_cc()
    assert dataset == constants.ILSVRC
    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = dataset
    if dataset.startswith(pre):
        dsname = dataset.replace('{}_'.format(pre), '')

    mk_ds = f"mkdir -p $SLURM_TMPDIR/datasets/wsol-done-right/{dsname} "

    root_data = f"{os.environ['SCRATCH']}/datasets/wsol-done-right/" \
                f"compressed_{dataset}"
    dest = f"$SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/ "

    px = 'srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 '

    chunks = list(range(constants.NBR_CHUNKS_TR[dataset]))
    buckets = list(chunk_it(chunks, constants.BUCKET_SZ))

    jobs = []

    for i in buckets[bucket]:
        name = f'train_chunk_{i}.tar.zst'
        c = join(root_data, name)
        c_p = f'cp {c} {dest}  '
        c_d = f'cd {dest}  '
        extr = f'tar -xf {c} --use-compress-program=zstd  -C {dest}'
        rm = f'rm {name} '

        cmd = f'( {mk_ds}  && {px} {extr}  ) & '
        l_comds = [mk_ds, extr]
        path_cmds = write_cmds_scratch(
            l_comds, f'bucket-{bucket}-chunk-{i}-{os.getpid()}.sh')

        cmd = f'bash {path_cmds}'
        try:
            p = subprocess.Popen(cmd, shell=True)
        except subprocess.SubprocessError as e:
            print("Failed to run: {}. Error: {}".format(cmd, e))
            p = None

        jobs.append(p)

    for p in jobs:
        if isinstance(p, subprocess.Popen):
            p.wait()
            # output, error = p.communicate()
            # print('ERROR', error)
            # if error:
            #     return -1, error.strip().decode("utf-8")

        elif p is None:
            return -1, 'Failed subprocess.Popen'

    print(f' train: process pid {os.getpid()} has succeeded. ls:')
    os.system(f"ls $SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/ ")
    return 0, 'Success'


def delete_train(bucket: int, dataset: str) -> Tuple[int, str]:
    assert is_cc()
    assert dataset == constants.ILSVRC
    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = dataset
    if dataset.startswith(pre):
        dsname = dataset.replace('{}_'.format(pre), '')

    cmd = f"rm -r $SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/train "

    l_comds = [cmd]
    path_cmds = write_cmds_scratch(
        l_comds, f'delete-train-bucket-{bucket}-{dsname}-{os.getpid()}.sh')

    cmd = f'bash {path_cmds}'
    try:
        p = subprocess.Popen(cmd, shell=True)
    except subprocess.SubprocessError as e:
        print("Failed to run: {}. Error: {}".format(cmd, e))
        p = None

    if isinstance(p, subprocess.Popen):
        p.wait()

    elif p is None:
        return -1, 'Failed subprocess.Popen'

    print(f' train: process pid {os.getpid()} has succeeded in deleting '
          f'previous bucket. '
          f'ls $SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/:')
    os.system(f"ls $SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/ ")
    print(f' train: process pid {os.getpid()} has succeeded in deleting '
          f'previous bucket. '
          f'ls $SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/train:')
    # os.system(f"ls $SLURM_TMPDIR/datasets/wsol-done-right/{dsname}/train ")
    return 0, 'Success'


if __name__ == '__main__':
    # check_files_exist()

    # create chunks
    # chunk_trainset(debug=True)
    # chunk_trainset(debug=False)

    # compress chunks.
    # compress_vld_tst()

    compress_chunks()
    # _uncompress_chunks_sanity_check()






























