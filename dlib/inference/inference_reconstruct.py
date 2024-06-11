from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import subprocess
from pathlib import Path
import datetime as dt
import argparse
import more_itertools as mit

import numpy as np
import pretrainedmodels.utils
import tqdm
import yaml
import munch
import pickle as pkl
from texttable import Texttable

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import imageio


import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
from dlib.utils import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import log_device
from dlib.models.select_model import define_model
from dlib.utils.utils_dataloaders import get_all_eval_loaders
from dlib.utils.utils_trainer import evaluate_single_ds
from dlib.utils.utils_trainer import Interpolate
from dlib.utils.utils_tracker import find_last_tracker

from dlib.utils import utils_image
from dlib.utils.shared import reformat_id


from dlib.utils.shared import is_gsys
from dlib.utils.shared import is_tay

from dlib.utils.utils_reproducibility import set_seed



core_pattern = 'passed.txt'

_ENV_NAME = constants._ENV_NAME

if is_gsys():
    virenv = "\nCONDA_BASE=$(conda info --base) \n" \
             "source $CONDA_BASE/etc/profile.d/conda.sh\n" \
             "conda activate {}\n".format(_ENV_NAME)

elif is_tay():
    virenv = f"\nsource /projets/AP92990/venvs" \
             f"/{_ENV_NAME}/bin/activate\n"
else:  # ???
    virenv = "\nCONDA_BASE=$(conda info --base) \n" \
             "source $CONDA_BASE/etc/profile.d/conda.sh\n" \
             "conda activate {}\n".format(_ENV_NAME)


PREAMBULE = "#!/usr/bin/env bash \n {}".format(virenv)
PREAMBULE += '\n# ' + '=' * 78 + '\n'
PREAMBULE += 'cudaid=$1\nexport CUDA_VISIBLE_DEVICES=$cudaid\n\n'


sz = 15
white = "rgb(255, 255, 255)"

base = join(root_dir, "dlib/visualization/fonts/Inconsolata")
font_regular = ImageFont.truetype(join(base, 'Inconsolata-Regular.ttf'), sz)
font_bold = ImageFont.truetype(
            join(base, 'Inconsolata-Bold.ttf'), sz)
DX = 10


def to_gif(limgs: list, out_path: str, fps: int = 4):
    # limgs[0].save(out_path, save_all=True, append_images=limgs[1:],
    #               optimize=False, duration=200, loop=0)
    imageio.mimwrite(out_path, limgs, duration=(1000 * 1/fps))


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


def tag_img(img, msg: str):
    draw = ImageDraw.Draw(img)
    x = 5
    draw, x = drawonit(draw, x, 0, msg, white, font_bold, DX)

    return img


def mk_fd(fd):
    os.makedirs(fd, exist_ok=True)


def evaluate():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--eval_ds_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--reconstruct_input", type=str, default=None)
    parser.add_argument("--exp_path", type=str, default=None)

    parsedargs = parser.parse_args()
    eval_ds_name = parsedargs.eval_ds_name
    exp_path = parsedargs.exp_path
    cudaid = parsedargs.cudaid
    split = parsedargs.split
    reconstruct_input = parsedargs.reconstruct_input

    assert reconstruct_input in constants.RECON_INPUTS, reconstruct_input
    assert split in constants.SPLITS, split
    assert os.path.isdir(exp_path)

    _CODE_FUNCTION = f'inf-reconstruct-eval_ds_name-{eval_ds_name}'

    _VERBOSE = True
    with open(join(exp_path, 'config_final.yml'), 'r') as fy:
        args: dict = yaml.safe_load(fy)
        args['c_cudaid'] = str(cudaid)
        args['reconstruct_input'] = reconstruct_input  # todo: modified.
        args = Dict2Obj(args)

    _DEFAULT_SEED = args.myseed
    os.environ['MYSEED'] = str(args.myseed)

    outd = join(exp_path, _CODE_FUNCTION)
    args.outd = outd
    args.outd_backup = outd
    args.eval_bsize = 2

    mk_fd(outd)

    msg = f'Task: {args.task} \t net_type: {args.netG["net_type"]} \t'

    log_backends = [
        ArbJSONStreamBackend(Verbosity.VERBOSE,
                             join(outd, "log.json")),
        ArbTextStreamBackend(Verbosity.VERBOSE,
                             join(outd, "log.txt")),
    ]

    if _VERBOSE:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
    DLLogger.init_arb(backends=log_backends, is_master=True, reset=True)
    DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))
    DLLogger.log(fmsg(f"Evaluate. Task {args.task}, eval ds name:"
                      f" {eval_ds_name}"))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_device(parsedargs)
    model = define_model(args)
    # todo: deal with multi-valid
    _path_model = join(exp_path, 'best-models', 'G-model.pth')
    assert os.path.isfile(_path_model), _path_model
    model.load_network(_path_model,
                       model.netG,
                       strict=True,
                       param_key='params')
    model.set_eval_mode()

    test_loaders = get_all_eval_loaders(args, eval_ds_name, n=-1)

    tracker, roi_tracker = find_last_tracker(outd, args)
    model.flush()
    nbr_to_plot = 100

    save_img_dir = join(outd, 'images', reconstruct_input, split, eval_ds_name)
    os.makedirs(save_img_dir, exist_ok=True)

    # debug: -------------------------------------------------------------------
    # generate_synthetic_via_noise(test_loaders[eval_ds_name], outd)
    # sys.exit()
    # --------------------------------------------------------------------------

    tracker, roi_tracker = evaluate_single_ds(args=args,
                                              model=model,
                                              loader=test_loaders[eval_ds_name],
                                              ds_name=eval_ds_name,
                                              tracker=tracker,
                                              roi_tracker=roi_tracker,
                                              current_step=-1,
                                              epoch=-1,
                                              split=split,
                                              nbr_to_plot=nbr_to_plot,
                                              save_img_dir=save_img_dir
                                              )

    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    _model_interp = Interpolate(
        task=args.task, scale=args.scale,
        scale_mode=args.basic_interpolation).to(device)
    _ds_name = f'{eval_ds_name}_{args.basic_interpolation}'

    save_img_dir = join(outd, 'images', reconstruct_input, split, _ds_name)
    os.makedirs(save_img_dir, exist_ok=True)

    tracker, roi_tracker = evaluate_single_ds(args=args,
                                              model=_model_interp,
                                              loader=test_loaders[eval_ds_name],
                                              ds_name=_ds_name,
                                              tracker=tracker,
                                              roi_tracker=roi_tracker,
                                              current_step=-1,
                                              epoch=-1,
                                              split=split,
                                              nbr_to_plot=nbr_to_plot,
                                              save_img_dir=save_img_dir
                                              )


    DLLogger.log(fmsg(f"Evaluate. Task {args.task}, eval set: {eval_ds_name}"))
    DLLogger.log(fmsg('Time: {}'.format(dt.datetime.now() - t0)))

    # end standard evaluation. -------------------------------------------------

def generate_synthetic_via_noise(eval_loader, outd: str):
    os.makedirs(outd, exist_ok=True)
    announce_msg('Generating synthetic low res via noise.')

    th = 7.

    for sigma in list(range(1, 21, 1)):
        sigma = float(sigma)
        sigma_d = join(outd, f'th-{th}' ,
                       f'synthetic-low-res-via-noise-sigma-{sigma}')
        os.makedirs(sigma_d, exist_ok=True)

        announce_msg(f'Processing sigma: {sigma}.')

        j = 0

        for data in tqdm.tqdm(eval_loader, total=len(eval_loader), ncols=80):
            hr = data['h_im']
            lr = data['l_im']

            hr = utils_image.tensor2uint82float(hr)  # [0, 255], tensor.
            lr = utils_image.tensor2uint82float(lr)  # [0, 255], tensor.

            hr_to_lr = F.interpolate(hr,
                                     size=lr.shape[2:],
                                     mode='bicubic')
            hr_to_lr = torch.clamp(hr_to_lr, min=0.0, max=255.)

            # create new samples
            roi_h_to_l = (hr_to_lr >= th)
            new_low = torch.normal(mean=hr_to_lr, std=sigma)
            new_low = torch.clamp(new_low, 0.0, 255.)
            new_low = new_low * roi_h_to_l.float() + (
                        1 - roi_h_to_l.float()) * hr_to_lr

            # store them.
            bsize = lr.shape[0]
            for i in range(bsize):
                img_id = data['h_id'][i]  # use H as id.
                img_lr = lr[i].squeeze().cpu().numpy()  # HW.
                img_lr = Image.fromarray(img_lr).convert('RGB')
                img_lr = tag_img(img_lr, 'real')

                img_lr_sim = new_low[i].squeeze().cpu().numpy()  # HW.
                img_lr_sim = Image.fromarray(img_lr_sim).convert('RGB')
                img_lr_sim = tag_img(img_lr_sim, f'simul-sigma:{int(sigma)}')

                h, w = lr.shape[2:]

                delta = 1
                img_big = Image.new("RGB", (w * 2 + delta, h), color='orange')
                widx = 0
                # for img in l_imgs:
                img_big.paste(img_lr_sim, (widx, 0), None)
                widx += w + delta
                img_big.paste(img_lr, (widx, 0), None)
                save_img_path = join(sigma_d, f'{reformat_id(img_id)}.png')
                img_big.save(save_img_path)

            j += 1

            if j >= 30:
                break

    announce_msg('Done generating synthetic low res via noise.')

def build_cmds():
    _NBRGPUS = 1
    # STD task.
    task = constants.RECONSTRUCT
    reconstruct_input = constants.RECON_IN_L_TO_HR
    reconstruct_type = constants.HIGH_RES
    search_dir = join(root_dir, constants.FULL_BEST_EXPS, task)

    scale = 2
    out_sz = 512

    holder_prior = {
        2: 256,  # in_sz
        4: 128,
        8: 64
    }
    in_sz = holder_prior[scale]

    cell = constants.CELL2
    # cell = constants.CELL1
    # cell = constants.CELL0

    split = constants.TESTSET

    trainsets = [
        eval(f'constants.CACO2_{constants.TRAINSET.upper()}_'
             f'X{scale}_IN_{in_sz}_OUT_{out_sz}_CELL_{cell}')
    ]

    testset = eval(f'constants.CACO2_{constants.TESTSET.upper()}_'
                   f'X{scale}_IN_{in_sz}_OUT_{out_sz}_CELL_{cell}')

    if split == constants.TESTSET:
        eval_ds_name = testset

    elif split == constants.TRAINSET:
        eval_ds_name = trainsets[0]

    else:
        raise NotImplementedError(split)

    net_type = constants.SRCNN

    passed_files = find_files_pattern(fd_in_=search_dir, pattern_=core_pattern)
    # filter by task
    tmp_passed = []
    for pf in passed_files:
        exp_fd = dirname(pf)
        with open(join(exp_fd, 'config_final.yml'), 'r') as yl:
            args = yaml.safe_load(yl)

        cnd = True
        cnd &= (args['task'] == task)
        cnd &= (args['scale'] == scale)
        cnd &= (args['test_dsets'] == testset)
        cnd &= (args['reconstruct_type'] == reconstruct_type)
        cnd &= (args['netG']['net_type'] == net_type)

        if cnd:
            tmp_passed.append(pf)

    passed_files = tmp_passed
    splitted_files = [list(c) for c in mit.divide(_NBRGPUS, passed_files)]
    assert len(splitted_files) == _NBRGPUS

    for i in range(_NBRGPUS):
        _script_path = join(root_dir, 'recons_c_{}.sh'.format(i))
        script = open(_script_path, 'w')
        script.write(PREAMBULE)
        for file in splitted_files[i]:
            exp_dir = dirname(file)

            cmd = f'python dlib/inference/inference_reconstruct.py ' \
                  f'--cudaid 0 ' \
                  f'--eval_ds_name {eval_ds_name} ' \
                  f'--split {split} ' \
                  f'--reconstruct_input {reconstruct_input} ' \
                  f'--exp_path {exp_dir} ' \
                  f'\n'
            script.write(cmd)
            print(cmd)

        script.close()
        os.system('chmod +x {}'.format(_script_path))

    print('Passed files {}'.format(len(passed_files)))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        build_cmds()
    else:
        evaluate()

