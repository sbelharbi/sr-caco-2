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
import yaml
import munch
import pickle as pkl

import torch

root_dir = dirname(abspath(__file__))
sys.path.append(root_dir)

from  dlib.utils import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import log_device
from dlib.utils import utils_config

from dlib.utils.utils_reproducibility import set_seed


from dlib.models.select_model import define_model
from dlib.utils.utils_dataloaders import get_all_eval_loaders
from dlib.utils.utils_trainer import evaluate
from dlib.utils.utils_tracker import find_last_tracker
from dlib.utils.utils_tracker import save_tracker

_SURVEY_ABLATIONS = 'SURVEY_ABLATIONS'


def evaluate_pretrained():
    """
    Evaluate test set.
    :return:
    """
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--exp_path", type=str, default=None)

    parsedargs = parser.parse_args()
    exp_path = parsedargs.exp_path

    assert os.path.isdir(exp_path)


    _VERBOSE = True
    with open(join(exp_path, 'config_model.yml'), 'r') as fy:
        args = yaml.safe_load(fy)

        args_dict = deepcopy(args)

        args['distributed'] = False
        args['data_root'] = utils_config.get_root_datasets(task=args['task'])

        args = Dict2Obj(args)

        args.outd = exp_path

    test_dsets = args.test_dsets
    assert len(test_dsets.split('+')) == 1, len(test_dsets.split('+'))
    split = test_dsets
    _CODE_FUNCTION = f'eval_test_{split}'

    _DEFAULT_SEED = args.myseed
    os.environ['MYSEED'] = str(args.myseed)

    outd = join(exp_path, _CODE_FUNCTION)

    os.makedirs(outd, exist_ok=True)

    msg = f'Task: {args.task}. ' \
          f'Trainset: {args.train_dsets} \t Method: {args.method}.'

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
    DLLogger.log(fmsg("Evaluate split {}".format(split)))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_device(parsedargs)
    torch.cuda.set_device(0)

    args.is_train = False
    model_weights_path = join(exp_path, 'best-models/G-model.pth')
    args.netG['checkpoint_path_netG'] = model_weights_path

    model = define_model(args)
    model.load()
    model.netG.eval()

    DLLogger.log(model.info_network())

    args.outd = exp_path
    args.outd_backup = exp_path
    args.is_master = True

    test_loaders = get_all_eval_loaders(args, args.test_dsets, n=-1)
    tracker, roi_tracker = find_last_tracker(outd, args)

    tracker, roi_tracker = evaluate(args=args,
                                    model=model,
                                    loaders=test_loaders,
                                    tracker=tracker,
                                    roi_tracker=roi_tracker,
                                    current_step=-1,
                                    epoch=-1,
                                    split=constants.TESTSET,
                                    use_best_models=True,
                                    nbr_to_plot=30
                                    )

    save_tracker(outd,
                 tracker=tracker,
                 roi_tracker=roi_tracker
                 )

    DLLogger.log(fmsg("Bye."))




if __name__ == '__main__':

    evaluate_pretrained()

