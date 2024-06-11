import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import argparse
from copy import deepcopy
import warnings
import subprocess
import fnmatch
import glob
import shutil
import datetime as dt

import yaml
import munch
import numpy as np
import torch
import torch.distributed as dist

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants, utils_config

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc
from dlib.utils.shared import get_tag_device
from dlib.utils.tools import get_tag
from dlib.utils import utils_reproducibility as reproducibility
from dlib.utils.utils_data_cc import move_datasets_scrach_to_node
from dlib.utils.shared import safe_str_var


def find_files_pattern(fd_in_, pattern_):
    assert os.path.exists(fd_in_), "Folder {} does not exist.".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def int_or_float(v):
    assert isinstance(v, str), type(v)

    try:
        val = int(v)

        return val
    except:
        argparse.ArgumentTypeError(f'Int/float value expected. Found: '
                                   f'v: {v}. Type: {type(v)}')

    try:
        val = float(v)
        return val

    except:
        argparse.ArgumentTypeError(f'Int/float value expected. Found: '
                                   f'v: {v}. Type: {type(v)}')

    raise argparse.ArgumentTypeError(f'Int/float value expected. Found: '
                                     f'{type(v)}')


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_args(args: dict, net_type: str):
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=None,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--myseed", type=int, default=None, help="Seed.")
    parser.add_argument("--debug_subfolder", type=str, default=None,
                        help="Name of subfold for debugging. Default: ''.")
    parser.add_argument("--task", type=str, default=None, help='task.')
    parser.add_argument("--reconstruct_type", type=str, default=None,
                        help='Type of reconstruction (for reconstruction '
                             'task only).')
    parser.add_argument("--reconstruct_input", type=str, default=None,
                        help='Type of input for reconstruction (for '
                             'reconstruction task only).')
    parser.add_argument("--method", type=str, default=None, help='method.')
    parser.add_argument("--is_train", type=str2bool, default=None,
                        help='is train?.')
    parser.add_argument("--n_channels", type=int, default=None,
                        help='number input image channels.')
    parser.add_argument("--train_dsets", type=str, default=None,
                        help='name train sets. if many, separate thgemn by +.')
    parser.add_argument("--valid_dsets", type=str, default=None,
                        help='name valid sets. if many, separate thgemn by +.')
    parser.add_argument("--test_dsets", type=str, default=None,
                        help='name test sets. if many, separate thgemn by +.')
    parser.add_argument("--h_size", type=int, default=None,
                        help='Size cropped output patch size. Input patch '
                             'size will be estimated based on the scale.')

    parser.add_argument("--valid_n_samples", type=int, default=None,
                        help='Number samples for each validset. Useful for '
                             'debug.')
    parser.add_argument("--scale", type=int, default=None,
                        help='Scale factor.')

    parser.add_argument("--train_n", type=float, default=None,
                        help='percentage of trainset to consider ]0, 1.].')

    parser.add_argument("--batch_size", type=int, default=None,
                        help='Training batch size.')
    parser.add_argument("--eval_bsize", type=int, default=None,
                        help='Eval batch size.')
    parser.add_argument("--num_workers", type=int, default=None,
                        help='Number workers for dataloader.')
    parser.add_argument("--exp_id", type=str, default=None, help="Exp id.")
    parser.add_argument("--verbose", type=str2bool, default=None,
                        help="Verbosity (bool).")
    parser.add_argument("--fd_exp", type=str, default=None,
                        help="Relative path to exp folder.")
    parser.add_argument("--save_dir_models", type=str, default=None,
                        help="Name folder where to store models.")
    parser.add_argument("--save_dir_imgs", type=str, default=None,
                        help="Name folder where to store predicted images.")
    parser.add_argument("--init_pretrained_path", type=str, default=None,
                        help="Absolute path to file holding pretrained "
                             "weights.")
    parser.add_argument("--basic_interpolation", type=str, default=None,
                        help="Interpolation method.")
    parser.add_argument("--use_interpolated_low", type=str2bool, default=None,
                        help="bool. in case of real low resolution, this flag "
                             "forces to use interpolated low resolution.")
    parser.add_argument("--inter_low_th", type=float, default=None,
                        help="Threshold used to get ROI (cells) to simulate "
                             "low resolution via interpolation + noise. for "
                             "CACO2 dataset.")
    parser.add_argument("--inter_low_sigma", type=float, default=None,
                        help="Standard deviation to create new sample via "
                             "Gaussian sampling of new low resolution images. "
                             "For CACO2 dataset")

    # Train ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--E_decay", type=float, default=None,
                        help="Decay of EMA of model.")
    parser.add_argument("--G_optimizer_type", type=str, default=None,
                        help="Optimizer type.")
    parser.add_argument("--G_optimizer_lr", type=float, default=None,
                        help="Learning rate.")
    parser.add_argument("--G_optimizer_wd", type=float, default=None,
                        help="Weight decay.")
    parser.add_argument("--G_optimizer_clipgrad", type=float, default=None,
                        help="Gradient clip norm.")
    parser.add_argument("--G_optimizer_reuse", type=str2bool, default=None,
                        help="Whether to use check points or not.")
    parser.add_argument("--G_optimizer_momentum", type=float, default=None,
                        help="Momentum wit SGD.")
    parser.add_argument("--G_optimizer_nesterov", type=str2bool, default=None,
                        help="Use/not Nesterov.")
    parser.add_argument("--G_optimizer_beta1", type=float, default=None,
                        help="Beta1 / Adam.")
    parser.add_argument("--G_optimizer_beta2", type=float, default=None,
                        help="Beta2/Adam.")
    parser.add_argument("--G_optimizer_eps_adam", type=float, default=None,
                        help="Eps / Adam.")
    parser.add_argument("--G_optimizer_amsgrad", type=str2bool, default=None,
                        help="Use/not Amsgrad with Adam.")
    parser.add_argument("--G_scheduler_type", type=str, default=None,
                        help="Type of Learning rate scheduler.")
    parser.add_argument("--G_scheduler_gamma", type=float, default=None,
                        help="Gama for learning rate schedule MultiStepLR.")
    parser.add_argument("--G_scheduler_min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--G_scheduler_step_size", type=int, default=None,
                        help="Step size for lr scheduler.")
    parser.add_argument("--G_regularizer_orthstep", type=float, default=None,
                        help="Apply/not SVD orth. regularization.")
    parser.add_argument("--G_regularizer_clipstep", type=float, default=None,
                        help="Clip weights (regularization).")
    parser.add_argument("--G_param_strict", type=str2bool, default=None,
                        help="Load weights flag.")
    parser.add_argument("--E_param_strict", type=str2bool, default=None,
                        help="Load EMA flag.")
    parser.add_argument("--checkpoint_eval", type=int_or_float, default=None,
                        help="Validation frequency [iterations].")
    parser.add_argument("--checkpoint_save", type=int_or_float, default=None,
                        help="Checkpointing frequency [iterations].")
    parser.add_argument("--test_epoch_freq", type=int, default=None,
                        help="Test frequency [epochs].")
    parser.add_argument("--plot_epoch_freq", type=int, default=None,
                        help="Plot stats frequency [epochs].")
    parser.add_argument("--synch_scratch_epoch_freq", type=int, default=None,
                        help="Synch frequency node-2-scratch [epochs][CC "
                             "only].")
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Max epoch.')

    parser.add_argument('--ppiw', type=str2bool, default=None,
                        help='Use or not per-pixel importance weight to '
                             'alleviate color-unbalance.')
    parser.add_argument('--ppiw_min_per_col_w', type=float, default=None,
                        help='Minimum value for per-color weight for '
                             're-normalization.')

    parser.add_argument('--sample_tr_patch', type=str, default=None,
                        help='How to sample train patches.')
    parser.add_argument('--sample_tr_patch_th_style', type=str, default=None,
                        help='How to get the threshold for estimating ROI for '
                             'sample_tr_patch if ROIs are needed.')
    parser.add_argument('--sample_tr_patch_th', type=float, default=None,
                        help='In case fixed threshold is required for '
                             'sample_tr_patch, what is its value [0, 255].')
    parser.add_argument(f'--augment', type=str2bool, default=None,
                        help='CSR-CNN: Noise augment or not the input in '
                             'dataset (yes/no).')
    parser.add_argument(f'--augment_nbr_steps', type=int, default=None,
                        help='CSR-CNN: Noise augment or not the input in '
                             'dataset. how many upscaling steps.')
    parser.add_argument(f'--augment_use_roi', type=str2bool, default=None,
                        help='CSR-CNN: Noise augment or not the input in '
                             'dataset: use or not roi (yes/no).')

    parser.add_argument(f'--eval_over_roi_also', type=str2bool, default=None,
                        help='Evaluate also over ROI only. This can provide a '
                             'better performance assessment.')
    parser.add_argument(f'--eval_over_roi_also_model_select', type=str2bool,
                        default=None,
                        help='Perform model selection over metrics measured '
                             'over ROI only. This can provided a better '
                             'selection')
    # additional random data augmentation.
    parser.add_argument(f'--da_blur', type=str2bool, default=None,
                        help='Random blur of random block.')
    parser.add_argument(f'--da_blur_prob', type=float, default=None,
                        help='Prob. to use this blurring.')
    parser.add_argument(f'--da_blur_area', type=float, default=None,
                        help='Area of the block.')
    parser.add_argument(f'--da_blur_sigma', type=float, default=None,
                        help='Sigma of the Gaussian kernel.')

    parser.add_argument(f'--da_dot_bin_noise', type=str2bool, default=None,
                        help='Add random binary noise to a random block.')
    parser.add_argument(f'--da_dot_bin_noise_prob', type=float, default=None,
                        help='Prob. to use this data augmentation (addition of '
                             'random.noise).')
    parser.add_argument(f'--da_dot_bin_noise_area', type=float, default=None,
                        help='Area of the block.')
    parser.add_argument(f'--da_dot_bin_noise_p', type=float, default=None,
                        help='Prob. of a pixel to be set to zero. '
                             '(1 - p) is the Bernoulli dist. param.')

    parser.add_argument(f'--da_add_gaus_noise', type=str2bool, default=None,
                        help='Add random Gaussian noise to a random block.')
    parser.add_argument(f'--da_add_gaus_noise_prob', type=float, default=None,
                        help='Prob. to use this data aug.')
    parser.add_argument(f'--da_add_gaus_noise_area', type=float, default=None,
                        help='Area of the block.')
    parser.add_argument(f'--da_add_gaus_noise_std', type=float, default=None,
                        help='Standard deviation of the Gaussian. It is '
                             'centered at 0.')

    # weights sparsity
    parser.add_argument("--w_sparsity", type=str2bool, default=None,
                        help="w_sparsity: use/not weights sparsity loss.")
    parser.add_argument("--w_sparsity_lambda", type=float, default=None,
                        help="w_sparsity: lambda.")

    # models

    if net_type == constants.SWINIR:
        # SWINIR
        nt = constants.SWINIR
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_window_size', type=int, default=None,
                            help='SwinIR: window size.')
        parser.add_argument(f'--{nt}_img_range', type=float, default=None,
                            help='SwinIR: image range.')
        parser.add_argument(f'--{nt}_depths', type=str, default=None,
                            help='SwinIR: depth. int+int+int+ ....')
        parser.add_argument(f'--{nt}_embed_dim', type=int, default=None,
                            help='SwinIR: Embedding dimension.')
        parser.add_argument(f'--{nt}_num_heads', type=str, default=None,
                            help='SwinIR: number of head int+int+int ....')
        parser.add_argument(f'--{nt}_mlp_ratio', type=int, default=None,
                            help='SwinIR: MLP ratio.')
        parser.add_argument(f'--{nt}_upsampler', type=str, default=None,
                            help='SwinIR: upsampler type.')
        parser.add_argument(f'--{nt}_resi_connection', type=str, default=None,
                            help='SwinIR: residual connection: 1/3conv.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='SwinIR: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='SwinIR: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='SwinIR: Weight init. gain.')

    elif net_type == constants.ACT:
        # ACT
        nt = constants.ACT
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_n_feats', type=int, default=None,
                            help='ACT: n_feats.')
        parser.add_argument(f'--{nt}_img_range', type=float, default=None,
                            help='ACT: image range.')
        parser.add_argument(f'--{nt}_n_resgroups', type=int, default=None,
                            help='ACT: n_resgroups.')
        parser.add_argument(f'--{nt}_n_resblocks', type=int, default=None,
                            help='ACT: n_resblocks.')
        parser.add_argument(f'--{nt}_reduction', type=int, default=None,
                            help='ACT: reduction.')
        parser.add_argument(f'--{nt}_n_heads', type=int, default=None,
                            help='ACT: n_heads.')
        parser.add_argument(f'--{nt}_n_layers', type=int, default=None,
                            help='ACT: n_layers.')
        parser.add_argument(f'--{nt}_n_fusionblocks', type=int, default=None,
                            help='ACT: n_fusionblocks.')
        parser.add_argument(f'--{nt}_dropout_rate', type=float, default=None,
                            help='ACT: dropout_rate.')
        parser.add_argument(f'--{nt}_token_size', type=int, default=None,
                            help='ACT: token_size.')
        parser.add_argument(f'--{nt}_expansion_ratio', type=int, default=None,
                            help='ACT: expansion_ratio.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='ACT: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='ACT: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='ACT: Weight init. gain.')

    elif net_type == constants.GRL:
        # GRL
        nt = constants.GRL
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_window_size', type=int, default=None,
                            help='GRL: window size.')
        parser.add_argument(f'--{nt}_img_range', type=float, default=None,
                            help='GRL: image range.')
        parser.add_argument(f'--{nt}_embed_dim', type=int, default=None,
                            help='GRL: Embedding dimension.')
        parser.add_argument(f'--{nt}_mlp_ratio', type=int, default=None,
                            help='GRL: MLP ratio.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='GRL: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='GRL: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='GRL: Weight init. gain.')

    elif net_type == constants.DFCAN:
        # DFCAN
        nt = constants.DFCAN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='DFCAN: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='DFCAN: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='DFCAN: Weight init. gain.')

    elif net_type == constants.SRFBN:
        # SRFBN
        nt = constants.SRFBN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_num_features', type=int, default=None,
                            help='SRFBN: num_features.')
        parser.add_argument(f'--{nt}_num_steps', type=int, default=None,
                            help='SRFBN: num_steps.')
        parser.add_argument(f'--{nt}_num_groups', type=int, default=None,
                            help='SRFBN: num_groups.')
        parser.add_argument(f'--{nt}_use_cl', type=str2bool, default=None,
                            help='SRFBN: use curriculum learning strategy.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='SRFBN: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='SRFBN: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='SRFBN: Weight init. gain.')

    elif net_type == constants.DBPN:
        # DBPN
        nt = constants.DBPN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_base_filter', type=int, default=None,
                            help='DBPN: base_filter.')
        parser.add_argument(f'--{nt}_feat', type=int, default=None,
                            help='DBPN: feat.')
        parser.add_argument(f'--{nt}_num_stages', type=int, default=None,
                            help='DBPN: num_stages.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='DBPN: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='DBPN: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='DBPN: Weight init. gain.')

    elif net_type == constants.MSLAPSR:
        # MSLAPSR
        nt = constants.MSLAPSR
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='MSLAPSR: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='MSLAPSR: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='MSLAPSR: Weight init. gain.')

    elif net_type == constants.OMNISR:
        # OMNISR
        nt = constants.OMNISR
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_num_feat', type=int, default=None,
                            help='OMNISR: num_feat.')
        parser.add_argument(f'--{nt}_res_num', type=int, default=None,
                            help='OMNISR: res_num.')
        parser.add_argument(f'--{nt}_bias', type=str2bool, default=None,
                            help='OMNISR: bias.')
        parser.add_argument(f'--{nt}_window_size', type=int, default=None,
                            help='OMNISR: window_size.')
        parser.add_argument(f'--{nt}_block_num', type=int, default=None,
                            help='OMNISR: block_num.')
        parser.add_argument(f'--{nt}_pe', type=str2bool, default=None,
                            help='OMNISR: pe.')
        parser.add_argument(f'--{nt}_ffn_bias', type=str2bool, default=None,
                            help='OMNISR: ffn_bias.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='OMNISR: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='OMNISR: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='OMNISR: Weight init. gain.')

    elif net_type == constants.PROSR:
        # PROSR
        nt = constants.PROSR
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_residual_denseblock', type=str2bool,
                            default=None, help='PROSR: residual_denseblock.')
        parser.add_argument(f'--{nt}_num_init_features', type=int, default=None,
                            help='PROSR: num_init_features.')
        parser.add_argument(f'--{nt}_bn_size', type=int, default=None,
                            help='PROSR: bn_size.')
        parser.add_argument(f'--{nt}_growth_rate', type=int, default=None,
                            help='PROSR: growth_rate.')
        parser.add_argument(f'--{nt}_ps_woReLU', type=str2bool, default=None,
                            help='PROSR: ps_woReLU.')
        parser.add_argument(f'--{nt}_level_compression', type=int, default=None,
                            help='PROSR: level_compression.')
        parser.add_argument(f'--{nt}_res_factor', type=float, default=None,
                            help='PROSR: res_factor.')
        parser.add_argument(f'--{nt}_max_num_feature', type=int, default=None,
                            help='PROSR: max_num_feature.')
        parser.add_argument(f'--{nt}_block_compression', type=float,
                            default=None, help='PROSR: block_compression.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='PROSR: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='PROSR: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='PROSR: Weight init. gain.')

    elif net_type == constants.ENLCN:
        # ENLCN
        nt = constants.ENLCN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_n_resblock', type=int, default=None,
                            help='ENLCN: n_resblock.')
        parser.add_argument(f'--{nt}_n_feats', type=int, default=None,
                            help='ENLCN: n_feats.')
        parser.add_argument(f'--{nt}_res_scale', type=float, default=None,
                            help='ENLCN: res_scale.')
        parser.add_argument(f'--{nt}_img_range', type=float, default=None,
                            help='ENLCN: image range.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='ENLCN: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='ENLCN: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='ENLCN: Weight init. gain.')

    elif net_type == constants.NLSN:
        # NLSN
        nt = constants.NLSN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_n_resblocks', type=int, default=None,
                            help='NLSN: n_resblocks.')
        parser.add_argument(f'--{nt}_n_feats', type=int, default=None,
                            help='NLSN: n_feats.')
        parser.add_argument(f'--{nt}_n_hashes', type=int, default=None,
                            help='NLSN: n_hashes.')
        parser.add_argument(f'--{nt}_chunk_size', type=int, default=None,
                            help='NLSN: chunk_size.')
        parser.add_argument(f'--{nt}_res_scale', type=float, default=None,
                            help='NLSN: res_scale.')
        parser.add_argument(f'--{nt}_img_range', type=float, default=None,
                            help='NLSN: image range.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='NLSN: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='NLSN: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='NLSN: Weight init. gain.')

    elif net_type == constants.DRRN:
        # DRRN
        nt = constants.DRRN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_num_residual_units', type=int,
                            default=None,
                            help='DRRN: num. of residual units.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='MemNet: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='MemNet: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='MemNet: Weight init. gain.')

    elif net_type == constants.MEMNET:
        # MEMNET
        nt = constants.MEMNET
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_num_memory_blocks', type=int, default=None,
                            help='MemNet: number of memory blocks.')
        parser.add_argument(f'--{nt}_num_residual_blocks', type=int,
                            default=None,
                            help='MemNet: num. of residual blocks.')

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='MemNet: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='MemNet: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='MemNet: Weight init. gain.')

    elif net_type == constants.VDSR:
        # VDSR
        nt = constants.VDSR
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='VDSR: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='VDSR: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='VDSR: Weight init. gain.')

    elif net_type == constants.SRCNN:
        # SRCNN
        nt = constants.SRCNN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_init_type', type=str, default=None,
                            help='SRCNN: Weights init. type.')
        parser.add_argument(f'--{nt}_init_bn_type', type=str, default=None,
                            help='SRCNN: BN init. type.')
        parser.add_argument(f'--{nt}_init_gain', type=float, default=None,
                            help='SRCNN: Weight init. gain.')

    elif net_type == constants.DSRSPLINES:

        # DSR-SPLINES
        nt = constants.DSRSPLINES
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_in_ksz', type=int, default=None,
                            help='DSR-Splines: Input kernel size.')
        parser.add_argument(f'--{nt}_splinenet_type', type=str, default=None,
                            help='DSR-Splines: Arch. type of spline.')
        parser.add_argument(f'--{nt}_n_splines_per_color', type=int,
                            default=None,
                            help='DSR-Splines: number of splines per '
                                 'color-plane.')
        parser.add_argument(f'--{nt}_use_local_residual', type=str2bool,
                            default=None,
                            help='DSR-Splines: Arch. use/not local residual.')
        parser.add_argument(f'--{nt}_use_global_residual', type=str2bool,
                            default=None,
                            help='DSR-Splines: Arch. use/not global residual.')

    elif net_type == constants.CSRCNN:

        # CSR-CNN
        nt = constants.CSRCNN
        nt = safe_str_var(nt)

        parser.add_argument(f'--{nt}_in_ksz', type=int, default=None,
                            help='CSR-CNN: Input kernel size.')
        parser.add_argument(f'--{nt}_ngroups', type=int, default=None,
                            help='CSR-CNN: number of convolution groups.')
        parser.add_argument(f'--{nt}_net_type', type=str,
                            default=None,
                            help='CSR-CNN: CNN type.')
        parser.add_argument(f'--{nt}_use_local_residual', type=str2bool,
                            default=None,
                            help='CSR-CNN: Arch. use/not local residual.')
        parser.add_argument(f'--{nt}_use_global_residual', type=str2bool,
                            default=None,
                            help='CSR-CNN: Arch. use/not global residual.')

        # unet
        parser.add_argument(f'--{nt}_outksz', type=int, default=None,
                            help='CSR-CNN: Output kernel size.')
        parser.add_argument(f'--{nt}_inner_channel', type=int, default=None,
                            help='CSR-CNN: Number of channels in each layer.'
                                 'It will be multiplied by "_channel_mults".')
        parser.add_argument(f'--{nt}_norm_groups', type=int, default=None,
                            help='CSR-CNN: Size of group to perform group'
                                 'normalization.')
        parser.add_argument(f'--{nt}_channel_mults', type=str, default=None,
                            help='CSR-CNN: string underf the form "a_b_c_...".'
                                 'Indicates the depth of unet and the '
                                 'multiplier of "_inner_channel" to determine '
                                 'the number of channels per layer.')
        parser.add_argument(f'--{nt}_res_blocks', type=int, default=None,
                            help='CSR-CNN: Number of residual blocks.')
        parser.add_argument(f'--{nt}_dropout', type=float, default=None,
                            help='CSR-CNN: dropout.')


    else:
        raise NotImplementedError(f'net-type: {net_type}')


    # ======================================================================
    #                              MODEL
    # ======================================================================
    parser.add_argument("--net_type", type=str, default=None,
                        help="model's name.")
    parser.add_argument("--net_task", type=str, default=None,
                        help="model's tas: regression or 'segmentation'.")

    # ======================================================================
    #                         ELB
    # ======================================================================
    parser.add_argument("--elb_init_t", type=float, default=None,
                        help="Init t for elb.")
    parser.add_argument("--elb_max_t", type=float, default=None,
                        help="Max t for elb.")
    parser.add_argument("--elb_mulcoef", type=float, default=None,
                        help="Multi. coef. for elb..")

    # Loss: start
    parser.add_argument("--l1", type=str2bool, default=None,
                        help="use/not l1 loss.")
    parser.add_argument("--l1_use_residuals", type=str2bool, default=None,
                        help="use/not residuals.")
    parser.add_argument("--l1_lambda", type=float, default=None,
                        help="Lambda l1 loss.")
    parser.add_argument("--l2", type=str2bool, default=None,
                        help="use/not l2 loss.")
    parser.add_argument("--l2_use_residuals", type=str2bool, default=None,
                        help="use/not residuals.")
    parser.add_argument("--l2_lambda", type=float, default=None,
                        help="Lambda l2 loss.")
    parser.add_argument("--l2sum", type=str2bool, default=None,
                        help="use/not l2sum loss.")
    parser.add_argument("--l2sum_use_residuals", type=str2bool, default=None,
                        help="use/not residuals.")
    parser.add_argument("--l2sum_lambda", type=float, default=None,
                        help="Lambda l2sum loss.")
    parser.add_argument("--ssim", type=str2bool, default=None,
                        help="use/not ssim loss.")
    parser.add_argument("--ssim_lambda", type=float, default=None,
                        help="Lambda ssim loss.")
    parser.add_argument("--ssim_window_s", type=int, default=None,
                        help="Window size of ssim loss.")
    parser.add_argument("--charbonnier", type=str2bool, default=None,
                        help="use/not charbonnier loss.")
    parser.add_argument("--charbonnier_use_residuals", type=str2bool,
                        default=None, help="use/not residuals.")
    parser.add_argument("--charbonnier_lambda", type=float, default=None,
                        help="Lambda charbonnier loss.")
    parser.add_argument("--charbonnier_eps", type=float, default=None,
                        help="eps charbonnier loss.")

    parser.add_argument("--boundpred", type=str2bool, default=None,
                        help="boundpred: use bounded prediction inequalities.")
    parser.add_argument("--boundpred_use_residuals", type=str2bool,
                        default=None, help="use/not residuals.")
    parser.add_argument("--boundpred_lambda", type=float, default=None,
                        help="boundpred: lambda.")
    parser.add_argument("--boundpred_eps", type=float, default=None,
                        help="boundpred: constant > 0 to add to create "
                             "inequalities.")
    parser.add_argument("--boundpred_restore_range", type=str2bool,
                        default=None,
                        help="boundpred: scale or not the prediction (and "
                             "target) range into [0, max_color].")

    parser.add_argument("--local_moments", type=str2bool, default=None,
                        help="local moments: use/not.")
    parser.add_argument("--local_moments_use_residuals", type=str2bool,
                        default=None, help="use/not residuals.")
    parser.add_argument("--local_moments_lambda", type=float, default=None,
                        help="local moments: lambda.")
    parser.add_argument("--local_moments_ksz", type=str, default=None,
                        help="local moments: kernel(s) size. separate by '_' "
                             "if many.")

    parser.add_argument("--img_grad", type=str2bool, default=None,
                        help="Image gradient loss: use/not.")
    parser.add_argument("--img_grad_use_residuals", type=str2bool, default=None,
                        help="Image gradient loss: residuals/image?")
    parser.add_argument("--img_grad_lambda", type=float, default=None,
                        help="Image gradient loss: lambda.")
    parser.add_argument("--img_grad_norm", type=str, default=None,
                        help="Image gradient loss: norm.")

    parser.add_argument("--norm_img_grad", type=str2bool, default=None,
                        help="Norm image gradient loss: use/not.")
    parser.add_argument("--norm_img_grad_use_residuals", type=str2bool,
                        default=None,
                        help="Norm image gradient loss: residuals/image?")
    parser.add_argument("--norm_img_grad_lambda", type=float, default=None,
                        help="Norm image gradient loss: lambda.")
    parser.add_argument("--norm_img_grad_type", type=str, default=None,
                        help="Norm image gradient loss: norm type.")

    parser.add_argument("--laplace", type=str2bool, default=None,
                        help="Laplacian filter loss: use/not.")
    parser.add_argument("--laplace_use_residuals", type=str2bool, default=None,
                        help="Laplacian filter loss: residuals/image?")
    parser.add_argument("--laplace_lambda", type=float, default=None,
                        help="Laplacian loss: lambda.")
    parser.add_argument("--laplace_norm", type=str, default=None,
                        help="Laplacian loss: norm.")

    parser.add_argument("--norm_laplace", type=str2bool, default=None,
                        help="Norm Laplacian filter loss: use/not.")
    parser.add_argument("--norm_laplace_use_residuals", type=str2bool,
                        default=None,
                        help="Norm Laplacian filter loss: residuals/image?")
    parser.add_argument("--norm_laplace_lambda", type=float, default=None,
                        help="Norm Laplacian loss: lambda.")
    parser.add_argument("--norm_laplace_type", type=str, default=None,
                        help="Norm Laplacian loss: norm type.")

    parser.add_argument("--loc_var", type=str2bool, default=None,
                        help="Local variation loss: use/not.")
    parser.add_argument("--loc_var_ksz", type=int, default=None,
                        help="Local variation loss: kernel size.")
    parser.add_argument("--loc_var_use_residuals", type=str2bool, default=None,
                        help="Local variation loss: residuals/image?")
    parser.add_argument("--loc_var_lambda", type=float, default=None,
                        help="Local variation loss: lambda.")
    parser.add_argument("--loc_var_norm", type=str, default=None,
                        help="Local variation loss: norm.")

    parser.add_argument("--norm_loc_var", type=str2bool, default=None,
                        help="Norm local variation loss: use/not.")
    parser.add_argument("--norm_loc_var_ksz", type=int, default=None,
                        help="Norm local variation loss: kernel size.")
    parser.add_argument("--norm_loc_var_use_residuals", type=str2bool,
                        default=None,
                        help="Norm local variation loss: residuals/image?")
    parser.add_argument("--norm_loc_var_lambda", type=float, default=None,
                        help="Norm local variation loss: lambda.")
    parser.add_argument("--norm_loc_var_type", type=str, default=None,
                        help="Norm local variation loss: norm type.")

    parser.add_argument("--hist", type=str2bool, default=None,
                        help="Histogram match loss.")
    parser.add_argument("--hist_lambda", type=float, default=None,
                        help="Histogram loss lambda.")
    parser.add_argument("--hist_sigma", type=float, default=None,
                        help="Histogram loss: sigma to estimate soft "
                             "histogram (> 0).")
    parser.add_argument("--hist_metric", type=str, default=None,
                        help="Histogram loss: metric to measure discrepancy "
                             "between 2 histograms.")

    parser.add_argument("--kde", type=str2bool, default=None,
                        help="KDE match loss.")
    parser.add_argument("--kde_lambda", type=float, default=None,
                        help="KDE loss lambda.")
    parser.add_argument("--kde_nbins", type=int, default=None,
                        help="KDE loss: number of bins.")
    parser.add_argument("--kde_kde_bw", type=float, default=None,
                        help="KDE loss: kde band width (gaussian variance) "
                             "(> 0).")
    parser.add_argument("--kde_metric", type=str, default=None,
                        help="kde loss: metric to measure discrepancy "
                             "between 2 kde.")

    parser.add_argument("--ce", type=str2bool, default=None,
                        help="cross-entropy loss: only for net task: segmt.")
    parser.add_argument("--ce_lambda", type=float, default=None,
                        help="cross-entropy loss: lambda loss.")


    # loss ends.

    # AMP ----------------------------------------------------------------------
    parser.add_argument("--amp", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "training.")
    parser.add_argument("--amp_eval", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "inference.")
    # DDP ----------------------------------------------------------------------
    parser.add_argument('--distributed', type=str2bool, default=None,
                        help="Distributed or not (multiple or single GPU).")
    parser.add_argument("--local_rank", type=int, default=None,
                        help='DDP. Local rank. Set too zero if you are using '
                             'one node. not CC().')
    parser.add_argument("--local_world_size", type=int, default=None,
                        help='DDP. Local world size: number of gpus per node. '
                             'Not CC().')
    parser.add_argument('--init_method', default=None,
                        type=str, help='DDP. init method. CC().')
    parser.add_argument('--dist_backend', default=None, type=str,
                        help='DDP. Distributed backend. CC()')
    parser.add_argument('--world_size', type=int, default=None,
                        help='DDP. World size. CC().')

    # model selection
    parser.add_argument('--model_select_mtr', type=str, default=None,
                        help='Model selection metric over validset.')

    input_parser = parser.parse_args()

    def warnit(name, vl_old, vl):
        """
        Warn that the variable with the name 'name' has changed its value
        from 'vl_old' to 'vl' through command line.
        :param name: str, name of the variable.
        :param vl_old: old value.
        :param vl: new value.
        :return:
        """
        if vl_old != vl:
            print("Changing {}: {}  -----> {}".format(name, vl_old, vl))
        else:
            print("{}: {}".format(name, vl_old))

    attributes = input_parser.__dict__.keys()

    for k in attributes:
        val_k = getattr(input_parser, k)
        if k in args.keys():
            if val_k is not None:
                warnit(k, args[k], val_k)
                args[k] = val_k
            else:
                warnit(k, args[k], args[k])

        else:
            foundit = False
            for kk in args.keys():
                if isinstance(args[kk], dict):
                    if k in args[kk].keys():
                        foundit = True
                        if val_k is not None:
                            warnit(k, args[kk][k], val_k)
                            args[kk][k] = val_k
                        else:
                            warnit(k, args[kk][k], args[kk][k])
            if not foundit:
                raise ValueError(f'key {k} not found in args.')

    # add the current seed to the os env. vars. to be shared across this
    # process.
    # this seed is expected to be local for this process and all its
    # children.
    # running a parallel process will not have access to this copy not
    # modify it. Also, this variable will not appear in the system list
    # of variables. This is the expected behavior.
    # TODO: change this way of sharing the seed through os.environ. [future]
    # the doc mentions that the above depends on `putenv()` of the
    # platform.
    # https://docs.python.org/3.7/library/os.html#os.environ
    os.environ['MYSEED'] = str(args["myseed"])
    max_seed = (2 ** 32) - 1
    msg = f"seed must be: 0 <= {int(args['myseed'])} <= {max_seed}"
    assert 0 <= int(args['myseed']) <= max_seed, msg

    task = args['task']
    if task in [constants.SUPER_RES, constants.RECONSTRUCT]:
        task = constants.SUPER_RES  # to get data/folds
        # todo: weak.

    args['splits_root'] = join(constants.RELATIVE_META_ROOT, task)
    args['data_root'] = utils_config.get_root_datasets(task=task)

    args['num_gpus'] = len(args['cudaid'].split(','))
    args['multi_valid'] = (len(args['valid_dsets'].split(constants.SEP)) > 1)

    # Update Models
    net_type = args['netG']['net_type']
    nt = safe_str_var(net_type)

    if net_type == constants.SWINIR:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']
        args['netG'][f'{nt}_img_size'] = args['h_size'] // args['scale']

        xx = args['netG'][f'{nt}_depths'].split(constants.SEP)
        xx = [int(x) for x in xx]
        args['netG'][f'{nt}_depths'] = xx

        zz = args['netG'][f'{nt}_num_heads'].split(constants.SEP)
        zz = [int(z) for z in zz]
        args['netG'][f'{nt}_num_heads'] = zz

    elif net_type == constants.ACT:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.GRL:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']
        args['netG'][f'{nt}_img_size'] = args['h_size'] // args['scale']

    elif net_type == constants.DFCAN:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.SRFBN:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.DBPN:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.ENLCN:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.NLSN:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.OMNISR:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.PROSR:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.MSLAPSR:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.MEMNET:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.DRRN:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.VDSR:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_chans'] = args['n_channels']

    elif net_type == constants.SRCNN:
        args['netG'][f'{nt}_in_chans'] = args['n_channels']


    elif net_type == constants.DSRSPLINES:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_planes'] = args['n_channels']

    elif net_type == constants.CSRCNN:
        args['netG'][f'{nt}_upscale'] = args['scale']
        args['netG'][f'{nt}_in_planes'] = args['n_channels']


    if args['t0'] is None:
        args['t0'] = dt.datetime.now()

    args['outd'], args['subpath'] = outfd(Dict2Obj(args))

    args['outd_backup'] = args['outd']
    if is_cc():
        args['outd'] = join(os.environ["SLURM_TMPDIR"],
                            basename(normpath(args['outd'])))

        os.makedirs(args['outd'], exist_ok=True)
        # todo: if exist in scratch, move it to node.

    for dx in [args['outd'], args['outd_backup']]:
        os.makedirs(join(dx, args['save_dir_models']), exist_ok=True)
        os.makedirs(join(dx, args['save_dir_imgs']), exist_ok=True)

    cmdr = not constants.OVERRUN
    if is_cc():
        cmdr &= os.path.isfile(join(args['outd_backup'], 'passed.txt'))
        os.makedirs(join(os.environ["SCRATCH"], constants.SCRATCH_COMM),
                    exist_ok=True)
    else:
        cmdr &= os.path.isfile(join(args['outd'], 'passed.txt'))

    if cmdr:
        warnings.warn(f"EXP {args['outd']} has already been done. EXITING.")
        sys.exit(0)

    # DDP. ---------------------------------------------------------------------
    if args['distributed']:
        ngpus_per_node = torch.cuda.device_count()

        if is_cc():  # multiple nodes. each w/ multiple gpus.
            local_rank = int(os.environ.get("SLURM_LOCALID"))
            rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

            current_device = local_rank
            torch.cuda.set_device(current_device)

            args['rank'] = rank
            args['local_rank'] = local_rank
            args['is_master'] = ((local_rank == 0) and (rank == 0))
            args['c_cudaid'] = current_device
            args['is_node_master'] = (local_rank == 0)

        else:  # single machine w/ multiple gpus.
            assert len(args['cudaid'].split(',')) > 1
            assert ngpus_per_node > 1

            args['local_rank'] = int(os.environ["LOCAL_RANK"])
            args['world_size'] = ngpus_per_node
            args['is_master'] = args['local_rank'] == 0
            args['is_node_master'] = args['local_rank'] == 0
            torch.cuda.set_device(args['local_rank'])
            args['c_cudaid'] = args['local_rank']
            args['world_size'] = ngpus_per_node
    else:
        current_device = 0
        torch.cuda.set_device(current_device)
        args['is_master'] = True
        args['is_node_master'] = True

    # --------------------------------------------------------------------------

    reproducibility.set_to_deterministic(seed=args["myseed"], verbose=True)

    args_dict = deepcopy(args)
    args = Dict2Obj(args)
    # sanity check ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # todo: do more checking.
    assert isinstance(args.train['checkpoint_eval'], int) or isinstance(
        args.train['checkpoint_eval'], float), type(
        args.train['checkpoint_eval'])
    assert isinstance(args.train['checkpoint_save'], int) or isinstance(
        args.train['checkpoint_save'], float), type(
        args.train['checkpoint_save'])

    assert args.train['checkpoint_eval'] > 0, args.train['checkpoint_eval']
    assert args.train['checkpoint_save'] > 0, args.train['checkpoint_save']

    assert isinstance(args.train_n, float), type(args.train_n)
    assert 0 < args.train_n <= 1., args.train_n

    if args.eval_over_roi_also_model_select:
        msg = f"{args.eval_over_roi_also_model_select} " \
              f"{args.eval_over_roi_also}"
        assert args.eval_over_roi_also, msg

    v_n =args.valid_n_samples
    assert (v_n == -1) or (v_n > 0), v_n

    if args.augment:
        assert args.method == constants.CSRCNN_MTH, args.method


    if args.netG['net_task'] == constants.SEGMENTATION:
        assert args.method == constants.CSRCNN_MTH, args.method

    task = args.netG['net_task']
    assert task in [constants.REGRESSION, constants.SEGMENTATION], task
    if args.ce:
        assert task == constants.SEGMENTATION, f'task: {task}. CE loss: ' \
                                               f'{args.ce}'

    assert args.method in constants.METHODS, args.method
    assert args.method == constants.NETTYPE_METHOD[args.netG['net_type']]

    if args.boundpred:
        assert args.boundpred_eps > 0, args.boundpred_eps

    if args.ppiw:
        assert 0.0 < args.ppiw_min_per_col_w < 1.0, args.ppiw_min_per_col_w

    sample_p = args.sample_tr_patch
    assert sample_p in constants.SAMPLE_PATCHES, sample_p
    msg = f"{args.sample_tr_patch_th_style} not in {constants.ROI_STYLE_TH}"
    assert args.sample_tr_patch_th_style in constants.ROI_STYLE_TH, msg

    if args.sample_tr_patch in [
        constants.SAMPLE_ROI, constants.SAMPLE_EDTXROI, constants.SAMPLE_EDT
    ] and args.sample_tr_patch_th_style == constants.TH_FIX:

        msg = f"{args.color_min} <= {args.sample_tr_patch_th} " \
              f"<= {args.color_max}"
        assert args.color_min <= args.sample_tr_patch_th <= args.color_max, msg


    # todo: update.
    assert any([args.l1,
                args.l2,
                args.l2sum,
                args.ssim,
                args.charbonnier,
                args.boundpred,
                args.local_moments,
                args.img_grad,
                args.norm_img_grad,
                args.laplace,
                args.norm_laplace,
                args.loc_var,
                args.norm_loc_var,
                args.hist,
                args.kde,
                args.ce])

    if args.loc_var and args.norm_loc_var:
        msg = f'{args.loc_var_ksz} {args.norm_loc_var_ksz}'
        assert args.loc_var_ksz == args.norm_loc_var_ksz, msg

    assert args.scale > 0
    assert args.netG['net_type'] in constants.MODELS

    assert args.model_select_mtr in constants.METRICS, args.model_select_mtr
    b_in = args.basic_interpolation
    assert b_in in constants.INTERPOLATION_MODES, b_in

    return args, args_dict


def outfd(args):

    tag = [('id', args.exp_id),
           ('tsk', args.task),
           ('x', args.scale),
           # ('trds', args.train_dsets),
           # ('vlds', args.valid_dsets),
           # ('tsds', args.test_dsets),
           ('netG', args.netG['net_type']),
           ('sd', args.myseed)
           ]

    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    tag2 = []

    if args.l1:
        tag2.append(("l1", 'yes'))

    if args.l2:
        tag2.append(("l2", 'yes'))

    if args.l2sum:
        tag2.append(("l2sum", 'yes'))

    if args.ssim:
        tag2.append(("ssim", 'yes'))

    if args.charbonnier:
        tag2.append(("charb", 'yes'))

    if tag2:
        tag2 = [(el[0], str(el[1])) for el in tag2]
        tag2 = '-'.join(['_'.join(el) for el in tag2])
        tag = "{}-{}".format(tag, tag2)

    parent_lv = "exps"
    if args.debug_subfolder not in ['', None, 'None']:
        parent_lv = join(parent_lv, args.debug_subfolder)

    subfd = join(args.task, args.netG['net_type'], args.train_dsets)
    _root_dir = root_dir
    if is_cc():
        _root_dir = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER)

    subpath = join(parent_lv, subfd, tag)

    OUTD = join(_root_dir, subpath)
    OUTD = expanduser(OUTD)

    os.makedirs(OUTD, exist_ok=True)

    return OUTD, subpath


def wrap_sys_argv_cmd(cmd: str, pre):
    splits = cmd.split(' ')
    el = splits[1:]
    pairs = ['{} {}'.format(i, j) for i, j in zip(el[::2], el[1::2])]
    pro = splits[0]
    sep = ' \\\n' + (len(pre) + len(pro) + 2) * ' '
    out = sep.join(pairs)
    return "{} {} {}".format(pre, pro, out)


def copy_code(dest,
              compress=False,
              verbose=False
              ):
    exts = tuple(["py", "sh", "yaml"])
    flds_files = ['.']

    # too expensive.
    # for fld in flds_files:
    #     files = glob.iglob(os.path.join(root_dir, fld, "*"))
    #     subfd = join(dest, fld) if fld != "." else dest
    #     if not os.path.exists(subfd):
    #         os.makedirs(subfd, exist_ok=True)
    #
    #     for file in files:
    #         if file.endswith(exts):
    #             if os.path.isfile(file):
    #                 shutil.copy(file, subfd)
    # cp dlib
    dirs = ["dlib", "cmds"]
    for dirx in dirs:
        cmds = [
            "cd {} && ".format(root_dir),
            "cp -r {} {} ".format(dirx, dest)
        ]
        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)

    if compress:
        head = dest.split(os.sep)[-1]
        if head == '':  # dest ends with '/'
            head = dest.split(os.sep)[-2]
        cmds = [
            "cd {} && ".format(dest),
            "cd .. && ",
            "tar -cf {}.tar.gz {}  && ".format(head, head),
            "rm -rf {}".format(head)
               ]

        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)


def amp_log(args: object):
    _amp = False
    if args.amp:
        DLLogger.log(fmsg('AMP: activated'))
        _amp = True

    if args.amp_eval:
        DLLogger.log(fmsg('AMP_EVAL: activated'))
        _amp = True

    if _amp:
        tag = get_tag_device(args=args)
        if 'P100' in get_tag_device(args=args):
            DLLogger.log(fmsg(f'AMP [train: {args.amp},'
                              f' eval: {args.amp_eval}] is ON but '
                              f'your GPU {tag} '
                              'does not seem to have tensor cores. Your code '
                              'may experience slowness. It is better to '
                              'deactivate AMP.'))


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_type", type=str, required=True,
                        help="model's name.")
    parsedargs, unkn = parser.parse_known_args()
    net_type = parsedargs.net_type

    args: dict = utils_config.get_config(net_type=net_type)

    args, args_dict = get_args(args=args, net_type=net_type)
    distributed = args.distributed

    if distributed:

        if is_cc():
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.init_method,
                                    world_size=args.world_size,
                                    rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend)

        group = dist.group.WORLD
        group_size = torch.distributed.get_world_size(group)
        args_dict['distributed'] = group_size > 1
        msg = f'this job sees more gpus {args_dict["world_size"]} ' \
              f'than it is allowed to use {group_size}. use:' \
              f'"export CUDA_VISIBLE_DEVICES=$cudaid" to keep only ' \
              f'used gpus visible.'
        assert group_size == args_dict['world_size'], msg
        args.distributed = group_size > 1
        assert group_size == args.world_size

    log_backends = [
        ArbJSONStreamBackend(
            Verbosity.VERBOSE, join(args.outd_backup, "log.json"),
            append_if_exist=True),
        ArbTextStreamBackend(
            Verbosity.VERBOSE, join(args.outd_backup, "log.txt"),
            append_if_exist=True),
    ]

    if args.verbose:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

    if distributed:
        DLLogger.init_arb(backends=log_backends, is_master=args.is_master,
                              flush_at_log=args.verbose)
    else:
        DLLogger.init_arb(backends=log_backends, is_master=True,
                          flush_at_log=args.verbose)

    DLLogger.log(fmsg("Start time: {}".format(args.t0)))

    # log important info.
    if args.use_interpolated_low:
        DLLogger.log(fmsg(f'use_interpolated_low: {args.use_interpolated_low}'))

    amp_log(args=args)

    outd = args.outd

    if args.is_master:
        if not os.path.exists(join(outd, "code/")):
            os.makedirs(join(outd, "code/"), exist_ok=True)

        with open(join(outd, "code/config.yml"), 'w') as fyaml:
            yaml.dump(args_dict, fyaml)

        with open(join(outd, "config.yml"), 'w') as fyaml:
            yaml.dump(args_dict, fyaml)

        str_cmd = wrap_sys_argv_cmd(" ".join(sys.argv), "time python")
        with open(join(outd, "code/cmd.sh"), 'w') as frun:
            frun.write("#!/usr/bin/env bash \n")
            frun.write(str_cmd)

        copy_code(join(outd, "code/"), compress=True, verbose=False)

    # transfer data in CC: scratch to node in case of distributed.
    status = 0
    if is_cc() and distributed and args.is_node_master:
        status = move_datasets_scrach_to_node(args)

        if (status == -1) and args.is_master:
            DLLogger.log(f'Error in transferring data from scratch to node. '
                         f'Exiting.')
    if distributed:
        dist.barrier()

    if status == -1:
        sys.exit()

    return args, args_dict
