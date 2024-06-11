import os
import sys
from os.path import join, dirname, abspath
import re
import glob
from typing import Union, List

import yaml

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.utils.tools import chunk_it
from dlib.utils.utils_init_default_args import init_net_g

import dlib.dllogger as DLLogger

__all__ = ['get_config', 'find_last_checkpoint', 'save_config',
           'delete_previous_checkpoints_except_last',
           'clean_previous_checkpoints_except_last', 'get_root_datasets']


def get_root_datasets(task: str):
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = f'{os.environ["EXDRIVE"]}/datasets'
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = f'{os.environ["DATASETSH"]}/{task}'
        elif os.environ['HOST_XXX'] == 'gsys':
            baseurl = f'{os.environ["DATASETSH"]}/{task}'
        elif os.environ['HOST_XXX'] == 'tay':
            baseurl = f'{os.environ["DATASETSH"]}/{task}'
        elif os.environ['HOST_XXX'] == 'ESON':
            baseurl = f'{os.environ["DATASETSH"]}/datasets'
        else:
            raise NotImplementedError

    elif "CC_CLUSTER" in os.environ.keys():
        if "SLURM_TMPDIR" in os.environ.keys():
            # if we are running within a job use the node disc:  $SLURM_TMPDIR
            baseurl = f'{os.environ["SLURM_TMPDIR"]}/datasets/{task}'
        else:
            # if we are not running within a job, use the scratch.
            # this case my happen if someone calls this function outside a job.
            baseurl = f'{os.environ["SCRATCH"]}/datasets/{task}'

    msg_unknown_host = "Sorry, it seems we are unable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "We recommend you to add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def get_nbr_bucket(ds: str) -> int:
    nbr_chunks = constants.NBR_CHUNKS_TR[ds]
    out = chunk_it(list(range(nbr_chunks)), constants.BUCKET_SZ)
    return len(list(out))


def get_config(net_type: str = constants.SWINIR):

    args = {
        # ======================================================================
        #                               GENERAL
        # ======================================================================
        "task": constants.SUPER_RES,  # task.
        "reconstruct_type": constants.LOW_RES,  # what to to learn to
        # reconstruct: low resolution or high resolution images.
        "reconstruct_input": constants.RECON_IN_FAKE,  # input data to
        # reconstruction.
        # reconstruct_type=low_res: ----------------------------
        # fake: microscope low resolution images are
        # artificially blurred. real: high resolution are interpolated to low
        # resolution.
        # during standard training/inference: use fake.
        # during application: use real low resolution.
        # -------------------------------------------------------
        # reconstruct_type=high_res: ----------------------------
        # input: high res, or low res (interpolated tp high res).
        "is_train": True,  # train or not mode.
        "myseed": 0,  # Seed for reproducibility. int in [0, 2**32 - 1].
        "cudaid": '0',  # str. cudaid. form: '0,1,2,3' for cuda devices.
        'num_gpus': 1,  # int. number of gpus. will be set automatically from
        # 'cudaid'.
        'n_channels': 3,  # int. number of channels of input image.
        "debug_subfolder": '',  # subfolder used for debug. if '', we do not
        # consider it.
        "train_dsets": '',  # name of the train datasets. use
        # '+' to separate them.
        "valid_dsets": '',  # name of the validation datasets. use
        # '+' to separate them.
        "test_dsets": '',  # name of the test datasets. use
        # '+' to separate them.
        'multi_valid': False,  # bool. true if validation is done over
        # multiple dataset. this could be the case when the validsets are the
        # testsets. this will be set automatically.
        "valid_n_samples": -1,  # useful for debug. int. how many samples to
        # consider for each VALIDSET. -1 means all. or > 0.
        "h_size": 96,  # int. size of output cropped patch. the size of the
        # input crop size will be determined by h_size // scale.
        "scale": 2,  # int. scale factor.
        "train_n": 1.,  # float in ]0., 1.]. percentage of train samples to
        # consider. useful for ablation study.
        "color_min": 0,  # range of colors (per plane): min value. uint8.
        "color_max": 255,  # range of colors (per plane): max value. uint8.
        "batch_size": 8,  # the batch size for training. in case of
        # multi-gpus, this will be the batch size per-gpu.
        "eval_bsize": 8,  # int. batch size for evaluation.
        "num_workers": 4,  # number of workers for dataloader of the trainset.
        "exp_id": "123456789",  # exp id. random number unique for the exp.
        "verbose": True,  # if true, we print messages on the go. else,
        # we flush everything at the end.
        'fd_exp': None,  # relative path to folder where the exp.
        'abs_fd_exp': None,  # absolute path to folder where the exp.
        't0': None,  # approximate time of starting the code.
        'tend': None,  # time when this code ends.
        'running_time': None,  # the time needed to run the entire code.
        'save_dir_models': 'models',  # folder's name where to store models.
        'save_dir_imgs': 'images',  # folder's name where to store predictions.
        'data_root': '',  # absolute path to data parent.
        'splits_root': '',  # relative folder from root where splits are stored.
        'model_select_mtr': constants.PSNR_MTR,  # metric used for model
        # selectipn over validation set.
        'basic_interpolation': constants.INTER_BICUBIC,  # simple
        # performance baseline: interpolcation.
        'use_interpolated_low': False,  # useful only for datasets with true
        # low-resolution. if true, the low resolution image is obtained via
        # interpolation not using the real low resolution. the interpolation
        # method is: basic_interpolation.
        # this is useful to simulate low resolution for caco2 dataset.
        # requires noise level and thresholding to get ROI.
        "inter_low_th": 7.,  # threshold used to estimate ROI in caco2
        # dataset. ROI==cells.
        "inter_low_sigma": 6.,  # tandard deviation for simulating low res of
        # caco2 dataset. we create new sample via N(I, sigma^2). where I is
        # the high resolution image downscaled to low resolution via
        # interpolation.
        'method': constants.NETTYPE_METHOD[net_type],  # name of the method.
        'netG': {
            'net_task': constants.REGRESSION,  # task of net.
            'net_type': net_type,
            'init_pretrained_path': '',  # path to pretrained weights G.
            'checkpoint_path_netG': '',  # weights path of a checkpoint G.
            'checkpoint_path_optimizerG': '',  # optimizer path of a
            # checkpoint G.
            'checkpoint_path_netE': ''  # todo
        },
        'train': {
            "E_decay": 0.0,  # use exponential moving average of the model.
            # set to 0 to disable it. .999

            "G_optimizer_type": constants.ADAM,  # adam, sgd.
            "G_optimizer_lr": 2e-4,  # learning rate.
            "G_optimizer_wd": 1e-4,  # weight decay.
            "G_optimizer_clipgrad": 0.0,  # clip-grad norm.
            "G_optimizer_reuse": True,  # use checkpoint of optimizer.
            "G_optimizer_momentum": 0.9,  # Momentum.
            "G_optimizer_nesterov": True,  # If True, Nesterov algorithm is
            # used.
            # ==================== ADAM =========================
            "G_optimizer_beta1": 0.9,  # beta1.
            "G_optimizer_beta2": 0.999,  # beta2
            "G_optimizer_eps_adam": 1e-08,  # eps. for numerical stability.
            "G_optimizer_amsgrad": False,  # Use amsgrad variant or not.

            "G_scheduler_type": constants.MULTISTEPLR,
            "G_scheduler_milestones": [500000000, 900000000],  # for only
            # constants.MULTISTEPLR.
            "G_scheduler_step_size": 3,  # int. for only constants.MYSTEPLR.
            # int for epochs.
            "G_scheduler_gamma": 0.5,  # gamma. float > 0.
            "G_scheduler_min_lr": 1e-4,  # min lr allowed. cant go below it.
            # applied only for constants.MYSTEPLR. float.

            "G_regularizer_orthstep": 0.0,
            "G_regularizer_clipstep": 0.0,

            "G_param_strict": True,
            "E_param_strict": True,

            "checkpoint_eval": 5000,  # frequency of validation [iterations].
            # int. > 0. or float ]0, 1]. if float, it is a percentage from
            # the total number of minibatch in trainset. e.g. 0.5 means
            # perform a validation once every half of the train minibatch is
            # processed.
            "checkpoint_save": 5000,  # frequency of checkpointing [
            # iterations]. int >0, or float ]0, 1]. if float, it is a
            # percentage from the total number of trainset minibatches.
            "test_epoch_freq": 50,  # frequency of test [epochs]
            "plot_epoch_freq": 5,  # frequency of plotting train stats [epochs]
            "synch_scratch_epoch_freq": 50,  # frequency of synchronizing
            # scratch folder. applied only for CC server. [epochs]

        },
        # ======================================================================
        #                          EVALUATION
        # ======================================================================
        "eval_over_roi_also": False,  # if true, we perform evaluation over
        # ROIs only. an ROI in an image is obtained via thresholding.
        "eval_over_roi_also_ths": constants.ROI_THRESH,  # list of the
        # thresholds. Each one is used to estimate ROI in an image,
        # then compute average performance over set of images. The final
        # metric is the average over all per-threshold performance (
        # marginalize thresholds).
        "eval_over_roi_also_model_select": False,  # if true, model selection
        # is performed based on metric 'eval_over_roi_also' over ROI and not
        # over full image. if se, we take full image.
        # ======================================================================
        #                 RANDOM ADDITIONAL DATA LOCAL AUGMENTATION
        # ======================================================================
        "da_blur": False,  # apply local blur to a random block.
        "da_blur_prob": 0.5,  # prob. to use this DA.
        "da_blur_area": 0.3,  # percentage of the image to apply this DA. (
        # area of the random block)
        "da_blur_sigma": 1.,  # sigma of the Gaussian kernel.

        "da_dot_bin_noise": False,  # Multiply a random block a binary random
        # noise sampled from Bernoulli dist.
        "da_dot_bin_noise_prob": 0.5,  # prob. to use this DA.
        "da_dot_bin_noise_area": 0.3,  # percentage of the image to apply this
        # DA. (area of the random block)
        "da_dot_bin_noise_p": 0.5,  # (1 - p) is the parameter of the
        # Bernoulli dist. p: prob. of a pixel to set to 0.

        "da_add_gaus_noise": False,  # Add random Gaussian noise to a random
        # block. Gaus(0, std).
        "da_add_gaus_noise_prob": 0.5,  # prob. to use this DA.
        "da_add_gaus_noise_area": 0.3,  # percentage of the image to apply this
        # DA. (area of the random block)
        "da_add_gaus_noise_std": 0.03,  # standard deviation of the Gaussin.
        # ======================================================================
        #                  WEIGHTS SPARSITY (l1)
        # ======================================================================
        # weight sparsity loss
        "w_sparsity": False,  # Weight sparsity (l1 norm).
        "w_sparsity_lambda": 1.,  # lambda.
        # ======================================================================
        #                          ELB
        # ======================================================================
        "elb_init_t": 1.,  # used for ELB.
        "elb_max_t": 10.,  # used for ELB.
        "elb_mulcoef": 1.01,  # used for ELB.
        # ======================================================================
        #                            CONSTRAINTS:
        #                     'SuperResolution', sr
        #                     'ConRanFieldFcams', crf_fc
        #                     'EntropyFcams', entropy_fc
        #                     'PartUncerknowEntropyLowCams', partuncertentro_lc
        #                     'PartCertKnowLowCams', partcert_lc
        #                     'MinSizeNegativeLowCams', min_sizeneg_lc
        #                     'MaxSizePositiveLowCams', max_sizepos_lc
        #                     'MaxSizePositiveFcams' max_sizepos_fc
        # ======================================================================
        "max_epochs": 1000000,  # number of training epochs.
        # per-pixel importance
        "ppiw": False,  # use/not per-pixel importance weight.
        "ppiw_min_per_col_w": 0.001,  # minimal weight per color. used to
        # re-normalize per-color weight for loss. max value is 1. has to be
        # in ]0, 1[.
        # data sampler.
        # noise augmentation. for CSR-CNN -------
        "augment": False,  # whether to augment input sample with noise.
        # applied only for CSR-CNN.
        "augment_nbr_steps": 2,  # augmentation: upscale low
        # resolution n steps. in each step, add noise.
        "augment_use_roi": False,  # add noise only to roi.
        # -----------
        "sample_tr_patch": constants.SAMPLE_UNIF,  # how to sample train
        # patches. see constants.SAMPLE_PATCHES.
        "sample_tr_patch_th_style": constants.TH_AUTO,  # if sample_tr_patch
        # requires ROI, how to estimate the threshold: auto, or fixed.
        "sample_tr_patch_th": constants.TH_AUTO,  # if
        # sample_tr_patch_th_style is fixed, what is its float value [0, 255].
        # -----------------------  Losses.
        "l1": False,  # l1
        "l1_use_residuals": False,  # it true, we use residuals instead of
        # the image IF the model supports residuals. if not, an error will be
        # thrown.
        "l1_lambda": 1.,  # lambda l1.

        "l2": False,  # l2
        "l2_use_residuals": False,  # it true, we use residuals instead of
        # the image IF the model supports residuals. if not, an error will be
        # thrown.
        "l2_lambda": 1.,  # lambda l2.

        "l2sum": False,  # l2 sum
        "l2sum_use_residuals": False,  # it true, we use residuals instead of
        # the image IF the model supports residuals. if not, an error will be
        # thrown.
        "l2sum_lambda": 1.,  # lambda l2 sum.

        "ssim": False,  # ssim
        "ssim_lambda": 1.,  # lambda ssim.
        "ssim_window_s": 11,  # window size.

        "charbonnier": False,  # charbonnier loss.
        "charbonnier_use_residuals": False,  # if true, we use residuals
        # instead of the image IF the model supports residuals. if not,
        # an error will be thrown.
        "charbonnier_lambda": 1.,  # lambda charbonnier loss.
        "charbonnier_eps": 1e-9,  # eps for chabonnier.

        "boundpred": False,
        "boundpred_use_residuals": False,  # it true, we use residuals
        # instead of the image IF the model supports residuals. if not,
        # an error will be thrown.
        "boundpred_lambda": 1.,
        "boundpred_eps": 1.,  # warning: this needs to be adjusted depending
        # whether "boundpred_restore_range" is on or off.
        "boundpred_restore_range": True,  # restore range (y, y_hat) into [0,
        # 255(==max_color)].

        "local_moments": False,  # match local stats. (kl.)
        "local_moments_use_residuals": False,  # it true, we use residuals
        # instead of the image IF the model supports residuals. if not,
        # an error will be thrown.
        "local_moments_lambda": 1.,
        "local_moments_ksz": '3',  # kernel size for locality. > 1 and odd. to
        # perform mutil-scale moments, using multiple kernels separated by
        # '_'. eg.g.: '3_5_7'.
        "img_grad": False,  # use 1st order image gradient.
        "img_grad_use_residuals": False,  # apply over residuals or image.
        "img_grad_lambda": 1.,  # lambda loss.
        "img_grad_norm": constants.NORM2,  # norm.

        "norm_img_grad": False,  # use norm 1st order image gradient.
        "norm_img_grad_use_residuals": False,  # apply over residuals or image.
        "norm_img_grad_lambda": 1.,  # lambda loss.
        "norm_img_grad_type": constants.NORM2,  # type norm: l1, l2.

        "laplace": False,  # use second order image gradient (Laplacian filter).
        "laplace_use_residuals": False,  # apply over residuals or image.
        "laplace_lambda": 1.,  # lambda loss.
        "laplace_norm": constants.NORM2,  # norm.

        "norm_laplace": False,  # use norm second order image gradient (
        # Laplacian filter).
        "norm_laplace_use_residuals": False,  # apply over residuals or image.
        "norm_laplace_lambda": 1.,  # lambda loss.
        "norm_laplace_type": constants.NORM2,  # type norm: l1, l2.

        "loc_var": False,  # measure local variation loss.
        "loc_var_ksz": 3,  # int. kernel size.
        "loc_var_use_residuals": False,  # residuals or image?
        "loc_var_lambda": 1.,  # lambda loss.
        "loc_var_norm": constants.NORM2,  # norm.

        "norm_loc_var": False,  # norm measure local variation loss.
        "norm_loc_var_ksz": 3,  # int. kernel size.
        "norm_loc_var_use_residuals": False,  # residuals or image?
        "norm_loc_var_lambda": 1.,  # lambda loss.
        "norm_loc_var_type": constants.NORM2,  # type norm: l1, l2.

        "hist": False,  # histogram matching loss.
        "hist_lambda": 1.,  # lambda loss.
        "hist_sigma": 1e5,  # sigma for soft histogram.
        "hist_metric": constants.NORM2,  # type norm: l1, l2, kl, bh. BH
        # needs ELB.

        "kde": False,  # kde matching loss.
        "kde_lambda": 1.,  # lambda loss.
        "kde_nbins": 256,  # number of bins loss.
        "kde_kde_bw": 1. / (255.**2),  # kde bandwidth (sigma gaussian).
        # variance.
        "kde_metric": constants.NORM2,  # type norm: l1, l2, bh. BH
        # needs ELB.

        "ce": False,  # cross-entropy. only for net task: segmentation.
        "ce_lambda": 1.,  # lambda loss.

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ GENERIC
        'amp': False,  # if true, use automatic mixed-precision for training
        'amp_eval': False,  # if true, amp is used for inference.
        # ======================================================================
        #                             DDP:
        # NOT CC(): means single machine.  CC(): multiple nodes.
        # ======================================================================
        'local_rank': 0,  # int. for not CC(). auto-set.
        'local_world_size': 1,  # int. for not CC(). number of gpus to use.
        'rank': 0,  # int. global rank. useful for CC(). 0 otherwise. will be
        # set automatically.
        'init_method': '',  # str. CC(). init method. needs to be defined.
        # will be be determined automatically.
        'dist_backend': constants.GLOO,  # str. CC() or not CC(). distributed
        # backend.
        'world_size': 1,  # init. CC(). total number of gpus. will be
        # determined automatically.
        'is_master': False,  # will be set automatically if this process is
        # the master.
        'is_node_master': False,  # will be set auto. true if this process is
        # has local rank = 0.
        'c_cudaid': 0,  # int. current cuda id. auto-set.
        'distributed': False,  # bool.
    }

    assert args['task'] in constants.TASKS
    args['netG']: dict = init_net_g(args['netG'], args)

    return args


def find_last_checkpoint(save_dir: str, net_type: str = 'G',
                         pretrained_path: str = ''):
    """
    Args:
        save_dir: model folder
        net_type: 'G' or 'D' or 'optimizerG' or 'optimizerD'
        pretrained_path: pretrained model path. If save_dir does not have any
        model, load from pretrained_path

    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(save_dir, f'*_{net_type}.pth'))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, f'{init_iter}_{net_type}.pth')

    else:
        init_iter = 0
        init_path = pretrained_path

    return init_iter, init_path


def delete_previous_checkpoints_except_last(save_dir: str, net_type: str = 'G'):

    file_list = glob.glob(os.path.join(save_dir, f'*_{net_type}.pth'))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))

        laster_itera = max(iter_exist)

        for itera in iter_exist:
            if itera != laster_itera:
                _path = os.path.join(save_dir, f'{itera}_{net_type}.pth')
                os.remove(_path)
                DLLogger.log(f'deleted checkpoint @{net_type}: {_path}')
    else:
        DLLogger.log(f'no checkpoint @{net_type} to delete.')


def clean_previous_checkpoints_except_last(save_dir: str, net_types: List[str]):
    for net_type in net_types:
        delete_previous_checkpoints_except_last(save_dir, net_type)


def save_config(args: Union[object, dict], save_dir: str):
    _args = args
    if not isinstance(args, dict):  # todo: weak test.
        _args = vars(args)

    with open(join(save_dir, 'config_final.yml'), 'w') as fout:
        yaml.dump(_args, fout)


if __name__ == '__main__':
    pass
