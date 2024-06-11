import os
import sys
from os.path import dirname, abspath
from copy import deepcopy

import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.optim import SGD

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger
from dlib import loss
from dlib.losses.elb import ELB
import dlib.learning.lr_scheduler as my_lr_scheduler
from dlib.utils import constants

__all__ = ['define_loss', 'define_optimizer', 'define_scheduler']


def define_loss(args):
    masterloss = loss.MasterLoss(cuda_id=args.c_cudaid)

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
                args.ce,
                args.w_sparsity])

    elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
              mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

    if args.l1:
        masterloss.add(
            loss.L1(cuda_id=args.c_cudaid,
                    lambda_=args.l1_lambda,
                    use_residuals=args.l1_use_residuals
                    )
        )

    if args.l2:
        masterloss.add(
            loss.L2(cuda_id=args.c_cudaid,
                    lambda_=args.l2_lambda,
                    use_residuals=args.l2_use_residuals
                    )
        )

    if args.l2sum:
        masterloss.add(
            loss.L2Sum(cuda_id=args.c_cudaid,
                       lambda_=args.l2sum_lambda,
                       use_residuals=args.l2sum_use_residuals
                    )
        )

    if args.ssim:
        lssim = loss.NegativeSsim(cuda_id=args.c_cudaid,
                                  lambda_=args.ssim_lambda)
        lssim.set_window_size(window_size=args.ssim_window_s)
        masterloss.add(lssim)

    if args.local_moments:
        llocal_mmts = loss.LocalMoments(
            cuda_id=args.c_cudaid,
            lambda_=args.local_moments_lambda,
            use_residuals=args.local_moments_use_residuals)
        ksz = args.local_moments_ksz.split('_')
        ksz = [int(k) for k in ksz]

        llocal_mmts.set_ksz(ksz=ksz)
        masterloss.add(llocal_mmts)

    if args.charbonnier:
        deepcopy(elb) if args.sr_elb else nn.Identity()
        lcharb = loss.Charbonnier(cuda_id=args.c_cudaid,
                                  lambda_=args.charbonnier_lambda,
                                  use_residuals=args.charbonnier_use_residuals
                                  )
        lcharb.set_eps(eps=args.charbonnier_eps)
        masterloss.add(lcharb)

    if args.boundpred:
        l_boundpred = loss.BoundedPrediction(
            cuda_id=args.c_cudaid,
            lambda_=args.boundpred_lambda,
            elb=deepcopy(elb),
            restore_range=args.boundpred_restore_range,
            color_max=args.color_max,
            use_residuals=args.boundpred_use_residuals
        )
        l_boundpred.set_eps(args.boundpred_eps)
        masterloss.add(l_boundpred)


    if args.img_grad:
        l_img_grad = loss.ImageGradientLoss(
            cuda_id=args.c_cudaid,
            lambda_=args.img_grad_lambda,
            use_residuals=args.img_grad_use_residuals
        )
        l_img_grad.set_it(norm_str=args.img_grad_norm)
        masterloss.add(l_img_grad)

    if args.norm_img_grad:
        l_norm_img_grad = loss.NormImageGradientLoss(
            cuda_id=args.c_cudaid,
            lambda_=args.norm_img_grad_lambda,
            use_residuals=args.norm_img_grad_use_residuals
        )
        l_norm_img_grad.set_it(norm_str=args.norm_img_grad_type)
        masterloss.add(l_norm_img_grad)


    if args.laplace:
        l_laplace = loss.LaplacianFilterLoss(
            cuda_id=args.c_cudaid,
            lambda_=args.laplace_lambda,
            use_residuals=args.laplace_use_residuals
        )
        l_laplace.set_it(norm_str=args.laplace_norm)
        masterloss.add(l_laplace)

    if args.norm_laplace:
        l_norm_laplace = loss.NormLaplacianFilterLoss(
            cuda_id=args.c_cudaid,
            lambda_=args.norm_laplace_lambda,
            use_residuals=args.norm_laplace_use_residuals
        )
        l_norm_laplace.set_it(norm_str=args.norm_laplace_type)
        masterloss.add(l_norm_laplace)


    if args.loc_var:
        l_loc_var = loss.LocalVariationLoss(
            cuda_id=args.c_cudaid,
            lambda_=args.loc_var_lambda,
            use_residuals=args.loc_var_use_residuals
        )
        l_loc_var.set_it(ksz=args.loc_var_ksz, norm_str=args.loc_var_norm)
        masterloss.add(l_loc_var)

    if args.norm_loc_var:
        l_norm_loc_var = loss.NormLocalVariationLoss(
            cuda_id=args.c_cudaid,
            lambda_=args.norm_loc_var_lambda,
            use_residuals=args.norm_loc_var_use_residuals
        )
        l_norm_loc_var.set_it(
            ksz=args.norm_loc_var_ksz, norm_str=args.norm_loc_var_type)
        masterloss.add(l_norm_loc_var)


    if args.hist:
        l_hist = loss.HistogramMatch(
            cuda_id=args.c_cudaid,
            lambda_=args.hist_lambda,
            elb=deepcopy(elb),
            color_min=args.color_min,
            color_max=args.color_max
        )
        l_hist.set_it(norm_str=args.hist_metric, sigma=args.hist_sigma)
        masterloss.add(l_hist)


    if args.kde:
        l_kde = loss.KDEMatch(
            cuda_id=args.c_cudaid,
            lambda_=args.kde_lambda,
            elb=deepcopy(elb),
            color_min=0,
            color_max=1
        )
        l_kde.set_it(norm_str=args.kde_metric, kde_bw=args.kde_kde_bw,
                     ndim=args.n_channels, nbins=args.kde_nbins)
        masterloss.add(l_kde)


    if args.ce:
        l_ce = loss.CrossEntropyL(
            cuda_id=args.c_cudaid,
            lambda_=args.ce_lambda,
            color_min=args.color_min,
            color_max=args.color_max
        )
        masterloss.add(l_ce)

    if args.w_sparsity:
        ws_loss = loss.WeightsSparsityLoss(
            cuda_id=args.c_cudaid,
            lambda_=args.w_sparsity_lambda
        )

        masterloss.add(ws_loss)


    assert len(masterloss.n_holder) > 1
    return masterloss



def define_optimizer(args, netG):
    opt_train = args.train
    G_optim_params = []
    for k, v in netG.named_parameters():
        if v.requires_grad:
            G_optim_params.append(v)
        else:
            DLLogger.log(f'Params [{k:s}] will not optimize.')

    optim_name = opt_train['G_optimizer_type']
    msg = f"{optim_name} {constants.OPTIMIZERS}"
    assert optim_name in constants.OPTIMIZERS, msg

    if optim_name == constants.ADAM:
        G_optimizer = Adam(G_optim_params,
                           lr=opt_train['G_optimizer_lr'],
                           betas=(opt_train['G_optimizer_beta1'],
                                  opt_train['G_optimizer_beta2']),
                           eps=opt_train['G_optimizer_eps_adam'],
                           amsgrad=opt_train['G_optimizer_amsgrad'],
                           weight_decay=opt_train['G_optimizer_wd'])
    elif optim_name == constants.SGD:
        G_optimizer = SGD(G_optim_params,
                          lr=opt_train['G_optimizer_lr'],
                          weight_decay=opt_train['G_optimizer_wd'],
                          momentum=opt_train['G_optimizer_momentum'],
                          nesterov=opt_train['G_optimizer_nesterov']
                          )
    else:
        raise NotImplementedError(f'optimizer: {optim_name}')

    return G_optimizer


def define_scheduler(args, G_optimizer) -> list:
    opt_train = args.train
    schedulers = []
    scheduler_name = opt_train['G_scheduler_type']
    assert scheduler_name in constants.STEPSLR, scheduler_name

    gamma = opt_train['G_scheduler_gamma']

    assert gamma >= 0, gamma
    assert isinstance(gamma, float), type(gamma)

    if scheduler_name == constants.MULTISTEPLR:
        milestones = opt_train['G_scheduler_milestones']
        schedulers.append(
            lr_scheduler.MultiStepLR(G_optimizer,
                                     milestones=milestones,
                                     gamma=gamma)
        )

    elif scheduler_name == constants.MYSTEPLR:
        step_size = opt_train['G_scheduler_step_size']
        min_lr = opt_train['G_scheduler_min_lr']

        assert step_size > 0, step_size
        assert isinstance(step_size, int), type(step_size)

        assert min_lr >= 0, min_lr
        assert isinstance(min_lr, float), min_lr

        schedulers.append(
            my_lr_scheduler.MyStepLR(optimizer=G_optimizer,
                                     step_size=step_size,
                                     gamma=gamma,
                                     last_epoch=-1,
                                     min_lr=min_lr
                                     )
        )
    else:
        raise NotImplementedError(f'LR scheduler: {scheduler_name}')

    return schedulers
