import warnings
import sys
import os
from os.path import dirname, abspath, join
from copy import deepcopy
import math

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.learning import lr_scheduler as my_lr_scheduler

from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import count_nb_params
from dlib.utils import constants
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag
from dlib.utils.shared import format_dict_2_str

import dlib
from dlib import create_model

from dlib.losses.elb import ELB
from dlib import losses


import dlib.dllogger as DLLogger


__all__ = [
    'get_loss',
    'get_pretrainde_classifier',
    'get_model',
    'get_optimizer'
]


def get_encoder_d_c(encoder_name):
    if encoder_name in [constants.VGG16]:
        vgg_encoders = dlib.encoders.vgg_encoders
        encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        decoder_channels = (256, 128, 64)
    else:
        encoder_depth = 5
        decoder_channels = (256, 128, 64, 32, 16)

    return encoder_depth, decoder_channels


def get_loss_std_cl(args):
    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    # image classification loss
    assert args.task == constants.STD_CL

    masterloss.add(losses.ClLoss(cuda_id=args.c_cudaid,
                                 support_background=support_background,
                                 multi_label_flag=multi_label_flag))

    return masterloss


def get_loss_fcam(args):
    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    assert args.task == constants.F_CL

    if not args.model['freeze_cl']:
        masterloss.add(losses.ClLoss(
            cuda_id=args.c_cudaid,
            support_background=support_background,
            multi_label_flag=multi_label_flag))

    elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
              mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

    if args.im_rec:
        masterloss.add(
            losses.ImgReconstruction(
                cuda_id=args.c_cudaid,
                lambda_=args.im_rec_lambda,
                elb=deepcopy(elb) if args.sr_elb else nn.Identity(),
                support_background=support_background,
                multi_label_flag=multi_label_flag)
        )

    if args.crf_fc:
        masterloss.add(losses.ConRanFieldFcams(
            cuda_id=args.c_cudaid,
            lambda_=args.crf_lambda,
            sigma_rgb=args.crf_sigma_rgb, sigma_xy=args.crf_sigma_xy,
            scale_factor=args.crf_scale,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.crf_start_ep, end_epoch=args.crf_end_ep,
        ))

    if args.entropy_fc:
        masterloss.add(losses.EntropyFcams(
            cuda_id=args.c_cudaid,
            lambda_=args.entropy_fc_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag))

    if args.max_sizepos_fc:
        masterloss.add(losses.MaxSizePositiveFcams(
            cuda_id=args.c_cudaid,
            lambda_=args.max_sizepos_fc_lambda,
            elb=deepcopy(elb),
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.max_sizepos_fc_start_ep,
            end_epoch=args.max_sizepos_fc_end_ep
        ))

    if args.sl_fc:
        sl_fcam = losses.SelfLearningFcams(
            cuda_id=args.c_cudaid,
            lambda_=args.sl_fc_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.sl_start_ep, end_epoch=args.sl_end_ep,
            seg_ignore_idx=args.seg_ignore_idx
        )

        masterloss.add(sl_fcam)

    assert len(masterloss.n_holder) > 1

    return masterloss


def get_loss_tcam(args):
    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    assert args.task == constants.TCAM

    if not args.model['freeze_cl']:
        masterloss.add(losses.ClLoss(
            cuda_id=args.c_cudaid,
            support_background=support_background,
            multi_label_flag=multi_label_flag))

    elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
              mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

    if args.crf_tc:
        masterloss.add(losses.ConRanFieldTcams(
            cuda_id=args.c_cudaid,
            lambda_=args.crf_tc_lambda,
            sigma_rgb=args.crf_tc_sigma_rgb,
            sigma_xy=args.crf_tc_sigma_xy,
            scale_factor=args.crf_tc_scale,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.crf_tc_start_ep,
            end_epoch=args.crf_tc_end_ep,
        ))

    if args.max_sizepos_tc:
        masterloss.add(losses.MaxSizePositiveTcams(
            cuda_id=args.c_cudaid,
            lambda_=args.max_sizepos_tc_lambda,
            elb=deepcopy(elb),
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.max_sizepos_tc_start_ep,
            end_epoch=args.max_sizepos_tc_end_ep
        ))

    if args.sl_tc:
        sl_tcam = losses.SelfLearningTcams(
            cuda_id=args.c_cudaid,
            lambda_=args.sl_tc_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.sl_tc_start_ep,
            end_epoch=args.sl_tc_end_ep,
            seg_ignore_idx=args.seg_ignore_idx
        )

        masterloss.add(sl_tcam)

    assert len(masterloss.n_holder) > 1

    return masterloss


def get_loss_cbox(args):
    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    assert args.task == constants.C_BOX

    assert any([args.cb_area_box,
                args.cb_cl_score,
                args.cb_pp_box,
                args.cb_seed])

    elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
              mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

    if args.cb_area_box:
        masterloss.add(losses.AreaBox(
            cuda_id=args.c_cudaid,
            lambda_=args.cb_area_box_l,
            start_epoch=args.cb_area_box_start_epoch,
            end_epoch=args.cb_area_box_end_epoch,
            elb=deepcopy(elb),
            multi_label_flag=multi_label_flag,
            cb_area_normed=args.cb_area_normed
        ))

    if args.cb_cl_score:
        masterloss.add(losses.ClScoring(
            cuda_id=args.c_cudaid,
            lambda_=args.cb_cl_score_l,
            start_epoch=args.cb_cl_score_start_epoch,
            end_epoch=args.cb_cl_score_end_epoch,
            elb=deepcopy(elb),
            multi_label_flag=multi_label_flag,
        ))

    if args.cb_pp_box:
        masterloss.add(losses.BoxBounds(
            cuda_id=args.c_cudaid,
            lambda_=args.cb_pp_box_l,
            start_epoch=args.cb_pp_box_start_epoch,
            end_epoch=args.cb_pp_box_end_epoch,
            elb=deepcopy(elb),
            multi_label_flag=multi_label_flag
        ))

    if args.cb_seed:
        masterloss.add(losses.SeedCbox(
            cuda_id=args.c_cudaid,
            lambda_=args.cb_seed_l,
            start_epoch=args.cb_seed_start_epoch,
            end_epoch=args.cb_seed_end_epoch,
            elb=deepcopy(elb),
            multi_label_flag=multi_label_flag,
            seg_ignore_idx=args.seg_ignore_idx
        ))

    assert len(masterloss.n_holder) > 1
    return masterloss


def get_loss(args):
    masterloss = None
    # image classification loss
    if args.task == constants.STD_CL:
        masterloss = get_loss_std_cl(args)
    # fcams
    elif args.task == constants.F_CL:
        masterloss = get_loss_fcam(args)
    # tcam
    elif args.task == constants.TCAM:
        masterloss = get_loss_tcam(args)
    # cbox
    elif args.task == constants.C_BOX:
        masterloss = get_loss_cbox(args)
    else:
        raise NotImplementedError

    masterloss.check_losses_status()
    masterloss.cuda(args.c_cudaid)

    DLLogger.log(message="Train loss: {}".format(masterloss))
    return masterloss


def get_aux_params(args):
    """
    Prepare the head params.
    :param args:
    :return:
    """
    assert args.spatial_pooling in constants.SPATIAL_POOLINGS
    return {
        "pooling_head": args.spatial_pooling,
        "classes": args.num_classes,
        "modalities": args.wc_modalities,
        "kmax": args.wc_kmax,
        "kmin": args.wc_kmin,
        "alpha": args.wc_alpha,
        "dropout": args.wc_dropout,
        "support_background": args.model['support_background'],
        "r": args.lse_r
    }


def get_pretrainde_classifier(args, pretrained_ch_pt: str):
    assert pretrained_ch_pt is not None, pretrained_ch_pt
    assert pretrained_ch_pt in [constants.BEST_LOC, constants.BEST_CL]

    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)

    aux_params = get_aux_params(args)
    model = create_model(
        task=constants.STD_CL,
        arch=constants.STDCLASSIFIER,
        encoder_name=p.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=p.in_channels,
        encoder_depth=encoder_depth,
        scale_in=p.scale_in,
        aux_params=aux_params
    )

    DLLogger.log("PRETRAINED CLASSIFIER `{}` was created. "
                 "Nbr.params: {}".format(model, count_nb_params(model)))
    auxp_str = format_dict_2_str(aux_params) if aux_params is not None else None
    log = f"Arch: {p.arch}\n" \
          f"encoder_name: {p.encoder_name}\n" \
          f"encoder_weights: {encoder_weights}\n" \
          f"classes: {classes}\n" \
          f"aux_params: \n{auxp_str}\n" \
          f"scale_in: {p.scale_in}\n" \
          f"freeze_cl: {p.freeze_cl}\n" \
          f"img_range: {args.img_range} \n"
    DLLogger.log(log)

    if pretrained_ch_pt == constants.BEST_CL:
        path_cl = args.model['folder_pre_trained_cl']

    elif pretrained_ch_pt == constants.BEST_LOC:
        path_cl = args.model['folder_pre_trained_seeder']
    else:
        raise NotImplementedError(pretrained_ch_pt)

    assert path_cl not in [None, 'None', '']

    msg = "You have asked to set the classifier " \
          " from {} .... [OK]".format(path_cl)
    warnings.warn(msg)
    DLLogger.log(msg)

    if args.task == constants.TCAM:
        assert pretrained_ch_pt is not None
        assert pretrained_ch_pt == constants.BEST_LOC
        assert pretrained_ch_pt == args.tcam_pretrained_seeder_ch_pt

    if args.task == constants.C_BOX:
        assert pretrained_ch_pt == args.cb_pretrained_cl_ch_pt

    if args.task == constants.F_CL:
        assert pretrained_ch_pt == constants.BEST_LOC

    encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                           map_location=get_cpu_device())
    model.encoder.super_load_state_dict(encoder_w, strict=True)

    header_w = torch.load(join(path_cl, 'classification_head.pt'),
                          map_location=get_cpu_device())
    model.classification_head.load_state_dict(header_w, strict=True)

    model.eval()
    model.freeze_classifier()
    model.assert_cl_is_frozen()

    return model


def get_model(args, eval=False):

    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)
    h = None

    if args.task == constants.STD_CL:
        aux_params = get_aux_params(args)
        model = create_model(
            task=args.task,
            arch=p.arch,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=p.in_channels,
            encoder_depth=encoder_depth,
            scale_in=p.scale_in,
            aux_params=aux_params
        )

    elif args.task == constants.F_CL:
        aux_params = get_aux_params(args)

        assert args.seg_mode == constants.BINARY_MODE
        seg_h_out_channels = 2

        model = create_model(
            task=args.task,
            arch=p.arch,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            seg_h_out_channels=seg_h_out_channels,
            scale_in=p.scale_in,
            aux_params=aux_params,
            freeze_cl=p.freeze_cl,
            im_rec=args.im_rec,
            img_range=args.img_range
        )

    elif args.task == constants.TCAM:
        aux_params = get_aux_params(args)

        assert args.seg_mode == constants.BINARY_MODE
        seg_h_out_channels = 2

        model = create_model(
            task=args.task,
            arch=p.arch,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            seg_h_out_channels=seg_h_out_channels,
            scale_in=p.scale_in,
            aux_params=aux_params,
            freeze_cl=p.freeze_cl,
            im_rec=args.im_rec,
            img_range=args.img_range
        )

    elif args.task == constants.C_BOX:
        aux_params = None
        seg_h_out_channels = 2
        h = math.ceil(args.crop_size * p.scale_domain)
        w = h

        model = create_model(
            task=args.task,
            arch=p.arch,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            in_channels=p.in_channels,
            h=h,
            w=w,
            scale_in=p.scale_in,
            freeze_enc=p.freeze_encoder
        )
    else:
        raise NotImplementedError

    DLLogger.log("`{}` was created. Nbr.params: {}".format(
        model,  count_nb_params(model)))
    auxp_str = format_dict_2_str(aux_params) if aux_params is not None else None
    log = f"Arch: {p.arch}\n" \
          f"task: {args.task}\n" \
          f"encoder_name: {p.encoder_name}\n" \
          f"encoder_weights: {encoder_weights}\n" \
          f"classes: {classes}\n" \
          f"aux_params: \n{auxp_str}\n" \
          f"scale_in: {p.scale_in}\n" \
          f"freeze_cl: {p.freeze_cl}\n" \
          f"freeze_encoder: {p.freeze_encoder}\n" \
          f"nbr_cl_h_w: {h}\n" \
          f"im_rec: {args.im_rec}\n" \
          f"img_range: {args.img_range} \n"
    DLLogger.log(log)
    DLLogger.log(model.get_info_nbr_params())

    path_file = args.model['path_pre_trained']
    if path_file not in [None, 'None']:
        msg = f"You have asked to load a specific pre-trained " \
              f"model from {path_file} .... [OK]"
        warnings.warn(msg)
        DLLogger.log(msg)
        pre_tr_state = torch.load(path_file, map_location=get_cpu_device())
        model.load_state_dict(pre_tr_state, strict=args.model['strict'])

    path_cl = args.model['folder_pre_trained_cl']
    if path_cl not in [None, 'None', '']:
        assert args.task in [constants.F_CL, constants.C_BOX, constants.TCAM]

        msg = f"You have asked to set the classifier " \
              f"from {path_cl} .... [OK]"
        warnings.warn(msg)
        DLLogger.log(msg)

        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                               map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        if args.task == constants.F_CL:
            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                  map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)

    if args.model['freeze_cl']:
        assert args.task in [constants.F_CL, constants.TCAM]
        assert args.model['folder_pre_trained_cl'] not in [None, 'None', '']

        model.freeze_classifier()
        model.assert_cl_is_frozen()

    if args.model['freeze_encoder']:
        assert args.task == constants.C_BOX
        model.freeze_encoder()
        model.assert_encoder_is_frozen()

    if eval:
        assert os.path.isdir(args.outd)
        tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
        path = join(args.outd, tag)
        cpu_device = get_cpu_device()

        if args.task == constants.STD_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'classification_head.pt'),
                                 map_location=cpu_device)
            model.classification_head.load_state_dict(weights, strict=True)

        elif args.task in [constants.F_CL, constants.TCAM]:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)
            if model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=cpu_device)
                model.reconstruction_head.load_state_dict(weights, strict=True)

        elif args.task == constants.C_BOX:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'box_head.pt'),
                                 map_location=cpu_device)
            model.box_head.load_state_dict(weights, strict=True)
        else:
            raise NotImplementedError

        msg = "EVAL-mode. Reset model weights to: {}".format(path)
        warnings.warn(msg)
        DLLogger.log(msg)

    return model


def standardize_optimizers_params(optm_dict):
    """
    Standardize the keys of a dict for the optimizer.
    all the keys starts with 'optn[?]__key' where we keep only the key and
    delete the initial.
    the dict should not have a key that has a dict as value. we do not deal
    with this case. an error will be raise.

    :param optm_dict: dict with specific keys.
    :return: a copy of optm_dict with standardized keys.
    """
    msg = "'optm_dict' must be of type dict. found {}.".format(type(optm_dict))
    assert isinstance(optm_dict, dict), msg
    new_optm_dict = deepcopy(optm_dict)
    loldkeys = list(new_optm_dict.keys())

    for k in loldkeys:
        if k.startswith('opt'):
            msg = "'{}' is a dict. it must not be the case." \
                  "otherwise, we have to do a recursive thing....".format(k)
            assert not isinstance(new_optm_dict[k], dict), msg

            new_k = k.split('__')[1]
            new_optm_dict[new_k] = new_optm_dict.pop(k)

    return new_optm_dict


def _get_model_params_for_opt(args, model):
    hparams = deepcopy(args.optimizer)
    hparams = standardize_optimizers_params(hparams)
    hparams = Dict2Obj(hparams)

    if args.task in [constants.F_CL, constants.C_BOX, constants.TCAM]:
        return [
            {'params': model.parameters(), 'lr': hparams.lr}
        ]

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    architecture = args.model['encoder_name']
    assert architecture in constants.BACKBONES

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['encoder.features.'],  # features
        'resnet': ['encoder.layer4.', 'classification_head.'],  # CLASSIFIER
        'inception': ['encoder.Mixed', 'encoder.Conv2d_1', 'encoder.Conv2d_2',
                      'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
    }

    param_features = []
    param_classifiers = []

    def param_features_substring_list(arch):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if arch.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}"
                       .format(arch))

    for name, parameter in model.named_parameters():

        if string_contains_any(
                name,
                param_features_substring_list(architecture)):
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_features.append(parameter)
            elif architecture == constants.RESNET50:
                param_classifiers.append(parameter)
        else:
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_classifiers.append(parameter)
            elif architecture == constants.RESNET50:
                param_features.append(parameter)

    return [
            {'params': param_features, 'lr': hparams.lr},
            {'params': param_classifiers,
             'lr': hparams.lr * hparams.lr_classifier_ratio}
    ]


def get_optimizer(args, model):
    """Instantiate an optimizer.
    Input:
        args: object. Contains the configuration of the exp that has been
        read from the yaml file.
        mode: a pytorch model with parameters.

    Output:
        optimizer: a pytorch optimizer.
        lrate_scheduler: a pytorch learning rate scheduler (or None).
    """
    hparams = deepcopy(args.optimizer)
    hparams = standardize_optimizers_params(hparams)
    hparams = Dict2Obj(hparams)

    op_col = {}

    params = _get_model_params_for_opt(args, model)
    opname = hparams.name_optimizer
    msg = f"{opname} {constants.OPTIMIZERS}"
    assert hparams.name_optimizer in constants.OPTIMIZERS, msg

    if hparams.name_optimizer == constants.SGD:
        optimizer = SGD(params=params,
                        momentum=hparams.momentum,
                        dampening=hparams.dampening,
                        weight_decay=hparams.weight_decay,
                        nesterov=hparams.nesterov)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['momentum'] = hparams.momentum
        op_col['dampening'] = hparams.dampening
        op_col['weight_decay'] = hparams.weight_decay
        op_col['nesterov'] = hparams.nesterov

    elif hparams.name_optimizer == constants.ADAM:
        optimizer = Adam(params=params,
                         betas=(hparams.beta1, hparams.beta2),
                         eps=hparams.eps_adam,
                         weight_decay=hparams.weight_decay,
                         amsgrad=hparams.amsgrad)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['beta1'] = hparams.beta1
        op_col['beta2'] = hparams.beta2
        op_col['weight_decay'] = hparams.weight_decay
        op_col['amsgrad'] = hparams.amsgrad
    else:
        raise ValueError("Unsupported optimizer `{}` .... "
                         "[NOT OK]".format(args.optimizer["name"]))

    if hparams.lr_scheduler:
        if hparams.name_lr_scheduler == "step":
            lrate_scheduler = lr_scheduler.StepLR(optimizer,
                                                  step_size=hparams.step_size,
                                                  gamma=hparams.gamma,
                                                  last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "cosine":
            lrate_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.t_max,
                eta_min=hparams.min_lr,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['T_max'] = hparams.T_max
            op_col['eta_min'] = hparams.eta_min
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "mystep":
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer,
                step_size=hparams.step_size,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch,
                min_lr=hparams.min_lr)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "mycosine":
            lrate_scheduler = my_lr_scheduler.MyCosineLR(
                optimizer,
                coef=hparams.coef,
                max_epochs=hparams.max_epochs,
                min_lr=hparams.min_lr,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['coef'] = hparams.coef
            op_col['max_epochs'] = hparams.max_epochs
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "multistep":
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=hparams.milestones,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['milestones'] = hparams.milestones
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        else:
            raise ValueError("Unsupported learning rate scheduler `{}` .... "
                             "[NOT OK]".format(
                                hparams.name_lr_scheduler))
    else:
        lrate_scheduler = None

    DLLogger.log("Optimizer:\n{}".format(format_dict_2_str(op_col)))

    return optimizer, lrate_scheduler
