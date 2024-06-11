import os
import sys
from os.path import join, dirname, abspath
import re
import glob
from typing import Union, Tuple
import math
from copy import deepcopy
import datetime as dt
import time

import yaml
from tqdm import tqdm

import torch.distributed as dist
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils.tools import chunk_it
from dlib.utils.tools import check_model_output_corruption
from dlib.utils.utils_init_default_args import init_net_g
from dlib.utils.shared import fmsg
import dlib.dllogger as DLLogger

from dlib.utils.utils_tracker import update_tracker_train
from dlib.utils.utils_tracker import save_tracker
from dlib.utils.utils_tracker import update_tracker_eval
from dlib.utils.utils_tracker import reset_tracker_eval
from dlib.utils.utils_tracker import is_last_perf_best_perf
from dlib.utils.utils_tracker import write_current_perf_eval
from dlib.utils.utils_tracker import current_perf_to_str
from dlib.utils.utils_tracker import plot_tracker_eval
from dlib.utils.utils_tracker import plot_tracker_train

from dlib.utils import utils_image
from dlib.utils.shared import is_cc
from dlib.utils.utils_exps import copy_exp_dir_node_to_scratch
from dlib.utils.utils_config import clean_previous_checkpoints_except_last
from dlib.utils.shared import reformat_id
from dlib.utils.utils_parallel import sync_tensor_across_gpus
from dlib.utils.utils_parallel import sync_non_tensor_value_across_gpus
from dlib.utils.utils_parallel import sync_dict_across_gpus


__all__ = ['train_valid', 'evaluate', 'ddp_barrier', 'evaluate_single_ds',
           'Interpolate']


def ddp_barrier(distributed: bool):
    if distributed:
        dist.barrier()


def _cautious_merge_dicts(dict1, dict2) -> dict:
    for k in dict2:
        assert k not in dict1, k
        dict1[k] = deepcopy(dict2[k])

    return dict1


def _detach(l: list):
    return [v.detach() for v in l]


def _update_loss_holder(l_holder, loss_fn) -> Tuple[list, list]:
    n_holder = deepcopy(loss_fn.n_holder)
    c_l_holder = _detach(loss_fn.l_holder)
    assert len(n_holder) == len(c_l_holder)

    if l_holder is None:
        l_holder = c_l_holder
    else:
        assert len(l_holder) == len(c_l_holder)
        l_holder = [l_holder[i] + c_l_holder[i] for i in range(len(l_holder))]

    return l_holder, n_holder


def _avg_list(l: list, n: int) -> list:
    return [v / float(n) for v in l]


class Interpolate(torch.nn.Module):
    def __init__(self, task: str, scale: int, scale_mode: str):
        super(Interpolate, self).__init__()

        self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.scale: int = scale

        assert task in constants.TASKS, f"{task} | {constants.TASKS}"
        self.task = task

        assert scale_mode in [constants.INTER_BICUBIC], scale_mode
        self.scale_mode: str = scale_mode

        self.L: torch.Tensor = None
        self.E: torch.Tensor = None
        self.H: torch.Tensor = None

    def feed_data(self, data, need_H=True):
        if self.task == constants.SUPER_RES:
            self.L = data['l_im'].to(self.device)
            if need_H:
                self.H = data['h_im'].to(self.device)

        elif self.task == constants.RECONSTRUCT:
            self.L = data['in_reconstruct'].to(self.device)
            if need_H:
                self.H = data['trg_reconstruct'].to(self.device)

        else:
            raise NotImplementedError(self.task)


    def forward(self):
        x = self.L
        assert x.ndim == 4, x.ndim

        if self.scale_mode == constants.INTER_BICUBIC:
            mode = 'bicubic'
        else:
            raise NotImplementedError(f'Not supported : {self.scale_mode}')

        if self.task == constants.SUPER_RES:
            scale = self.scale

        elif self.task == constants.RECONSTRUCT:
            scale = 1

        else:
            raise NotImplementedError(self.task)


        out = F.interpolate(input=x,
                            scale_factor=scale,
                            mode=mode,
                            antialias=True
                            )

        # todo: weak. assumes data in [0., 1.].
        out = torch.clamp(out, 0.0, 1.0)
        self.E = out

    def set_eval_mode(self):
        self.eval()

    def set_train_mode(self):
        pass

    def test(self):
        self.eval()
        with torch.no_grad():
            self.forward()

    def current_visuals(self, need_H=True):
        out_dict = dict()
        out_dict['L'] = self.L.detach().float()
        out_dict['E'] = self.E.detach().float()
        if need_H:
            out_dict['H'] = self.H.detach().float()

        return out_dict


def _validate(args: object,
              model,
              valid_loaders: dict,
              split: str,
              tracker: dict,
              roi_tracker: dict,
              current_step: int,
              current_epoch: int
              ) -> Tuple[dict, dict]:

    torch.cuda.empty_cache()

    multi_valid = args.multi_valid
    model_is_interp = isinstance(model, Interpolate)

    for ds_name in valid_loaders:
        _vl_loader = valid_loaders[ds_name]

        if model_is_interp:
            ds_name = f'{ds_name}_{args.basic_interpolation}'

        save_img_dir = join(args.outd, args.save_dir_imgs, split, ds_name)
        os.makedirs(save_img_dir, exist_ok=True)
        track_evolution_img = True

        out_ = fast_eval(model=model,
                         data_loader=_vl_loader,
                         ds_name=ds_name,
                         split=split,
                         tracker=tracker,
                         roi_tracker=roi_tracker,
                         args=args,
                         current_step=current_step,
                         epoch=current_epoch,
                         tqdm_pos=2,
                         nbr_to_plot=4,
                         save_img_dir=save_img_dir,
                         track_evolution_img=track_evolution_img
                         )
        tracker, details, roi_tracker, roi_details = out_

        is_last_best = is_last_perf_best_perf(tracker,
                                              roi_tracker,
                                              args.eval_over_roi_also,
                                              args.eval_over_roi_also_model_select,
                                              split=split,
                                              ds_name=ds_name,
                                              metric=args.model_select_mtr)

        if args.is_master:

            _dir = join(args.outd_backup, 'best-models')
            os.makedirs(_dir, exist_ok=True)

            # log best --
            if is_last_best:
                if not model_is_interp:
                    _f_name = f'{ds_name}.pth' if multi_valid else \
                        'model.pth'
                    model.save_best(_dir, p_name_file=_f_name)

                _f_yml = f'details_{ds_name}.yml'
                with open(join(_dir, _f_yml), 'w') as fd:
                    yaml.dump(details, fd)

                if args.eval_over_roi_also:
                    _f_yml = f'roi_details_{ds_name}.yml'
                    with open(join(_dir, _f_yml), 'w') as fd:
                        yaml.dump(roi_details, fd)

            status_perf = write_current_perf_eval(
                tracker,
                split=split,
                ds_name=ds_name,
                save_dir=_dir if is_last_best else None,
                name_f=f'{ds_name}.yaml' if is_last_best else None,
                current_step=current_step,
                current_epoch=current_epoch
            )
            roi_status_perf = None
            if args.eval_over_roi_also:
                name_f = f'roi-{ds_name}.yaml'
                roi_status_perf = write_current_perf_eval(
                    roi_tracker,
                    split=split,
                    ds_name=ds_name,
                    save_dir=_dir if is_last_best else None,
                    name_f=name_f if is_last_best else None,
                    current_step=current_step,
                    current_epoch=current_epoch
                    )

            msg = current_perf_to_str(
                status=status_perf,
                roi_status=roi_status_perf,
                master_mtr=args.model_select_mtr,
                model_select_roi=args.eval_over_roi_also_model_select
            )
            DLLogger.log(msg)

    torch.cuda.empty_cache()

    return tracker, roi_tracker


def train_valid(args: object,
                model,
                train_loader,
                train_sampler,
                valid_loaders: dict,
                test_loaders: dict,
                tracker: dict,
                roi_tracker: dict,
                current_step: int
                ):

    train_size = int(math.ceil(
        len(train_loader.dataset) / (args.batch_size * args.num_gpus)))
    current_epoch = math.floor(current_step / float(train_size))

    if current_step == 0:  # interpolation perf.
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        _model_interp = Interpolate(
            task=args.task, scale=args.scale,
            scale_mode=args.basic_interpolation).to(device)

        tracker, roi_tracker = _validate(args=args,
                                         model=_model_interp,
                                         valid_loaders=valid_loaders,
                                         split=constants.VALIDSET,
                                         tracker=tracker,
                                         roi_tracker=roi_tracker,
                                         current_step=0,
                                         current_epoch=0
                                         )

        # todo: validate on step=0.

    max_seed = (2 ** 32) - 1

    for epoch in tqdm(range(current_epoch, args.max_epochs, 1),
                      total=(args.max_epochs - current_epoch),
                      ncols=80,
                      desc="TR-EPOCH",
                      position=0,
                      leave=True
                      ):
        n_holder = None
        l_holder = None

        ddp_barrier(args.distributed)

        torch.cuda.empty_cache()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        t_init_epoch = dt.datetime.now()

        n_check_eval = args.train['checkpoint_eval']
        assert n_check_eval > 0, n_check_eval

        if n_check_eval < 1:
            assert isinstance(n_check_eval, float), type(n_check_eval)
            n_mbatchs = len(train_loader)
            n_check_eval = max(int(n_check_eval * n_mbatchs), 1)

        else:
            assert isinstance(n_check_eval, int), type(n_check_eval)

        n_checkpoint_save = args.train['checkpoint_save']
        assert n_checkpoint_save > 0, n_checkpoint_save

        if n_checkpoint_save < 1:
            assert isinstance(n_checkpoint_save, float), type(n_checkpoint_save)
            n_mbatchs = len(train_loader)
            n_checkpoint_save = max(int(n_checkpoint_save * n_mbatchs), 1)

        else:
            assert isinstance(n_checkpoint_save, int), type(n_checkpoint_save)


        for i, train_data in tqdm(enumerate(train_loader), ncols=80,
                                  total=len(train_loader), position=1,
                                  leave=False, desc='ITER-STEP'):

            current_step += 1

            c_seed = args.myseed + current_step
            c_seed = int(c_seed % max_seed)
            set_seed(seed=c_seed, verbose=False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            model.set_train_mode()

            ddp_barrier(args.distributed)

            model.feed_data(train_data)
            model.optimize_parameters(epoch, current_step)
            model.update_learning_rate()

            tracker = update_tracker_train(tracker,
                                           n_losses=model.loss_fn.n_holder,
                                           v_losses=model.loss_fn.l_holder,
                                           period=constants.PR_ITER
                                           )

            # todo: synch loss between gpus.
            l_holder, n_holder = _update_loss_holder(l_holder, model.loss_fn)


            cnd = (current_step % n_check_eval == 0)
            if args.distributed and (args.eval_bsize > 1):
                pass  # do multi-gpu eval.
            else:
                cnd &= args.is_master  # prevent multi-gpu eval.

            ddp_barrier(args.distributed)

            if cnd:
                tracker, roi_tracker = _validate(args=args,
                                                 model=model,
                                                 valid_loaders=valid_loaders,
                                                 split=constants.VALIDSET,
                                                 tracker=tracker,
                                                 roi_tracker=roi_tracker,
                                                 current_step=current_step,
                                                 current_epoch=epoch
                                                 )

            ddp_barrier(args.distributed)
            cnd = (current_step % n_checkpoint_save == 0)
            cnd &= args.is_master

            if cnd:
                # store in scratch in case of cc.
                model.save(current_step)
                clean_previous_checkpoints_except_last(model.save_dir,
                                                       ['G', 'optimizerG']
                                                       )

                save_tracker(args.outd_backup,
                             tracker=tracker,
                             roi_tracker=roi_tracker
                             )


        epoch_loss = _avg_list(l_holder, len(train_loader))
        tracker = update_tracker_train(tracker,
                                       n_losses=n_holder,
                                       v_losses=epoch_loss,
                                       period=constants.PR_EPOCH
                                       )
        DLLogger.log(f'Epoch {epoch}. Total TR loss: {epoch_loss[0]:.5f}')

        # test over the last best model found using validation.
        cnd_tst = epoch > 0
        cnd_tst &= (epoch % args.train['test_epoch_freq'] == 0)
        if cnd_tst:
            model.flush()
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

        # plot tracker
        if (epoch % args.train['plot_epoch_freq'] == 0) and args.is_master:
            p = join(args.outd, f'tracker_{constants.VALIDSET}.png')
            plot_tracker_eval(tracker=tracker,
                              roi_tracker=roi_tracker,
                              eval_over_roi_also=args.eval_over_roi_also,
                              eval_over_roi_also_model_select=args.eval_over_roi_also_model_select,
                              split=constants.VALIDSET,
                              path_store_figure=p,
                              args=args
                              )

            p = join(args.outd,
                     f'tracker_{constants.TRAINSET}-{constants.PR_ITER}.png')
            plot_tracker_train(tracker=tracker,
                               split=constants.TRAINSET,
                               path_store_figure=p,
                               args=args,
                               period=constants.PR_ITER
                               )

            p =join(args.outd,
                    f'tracker_{constants.TRAINSET}-{constants.PR_EPOCH}.png')
            plot_tracker_train(tracker=tracker,
                               split=constants.TRAINSET,
                               path_store_figure=p,
                               args=args,
                               period=constants.PR_EPOCH
                               )

        if is_cc() and args.is_master and (
                epoch % args.train['synch_scratch_epoch_freq'] == 0):
            copy_exp_dir_node_to_scratch(args)

        model.loss_fn.update_t()
        delta_t = dt.datetime.now() - t_init_epoch
        DLLogger.log(fmsg('Train epoch runtime: {}'.format(delta_t)))
        torch.cuda.empty_cache()

    # end training: ------------------------------------------------------------
    # test
    model.flush()
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

    # plot tracker
    if args.is_master:
        p = join(args.outd, f'tracker_{constants.VALIDSET}.png')
        plot_tracker_eval(tracker=tracker,
                          roi_tracker=roi_tracker,
                          eval_over_roi_also=args.eval_over_roi_also,
                          eval_over_roi_also_model_select=args.eval_over_roi_also_model_select,
                          split=constants.VALIDSET,
                          path_store_figure=p,
                          args=args
                          )

        p =join(args.outd,
                f'tracker_{constants.TRAINSET}-{constants.PR_ITER}.png')
        plot_tracker_train(tracker=tracker,
                           split=constants.TRAINSET,
                           path_store_figure=p,
                           args=args,
                           period=constants.PR_ITER
                           )

        p = join(args.outd,
                 f'tracker_{constants.TRAINSET}-{constants.PR_EPOCH}.png')
        plot_tracker_train(tracker=tracker,
                           split=constants.TRAINSET,
                           path_store_figure=p,
                           args=args,
                           period=constants.PR_EPOCH
                           )

        save_tracker(args.outd,
                     tracker=tracker,
                     roi_tracker=roi_tracker
                     )


def fast_eval(model,
              data_loader,
              ds_name: str,
              split: str,
              tracker: dict,
              roi_tracker: dict,
              args: object,
              current_step: int,
              epoch: int,
              tqdm_pos: int = 0,
              nbr_to_plot: int = 2,
              save_img_dir: str  = '',
              track_evolution_img: bool = False
              ) -> Tuple[dict, dict, dict, dict]:

    # reset test tracker. we do not track the test performance.
    # testing during training is allowed but using only the best model found
    # over the valid set. it can be done if you want to see the best
    # performance obtained so far with the best selected model so far since
    # the training can be way too long.
    if split == constants.TESTSET:
        reset_tracker_eval(tracker=tracker, split=split, ds_name=ds_name)
        reset_tracker_eval(tracker=roi_tracker, split=split, ds_name=ds_name)

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.set_eval_mode()

    border = args.scale
    # full image metrics
    avg_psnr = 0.0
    avg_mse = 0.0
    avg_nrmse = 0.0
    avg_ssim = 0.0
    avg_psnr_y = 0.0
    details = dict()

    # ROI only metrics
    roi_avg_psnr = 0.0
    roi_avg_mse = 0.0
    roi_avg_nrmse = 0.0
    roi_avg_ssim = 0.0
    roi_avg_psnr_y = 0.0
    roi_details = dict()

    idx = 0

    desc = f'Eval: {ds_name} (split: {split})'
    DLLogger.log(fmsg(desc))

    t_init_epoch = dt.datetime.now()

    for test_data in tqdm(data_loader, total=len(data_loader), ncols=80,
                          desc=desc, position=tqdm_pos, leave=(tqdm_pos == 0)):

        set_seed(seed=args.myseed, verbose=False)

        # todo: set a condition where to draw or not and how many.

        if isinstance(model, Interpolate):
            model.feed_data(test_data)
            model.test()

        else:
            model = _forward_with_padding(test_data, model, args)
            check_model_output_corruption(model.E)

        visuals = model.current_visuals()

        with torch.no_grad():
            # full image metrics
            _mtrs = _compute_metrics(test_data,
                                     visuals,
                                     idx,
                                     nbr_to_plot,
                                     split,
                                     ds_name,
                                     args,
                                     current_step,
                                     border,
                                     roi_th=None,
                                     plot=True,
                                     save_img_dir=save_img_dir,
                                     track_evolution_img=track_evolution_img
                                     )

            avg_psnr += _mtrs[constants.PSNR_MTR]
            avg_mse += _mtrs[constants.MSE_MTR]
            avg_nrmse += _mtrs[constants.NRMSE_MTR]
            avg_ssim += _mtrs[constants.SSIM_MTR]
            avg_psnr_y += _mtrs[constants.PSNR_Y_MTR]
            details = _cautious_merge_dicts(details, _mtrs['details'])

            # ROI metrics
            if args.eval_over_roi_also:
                nth = len(args.eval_over_roi_also_ths)
                assert nth > 0, nth
                _roi_mtrs = marginalize_roi_th_perf(test_data,
                                                    visuals,
                                                    idx,
                                                    nbr_to_plot,
                                                    split,
                                                    ds_name,
                                                    args,
                                                    current_step,
                                                    border
                                                    )
                roi_avg_psnr += _roi_mtrs[constants.PSNR_MTR]
                roi_avg_mse += _roi_mtrs[constants.MSE_MTR]
                roi_avg_nrmse += _roi_mtrs[constants.NRMSE_MTR]
                roi_avg_ssim += _roi_mtrs[constants.SSIM_MTR]
                roi_avg_psnr_y += _roi_mtrs[constants.PSNR_Y_MTR]
                roi_details = _cautious_merge_dicts(roi_details,
                                                    _roi_mtrs['details'])


        idx = idx + test_data['l_im'].shape[0]

    # synch
    if args.distributed and (args.eval_bsize > 1):
        avg_psnr = sync_tensor_across_gpus(avg_psnr.view(1, )).sum()
        avg_mse = sync_tensor_across_gpus(avg_mse.view(1, )).sum()
        avg_nrmse = sync_tensor_across_gpus(avg_nrmse.view(1, )).sum()
        avg_ssim = sync_tensor_across_gpus(avg_ssim.view(1, )).sum()
        avg_psnr_y = sync_tensor_across_gpus(avg_psnr_y.view(1, )).sum()
        idx = sync_non_tensor_value_across_gpus(float(idx))

        details = _sync_details_across_gpu(details, data_loader)

        if args.eval_over_roi_also:
            roi_avg_psnr = sync_tensor_across_gpus(roi_avg_psnr.view(1, )).sum()
            roi_avg_mse = sync_tensor_across_gpus(roi_avg_mse.view(1, )).sum()
            roi_avg_nrmse = sync_tensor_across_gpus(
                roi_avg_nrmse.view(1, )).sum()
            roi_avg_ssim = sync_tensor_across_gpus(roi_avg_ssim.view(1, )).sum()
            roi_avg_psnr_y = sync_tensor_across_gpus(
                roi_avg_psnr_y.view(1, )).sum()
            roi_details = _sync_details_across_gpu(roi_details, data_loader)


        ddp_barrier(args.distributed)

    avg_psnr = (avg_psnr / idx).item()
    avg_mse = (avg_mse / idx).item()
    avg_nrmse = (avg_nrmse / idx).item()
    avg_ssim = (avg_ssim / idx).item()
    avg_psnr_y = (avg_psnr_y / idx).item()

    if args.eval_over_roi_also:
        roi_avg_psnr = (roi_avg_psnr / idx).item()
        roi_avg_mse = (roi_avg_mse / idx).item()
        roi_avg_nrmse = (roi_avg_nrmse / idx).item()
        roi_avg_ssim = (roi_avg_ssim / idx).item()
        roi_avg_psnr_y = (roi_avg_psnr_y / idx).item()


    delta_t = dt.datetime.now() - t_init_epoch
    DLLogger.log(fmsg(f'Eval time Split: {split}, dataset: {ds_name}:  '
                      f'{delta_t}'))

    # update tracker(s).
    mtr_val = {
        constants.PSNR_MTR: avg_psnr,
        constants.MSE_MTR: avg_mse,
        constants.NRMSE_MTR: avg_nrmse,
        constants.SSIM_MTR: avg_ssim,
        constants.PSNR_Y_MTR: avg_psnr_y
    }

    roi_mtr_val = {
        constants.PSNR_MTR: roi_avg_psnr,
        constants.MSE_MTR: roi_avg_mse,
        constants.NRMSE_MTR: roi_avg_nrmse,
        constants.SSIM_MTR: roi_avg_ssim,
        constants.PSNR_Y_MTR: roi_avg_psnr_y
    }

    if not args.eval_over_roi_also:

        tracker, _ = fast_update_tracker(args=args,
                                        tracker=tracker,
                                        mtr_val=mtr_val,
                                        split=split,
                                        ds_name=ds_name,
                                        idx_best=None
                                        )
    else:
        if args.eval_over_roi_also_model_select:
            roi_tracker, idx_best = fast_update_tracker(args=args,
                                                        tracker=roi_tracker,
                                                        mtr_val=roi_mtr_val,
                                                        split=split,
                                                        ds_name=ds_name,
                                                        idx_best=None
                                                        )

            tracker, _ = fast_update_tracker(args=args,
                                             tracker=tracker,
                                             mtr_val=mtr_val,
                                             split=split,
                                             ds_name=ds_name,
                                             idx_best=idx_best
                                             )


        else:
            tracker, idx_best = fast_update_tracker(args=args,
                                                    tracker=tracker,
                                                    mtr_val=mtr_val,
                                                    split=split,
                                                    ds_name=ds_name,
                                                    idx_best=None
                                                    )

            roi_tracker, _ = fast_update_tracker(args=args,
                                                 tracker=roi_tracker,
                                                 mtr_val=roi_mtr_val,
                                                 split=split,
                                                 ds_name=ds_name,
                                                 idx_best=idx_best
                                                 )


    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model.set_train_mode()

    return tracker, details, roi_tracker, roi_details

def fast_update_tracker(args: object,
                        tracker: dict,
                        mtr_val: dict,
                        split: str,
                        ds_name: str,
                        idx_best: int = None
                        ) -> Tuple[dict, int]:

    # update master metric first.
    tracker, _idx_best = update_tracker_eval(tracker=tracker,
                                             split=split,
                                             ds_name=ds_name,
                                             metric=args.model_select_mtr,
                                             value=mtr_val[
                                                 args.model_select_mtr],
                                             idx_best=idx_best
                                             )
    local_idx_best = _idx_best

    if idx_best is not None:
        local_idx_best = idx_best

    mtrs_non_master = set(mtr_val.keys())
    mtrs_non_master.remove(args.model_select_mtr)  # inplace op!!!!!

    for k in mtrs_non_master:
        tracker, _ = update_tracker_eval(tracker=tracker,
                                         split=split,
                                         ds_name=ds_name,
                                         metric=k,
                                         value=mtr_val[k],
                                         idx_best=local_idx_best
                                         )

    return tracker, local_idx_best

def _sync_details_across_gpu(details: dict, data_loader) -> dict:

    holder = dict()
    mtrs = list(details[list(details.keys())[0]].keys())

    for mtr in mtrs:
        tmp = dict()
        for im_id in details:
            float_id = data_loader.dataset.im_h_ids_to_float[im_id]
            tmp[float_id] = details[im_id][mtr]

        holder[mtr] = sync_dict_across_gpus(tmp, move_sync_vals_to_cpu=True)

    out = dict()
    for mtr in mtrs:
        for f_im_id in holder[mtr]:
            im_id = data_loader.dataset.float_to_im_h_ids[f_im_id]
            val = holder[mtr][f_im_id].item()  # assumption: single value.

            if im_id in out:
                out[im_id][mtr] = val
            else:
                out[im_id] = {
                    mtr: val
                }

    return out


def _forward_with_padding(test_data: dict, model, args: object):

    if args.netG['net_type'] == constants.SWINIR:
        _test_data = deepcopy(test_data)
        model.feed_data(_test_data)

        _, _, h_old, w_old = _test_data['l_im'].shape
        nt = args.netG['net_type']

        wsz = eval("args.netG[f'{nt}_window_size']")
        im_l = model.L
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old

        # im_l = torch.cat(
        #     [im_l, torch.flip(im_l, [2])], 2)[:, :, :h_old + h_pad, :]
        # im_l = torch.cat(
        #     [im_l, torch.flip(im_l, [3])], 3)[:, :, :, :w_old + w_pad]

        # fast, less memory. flip only the pad.
        im_l = torch.cat([im_l, torch.flip(im_l[:, :, h_old - h_pad:, :], [2])],
                         2)
        im_l = torch.cat([im_l, torch.flip(im_l[:, :, :, w_old - w_pad:], [3])],
                         3)

        model.L = im_l
        model.test()
        model.E = model.E[..., :h_old * args.scale, :w_old * args.scale]
        model.L = model.L[..., :h_old, :w_old]
    else:
        model.feed_data(test_data)
        model.test()

    return model


def _rgb_tensor(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 4, t.ndim
    assert t.shape[1] in [1, 3], t.shape[1]

    if t.shape[1] == 3:
        return t
    return t.repeat(1, 3, 1, 1)


def marginalize_roi_th_perf(test_data: dict,
                            visuals: dict,
                            idx: int,
                            nbr_to_plot: int,
                            split: str,
                            ds_name: str,
                            args: object,
                            current_step: int,
                            border: int,
                            ) -> dict:
    assert args.eval_over_roi_also, args.eval_over_roi_also

    roi_mtrs = None

    nth = len(args.eval_over_roi_also_ths)
    assert nth > 0, nth
    for roi_th in args.eval_over_roi_also_ths:
        _mtrs = _compute_metrics(test_data,
                                 visuals,
                                 idx,
                                 nbr_to_plot,
                                 split,
                                 ds_name,
                                 args, current_step,
                                 border,
                                 roi_th=roi_th,
                                 plot=False
                                 )



        if roi_mtrs is None:
            roi_mtrs = _mtrs

        else:
            for k in _mtrs:
                if k != 'details':
                    roi_mtrs[k] = roi_mtrs[k] + _mtrs[k]

                else:
                    for kk in _mtrs[k]:
                        for m in _mtrs[k][kk]:
                            roi_mtrs[k][kk][m] = roi_mtrs[k][kk][m] + _mtrs[
                                k][kk][m]

    # average over thresholds
    nth = float(nth)
    for k in _mtrs:
        if k != 'details':
            roi_mtrs[k] = roi_mtrs[k] / nth

        else:
            for kk in _mtrs[k]:
                for m in _mtrs[k][kk]:
                    roi_mtrs[k][kk][m] = roi_mtrs[k][kk][m] / nth

    return roi_mtrs


def check_negative_non_float(vls: torch.Tensor, name: str):
    _inf = torch.isinf(vls).sum()
    _nan = torch.isnan(vls).sum()
    _neg = (vls < 0).sum()

    msg = f"inf {name}: {_inf}"
    try:
        assert _inf == 0, msg
    except AssertionError:
        DLLogger.log(f'Terminated due to error: {msg}')
        sys.exit()

    msg = f"nan {name}: {_nan}"

    try:
        assert _nan == 0, msg
    except AssertionError:
        DLLogger.log(f'Terminated due to error: {msg}')
        sys.exit()

    msg = f"neg {name}: {_neg}"
    try:
        assert _neg == 0, msg
    except AssertionError:
        DLLogger.log(f'Terminated due to error: {msg}')
        sys.exit()


def _compute_metrics(test_data: dict,
                     visuals: dict,
                     idx: int,
                     nbr_to_plot: int,
                     split: str,
                     ds_name: str,
                     args: object,
                     current_step: int,
                     border: int,
                     roi_th: int = None,
                     plot: bool = True,
                     save_img_dir: str  = '',
                     track_evolution_img: bool = False
                     ) -> dict:
    _details = dict()

    E_img = utils_image.tensor2uint82float(visuals['E'])  # BCHW [0, 255]
    H_img = utils_image.tensor2uint82float(visuals['H'])  # BCHW [0, 255]
    roi = None

    if roi_th is not None:
        assert args.eval_over_roi_also, args.eval_over_roi_also
        assert isinstance(roi_th, int), type(roi_th)
        b, c, h, w = H_img.shape
        assert c == 1, f" dont support c == {c} > 1."
        roi = (H_img >= roi_th).float()

    assert E_img.shape == H_img.shape

    # todo: add more metrics? depending on valid or test? valid: light
    #  metrics. test: full metrics.

    # ==========================================================================
    #                           COMPUTE PERF.
    # ==========================================================================

    # PSNR
    _psnr: torch.Tensor = utils_image.mbatch_gpu_calculate_psnr(
        E_img, H_img, border=border, roi=roi)

    check_negative_non_float(_psnr, name=constants.PSNR_MTR)

    # PSNR_Y
    assert E_img.shape[1] in [1, 3], E_img.shape[1]  # support only grey/RGB.
    _psnr_y: torch.Tensor = utils_image.mbatch_gpu_calculate_psnr(
        utils_image.mb_gpu_rgb2ycbcr(
            _rgb_tensor(E_img) / 255.0, only_y=True) * 255.0,
        utils_image.mb_gpu_rgb2ycbcr(
            _rgb_tensor(H_img) / 255.0, only_y=True) * 255.0,
        border=border,
        roi=roi
    )

    check_negative_non_float(_psnr_y, name=constants.PSNR_Y_MTR)

    # MSE
    _mse: torch.Tensor = utils_image.mbatch_gpu_calculate_mse(
        E_img, H_img, border=border, roi=roi)

    check_negative_non_float(_mse, name=constants.MSE_MTR)

    # NRMSE
    _nrmse: torch.Tensor = utils_image.mbatch_gpu_calculate_nrmse(
        img=E_img, y=H_img, border=border, roi=roi)

    check_negative_non_float(_nrmse, name=constants.NRMSE_MTR)

    # SSIM
    _ssim: torch.Tensor = utils_image.mbatch_gpu_calculate_ssim(
        E_img, H_img, border=border, roi=roi)

    check_negative_non_float(_ssim, name=constants.SSIM_MTR)

    # ==========================================================================
    #                        END COMPUTE PERF.
    # ==========================================================================

    out = {
        constants.PSNR_MTR: _psnr.sum(),
        constants.MSE_MTR: _mse.sum(),
        constants.NRMSE_MTR: _nrmse.sum(),
        constants.SSIM_MTR: _ssim.sum(),
        constants.PSNR_Y_MTR: _psnr_y.sum(),
        'details': dict()
    }

    _psnr = _psnr.detach().cpu()
    _psnr_y = _psnr_y.detach().cpu()
    _mse = _mse.detach().cpu()
    _nrmse = _nrmse.detach().cpu()
    _ssim = _ssim.detach().cpu()

    # plot + stats.
    bsize = test_data['l_im'].shape[0]
    _idx = idx
    for i in range(bsize):
        img_id = test_data['h_id'][i]  # use H as id.
        assert img_id not in _details

        _details[img_id] = {
            constants.PSNR_MTR: _psnr[i].item(),
            constants.MSE_MTR: _mse[i].item(),
            constants.NRMSE_MTR: _nrmse[i].item(),
            constants.SSIM_MTR: _ssim[i].item(),
            constants.PSNR_Y_MTR: _psnr_y[i].item()
        }

        if (_idx < nbr_to_plot) and args.is_master and plot:
            img_dir = save_img_dir
            assert os.path.isdir(save_img_dir), save_img_dir

            if track_evolution_img:
                img_dir = join(img_dir, reformat_id(img_id))
                os.makedirs(img_dir, exist_ok=True)
                save_img_path = join(
                    img_dir, f'{reformat_id(img_id)}_{current_step}.png')
            else:
                save_img_path = join(img_dir, f'{reformat_id(img_id)}.png')

            # if split == constants.TESTSET:
            #     img_dir = join(args.outd, args.save_dir_imgs, split, ds_name)
            #     save_img_path = join(img_dir, f'{reformat_id(img_id)}.png')
            # else:
            #     img_dir = join(args.outd, args.save_dir_imgs, split, ds_name,
            #                    reformat_id(img_id))
            #     save_img_path = join(
            #         img_dir, f'{reformat_id(img_id)}_{current_step}.png')

            # os.makedirs(img_dir, exist_ok=True)

            # todo: deal with rgb.
            _E_img = E_img[i].squeeze().cpu().numpy()  # CHW or HW.
            utils_image.cv2_imsave_rgb_in(_E_img, save_img_path)

        _idx += 1

    out['details'] = _details

    return out


def evaluate_single_ds(args: object,
                       model,
                       loader: dict,
                       ds_name: str,
                       tracker: dict,
                       roi_tracker: dict,
                       current_step: int,
                       epoch: int,
                       split: str,
                       nbr_to_plot: int = 10,
                       save_img_dir: str = ''
                       ) -> Tuple[dict, dict]:

    if not os.path.isdir(save_img_dir):
        save_img_dir = join(args.outd, args.save_dir_imgs, split, ds_name)
        os.makedirs(save_img_dir, exist_ok=True)

    track_evolution_img = False

    out_ = fast_eval(model=model,
                     data_loader=loader,
                     ds_name=ds_name,
                     split=split,
                     tracker=tracker,
                     roi_tracker=roi_tracker,
                     args=args,
                     current_step=current_step,
                     epoch=epoch,
                     nbr_to_plot=nbr_to_plot,
                     save_img_dir=save_img_dir,
                     track_evolution_img=track_evolution_img
                     )
    tracker, details, roi_tracker, roi_details = out_

    if args.is_master:

        _dir = join(args.outd_backup, 'best-models')
        os.makedirs(_dir, exist_ok=True)
        _f_name = f'details_{ds_name}.yml'
        with open(join(_dir, _f_name), 'w') as fd:
            yaml.dump(details, fd)

        if args.eval_over_roi_also:
            _f_name = f'roi_details_{ds_name}.yml'
            with open(join(_dir, _f_name), 'w') as fd:
                yaml.dump(roi_details, fd)



        status_perf = write_current_perf_eval(tracker,
                                              split=split,
                                              ds_name=ds_name,
                                              save_dir=_dir,
                                              name_f=f'{ds_name}.yaml',
                                              current_step=current_step,
                                              current_epoch=epoch
                                              )

        roi_status_perf = None
        if args.eval_over_roi_also:
            name_f = f'roi-{ds_name}.yaml'
            roi_status_perf = write_current_perf_eval(roi_tracker,
                                                      split=split,
                                                      ds_name=ds_name,
                                                      save_dir=_dir,
                                                      name_f=name_f,
                                                      current_step=current_step,
                                                      current_epoch=epoch
                                                      )


        msg = current_perf_to_str(
            status=status_perf,
            roi_status=roi_status_perf,
            master_mtr=args.model_select_mtr,
            model_select_roi=args.eval_over_roi_also_model_select
        )
        DLLogger.log(msg)

    return tracker, roi_tracker


def evaluate(args: object,
             model,
             loaders: dict,
             tracker: dict,
             roi_tracker: dict,
             current_step: int,
             epoch: int,
             split: str,
             use_best_models: bool = True,
             nbr_to_plot: int = 10
             ) -> Tuple[dict, dict]:

    torch.cuda.empty_cache()

    assert split == constants.TESTSET, split

    ddp_barrier(args.distributed)
    DLLogger.log(fmsg(f'Eval {split}: '
                      f'{f"{constants.SEP}".join(list(loaders.keys()))}'))

    # save current model to not override it.
    if args.is_master:
        dir_best = join(args.outd_backup, 'best-models')
        os.makedirs(dir_best, exist_ok=True)
        model.save_current(save_dir=dir_best)

    ddp_barrier(args.distributed)


    for ds_name in loaders:

        if use_best_models:
            _dir = join(args.outd_backup, 'best-models')

            if args.multi_valid:
                _f_name = f'G-{ds_name}.pth'
                _f_name = _f_name.replace(split, constants.VALIDSET)
            else:
                _f_name = 'G-model.pth'

            _path_model = join(_dir, _f_name)

            print(_path_model)

            # no best model/checkpoint has found -> skip.
            if not os.path.isfile(_path_model):
                DLLogger.log(
                    fmsg(f'No best model/checkpoint found for eval '
                         f'over {split} @ {ds_name}: Skipping.'
                         f'Model not found @: {_path_model}'))
                continue

            model.load_network(_path_model,
                               model.netG,
                               strict=True,
                               param_key='params')

        cnd = True
        if args.distributed and (args.eval_bsize > 1):
            pass  # do multi-gpu eval.
        else:
            cnd &= args.is_master  # prevent multi-gpu eval.

        ddp_barrier(args.distributed)


        if cnd:
            tracker, roi_tracker = evaluate_single_ds(args=args,
                                                      model=model,
                                                      loader=loaders[ds_name],
                                                      ds_name=ds_name,
                                                      tracker=tracker,
                                                      roi_tracker=roi_tracker,
                                                      current_step=current_step,
                                                      epoch=epoch,
                                                      split=split,
                                                      nbr_to_plot=nbr_to_plot
                                                      )

            # interpolation perf.
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
            _model_interp = Interpolate(
                task=args.task, scale=args.scale,
                scale_mode=args.basic_interpolation).to(device)
            _ds_name = f'{ds_name}_{args.basic_interpolation}'

            tracker, roi_tracker = evaluate_single_ds(args=args,
                                                      model=_model_interp,
                                                      loader=loaders[ds_name],
                                                      ds_name=_ds_name,
                                                      tracker=tracker,
                                                      roi_tracker=roi_tracker,
                                                      current_step=current_step,
                                                      epoch=epoch,
                                                      split=split,
                                                      nbr_to_plot=nbr_to_plot
                                                      )

        ddp_barrier(args.distributed)

    # turn back to the current model
    dir_best = join(args.outd_backup, 'best-models')
    os.makedirs(dir_best, exist_ok=True)

    # todo: check if there are other nets.
    model.load_current(save_dir=dir_best)

    ddp_barrier(args.distributed)
    torch.cuda.empty_cache()
    return tracker, roi_tracker
