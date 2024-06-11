import os
import sys
from os.path import dirname, abspath, join
from typing import Union, Tuple, Dict, Any
from copy import deepcopy
import pickle as pkl
import math
import datetime as dt


import numpy as np
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import torch.distributed as dist

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.datasets.wsol_loader import get_data_loader

from dlib.utils.utils_reproducibility import set_seed
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag

from dlib.cams.fcam_seeding import MBSeederSLFCAMS
from dlib.cams.tcam_seeding import TCAMSeeder

from dlib.cams.fcam_seeding import SeederCBOX

from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.shared import is_cc
from dlib.datasets.ilsvrc_manager import prepare_next_bucket
from dlib.datasets.ilsvrc_manager import prepare_vl_tst_sets
from dlib.datasets.ilsvrc_manager import delete_train

from dlib.parallel import sync_tensor_across_gpus
from dlib.parallel import MyDDP as DDP

from dlib.box import BoxStats
from dlib.filtering import GaussianFiltering

from dlib import losses


__all__ = ['Basic', 'Trainer']


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        val = constants.VALIDSET
        self.value_per_epoch = [] \
            if split == val else [-np.inf if higher_is_better else np.inf]
        # todo: replace with self.value_per_epoch = []

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)
        # todo: change to
        # idx = [i for i, x in enumerate(
        #  self.value_per_epoch) if x == self.best_value]
        # assert len(idx) > 0
        # self.best_epoch = idx[-1]


class Basic(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)
    _EVAL_METRICS = ['loss',
                     constants.CLASSIFICATION_MTR,
                     constants.LOCALIZATION_MTR,
                     constants.FAILD_BOXES_MTR]
    _BEST_CRITERION_METRIC = constants.LOCALIZATION_MTR

    _NUM_CLASSES_MAPPING = {
        constants.CUB: constants.NUMBER_CLASSES[constants.CUB],
        constants.ILSVRC: constants.NUMBER_CLASSES[constants.ILSVRC],
        constants.OpenImages: constants.NUMBER_CLASSES[constants.OpenImages],
        constants.YTOV1: constants.NUMBER_CLASSES[constants.YTOV1]
    }

    # @property
    # def _BEST_CRITERION_METRIC(self):
    #     assert self.inited
    #     assert self.args is not None
    #
    #     return 'localization'
    #
    #     if self.args.task == constants.STD_CL:
    #         return 'classification'
    #     elif self.args.task == constants.F_CL:
    #         return 'localization'
    #     else:
    #         raise NotImplementedError

    def __init__(self, args):
        self.args = args
        
    def _set_performance_meters(self):
        self._EVAL_METRICS += [f'{constants.LOCALIZATION_MTR}_IOU_{threshold}'
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top1_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top5_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict


def plot_perf_curves_top_1_5(curves: dict, fdout: str, title: str):

    x_label = r'$\tau$'
    y_label = 'BoxAcc'

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

    for i, top in enumerate(['top1', 'top5']):

        iouthres = sorted(list(curves[top].keys()))
        for iout in iouthres:
            axes[0, i].plot(curves['x'], curves[top][iout],
                            label=r'{}: $\sigma$={}'.format(top, iout))

        axes[0, i].xaxis.set_tick_params(labelsize=5)
        axes[0, i].yaxis.set_tick_params(labelsize=5)
        axes[0, i].set_xlabel(x_label, fontsize=8)
        axes[0, i].set_ylabel(y_label, fontsize=8)
        axes[0, i].grid(True)
        axes[0, i].legend(loc='best')
        axes[0, i].set_title(top)

    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.show()
    fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                dpi=300)


def plot_perf_curves_top_1_5(curves: dict, fdout: str, title: str):

    x_label = r'$\tau$'
    y_label = 'BoxAcc'

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

    for i, top in enumerate(['top1', 'top5']):

        iouthres = sorted(list(curves[top].keys()))
        for iout in iouthres:
            axes[0, i].plot(curves['x'], curves[top][iout],
                            label=r'{}: $\sigma$={}'.format(top, iout))

        axes[0, i].xaxis.set_tick_params(labelsize=5)
        axes[0, i].yaxis.set_tick_params(labelsize=5)
        axes[0, i].set_xlabel(x_label, fontsize=8)
        axes[0, i].set_ylabel(y_label, fontsize=8)
        axes[0, i].grid(True)
        axes[0, i].legend(loc='best')
        axes[0, i].set_title(top)

    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.show()
    fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                dpi=300)


class Trainer(Basic):

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 loss: losses.MasterLoss,
                 classifier=None):
        super(Trainer, self).__init__(args=args)

        self.device = torch.device(args.c_cudaid)
        self.args = args
        self.performance_meters = self._set_performance_meters()
        self.model = model

        if isinstance(model, DDP):
            self._pytorch_model = self.model.module
        else:
            self._pytorch_model = self.model

        self.loss: losses.MasterLoss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if is_cc() and self.args.ds_chunkable:
            if self.args.distributed:
                dist.barrier()
            if self.args.is_node_master:
                status1 = prepare_vl_tst_sets(dataset=self.args.dataset)
                if (status1[0] == -1) and self.args.is_master:
                    DLLogger.log(f'Error in preparing valid/test. '
                                 f'{status1[1]}. Exiting.')

                status2 = prepare_next_bucket(bucket=0,
                                              dataset=self.args.dataset)
                if (status2[0] == -1) and self.args.is_master:
                    DLLogger.log(f'Error in preparing bucket '
                                 f'{0}. {status2[1]}. Exiting.')

                if (status1[0] == -1) or (status2[0] == -1):
                    sys.exit()
            if self.args.distributed:
                dist.barrier()

        self.loaders, self.train_sampler = get_data_loader(
            args=self.args,
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            dataset=self.args.dataset,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            tr_bucket=0
        )

        self.sl_mask_builder = None
        if args.task == constants.F_CL:
            self.sl_mask_builder: MBSeederSLFCAMS = self._get_sl(args)
        if args.task == constants.TCAM:
            self.sl_mask_builder: TCAMSeeder = self._get_sl(args)

        self.roi_thresholds = None
        if args.task in [constants.F_CL, constants.TCAM]:
            self.roi_thresholds: dict = self._load_roi_thresholds(args)

        self.fg_mask_seed_builder = None
        self.bg_mask_seed_builder = None
        self.kde = None

        if args.task == constants.C_BOX:
            self.mask_seed_builder: SeederCBOX = self._get_cbox_seeder(args)

        self.epoch = 0
        self.counter = 0
        self.seed = int(args.MYSEED)
        self.default_seed = int(args.MYSEED)

        self.best_model_loc = deepcopy(self.model).to(self.cpu_device).eval()
        self.best_model_cl = deepcopy(self.model).to(self.cpu_device).eval()

        self.perf_meters_backup = None
        self.inited = True

        self.classifier = classifier
        self.box_stats = None
        self.blur_op = None
        if args.task == constants.C_BOX:
            assert classifier is not None
            self.classifier.eval()
            self.classifier.freeze_classifier()
            self.classifier.assert_cl_is_frozen()

            self.box_stats = BoxStats(scale_domain=args.model['scale_domain'],
                                      h=args.crop_size,
                                      w=args.crop_size).cuda(args.c_cudaid)

            self.blur_op: GaussianFiltering = GaussianFiltering(
                blur_ksize=args.cb_cl_score_blur_ksize,
                blur_sigma=args.cb_cl_score_blur_sigma,
                device=torch.device(self.args.c_cudaid)).cuda(args.c_cudaid)

        self.std_cam_extractor = None
        if args.task in [constants.F_CL, constants.TCAM]:
            assert classifier is not None
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=classifier, args=args)

        self.fcam_argmax = False
        self.fcam_argmax_previous = False

        self.vl_size_priors = None
        if self._is_prior_size_needed():
            self.vl_size_priors: dict = deepcopy(
                self.loaders[constants.VALIDSET].dataset.build_size_priors())

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()

        self.pre_forward_info = dict()

    @staticmethod
    def _load_roi_thresholds(args) -> dict:
        roi_ths_paths: dict = args.std_cams_thresh_file
        out = dict()
        for split in roi_ths_paths:
            out[split] = dict()
            if os.path.isfile(roi_ths_paths[split]):
                with open(roi_ths_paths[split], 'r') as froit:
                    content = froit.readlines()
                    content = [c.rstrip('\n') for c in content]
                    for line in content:
                        z = line.split(',')
                        assert len(z) == 2  # id, th
                        id_sample, th = z
                        assert id_sample not in out
                        out[split][id_sample] = float(th)
            else:
                out[split] = None

        return out

    @staticmethod
    def _build_std_cam_extractor(classifier, args):
        classifier.eval()
        return build_std_cam_extractor(classifier=classifier, args=args)

    @staticmethod
    def _get_sl(args):
        if args.task == constants.F_CL:
            return MBSeederSLFCAMS(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    min_p=args.sl_min_p,
                    fg_erode_k=args.sl_fg_erode_k,
                    fg_erode_iter=args.sl_fg_erode_iter,
                    support_background=args.model['support_background'],
                    multi_label_flag=args.multi_label_flag,
                    seg_ignore_idx=args.seg_ignore_idx)

        elif args.task == constants.TCAM:
            return TCAMSeeder(
                seed_tech=args.sl_tc_seed_tech,
                min_=args.sl_tc_min,
                max_=args.sl_tc_max,
                ksz=args.sl_tc_ksz,
                max_p=args.sl_tc_max_p,
                min_p=args.sl_tc_min_p,
                fg_erode_k=args.sl_tc_fg_erode_k,
                fg_erode_iter=args.sl_tc_fg_erode_iter,
                support_background=args.model['support_background'],
                multi_label_flag=args.multi_label_flag,
                seg_ignore_idx=args.seg_ignore_idx,
                cuda_id=args.c_cudaid
            )

        else:
            raise NotImplementedError(args.task)

    def _get_cbox_seeder(self, args):
        return SeederCBOX(n=args.cb_seed_n,
                          bg_low_z=args.cb_seed_bg_low_z,
                          bg_up_z=args.cb_seed_bg_up_z,
                          fg_erode_k=args.cb_seed_erode_k,
                          fg_erode_iter=args.cb_seed_erode_iter,
                          ksz=args.cb_seed_ksz,
                          seg_ignore_idx=args.seg_ignore_idx,
                          device=self.device
                          )

    def prepare_std_cams_disq(self, std_cams: torch.Tensor,
                              image_size: Tuple) -> torch.Tensor:

        assert std_cams.ndim == 4
        cams = std_cams.detach()

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        # Quick fix: todo...
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        return cams

    def get_std_cams_minibatch(self, images, targets) -> torch.Tensor:
        # used only for task f_cl
        assert self.args.task == constants.F_CL
        assert images.ndim == 4
        image_size = images.shape[2:]

        cams = None
        for idx, (image, target) in enumerate(zip(images, targets)):
            cl_logits = self.classifier(image.unsqueeze(0))
            cam = self.std_cam_extractor(
                class_idx=target.item(), scores=cl_logits, normalized=True)
            # h`, w`
            # todo: set to false (normalize).

            cam = cam.detach().unsqueeze(0).unsqueeze(0)

            if cams is None:
                cams = cam
            else:
                cams = torch.vstack((cams, cam))

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)

        return cams

    def is_seed_required(self, _epoch: int) -> bool:
        if self.args.task == constants.F_CL:
            cmd = (self.args.task == constants.F_CL)
            cmd &= self.args.sl_fc
            cmd &= ('self_learning_fcams' in self.loss.n_holder)
            cmd2 = False
            for _l in self.loss.losses:
                if isinstance(_l, losses.SelfLearningFcams):
                    cmd2 = _l.is_on(_epoch=_epoch)

            return cmd and cmd2

        elif self.args.task == constants.TCAM:
            cmd = (self.args.task == constants.TCAM)
            cmd &= self.args.sl_tc
            cmd &= ('self_learning_tcams' in self.loss.n_holder)
            cmd2 = False
            for _l in self.loss.losses:
                if isinstance(_l, losses.SelfLearningTcams):
                    cmd2 = _l.is_on(_epoch=_epoch)

            return cmd and cmd2

        elif self.args.task == constants.C_BOX:
            cmd = (self.args.task == constants.C_BOX)
            cmd &= ('seed_cbox' in self.loss.n_holder)
            cmd &= self.args.cb_seed
            cmd2 = False
            for _l in self.loss.losses:
                if isinstance(_l, losses.SeedCbox):
                    cmd2 = _l.is_on(_epoch=_epoch)

            return cmd and cmd2

        return False

    def _is_prior_size_needed(self) -> bool:
        cmd = (self.args.task == constants.C_BOX)
        cmd &= (self.args.dataset in [constants.CUB,
                                      constants.ILSVRC,
                                      constants.YTOV1])
        return cmd

    def _gen_rand_init_box(self, h: int, w: int, minsz: float):
        assert 0 < self.args.cb_init_box_size <= 1.
        assert isinstance(self.args.cb_init_box_size, float)

        assert isinstance(self.args.cb_init_box_var, float)
        assert self.args.cb_init_box_var >= 0

        maxsz = 0.99

        m = self.args.cb_init_box_size
        v = self.args.cb_init_box_var

        s = np.random.normal(loc=m, scale=v, size=(1,)).item()
        s = min(max(s, minsz), maxsz)

        x_hat_0 = max(h / 2. - h * np.sqrt(s) / 2., .0)
        x_hat_1 = min(h / 2. + h * np.sqrt(s) / 2., h - 1)

        y_hat_0 = max(w / 2. - w * np.sqrt(s) / 2., .0)
        y_hat_1 = min(w / 2. + w * np.sqrt(s) / 2., w - 1)
        return x_hat_0, x_hat_1, y_hat_0, y_hat_1

    def _cbox_filter_valid_tensors(self, tensor: torch.Tensor,
                                   valid: torch.Tensor
                                   ) -> Union[torch.Tensor, None]:

        idx_valid = torch.nonzero(valid.contiguous().view(-1, ),
                                  as_tuple=False).squeeze()
        if idx_valid.numel() == 0:
            return None

        _z = tensor[idx_valid]  # ?
        if idx_valid.numel() == 1:
            _z = _z.unsqueeze(0)

        return _z

    def _pre_forward(self, output, images: torch.Tensor,
                     vl_size_priors: Dict[str, Any]):

        n, c, h, w = images.shape
        ratio = float(h * w)

        if self.args.task == constants.STD_CL:
            pass

        elif self.args.task in [constants.F_CL, constants.TCAM]:
            pass

        elif self.args.task == constants.C_BOX:

            # todo: clean later.

            box = output
            _box = box.detach()
            zz = self.box_stats(box=_box, eval=True)
            x_hat, y_hat, valid, area, mask_fg, mask_bg = zz

            _area = area / ratio

            # imgs_fg = self.get_fg_imgs(images=images, blured_imgs=blured_imgs,
            #                            mask_fg=mask_fg, mask_bg=mask_bg)
            # imgs_bg = self.get_bg_imgs(images=images, blured_imgs=blured_imgs,
            #                            mask_fg=mask_fg, mask_bg=mask_bg)

            # logits_fg = self.classifier(imgs_fg)
            # logits_bg = self.classifier(imgs_bg)
            # logits_clean = self.classifier(images)

            # raw_imgs_device = raw_imgs.cuda(self.args.c_cudaid)

            self.pre_forward_info['x_hat'] = x_hat.detach()
            self.pre_forward_info['y_hat'] = y_hat.detach()

            c_data = constants.MIN_SIZE_DATA
            c_cont = constants.MIN_SIZE_CONST
            for i in range(n):

                if self.args.cb_pp_box_min_size_type == c_cont:
                    minsz = self.args.cb_pp_box_min_size
                elif self.args.cb_pp_box_min_size_type == c_data:
                    minsz = vl_size_priors['min_s'][i]
                else:
                    raise NotImplementedError

                if (valid[i] == 0) or (_area[i] < minsz):
                    z = self._gen_rand_init_box(h, w, minsz)
                    self.pre_forward_info['x_hat'][i][0] = z[0]
                    self.pre_forward_info['x_hat'][i][1] = z[1]
                    self.pre_forward_info['y_hat'][i][0] = z[2]
                    self.pre_forward_info['y_hat'][i][1] = z[3]
        else:
            raise NotImplementedError

    def _wsol_training(self,
                       images: torch.Tensor,
                       raw_imgs: torch.Tensor,
                       targets: torch.Tensor,
                       std_cams: torch.Tensor,
                       blured_imgs: torch.Tensor,
                       vl_size_priors: Dict[str, Any],
                       roi_thresholds: torch.Tensor = None):
        y_global = targets

        output = self.model(images)

        with torch.no_grad():
            self._pre_forward(output=output, images=images,
                              vl_size_priors=vl_size_priors)

        if self.args.task == constants.STD_CL:
            cl_logits = output
            loss = self.loss(epoch=self.epoch, cl_logits=cl_logits,
                             glabel=y_global)
            logits = cl_logits

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output

            if self.is_seed_required(_epoch=self.epoch):
                if std_cams is None:
                    cams_inter = self.get_std_cams_minibatch(images=images,
                                                             targets=targets)
                else:
                    cams_inter = std_cams

                with torch.no_grad():
                    seeds = self.sl_mask_builder(cams_inter)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds
            )
            logits = cl_logits

        elif self.args.task == constants.TCAM:
            cl_logits, fcams, im_recon = output
            if roi_thresholds is not None:
                # range [0, 255]. the chance that mean(th) > 1. is high as a
                # way to ensure that thresholds are in that range.
                # todo: weak assertion.
                assert roi_thresholds.mean() > 1., roi_thresholds.mean()

            if self.is_seed_required(_epoch=self.epoch):
                if std_cams is None:
                    cams_inter = self.get_std_cams_minibatch(images=images,
                                                             targets=targets)
                else:
                    cams_inter = std_cams

                with torch.no_grad():
                    seeds = self.sl_mask_builder(x=cams_inter,
                                                 roi_thresh=roi_thresholds)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds
            )
            logits = cl_logits

        elif self.args.task == constants.C_BOX:
            box = output
            zz = self.box_stats(box=box, eval=False)
            x_hat, y_hat, valid, area, mask_fg, mask_bg = zz

            logits_fg = None
            logits_bg = None
            logits_clean = None

            imgs_fg = self.get_fg_imgs(images=images,
                                       blured_imgs=blured_imgs,
                                       mask_fg=mask_fg, mask_bg=mask_bg)
            logits_fg = self.classifier(imgs_fg)

            if self.args.cb_cl_score:
                imgs_bg = self.get_bg_imgs(images=images,
                                           blured_imgs=blured_imgs,
                                           mask_fg=mask_fg, mask_bg=mask_bg)
                logits_bg = self.classifier(imgs_bg)
                logits_clean = self.classifier(images)


            cams_inter, seeds = None, None
            if self.is_seed_required(_epoch=self.epoch):
                assert std_cams is not None
                cams_inter = std_cams
                with torch.no_grad():
                    seeds = self.mask_seed_builder(cams_inter)

                # seeds = self._cbox_filter_valid_tensors(seeds, valid)

            loss = self.loss(
                epoch=self.epoch,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                seeds=seeds,
                raw_scores=box,
                x_hat=x_hat,
                y_hat=y_hat,
                valid=valid,
                area=area,
                mask_fg=mask_fg,
                mask_bg=mask_bg,
                logits_fg=logits_fg,
                logits_bg=logits_bg,
                logits_clean=logits_clean,
                pre_x_hat=self.pre_forward_info['x_hat'],
                pre_y_hat=self.pre_forward_info['y_hat'],
                vl_size_priors=vl_size_priors
            )

            logits = logits_fg
        else:
            raise NotImplementedError

        return logits, loss

    def on_epoch_start(self):
        torch.cuda.empty_cache()

        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.model.train(mode=True)

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

    def on_epoch_end(self):
        self.loss.update_t()
        # todo: temp. delete later.
        self.loss.check_losses_status()

        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        DLLogger.log(fmsg('Train epoch runtime: {}'.format(delta_t)))

        torch.cuda.empty_cache()

    def random(self):
        self.counter = self.counter + 1
        self.seed = self.seed + self.counter
        set_seed(seed=self.seed, verbose=False)

    def reload_data_bucket(self, tr_bucket: int):

        loaders, train_sampler = get_data_loader(
            args=self.args,
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            dataset=self.args.dataset,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            tr_bucket=tr_bucket
        )

        self.train_sampler = train_sampler
        self.loaders[constants.TRAINSET] = loaders[constants.TRAINSET]

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch + tr_bucket)

    @staticmethod
    def _fill_minibatch(_x: torch.Tensor, mbatchsz: int) -> torch.Tensor:
        assert isinstance(_x, torch.Tensor)
        assert isinstance(mbatchsz, int)
        assert mbatchsz > 0
        assert _x.shape[0] <= mbatchsz, f'{_x.shape[0]}, {mbatchsz}'

        if _x.shape[0] == mbatchsz:
            return _x

        s = _x.shape[0]
        t = math.ceil(float(mbatchsz) / s)
        v = torch.cat(t * [_x])
        assert v.shape[1:] == _x.shape[1:]

        out = v[:mbatchsz]
        assert out.shape[0] == mbatchsz
        return out

    @staticmethod
    def _fill_minibatch_list(_x: list, mbatchsz: int) -> list:
        assert isinstance(_x, list)
        assert isinstance(mbatchsz, int)
        assert mbatchsz > 0
        assert len(_x) <= mbatchsz, f'{len(_x)}, {mbatchsz}'

        if len(_x) == mbatchsz:
            return _x

        s = len(_x)
        t = math.ceil(float(mbatchsz) / s)
        v = t * _x

        out = v[:mbatchsz]
        assert len(out) == mbatchsz
        return out

    def train(self, split, epoch):
        assert split == constants.TRAINSET

        self.epoch = epoch
        self.random()
        self.on_epoch_start()

        nbr_tr_bucket = self.args.nbr_buckets
        if not self.args.ds_chunkable:
            nbr_tr_bucket = 1

        loader = self.loaders[split]
        all_roi_thresholds: dict = self.roi_thresholds[split]

        total_loss = None
        num_correct = 0
        num_images = 0

        mbatchsz = 0

        scaler = GradScaler(enabled=self.args.amp)

        for bucket in range(nbr_tr_bucket):

            status = 0
            if self.args.ds_chunkable:
                if is_cc():
                    if self.args.distributed:
                        dist.barrier()
                    if self.args.is_node_master:
                        if bucket > 0:
                            delete_train(bucket=bucket - 1,
                                         dataset=self.args.dataset)

                        status = prepare_next_bucket(bucket=bucket,
                                                     dataset=self.args.dataset)
                        if (status == -1) and self.args.is_master:
                            DLLogger.log(f'Error in preparing bucket '
                                         f'{bucket}. Exiting.')

                if self.args.distributed:
                    dist.barrier()
                if status == -1:
                    sys.exit()
                self.reload_data_bucket(tr_bucket=bucket)
                loader = self.loaders[split]

            if self.args.distributed:
                dist.barrier()

            for batch_idx, (
                    images, targets, images_id, raw_imgs, std_cams) in tqdm(
                    enumerate(loader), ncols=constants.NCOLS,
                    total=len(loader), desc=f'BUCKET {bucket}/{nbr_tr_bucket}'):

                self.random()
                if (batch_idx == 0) and (bucket == 0):
                    mbatchsz = images.shape[0]

                vl_size_priors: dict = None
                if self._is_prior_size_needed():
                    vl_size_priors: Dict[str, Any] = \
                        self._build_mbatch_size_prior(targets)
                    for kz in vl_size_priors:
                        vl_size_priors[kz] = self._fill_minibatch(
                            vl_size_priors[kz], mbatchsz).cuda(
                            self.args.c_cudaid)

                images = self._fill_minibatch(images, mbatchsz)
                targets = self._fill_minibatch(targets, mbatchsz)
                raw_imgs = self._fill_minibatch(raw_imgs, mbatchsz)
                # images_id: tuple.
                images_id: list = list(images_id)
                images_id = self._fill_minibatch_list(images_id, mbatchsz)
                roi_thresholds = None
                if all_roi_thresholds is not None:
                    _val = [all_roi_thresholds[_idx] for _idx in images_id]
                    roi_thresholds = torch.tensor(
                        _val, dtype=torch.float, requires_grad=False).cuda(
                        self.args.c_cudaid)
                    # expected range [0, 1].
                    # todo: weak assertion.
                    assert 0 <= roi_thresholds.mean() <= 1.

                    roi_thresholds = roi_thresholds * 255.  # scale up.

                images = images.cuda(self.args.c_cudaid)
                targets = targets.cuda(self.args.c_cudaid)

                blured_imgs = None
                if self.args.task == constants.C_BOX:
                    blured_imgs = self.blur_op(images=images)

                if std_cams.ndim == 1:
                    std_cams = None
                else:
                    assert std_cams.ndim == 4
                    std_cams = self._fill_minibatch(std_cams, mbatchsz)
                    std_cams = std_cams.cuda(self.args.c_cudaid)

                    with autocast(enabled=self.args.amp):
                        with torch.no_grad():
                            std_cams = self.prepare_std_cams_disq(
                                std_cams=std_cams, image_size=images.shape[2:])

                self.optimizer.zero_grad(set_to_none=True)

                # with torch.autograd.set_detect_anomaly(True):
                with autocast(enabled=self.args.amp):
                    logits, loss = self._wsol_training(
                        images, raw_imgs, targets, std_cams, blured_imgs,
                        vl_size_priors, roi_thresholds=roi_thresholds)

                with torch.no_grad():
                    pred = logits.argmax(dim=1).detach()

                    if total_loss is None:
                        total_loss = loss.detach().squeeze() * images.size(0)
                    else:
                        total_loss += loss.detach().squeeze() * images.size(0)

                    num_correct += (pred == targets).sum().detach()
                    num_images += images.shape[0]

                if loss.requires_grad and torch.isfinite(loss).item():
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

        if self.args.distributed:
            num_correct = sync_tensor_across_gpus(num_correct.view(1, )).sum()
            nxx = torch.tensor([num_images], dtype=torch.float,
                               requires_grad=False, device=torch.device(
                    self.args.c_cudaid)).view(1, )
            num_images = sync_tensor_across_gpus(nxx).sum().item()
            total_loss = sync_tensor_across_gpus(total_loss.view(1, )).sum()
            dist.barrier()

        loss_average = total_loss.item() / float(num_images)
        classification_acc = num_correct.item() / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        self.on_epoch_end()

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def _build_mbatch_size_prior(self,
                                 glabels: torch.Tensor) -> Dict[str, Any]:
        assert self.vl_size_priors is not None
        assert glabels.ndim == 1

        out: Dict[str, Any] = dict()
        k_labels = list(self.vl_size_priors.keys())
        for k in self.vl_size_priors[k_labels[0]]:
            out[k] = torch.zeros_like(glabels, dtype=torch.float32,
                                      requires_grad=False)

        gl: np.ndarray = glabels.cpu().numpy()
        for i in range(gl.size):
            label = gl[i]
            for k in out:
                out[k][i] = self.vl_size_priors[label][k]

        return out

    def print_performances(self, checkpoint_type=None):
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    DLLogger.log(
                        "Split {}, metric {}, current value: {}".format(
                         split, metric, current_performance))
                    if split != constants.TESTSET:
                        DLLogger.log(
                            "Split {}, metric {}, best value: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_value))
                        DLLogger.log(
                            "Split {}, metric {}, best epoch: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_epoch))

    def serialize_perf_meter(self) -> dict:
        return {
            split: {
                metric: vars(self.performance_meters[split][metric])
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }

    def save_performances(self, epoch=None, checkpoint_type=None):
        tag = '' if checkpoint_type is None else '_{}'.format(checkpoint_type)

        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = '_Argmax_True'

        log_path = join(self.args.outd, 'performance_log{}{}.pickle'.format(
            tag, tagargmax))
        with open(log_path, 'wb') as f:
            pkl.dump(self.serialize_perf_meter(), f)

        log_path = join(self.args.outd, 'performance_log{}{}.txt'.format(
            tag, tagargmax))
        with open(log_path, 'w') as f:
            f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
                checkpoint_type, epoch, tagargmax))

            for split in self._SPLITS:
                for metric in self._EVAL_METRICS:

                    f.write("REPORT EPOCH/{}: split: {}/metric {}: {} \n"
                            "".format(epoch, split, metric,
                                      self.performance_meters[split][
                                          metric].current_value))
                    f.write(
                        "REPORT EPOCH/{}: split: {}/metric {}: {}_best "
                        "\n".format(epoch, split, metric,
                                    self.performance_meters[split][
                                        metric].best_value))
    @staticmethod
    def get_fg_imgs(images: torch.Tensor, blured_imgs: torch.Tensor,
                    mask_fg: torch.Tensor, mask_bg: torch.Tensor):
        assert images.ndim == 4
        assert mask_fg.shape[0] == images.shape[0]
        assert mask_fg.shape[1] == 1
        assert mask_fg.shape[2:] == images.shape[2:]
        assert mask_fg.shape == mask_bg.shape

        return mask_fg * images + mask_bg * blured_imgs

    @staticmethod
    def get_bg_imgs(images: torch.Tensor, blured_imgs: torch.Tensor,
                    mask_fg: torch.Tensor, mask_bg: torch.Tensor):
        assert images.ndim == 4
        assert mask_fg.shape[0] == images.shape[0]
        assert mask_fg.shape[1] == 1
        assert mask_fg.shape[2:] == images.shape[2:]
        assert mask_fg.shape == mask_bg.shape

        return mask_bg * images + mask_fg * blured_imgs

    def cl_forward(self, images: torch.Tensor, blured_imgs: torch.Tensor=None):
        output = self.model(images)

        if self.args.task == constants.STD_CL:
            cl_logits = output

        elif self.args.task in [constants.F_CL, constants.TCAM]:
            cl_logits, fcams, im_recon = output

        elif self.args.task == constants.C_BOX:
            box = output
            _, _, valid, _, mask_fg, mask_bg = self.box_stats(box=box,
                                                              eval=True)
            imgs_fg = self.get_fg_imgs(images=images, blured_imgs=blured_imgs,
                                       mask_fg=mask_fg, mask_bg=mask_bg)
            cl_logits = self.classifier(imgs_fg)
        else:
            raise NotImplementedError

        return cl_logits

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids, _, _) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)
            with torch.no_grad():
                blured_imgs = None
                with autocast(enabled=self.args.amp_eval):
                    if self.args.task == constants.C_BOX:
                        blured_imgs = self.blur_op(images=images)

                    cl_logits = self.cl_forward(images=images,
                                                blured_imgs=blured_imgs
                                                ).detach()

                pred = cl_logits.argmax(dim=1)
                num_correct += (pred == targets).sum().detach()
                num_images += images.size(0)

        # sync
        if self.args.distributed:
            num_correct = sync_tensor_across_gpus(num_correct.view(1, )).sum()
            nx = torch.tensor([num_images], dtype=torch.float,
                              requires_grad=False, device=torch.device(
                    self.args.c_cudaid)).view(1, )
            num_images = sync_tensor_across_gpus(nx).sum().item()
            dist.barrier()

        classification_acc = num_correct / float(num_images) * 100
        if self.args.distributed:
            dist.barrier()

        torch.cuda.empty_cache()
        return classification_acc.item()

    def evaluate(self, epoch, split, checkpoint_type=None, fcam_argmax=False):
        torch.cuda.empty_cache()

        if fcam_argmax:
            assert self.args.task in [constants.F_CL, constants.TCAM]

        self.fcam_argmax_previous = self.fcam_argmax
        self.fcam_argmax = fcam_argmax
        tagargmax = ''
        if self.args.task in [constants.F_CL, constants.TCAM]:
            tagargmax = 'Argmax {}'.format(fcam_argmax)

        DLLogger.log(fmsg("Evaluate: Epoch {} Split {} {}".format(
            epoch, split, tagargmax)))

        outd = None
        if split == constants.TESTSET:
            assert checkpoint_type is not None
            if fcam_argmax:
                outd = join(self.args.outd, checkpoint_type, 'argmax-true',
                            split)
            else:
                outd = join(self.args.outd, checkpoint_type, split)
            if not os.path.isdir(outd):
                os.makedirs(outd, exist_ok=True)

        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()
        self._pytorch_model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split][constants.CLASSIFICATION_MTR].update(
            accuracy)

        cam_curve_interval = self.args.cam_curve_interval
        cmdx = (split == constants.VALIDSET)
        cmdx &= self.args.dataset in [constants.CUB,
                                      constants.ILSVRC,
                                      constants.YTOV1]
        if cmdx:
            cam_curve_interval = constants.VALID_FAST_CAM_CURVE_INTERVAL

        cam_computer = CAMComputer(
            args=deepcopy(self.args),
            model=self._pytorch_model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=split,
            cam_curve_interval=cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=outd,
            fcam_argmax=fcam_argmax,
            classifier=self.classifier,
            box_stats=self.box_stats,
            blur_op=self.blur_op
        )

        t0 = dt.datetime.now()

        cam_performance = cam_computer.compute_and_evaluate_cams()

        DLLogger.log(fmsg("CAM EVALUATE TIME of {} split: {}".format(
            split, dt.datetime.now() - t0)))

        if self.args.task == constants.C_BOX:
            failed_bbox = cam_computer.get_failed_boxes_mtr()
            self.performance_meters[split][
                constants.FAILD_BOXES_MTR].update(failed_bbox)

        if split == constants.TESTSET and self.args.is_master:
            cam_computer.draw_some_best_pred(rename_ordered=True)

        if self.args.multi_iou_eval or (self.args.dataset ==
                                        constants.OpenImages):
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[split][constants.LOCALIZATION_MTR].update(
            loc_score)

        if self.args.dataset in [constants.CUB,
                                 constants.ILSVRC,
                                 constants.YTOV1]:
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    f'{constants.LOCALIZATION_MTR}_IOU_{IOU_THRESHOLD}'].update(
                    cam_performance[idx])

                self.performance_meters[split][
                    'top1_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top1[idx])

                self.performance_meters[split][
                    'top5_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top5[idx])

            if split == constants.TESTSET and self.args.is_master:
                curve_top_1_5 = cam_computer.evaluator.curve_top_1_5
                with open(join(outd, 'curves_top_1_5.pkl'), 'wb') as fc:
                    pkl.dump(curve_top_1_5, fc, protocol=pkl.HIGHEST_PROTOCOL)

                title = get_tag(self.args, checkpoint_type=checkpoint_type)
                title = 'Top1/5: {}'.format(title)

                if fcam_argmax:
                    title += '_argmax_true'
                else:
                    title += '_argmax_false'
                plot_perf_curves_top_1_5(curves=curve_top_1_5, fdout=outd,
                                              title=title)

        if split == constants.TESTSET and self.args.is_master:

            curves = cam_computer.evaluator.curve_s
            with open(join(outd, 'curves.pkl'), 'wb') as fc:
                pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

            title = get_tag(self.args, checkpoint_type=checkpoint_type)

            if fcam_argmax:
                title += '_argmax_true'
            else:
                title += '_argmax_false'
            self.plot_perf_curves(curves=curves, fdout=outd, title=title)

            with open(join(outd, f'thresholds-{checkpoint_type}.yaml'),
                      'w') as fth:
                yaml.dump({
                    'iou_threshold_list':
                        cam_computer.evaluator.iou_threshold_list,
                    'best_tau_list': cam_computer.evaluator.best_tau_list
                }, fth)

        torch.cuda.empty_cache()

    def plot_perf_curves_top_1_5(self, curves: dict, fdout: str, title: str):

        x_label = r'$\tau$'
        y_label = 'BoxAcc'

        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

        for i, top in enumerate(['top1', 'top5']):

            iouthres = sorted(list(curves[top].keys()))
            for iout in iouthres:
                axes[0, i].plot(curves['x'], curves[top][iout],
                                label=r'{}: $\sigma$={}'.format(top, iout))

            axes[0, i].xaxis.set_tick_params(labelsize=5)
            axes[0, i].yaxis.set_tick_params(labelsize=5)
            axes[0, i].set_xlabel(x_label, fontsize=8)
            axes[0, i].set_ylabel(y_label, fontsize=8)
            axes[0, i].grid(True)
            axes[0, i].legend(loc='best')
            axes[0, i].set_title(top)

        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                    dpi=300)

    @staticmethod
    def plot_perf_curves(curves: dict, fdout: str, title: str):

        bbox = True
        x_label = r'$\tau$'
        y_label = 'BoxAcc'
        if 'y' in curves:
            bbox = False
            x_label = 'Recall'
            y_label = 'Precision'

        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)

        if bbox:
            iouthres = sorted([kk for kk in curves.keys() if kk != 'x'])
            for iout in iouthres:
                ax.plot(curves['x'], curves[iout],
                        label=r'$\sigma$={}'.format(iout))
        else:
            ax.plot(curves['x'], curves['y'], color='tab:orange',
                    label='Precision/Recall')

        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.grid(True)
        plt.legend(loc='best')
        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_perf.png'), bbox_inches='tight',
                    dpi=300)

    def capture_perf_meters(self):
        self.perf_meters_backup = deepcopy(self.performance_meters)

    def switch_perf_meter_to_captured(self):
        self.performance_meters = deepcopy(self.perf_meters_backup)
        self.fcam_argmax = self.fcam_argmax_previous

    def save_args(self):
        self._save_args(path=join(self.args.outd, 'config_obj_final.yaml'))

    def _save_args(self, path):
        _path = path
        with open(_path, 'w') as f:
            self.args.tend = dt.datetime.now()
            yaml.dump(vars(self.args), f)

    @property
    def cpu_device(self):
        return get_cpu_device()

    def save_best_epoch(self, split):
        self.args.best_epoch_loc = self.performance_meters[split][
            constants.LOCALIZATION_MTR].best_epoch

        self.args.best_epoch_cl = self.performance_meters[split][
            constants.CLASSIFICATION_MTR].best_epoch

    def save_checkpoints(self, split):
        best_epoch = self.performance_meters[split][
            constants.LOCALIZATION_MTR].best_epoch

        self._save_model(checkpoint_type=constants.BEST_LOC, epoch=best_epoch)

        best_epoch = self.performance_meters[split][
            constants.CLASSIFICATION_MTR].best_epoch
        self._save_model(checkpoint_type=constants.BEST_CL, epoch=best_epoch)

    def _save_model(self, checkpoint_type, epoch):
        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]

        if checkpoint_type == constants.BEST_LOC:
            _model = deepcopy(self.best_model_loc).to(self.cpu_device).eval()
        elif checkpoint_type == constants.BEST_CL:
            _model = deepcopy(self.best_model_cl).to(self.cpu_device).eval()
        else:
            raise NotImplementedError

        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        if self.args.task == constants.STD_CL:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.classification_head.state_dict(),
                       join(path, 'classification_head.pt'))

        elif self.args.task in [constants.F_CL, constants.TCAM]:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.decoder.state_dict(), join(path, 'decoder.pt'))
            torch.save(_model.segmentation_head.state_dict(),
                       join(path, 'segmentation_head.pt'))
            if _model.reconstruction_head is not None:
                torch.save(_model.reconstruction_head.state_dict(),
                           join(path, 'reconstruction_head.pt'))

        elif self.args.task == constants.C_BOX:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.box_head.state_dict(), join(path, 'box_head.pt'))

        else:
            raise NotImplementedError

        self._save_args(path=join(path, 'config_model.yaml'))
        DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                             " {}".format(checkpoint_type, epoch, tag, path))

    def model_selection(self, epoch, split):
        assert split == constants.VALIDSET

        if (self.performance_meters[split][constants.LOCALIZATION_MTR]
                .best_epoch) == epoch:
            self.best_model_loc = deepcopy(self.model).to(
                self.cpu_device).eval()

        if (self.performance_meters[split][constants.CLASSIFICATION_MTR]
                .best_epoch) == epoch:
            self.best_model_cl = deepcopy(self.model).to(self.cpu_device).eval()

    def load_checkpoint(self, checkpoint_type):
        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]
        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)

        if self.args.task == constants.STD_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'classification_head.pt'),
                                 map_location=self.device)
            self.model.classification_head.load_state_dict(weights,
                                                           strict=True)

        elif self.args.task in [constants.F_CL, constants.TCAM]:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=self.device)
            self.model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=self.device)
            self.model.segmentation_head.load_state_dict(weights, strict=True)

            if self.model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=self.device)
                self.model.reconstruction_head.load_state_dict(weights,
                                                               strict=True)
        elif self.args.task == constants.C_BOX:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'box_head.pt'),
                                 map_location=self.device)
            self.model.box_head.load_state_dict(weights, strict=True)

        else:
            raise NotImplementedError

        DLLogger.log("Checkpoint {} loaded.".format(path))

    def report_train(self, train_performance, epoch, split=constants.TRAINSET):
        DLLogger.log('REPORT EPOCH/{}: {}/classification: {}'.format(
            epoch, split, train_performance['classification_acc']))
        DLLogger.log('REPORT EPOCH/{}: {}/loss: {}'.format(
            epoch, split, train_performance['loss']))

    def report(self, epoch, split, checkpoint_type=None):
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for metric in self._EVAL_METRICS:
            DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: {} ".format(
                epoch, split, metric,
                self.performance_meters[split][metric].current_value))
            DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: "
                         "{}_best ".format(
                          epoch, split, metric,
                          self.performance_meters[split][metric].best_value))

    def adjust_learning_rate(self):
        self.lr_scheduler.step()

    def plot_meter(self, metrics: dict, filename: str, title: str = '',
                   xlabel: str = '', best_iter: int = None):

        ncols = 4
        ks = list(metrics.keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = metrics[ks[t]]['value_per_epoch']
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                axes[i, j].set_title(ks[t], fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                # axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                if best_iter is not None:
                    axes[i, j].plot([x[best_iter]], [val[best_iter]],
                                    marker='o',
                                    markersize=5,
                                    color="red")
                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    @staticmethod
    def clean_metrics(metric: dict) -> dict:
        _metric = deepcopy(metric)
        l = []
        for k in _metric.keys():
            cd = (_metric[k]['value_per_epoch'] == [])
            cd |= (_metric[k]['value_per_epoch'] == [np.inf])
            cd |= (_metric[k]['value_per_epoch'] == [-np.inf])

            if cd:
                l.append(k)

        for k in l:
            _metric.pop(k, None)

        return _metric

    def plot_perfs_meter(self):
        meters = self.serialize_perf_meter()
        xlabel = 'epochs'

        best_epoch = self.performance_meters[constants.VALIDSET][
            self._BEST_CRITERION_METRIC].best_epoch

        for split in [constants.TRAINSET, constants.VALIDSET]:
            title = 'DS: {}, Split: {}, box_v2_metric: {}. Best iter.:' \
                    '{} {}'.format(
                     self.args.dataset, split, self.args.box_v2_metric,
                     best_epoch, xlabel)
            filename = '{}-{}-boxv2-{}'.format(
                self.args.dataset, split, self.args.box_v2_metric)
            self.plot_meter(
                self.clean_metrics(meters[split]), filename=filename,
                title=title, xlabel=xlabel, best_iter=best_epoch)
