from collections import OrderedDict
import os
import sys
from os.path import dirname, abspath, join
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.models.select_network import define_G
from dlib.models.model_base import ModelBase

from dlib.losses.loss import CharbonnierLoss
from dlib.losses.loss_ssim import SSIMLoss

from dlib.utils.shared import safe_str_var
from dlib.utils.utils_model import test_mode
from dlib.utils.utils_regularizers import regularizer_orth
from dlib.utils.utils_regularizers import regularizer_clip
from dlib.utils import utils_instance as instance
from dlib.utils.tools import check_corruption

import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils import constants


# credit: https://github.com/cszn/KAIR


class ModelPlain(ModelBase):
    def __init__(self, args: object):
        super(ModelPlain, self).__init__(args)
        self.opt_train = self.args.train
        self.netG = define_G(args)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(args).to(self.device).eval()

        self.loss_fn = None
        self.args = args
    
    def flush(self):
        super(ModelPlain, self).flush()
        if hasattr(self.netG, 'flush'):
            self.netG.flush()
        # todo: NetE.

    def init_train(self):
        self.load()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.load_optimizers()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    def load(self):
        load_path_G = self.args.netG['checkpoint_path_netG']
        if os.path.isfile(load_path_G):
            DLLogger.log(fmsg(f'Loading model for G [{load_path_G:s}] ...'))
            self.load_network(load_path_G, self.netG,
                              strict=self.opt_train['G_param_strict'],
                              param_key='params')

        load_path_E = self.args.netG['checkpoint_path_netE']
        if self.opt_train['E_decay'] > 0:
            if os.path.isfile(load_path_E):
                DLLogger.log(fmsg(f'Loading model for E [{load_path_E:s}] ...'))
                self.load_network(load_path_E, self.netE,
                                  strict=self.opt_train['E_param_strict'],
                                  param_key='params_ema')
            else:
                DLLogger.log('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    def load_optimizers(self):
        load_path_optimizerG = self.args.netG['checkpoint_path_optimizerG']
        if os.path.isfile(load_path_optimizerG) and self.opt_train[
            'G_optimizer_reuse']:
            DLLogger.log(f'Loading optimizerG [{load_path_optimizerG:s}] ...')
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG',
                                iter_label)

    def save_current(self, save_dir: str):
        p_name_file = 'current_model.pth'
        os.makedirs(save_dir, exist_ok=True)

        self.save_network_path(self.netG, join(save_dir, f'G-{p_name_file}'))
        if self.opt_train['E_decay'] > 0:
            self.save_network_path(self.netE, join(save_dir,
                                                   f'E-{p_name_file}'))

    def load_current(self, save_dir: str):
        name = 'current_model.pth'
        load_path_G = join(save_dir, f"G-{name}")
        if os.path.isfile(load_path_G):
            DLLogger.log(fmsg(f'Loading current model for G [{load_path_G:s}] '
                              f'...'))
            self.load_network(load_path_G,
                              self.netG,
                              strict=True,
                              param_key='params')

        load_path_E = join(save_dir, f"E-{name}")
        if self.opt_train['E_decay'] > 0:
            if os.path.isfile(load_path_E):
                DLLogger.log(
                    fmsg(f'Loading current model for E [{load_path_E:s}] ...'))
                self.load_network(load_path_E,
                                  self.netE,
                                  strict=True,
                                  param_key='params_ema')

            self.netE.eval()

    def save_best(self, save_dir: str, p_name_file: str):
        os.makedirs(save_dir, exist_ok=True)

        self.save_network_path(self.netG, join(save_dir, f'G-{p_name_file}'))
        if self.opt_train['E_decay'] > 0:
            self.save_network_path(self.netE, join(save_dir,
                                                   f'E-{p_name_file}'))

    def define_loss(self):
        self.loss_fn = instance.define_loss(args=self.args)

    def define_optimizer(self):
        self.G_optimizer = instance.define_optimizer(args=self.args,
                                                     netG=self.netG)

    def define_scheduler(self):
        self.schedulers = instance.define_scheduler(
            args=self.args, G_optimizer=self.G_optimizer)

    def feed_data(self, data, need_H=True):

        if self.task == constants.SUPER_RES:
            self.L = data['l_im'].to(self.device)
            self.L_to_H = data['l_to_h_img'].to(self.device)
            self.Aug_L_to_H = data['l_to_h_img_aug'].to(self.device)

            if need_H:
                self.H = data['h_im'].to(self.device)
                if 'h_per_pixel_weight' in data:
                    self.h_per_pixel_weight = data['h_per_pixel_weight'].to(
                        self.device)

        elif self.task == constants.RECONSTRUCT:

            self.L = data['in_reconstruct'].to(self.device)
            assert need_H
            self.H = data['trg_reconstruct'].to(self.device)
            assert self.L.shape == self.H.shape, f"{self.L.shape} | " \
                                                 f"{self.H.shape}"

            self.L_to_H = data['in_reconstruct'].to(self.device)
            self.Aug_L_to_H = data['in_reconstruct'].to(self.device)

    def netG_forward(self):
        x = self.L
        nt = self.args.netG['net_type']
        nt = safe_str_var(nt)


        if self.args.method == constants.CSRCNN_MTH:
            nt_full = self.args.netG[f'{nt}_net_type']

            if nt_full != constants.NET_TYPE_PYRAMID:

                if self.args.augment:
                    x = self.Aug_L_to_H
                else:
                    x = self.L_to_H

        if self.args.method == constants.SRCNN_MTH:
            x = self.L_to_H

        if isinstance(self.netG, DDP) and not self.netG.training:
            self.E = self.netG.module(x)  # avoid timeout in eval mode.
        else:
            self.E = self.netG(x)

    def loss_srfbn(self, epoch: int) -> torch.Tensor:
        """
        Compute loss with curriculum learning strategy.
        :param epoch:
        :return:
        """
        nt = self.args.netG['net_type']
        assert nt == constants.SRFBN, nt

        nt = safe_str_var(nt)

        use_cl = self.args.netG[f'{nt}_use_cl']

        if use_cl:
            G_loss = 0.0
            t = 0.
            for e in self.netG.intermediate_outs:
                G_loss = G_loss + self.loss_fn(
                    epoch=epoch, y_pred=e, y_target=self.H,
                    trg_per_pixel_weight=self.h_per_pixel_weight,
                    model=self.netG
                )
                t = t + 1.

            G_loss = G_loss / float(t)

        else:
            G_loss = self.loss_fn(
                epoch=epoch, y_pred=self.E, y_target=self.H,
                trg_per_pixel_weight=self.h_per_pixel_weight,
                model=self.netG)

        return G_loss

    def loss_prosr(self, epoch: int) -> torch.Tensor:
        """
        PROSR network.

        Computer intermediate losses in addition to the final output loss.
        :param epoch:
        :return:
        """
        nt = self.args.netG['net_type']
        assert nt == constants.PROSR, nt

        # final output loss.
        G_loss = self.loss_fn(
            epoch=epoch, y_pred=self.E, y_target=self.H,
            trg_per_pixel_weight=self.h_per_pixel_weight,
            model=self.netG)

        for out in self.netG.intermediate_outs:

            assert out.ndim == 4, out.ndim
            #todo: weak. should used the true low resolution.
            gt = F.interpolate(
                input=self.H,
                size=out.shape[2:],
                mode='bicubic',
                align_corners=True
            )
            # todo: weak: original range may not be [0, 1]
            gt = torch.clamp(gt, min=0.0, max=1.)

            G_loss = G_loss + self.loss_fn(
                epoch=epoch, y_pred=out, y_target=gt,
                trg_per_pixel_weight=self.h_per_pixel_weight,
                model=self.netG
            )

        if len(self.netG.intermediate_outs) > 0:
            G_loss = G_loss / (len(self.netG.intermediate_outs) + 1.)

        return G_loss

    def loss_mslaprs(self, epoch: int) -> torch.Tensor:
        """
        MSLAPSR network.

        Computer intermediate losses in addition to the final output loss.
        :param epoch:
        :return:
        """
        nt = self.args.netG['net_type']
        assert nt == constants.MSLAPSR, nt

        # final output loss.
        G_loss = self.loss_fn(
            epoch=epoch, y_pred=self.E, y_target=self.H,
            trg_per_pixel_weight=self.h_per_pixel_weight,
            model=self.netG)

        for out in self.netG.intermediate_outs:

            assert out.ndim == 4, out.ndim
            #todo: weak. should used the true low resolution.
            gt = F.interpolate(
                input=self.H,
                size=out.shape[2:],
                mode='bicubic',
                align_corners=True
            )
            # todo: weak: original range may not be [0, 1]
            gt = torch.clamp(gt, min=0.0, max=1.)

            G_loss = G_loss + self.loss_fn(
                epoch=epoch, y_pred=out, y_target=gt,
                trg_per_pixel_weight=self.h_per_pixel_weight,
                model=self.netG
            )

        if len(self.netG.intermediate_outs) > 0:
            G_loss = G_loss / (len(self.netG.intermediate_outs) + 1.)

        return G_loss

    def optimize_parameters(self, epoch: int, current_step: int):

        net_type = self.args.netG['net_type']

        scaler = GradScaler(enabled=self.args.amp)

        self.G_optimizer.zero_grad()

        with autocast(enabled=self.args.amp):
            self.netG_forward()

            if net_type == constants.SRFBN:
                G_loss = self.loss_srfbn(epoch=epoch)

            elif net_type == constants.MSLAPSR:
                G_loss = self.loss_mslaprs(epoch=epoch)

            elif net_type == constants.PROSR:
                G_loss = self.loss_prosr(epoch=epoch)

            else:
                G_loss = self.loss_fn(
                    epoch=epoch, y_pred=self.E, y_target=self.H,
                    trg_per_pixel_weight=self.h_per_pixel_weight,
                    model=self.netG)

        if not G_loss.requires_grad or not torch.isfinite(G_loss).item():
            # skip update.
            return 0

        scaler.scale(G_loss).backward()

        if self.opt_train['G_optimizer_clipgrad']:
            G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad']
        else:
            G_optimizer_clipgrad = 0

        if G_optimizer_clipgrad > 0:
            scaler.unscale_(self.G_optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=G_optimizer_clipgrad,
                norm_type=2)

        scaler.step(self.G_optimizer)
        scaler.update()

        if self.opt_train['G_regularizer_orthstep']:
            G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep']
        else:
            G_regularizer_orthstep = 0

        cnd = (G_regularizer_orthstep > 0)
        # todo: check the need for this cond.
        # cnd &= (current_step % self.args.train['checkpoint_save'] != 0)

        if cnd and (current_step % G_regularizer_orthstep == 0):
            self.netG.apply(regularizer_orth)

        if self.opt_train['G_regularizer_clipstep']:
            G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep']
        else:
            G_regularizer_clipstep = 0

        cnd = (G_regularizer_clipstep > 0)
        # todo: check the need for this cond.
        # cnd &= (current_step % self.args.train['checkpoint_save'] != 0)

        if cnd and (current_step % G_regularizer_clipstep == 0):
            self.netG.apply(regularizer_clip)

        # todo: delete
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

        check_corruption(self)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    def testx8(self):
        # todo: delete??
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.args.scale,
                               modulo=1)
        self.netG.train()

    def set_eval_mode(self):
        self.netG.eval()

    def set_train_mode(self):
        self.netG.train(mode=True)

    def current_log(self):
        # todo: delete
        return self.log_dict

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float()
        out_dict['E'] = self.E.detach().float()
        if need_H:
            out_dict['H'] = self.H.detach().float()
        return out_dict

    def current_results(self, need_H=True):
        # batch.
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float()
        out_dict['E'] = self.E.detach().float()
        if need_H:
            out_dict['H'] = self.H.detach().float()
        return out_dict

    def print_network(self):
        msg = self.describe_network(self.netG)
        DLLogger.log(fmsg('Net info:'))
        DLLogger.log(msg)

    def print_params(self):
        msg = self.describe_params(self.netG)
        DLLogger.log(fmsg('Net params:'))
        DLLogger.log(msg)

    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
