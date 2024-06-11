import os
import sys
from os.path import dirname, abspath, join
from copy import deepcopy
import datetime as dt

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.utils_bnorm import merge_bn
from dlib.utils.utils_bnorm import tidy_sequential
from dlib.utils import constants

from dlib.utils.utils_reproducibility import set_seed
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


# credit: https://github.com/cszn/KAIR


class ModelBase():
    def __init__(self, args: object):
        self.args = args
        self.task = args.task
        self.save_dir = join(args.outd_backup, args.save_dir_models)
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.is_train = args.is_train
        self.schedulers = []

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()

        self.E = None
        self.H = None
        self.L = None

        # CSR-CNN.
        self.L_to_H = None
        self.Aug_L_to_H = None
        self.h_per_pixel_weight = None

    def flush(self):
        self.E = None
        self.H = None
        self.L = None

        # CSR-CNN.
        self.L_to_H = None
        self.Aug_L_to_H = None
        self.h_per_pixel_weight = None

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def save_best(self):
        pass

    def save_current(self):
        pass

    def load_current(self):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    def feed_data(self):
        pass

    def optimizer_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def set_eval_mode(self):
        pass

    def set_train_mode(self):
        pass

    @staticmethod
    def get_bare_model(network):
        # todo: use myddp if need access to model attributes.
        if isinstance(network, DDP):
            return network.module

        return network

    def model_to_device(self, network):
        network = network.to(self.device)

        # todo: use myddp if need access to model attributes.
        if self.args.distributed:
            network = DDP(network, device_ids=[torch.cuda.current_device()])

        return network

    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += f'Networks name: {network.__class__.__name__}\n'
        msg += f'Params number: ' \
               f'{sum(map(lambda x: x.numel(), network.parameters()))}\n'
        # msg += f'Net structure:\n{str(network)}\n'
        return msg

    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}\n'.format(
            'mean', 'min', 'max', 'std', 'shape', 'param_name')
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += f' | {v.mean():>6.3f} | {v.min():>6.3f} | ' \
                       f'{v.max():>6.3f} | {v.std():>6.3f} | {v.shape} ' \
                       f'|| {name:s}\n'
        return msg

    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = f'{iter_label}_{network_label}.pth'
        save_path = os.path.join(save_dir, save_filename)

        self.save_network_path(network, save_path)

    def save_network_path(self, network, path: str):
        network = self.get_bare_model(network)
        net_type = self.args.netG['net_type']
        if net_type in [constants.DSRSPLINES, constants.CSRCNN]:
            network.flush()

        cpu_device = torch.device('cpu')
        _network = deepcopy(network).to(cpu_device).eval()
        torch.save(_network.state_dict(), path)

    def load_network(self, load_path, network, strict=True, param_key='params'):

        network = self.get_bare_model(network)
        map_location = next(network.parameters()).device

        if strict:
            state_dict = torch.load(load_path, map_location=map_location)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path, map_location=map_location)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old), (key, param)) in zip(
                    state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)

    @staticmethod
    def save_optimizer(save_dir, optimizer, optimizer_label, iter_label):
        save_filename = f'{iter_label}_{optimizer_label}.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    @staticmethod
    def load_optimizer(load_path, optimizer):
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        optimizer.load_state_dict(torch.load(load_path, map_location=device))

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data,
                                                 alpha=1-decay)

    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)



