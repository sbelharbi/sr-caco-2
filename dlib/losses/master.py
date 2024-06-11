import sys
from os.path import dirname, abspath

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.core import ElementaryLoss


__all__ = ['MasterLoss']


class MasterLoss(nn.Module):
    def __init__(self, cuda_id: int, name=None):
        super().__init__()
        self._name = name

        self.losses = []
        self.l_holder = []
        self.n_holder = [self.__name__]
        self._device = torch.device(cuda_id)

    def add(self, loss_: ElementaryLoss):
        self.losses.append(loss_)
        self.n_holder.append(loss_.__name__)

    def update_t(self):
        for loss in self.losses:
            loss.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

    def forward(self, **kwargs):
        assert self.losses != []

        self.l_holder = []
        for loss in self.losses:

            self.l_holder.append(loss(**kwargs).to(self._device))

        loss = sum(self.l_holder)
        self.l_holder = [loss] + self.l_holder
        return loss

    def to_device(self):
        for loss in self.losses:
            loss.to(self._device)

    def check_losses_status(self):
        print('-' * 60)
        print('Losses status:')

        for i, loss in enumerate(self.losses):
            if hasattr(loss, 'is_on'):
                print(self.n_holder[i+1], ': ... ',
                      loss.is_on(),
                      "({}, {})".format(loss.start_epoch, loss.end_epoch))
        print('-' * 60)

    def __str__(self):
        return "{}():".format(
            self.__class__.__name__, ", ".join(self.n_holder))


if __name__ == "__main__":
    from dlib.utils.utils_reproducibility import set_seed
    from dlib.losses.cbox import SeedCbox
    set_seed(seed=0)
    b, c = 10, 4
    cudaid = 1
    torch.cuda.set_device(cudaid)

    loss = MasterLoss(cuda_id=cudaid)
    print(loss.__name__, loss, loss.l_holder, loss.n_holder)
    loss.add(SeedCbox(cuda_id=cudaid))
    for l in loss.losses:
        print(l, isinstance(l, SeedCbox))

    for e in loss.n_holder:
        print(e)

