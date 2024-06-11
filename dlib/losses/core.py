import sys
from os.path import dirname, abspath

import re
import torch.nn as nn
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB

__all__ = [
    'ElementaryLoss'
]


class ElementaryLoss(nn.Module):
    def __init__(self,
                 cuda_id,
                 name=None,
                 lambda_=1.,
                 elb=nn.Identity(),
                 support_background=False,
                 multi_label_flag=False,
                 sigma_rgb=15.,
                 sigma_xy=100.,
                 scale_factor=0.5,
                 start_epoch=None,
                 end_epoch=None,
                 seg_ignore_idx=-255
                 ):
        super(ElementaryLoss, self).__init__()
        self._name = name
        self.lambda_ = lambda_
        self.elb = elb
        self.support_background = support_background

        assert not multi_label_flag
        self.multi_label_flag = multi_label_flag

        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

        if end_epoch == -1:
            end_epoch = None

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.c_epoch = 0

        self.loss = None
        self._device = torch.device(cuda_id)

        self._zero = torch.tensor([0.0], device=self._device,
                                  requires_grad=False, dtype=torch.float)

        self.seg_ignore_idx = seg_ignore_idx

    def is_on(self, _epoch=None):
        if _epoch is None:
            c_epoch = self.c_epoch
        else:
            assert isinstance(_epoch, int)
            c_epoch = _epoch

        if (self.start_epoch is None) and (self.end_epoch is None):
            return True

        l = [c_epoch, self.start_epoch, self.end_epoch]
        if all([isinstance(z, int) for z in l]):
            return self.start_epoch <= c_epoch <= self.end_epoch

        if self.start_epoch is None and isinstance(self.end_epoch, int):
            return c_epoch <= self.end_epoch

        if isinstance(self.start_epoch, int) and self.end_epoch is None:
            return c_epoch >= self.start_epoch

        return False

    def unpacke_low_cams(self, cams_low, glabel):
        n = cams_low.shape[0]
        select_lcams = [None for _ in range(n)]

        for i in range(n):
            llabels = [glabel[i]]

            if self.support_background:
                llabels = [xx + 1 for xx in llabels]
                llabels = [0] + llabels

            for l in llabels:
                tmp = cams_low[i, l, :, :].unsqueeze(
                        0).unsqueeze(0)
                if select_lcams[i] is None:
                    select_lcams[i] = tmp
                else:
                    select_lcams[i] = torch.cat((select_lcams[i], tmp), dim=1)

        return select_lcams

    def update_t(self):
        if isinstance(self.elb, ELB):
            self.elb.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            out = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            return out
        else:
            return self._name

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                # fcam + cbox.
                seeds=None
                ):
        self.c_epoch = epoch

