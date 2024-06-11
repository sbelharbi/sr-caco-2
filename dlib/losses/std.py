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


__all__ = ['ClLoss']


class ClLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ClLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                # c-box
                raw_scores: torch.Tensor = None,
                x_hat: torch.Tensor = None,
                y_hat: torch.Tensor = None,
                valid: torch.Tensor = None,
                area: torch.Tensor = None,
                mask_fg: torch.Tensor = None,
                mask_bg: torch.Tensor = None,
                logits_fg: torch.Tensor = None,
                logits_bg: torch.Tensor = None,
                logits_clean: torch.Tensor = None,
                pre_x_hat: torch.Tensor = None,
                pre_y_hat: torch.Tensor = None
                ):
        super(ClLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        return self.loss(input=cl_logits, target=glabel) * self.lambda_