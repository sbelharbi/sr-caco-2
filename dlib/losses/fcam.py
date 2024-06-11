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
from dlib.losses.entropy import Entropy
from dlib.crf.dense_crf_loss import DenseCRFLoss


__all__ = [
    'ImgReconstruction',
    'SelfLearningFcams',
    'ConRanFieldFcams',
    'EntropyFcams',
    'MaxSizePositiveFcams'
]


class ImgReconstruction(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ImgReconstruction, self).__init__(**kwargs)

        self.loss = nn.MSELoss(reduction="none").to(self._device)

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
                maps_raw: torch.Tensor = None,
                x_hat: torch.Tensor = None,
                x_var: torch.Tensor = None,
                y_hat: torch.Tensor = None,
                y_var: torch.Tensor = None,
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
        super(ImgReconstruction, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        n = x_in.shape[0]
        loss = self.elb(self.loss(x_in, im_recon).view(n, -1).mean(
            dim=1).view(-1, ))
        return self.lambda_ * loss.mean()


class SelfLearningFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SelfLearningFcams, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

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
                maps_raw: torch.Tensor = None,
                x_hat: torch.Tensor = None,
                x_var: torch.Tensor = None,
                y_hat: torch.Tensor = None,
                y_var: torch.Tensor = None,
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
        super(SelfLearningFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=fcams, target=seeds) * self.lambda_


class ConRanFieldFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldFcams, self).__init__(**kwargs)

        self.loss = DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

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
                maps_raw: torch.Tensor = None,
                x_hat: torch.Tensor = None,
                x_var: torch.Tensor = None,
                y_hat: torch.Tensor = None,
                y_var: torch.Tensor = None,
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
        super(ConRanFieldFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)


class EntropyFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EntropyFcams, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

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
                maps_raw: torch.Tensor = None,
                x_hat: torch.Tensor = None,
                x_var: torch.Tensor = None,
                y_hat: torch.Tensor = None,
                y_var: torch.Tensor = None,
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
        super(EntropyFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert fcams.ndim == 4
        bsz, c, h, w = fcams.shape

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        probs = fcams_n.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c)

        return self.lambda_ * self.loss(probs).mean()


class MaxSizePositiveFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxSizePositiveFcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

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
                maps_raw: torch.Tensor = None,
                x_hat: torch.Tensor = None,
                x_var: torch.Tensor = None,
                y_hat: torch.Tensor = None,
                y_var: torch.Tensor = None,
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
        super(MaxSizePositiveFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1. / 2.)
