import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple, Sequence, Dict

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.core import ElementaryLoss


__all__ = [
    'AreaBox',
    'BoxBounds',
    'ClScoring',
    'SeedCbox'
]


class AreaBox(ElementaryLoss):
    def __init__(self, **kwargs):
        super(AreaBox, self).__init__(**kwargs)

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
                # fcam + cbox.
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
                ) -> torch.Tensor:
        super(AreaBox, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert area.ndim == 2
        assert valid.ndim == 2
        assert area.shape == valid.shape
        assert area.shape[1] == 1
        assert mask_fg.ndim == 4
        assert mask_fg.shape[0] == area.shape[0]
        assert mask_fg.shape[1] == 1
        assert mask_fg.shape[2:] == raw_img.shape[2:]

        idx_valid = torch.nonzero(valid.contiguous().view(-1, ),
                                  as_tuple=False).squeeze()

        if idx_valid.numel() == 0:
            return self._zero

        _v_areas = area.contiguous().view(-1, )[
            idx_valid].contiguous().view(-1, 1)  # ?, 1 | ? > 0
        b, c, h, w = mask_fg.shape
        if self.cb_area_normed:
            t = 1.
            _v_areas = _v_areas / float(h * w)
        else:
            t = float(h * w)

        a = torch.vstack((- _v_areas,
                          _v_areas - t))  # 2 * ?, 1

        loss = self.elb(a.view(-1, ))

        return self.lambda_ * loss


class ClScoring(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ClScoring, self).__init__(**kwargs)

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
                # fcam + cbox.
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
                pre_y_hat: torch.Tensor = None,
                vl_size_priors: dict = None
                ) -> torch.Tensor:
        super(ClScoring, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert logits_fg.ndim == 2
        assert logits_bg.shape == logits_fg.shape
        assert logits_clean.shape == logits_fg.shape

        assert valid.ndim == 2
        assert valid.shape[1] == 1
        assert valid.shape[0] == logits_fg.shape[0]

        assert glabel.shape[0] == logits_fg.shape[0]

        idx_valid = torch.nonzero(valid.contiguous().view(-1, ),
                                  as_tuple=False).squeeze()
        if idx_valid.numel() == 0:
            return self._zero

        _logits_fg = logits_fg[idx_valid]  # ?, c | c
        _logits_bg = logits_bg[idx_valid]  # ?, c | c
        _logits_clean = logits_clean[idx_valid]  # ?, c | c
        _glabel = glabel[idx_valid]  # ?

        if idx_valid.numel() == 1:
            _logits_fg = _logits_fg.unsqueeze(0)  # ?, c | ? > 0
            _logits_bg = _logits_bg.unsqueeze(0)  # ?, c | ? > 0
            _logits_clean = _logits_clean.unsqueeze(0)  # ?, c | ? > 0
            _glabel = _glabel.view(1, )  # ?

        _fg = _logits_fg.gather(1, _glabel.view(-1, 1))  # ?, 1
        _bg = _logits_bg.gather(1, _glabel.view(-1, 1))  # ?, 1
        _cl = _logits_clean.gather(1, _glabel.view(-1, 1))  # ?, 1

        e = torch.vstack((_cl - _fg,
                          _bg - _cl))  # 2 * ?, 1

        loss = self.elb(e.view(-1, ))  # []
        print(f'loss classifier response {loss}')

        return self.lambda_ * loss


class SeedCbox(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SeedCbox, self).__init__(**kwargs)

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
                # fcam + cbox.
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
                pre_y_hat: torch.Tensor = None,
                vl_size_priors: dict = None
                ):
        super(SeedCbox, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        idx_valid = torch.nonzero(valid.contiguous().view(-1, ),
                                  as_tuple=False).squeeze()
        if idx_valid.numel() == 0:
            return self._zero

        _seeds = seeds[idx_valid]  # ?, h, w | h, w
        _mask_fg = mask_fg[idx_valid]  # ?, 1, h, w | 1, h, w
        _mask_bg = mask_bg[idx_valid]  # ?, 1, h, w | 1, h, w

        if idx_valid.numel() == 1:
            _seeds = _seeds.unsqueeze(0)  # ?, h, w | ? > 0
            _mask_fg = _mask_fg.unsqueeze(0)  # ?, 1, h, w | ? > 0
            _mask_bg = _mask_bg.unsqueeze(0)  # ?, 1, h, w | ? > 0

        seg = torch.cat((_mask_bg, _mask_fg), dim=1)  # ?, 2, h, w

        return self.lambda_ * self.loss(input=seg, target=_seeds)


class BoxBounds(ElementaryLoss):
    def __init__(self, **kwargs):
        super(BoxBounds, self).__init__(**kwargs)

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
                # fcam + cbox.
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
                pre_y_hat: torch.Tensor = None,
                vl_size_priors: dict = None
                ) -> torch.Tensor:
        super(BoxBounds, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert x_hat.ndim == 2
        assert y_hat.ndim == 2

        assert pre_x_hat.shape == x_hat.shape
        assert pre_y_hat.shape == y_hat.shape
        assert pre_y_hat.shape == pre_x_hat.shape

        # print(x_hat[:10, :])
        # print(pre_x_hat[:10, :])
        # print(y_hat[:10, :])
        # print(pre_y_hat[:10, :])
        # input('ok')

        p = torch.cat((x_hat.contiguous().view(-1, ),
                       y_hat.contiguous().view(-1, )), dim=0)  # n

        pre = torch.cat((pre_x_hat.contiguous().view(-1, ),
                         pre_y_hat.contiguous().view(-1, )), dim=0)  # n

        box_diff = pre - p
        abs_box_diff = torch.abs(box_diff)
        smoothL1_sign = (abs_box_diff < 1.).detach().float()
        _loss = torch.pow(box_diff, 2) * 0.5 * smoothL1_sign + (
                abs_box_diff - 0.5) * (1. - smoothL1_sign)

        loss = _loss.mean()

        return self.lambda_ * loss
