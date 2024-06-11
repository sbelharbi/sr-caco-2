import sys
from os.path import dirname, abspath
from typing import Tuple

import re
import torch.nn as nn
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.loss.elb import ELB

__all__ = ['ElementaryLoss']


class ElementaryLoss(nn.Module):
    def __init__(self,
                 cuda_id,
                 name: str = None,
                 lambda_: float = 1.,
                 elb=nn.Identity(),
                 start_epoch: int = None,
                 end_epoch: int = None,
                 restore_range: bool = False,
                 color_min: int = 0,
                 color_max: int = 255,
                 use_residuals: bool = False
                 ):
        super(ElementaryLoss, self).__init__()
        self._name = name
        self.lambda_ = lambda_
        self.elb = elb

        if end_epoch == -1:
            end_epoch = None

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.c_epoch = 0

        self.loss = None
        self._device = torch.device(cuda_id)

        self._zero = torch.tensor([0.0], device=self._device,
                                  requires_grad=False, dtype=torch.float)

        self.restore_range = restore_range

        assert isinstance(color_min, int), type(color_min)
        assert isinstance(color_max, int), type(color_max)
        assert color_min < color_max, f'{color_min}, {color_max}'

        self.color_min = color_min
        self.color_max = color_max
        self.use_residuals = use_residuals

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

    @staticmethod
    def sanity_check_trg_per_pixel_weight(
            trg_per_pixel_weight: torch.Tensor, y_target: torch.Tensor):

        assert y_target.ndim == 4, y_target.ndim
        assert trg_per_pixel_weight.ndim == 4, trg_per_pixel_weight.ndim

        b, c, h, w = y_target.shape
        _b, _c, _h, _w = trg_per_pixel_weight.shape
        assert _c == 1, _c
        assert [b, h, w] == [_b, _h, _w], f'{[b, h, w]}, {[_b, _h, _w]}'

    @staticmethod
    def check_residual(model):
        assert model is not None
        assert model.x_interp is not None
        assert model.global_residual is not None

    def switch_to_residuals(self,
                            model: nn.Module,
                            y_target: torch.Tensor,
                            y_pred: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_residuals:
            self.check_residual(model)
            r = y_target - model.x_interp
            r_hat = model.global_residual

            _y_pred = r_hat
            _y_target = r

        else:
            _y_pred = y_pred
            _y_target = y_target

        return _y_pred, _y_target

    def forward(self,
                epoch: int,
                y_pred: torch.Tensor = None,
                y_target: torch.Tensor = None,
                trg_per_pixel_weight: torch.Tensor = None,
                model: nn.Module = None
                ):
        self.c_epoch = epoch

