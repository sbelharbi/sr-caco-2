import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple, Sequence

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


import torch


__all__ = ['BoxStats']


class BoxStats(torch.nn.Module):
    def __init__(self, scale_domain: float, h: int, w: int):
        super(BoxStats, self).__init__()

        assert scale_domain > 0
        assert isinstance(h, int)
        assert isinstance(w, int)
        assert h > 0
        assert w > 0

        self.h = h
        self.w = w
        self.scale_domain = float(scale_domain)

    def _get_float_full_hgrid(self, csize: int, s: int,
                              n: int, device):
        g = torch.arange(start=0, end=s, step=1, dtype=torch.float32,
                         device=device, requires_grad=False)

        g = g.view(-1, 1).repeat(1, n)  # s, n
        g = g.repeat(csize, 1, 1)  # c, s, n
        return g

    def _get_float_full_wgrid(self, csize: int, s: int,
                              n: int, device):
        g = torch.arange(start=0, end=s, step=1, dtype=torch.float32,
                         device=device, requires_grad=False)

        g = g.view(1, -1).repeat(n, 1)  # n, s
        g = g.repeat(csize, 1, 1)  # c, n, s
        return g

    def get_valid_box(self,
                      x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[1] == 2
        assert y.shape[1] == 2
        assert x.shape == y.shape

        v: torch.Tensor = (x[:, 1] > x[:, 0]).float()
        v = v * (y[:, 1] > y[:, 0]).float()  # b
        v = v * (x[:, 0] >= 0).float() * (x[:, 1] < self.h).float()
        v = v * (y[:, 0] >= 0).float() * (y[:, 1] < self.w).float()
        v = v.view(-1, 1)  # b, 1

        return v

    def get_area(self,
                 x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[1] == 2
        assert y.shape[1] == 2
        assert x.shape == y.shape

        area = x[:, 1] - x[:, 0]
        area = area * (y[:, 1] - y[:, 0])  # b
        area = area.view(-1, 1)  # b, 1

        return area

    def get_mask_fg(self,
                    x: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:

        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[1] == 2
        assert y.shape[1] == 2
        assert x.shape == y.shape

        b = x.shape[0]
        h, w = self.h, self.w
        device = x.device

        grid_h = self._get_float_full_hgrid(
            csize=1, s=h, n=w, device=device).unsqueeze(0)  # 1, 1, h, w
        grid_w = self._get_float_full_wgrid(
            csize=1, s=w, n=h, device=device).unsqueeze(0)  # 1, 1, h, w

        x1 = grid_h - x[:, 0].view(-1, 1, 1, 1)  # b, 1, h, w
        x2 = x[:, 1].view(-1, 1, 1, 1) - grid_h  # b, 1, h, w

        y1 = grid_w - y[:, 0].view(-1, 1, 1, 1)  # b, 1, h, w
        y2 = y[:, 1].view(-1, 1, 1, 1) - grid_w  # b, 1, h, w

        delta = torch.abs(x1) * torch.abs(x2) * torch.abs(y1) * torch.abs(y2)
        phi = torch.relu(x1) * torch.relu(x2) * torch.relu(y1) * torch.relu(y2)

        delta_d = delta.detach()

        mask = torch.zeros_like(input=phi, requires_grad=True)  # b, 1, h, w
        mask = (mask * 0.).type(phi.dtype)  # non-leaf variable
        mpos = delta_d > 0
        mzero = delta_d == 0
        mask[mpos] = phi[mpos] / delta_d[mpos]
        mask[mzero] = phi[mzero]

        return mask

    def get_mask_bg(self,
                    x: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:

        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[1] == 2
        assert y.shape[1] == 2
        assert x.shape == y.shape

        b = x.shape[0]
        h, w = self.h, self.w
        device = x.device

        grid_h = self._get_float_full_hgrid(
            csize=1, s=h, n=w, device=device).unsqueeze(0)  # 1, 1, h, w
        grid_w = self._get_float_full_wgrid(
            csize=1, s=w, n=h, device=device).unsqueeze(0)  # 1, 1, h, w

        x1 = x[:, 0].view(-1, 1, 1, 1) - grid_h  # b, 1, h, w
        x2 = grid_h - x[:, 1].view(-1, 1, 1, 1)  # b, 1, h, w

        y1 = y[:, 0].view(-1, 1, 1, 1) - grid_w  # b, 1, h, w
        y2 = grid_w - y[:, 1].view(-1, 1, 1, 1)  # b, 1, h, w

        Delta = (x1 > 0).float() * torch.abs(x1)
        Delta = Delta + (x2 > 0).float() * torch.abs(x2)
        Delta = Delta + (y1 > 0).float() * torch.abs(y1)
        Delta = Delta + (y2 > 0).float() * torch.abs(y2)

        psi = torch.relu(x1) + torch.relu(x2) + torch.relu(y1) + torch.relu(y2)

        Delta_d = Delta.detach()

        mask = torch.zeros_like(input=psi, requires_grad=True)  # b, 1, h, w
        mask = (mask * 0.).type(psi.dtype)  # non-leaf variable
        mpos = Delta_d > 0
        mzero = Delta_d == 0
        mask[mpos] = psi[mpos] / Delta_d[mpos]
        mask[mzero] = psi[mzero]

        return mask

    def _get_x_y(self, box: torch.Tensor, eval: bool = False
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert box.ndim == 2
        assert box.shape[1] == 4  # x1, y1, x2, y2.

        x = torch.cat((box[:, 0].view(-1, 1),
                       box[:, 2].view(-1, 1)), dim=1) / self.scale_domain
        y = torch.cat((box[:, 1].view(-1, 1),
                       box[:, 3].view(-1, 1)), dim=1) / self.scale_domain

        if eval:
            _x = torch.clamp(x, min=0., max=self.h - 1.)
            _y = torch.clamp(y, min=0., max=self.w - 1.)

            return _x, _y
        return x, y

    def forward(self, box: torch.Tensor,
                eval: bool = False) -> Sequence[torch.Tensor]:
        # box: n, 4
        assert box.ndim == 2
        assert box.shape[1] == 4  # x1, y1, x2, y2.

        x, y = self._get_x_y(box, eval)

        valid = self.get_valid_box(x=x, y=y)
        area = self.get_area(x=x, y=y)
        mask_fg = self.get_mask_fg(x=x, y=y)
        mask_bg = self.get_mask_bg(x=x, y=y)

        return x, y, valid, area, mask_fg, mask_bg


def draw_mask(mask, title: str):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.matshow(mask, cmap='gray')
    plt.title(title)
    plt.show()


def draw_both(outdir, h, w, x1, y1, x2, y2, sigma):
    _x1 = y1
    _y1 = x1
    _x2 = y2
    _y2 = x2

    DPI = 300
    fontp = {'family': 'serif',
             'color': 'red',
             'weight': 'normal',
             'size': 16,
             }

    fontc = {'family': 'serif',
             'color': 'red',
             'weight': 'normal',
             'size': 16,
             }

    linewidth = 2.
    bboxlinewidth = 3.


    x_ = np.linspace(0, w, w)
    y_ = np.linspace(0, h, h)

    X, Y = np.meshgrid(x_, y_)
    gaussian1 = np.exp(
        -(((X - _x1) ** 2) / (2 * sigma) + ((Y - _y1) ** 2) / (2 * sigma)))
    gaussian2 = np.exp(
        -(((X - _x2) ** 2) / (2 * sigma) + ((Y - _y2) ** 2) / (2 * sigma)))
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.matshow(gaussian1 + gaussian2, cmap="jet")

    point1x1 = [_x1, 0]
    point1y1 = [0, _y1]
    point1 = [_x1, _y1]

    point2x2 = [_x2, 0]
    point2y2 = [0, _y2]
    point2 = [_x2, _y2]

    plt.plot([point1x1[0], point1[0]], [point1x1[1], point1[1]], 'r--',
             linewidth=linewidth)
    plt.plot([point1y1[0], point1[0]], [point1y1[1], point1[1]], 'r--',
             linewidth=linewidth)

    plt.text(_x1 - 15, _y1 + 10, r'$p_1$', fontdict=fontp)
    plt.text(_x1 - 13, 10, r'$\hat{y}_1$', fontdict=fontc)
    plt.text(2, _y1 - 4, r'$\hat{x}_1$', fontdict=fontc)

    plt.plot([point2x2[0], point2[0]], [point2x2[1], point2[1]], 'r--',
             linewidth=linewidth)
    plt.plot([point2y2[0], point2[0]], [point2y2[1], point2[1]], 'r--',
             linewidth=linewidth)

    plt.text(_x2 - 15, _y2 + 10, r'$p_2$', fontdict=fontp)
    plt.text(_x2 - 13, 10, r'$\hat{y}_2$', fontdict=fontc)
    plt.text(2, _y2 - 4, r'$\hat{x}_2$', fontdict=fontc)

    point3 = [_x2, _y1]
    point4 = [_x1, _y2]

    plt.plot([point1[0], point3[0]], [point1[1], point3[1]], 'tab:orange',
             linewidth=bboxlinewidth)
    plt.plot([point1[0], point4[0]], [point1[1], point4[1]], 'tab:orange',
             linewidth=bboxlinewidth)

    plt.plot([point2[0], point3[0]], [point2[1], point3[1]], 'tab:orange',
             linewidth=bboxlinewidth)
    plt.plot([point2[0], point4[0]], [point2[1], point4[1]], 'tab:orange',
             linewidth=bboxlinewidth)

    plt.show()
    fig.savefig(join(outdir, 'both.png'), bbox_inches='tight', dpi=DPI,
                transparent=True)

def test_BoxStats():
    set_seed(0)
    torch.backends.cudnn.benchmark = True

    cuda = "0"
    device = torch.device(
        f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    h = 224
    w = 224
    x1 = 30
    x2 = 50
    y1 = 100
    y2 = 120
    d = 5
    c = 2
    b = 32
    scale_domain = 1.
    _x1 = torch.zeros((b, 1), device=device, requires_grad=True) * 0. + x1
    _y1 = torch.zeros((b, 1), device=device, requires_grad=True) * 0. + y1
    _x2 = torch.zeros((b, 1), device=device, requires_grad=True) * 0. + x2
    _y2 = torch.zeros((b, 1), device=device, requires_grad=True) * 0. + y2
    box = torch.cat((_x1, _y1, _x2, _y2), dim=1)
    sts = BoxStats(scale_domain=scale_domain, h=h, w=w)
    x_hat, y_hat, v, area, mask_fg, mask_bg = sts(box)

    print(f'true x1: {x1}  predicted {x_hat[0, 0]}')
    print(f'true x2: {x2}  predicted {x_hat[0, 1]}')
    print(f'true y1: {y1}  predicted {y_hat[0, 0]}')
    print(f'true y2: {y2}  predicted {y_hat[0, 1]}')
    print(f'true area: {(x2 - x1) * (y2 - y1)}  predicted area: {area[0]}')

    # plot
    draw_both('', h=h, w=w, x1=x_hat[0, 0].item(), y1=y_hat[0, 0].item(),
              x2=x_hat[0, 1].item(),
              y2=y_hat[0, 1].item(), sigma=0.1)

    draw_mask(mask_fg[0].squeeze().cpu().detach().numpy().astype(np.uint8) *
              255, title='Foreground')
    draw_mask(mask_bg[0].squeeze().cpu().detach().numpy().astype(np.uint8) *
              255, title='Background')


if __name__ == '__main__':
    import math

    from dlib.utils.shared import announce_msg
    from dlib.utils.utils_reproducibility import set_seed

    from os.path import join
    import matplotlib.pyplot as plt
    import numpy as np

    test_BoxStats()



