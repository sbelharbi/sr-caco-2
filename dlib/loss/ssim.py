import os.path
import sys
from os.path import dirname, abspath, join
from math import exp

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.loss.core import ElementaryLoss

# ref: https://github.com/Po-Hsun-Su/pytorch-ssim

__all__ = ['SSIMLoss']


def _gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(
            window_size)])
    return gauss/gauss.sum()


def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2,
                       groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/(
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = _create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel,
                     self.size_average)


def _ssim_value(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = _create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def tensor2uint82float(img: torch.Tensor) -> torch.Tensor:
    _img = (img.float().clamp(0, 1) * 255.0).round().clamp(0, 255).float()

    return _img

def micro():
    import sys
    from os.path import dirname, abspath, join
    from math import exp

    import torch.nn as nn
    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable
    import numpy as np
    import cv2
    from torch import optim
    from skimage import io
    import matplotlib.pyplot as plt

    from dlib.utils import utils_image

    sr_path = join(root_dir, 'data/debug/input/caco2/hr_div_1/'
                             'tile_HighRes1024-1_0_0_512_1792_2304_CELL2.tif')
    lr_path = join(root_dir, 'data/debug/input/caco2/hr_div_2/'
                             'tile_LowRes512-1_0_0_256_896_1152_CELL2.tif')
    assert os.path.isfile(sr_path), sr_path
    assert os.path.isfile(lr_path), lr_path
    np_lr = cv2.imread(lr_path, 0)
    np_sr = cv2.imread(sr_path, 0)
    print("sr sum in [0, 255]", np_sr.sum())

    sr = torch.from_numpy(np_sr).float().unsqueeze(0) / 255.0
    sr = sr.unsqueeze(0)
    lr = torch.from_numpy(np_lr).float().unsqueeze(0) / 255.0
    lr = lr.unsqueeze(0)
    hr = torch.rand(sr.size())

    print('cpu high', sr.min(), sr.max())
    print('cpu', lr.min(), lr.max())
    up = F.interpolate(
        input=lr,
        # size=[512, 512],
        scale_factor=2,
        mode='bicubic',
        # align_corners=True,
        antialias=True
    )
    print('cpu', up.min(), up.max())
    up = torch.clamp(up, 0.0, 1.0)
    print('cpu', up.min(), up.max())
    # hr = up * 1.

    if torch.cuda.is_available():
        sr = sr.cuda()
        lr = lr.cuda()
        hr = hr.cuda()
        up = up.cuda()

    print('gpu', lr.min(), lr.max())
    up = F.interpolate(
        input=lr,
        # size=[512, 512],
        scale_factor=2,
        mode='bicubic',
        # align_corners=True,
        antialias=True
    )
    print('gpu', up.min(), up.max())
    up = torch.clamp(up, 0.0, 1.0)
    print('gpu', up.min(), up.max(), up.sum())

    sr = Variable(sr, requires_grad=False)
    hr = Variable(hr, requires_grad=True)

    mse = nn.MSELoss(reduction='mean')
    loss = SSIMLoss()
    optimizer = optim.Adam([hr], lr=0.01)
    i = 0
    while i < 10:
        if i == -1:
            img = hr.detach().cpu().squeeze().float().numpy()
            img = np.uint8(np.clip(img * 255, 0, 255))
            plt.imshow(img)


        optimizer.zero_grad()
        down = F.interpolate(
            input=F.sigmoid(hr),
            size=[256, 256],
            mode='bilinear',
            # align_corners=True
        )
        if i == -1:
            plt.figure()
            d = down.detach().cpu().squeeze().float().numpy()
            plt.imshow(np.uint8(np.clip(d * 255, 0, 255)))
            plt.show()

        # l =  - loss(down, lr) + mse(down, lr)
        l = - loss(down, lr)

        print('loss: {:<4.8f}'.format(l))
        l.backward()
        optimizer.step()
        i += 1
        print(hr.min(), hr.max())


    img = F.sigmoid(hr).detach().cpu().squeeze().float().numpy()
    print("HR", img.min(), img.max())
    img = np.uint8(np.clip(img * 255, 0, 255))
    plt.imshow(img)
    # cv2.imwrite('out.tif', img)
    # plt.figure()
    img_up = up.detach().cpu().squeeze().float().numpy()
    img_up = np.uint8(np.clip(img_up * 255, 0, 255))

    img_up = tensor2uint82float(up.detach().cpu().squeeze()).numpy()
    print("img_up", img_up.min(), img_up.max(), img_up.sum())
    print("sr", np_sr.min(), np_sr.max(), np_sr.sum())
    print('unique up', np.unique(img_up))
    # plt.imshow(img_up)
    plt.figure()
    img_down = down.detach().cpu().squeeze().float().numpy()
    print("image down", img_down.min(), img_down.max())
    plt.imshow(np.uint8(np.clip(img_down * 255, 0, 255)))
    # plt.figure()
    # plt.imshow(np_sr)

    plt.show()


    psnr_mdl: torch.Tensor = utils_image.mbatch_gpu_calculate_psnr(
        torch.from_numpy(img).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(np_sr).unsqueeze(0).unsqueeze(0),
        border=0, roi=None)
    psnr_interp = utils_image.mbatch_gpu_calculate_psnr(
        torch.from_numpy(img_up).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(np_sr).unsqueeze(0).unsqueeze(0),
        border=0, roi=None)
    print(f"PSNR (model): {psnr_mdl}")
    print(f"PSNR (interp): {psnr_interp}")

def default():
    import cv2
    from torch import optim
    from skimage import io
    import matplotlib.pyplot as plt

    path_img = join(root_dir,
                    'data/debug/input/Black_Footed_Albatross_0002_55.jpg')
    npImg1 = cv2.imread(path_img)

    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)
    print(_ssim_value(img1.repeat(10, 1, 1, 1), img2.repeat(10, 1, 1, 1),
                      size_average=False
                      ).shape)

    ssim_value = _ssim_value(img1, img2).item()
    print("Initial ssim:", ssim_value)

    ssim_loss = SSIMLoss()
    # ssim_loss = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam([img2], lr=0.01)
    i = 0
    while ssim_value < 0.99:
        optimizer.zero_grad()
        ssim_out = -ssim_loss(img1, img2)
        ssim_value = -ssim_out.item()
        print('{:<4.4f}'.format(ssim_value))
        ssim_out.backward()
        optimizer.step()
        i += 1
        if i >= 100:
            break
    img = np.transpose(img2.detach().cpu().squeeze().float().numpy(), (1, 2, 0))
    plt.imshow(np.uint8(np.clip(img * 255, 0, 255)))
    plt.figure()
    plt.imshow(npImg1)
    plt.show()


if __name__ == '__main__':
    micro()

