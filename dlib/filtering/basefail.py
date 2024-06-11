import sys
import os
import time
from os.path import dirname, abspath, join
import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F

from torch.cuda.amp import custom_fwd
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

sys.path.append(
    join(root_dir,
         "crf/crfwrapper/bilateralfilter/build/lib.linux-x86_64-3.7")
)

from bilateralfilter import bilateralfilter, bilateralfilter_batch


class BilateralFilteringFunc(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,
                images_raw,
                images_normalized,
                sigma_rgb,
                sigma_xy
                ):

        device = images_normalized.device

        n, k, h, w = images_normalized.shape
        ctx.N, ctx.K, ctx.H, ctx.W = n, k, h, w

        images = images_raw.numpy().flatten()
        images_tensor_np = images_normalized.cpu().numpy().flatten()
        AS = np.zeros(images_tensor_np.shape, dtype=np.float32)
        bilateralfilter_batch(images, images_tensor_np, AS, ctx.N, ctx.K,
                              ctx.H, ctx.W, sigma_rgb, sigma_xy)

        filtered_img = torch.from_numpy(AS).to(device)
        print(filtered_img.shape)
        filtered_img = filtered_img.contiguous().view(ctx.N, ctx.K, ctx.H,
                                                      ctx.W)

        return filtered_img / filtered_img.max()


class BilateralFiltering(nn.Module):
    def __init__(self, sigma_rgb, sigma_xy):
        """
        Init. function.
        :param sigma_rgb: float. sigma for the bilateheral filtering (
        appearance kernel): color similarity.
        :param sigma_xy: float. sigma for the bilateral filtering
        (appearance kernel): proximity.
        """
        super(BilateralFiltering, self).__init__()
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy

    def forward(self, images_raw, images_normalized):
        """
        Forward loss.
        Image and segmentation are scaled with the same factor.

        :param images_raw: torch tensor of the image (values in [0, 255]). shape
        N*C*H*W. DEVICE: CPU.
        :param images_normalized: same as image_raw but normalized. device
        :return: filtered images_normalized on the same device.
        """
        assert images_raw.shape == images_normalized.shape
        filtered = BilateralFilteringFunc.apply(
            images_raw,
            images_normalized,
            self.sigma_rgb,
            self.sigma_xy
        )
        assert filtered.shape == images_normalized.shape

        return filtered

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}'.format(self.sigma_rgb, self.sigma_xy)


def test_BilateralFiltering():
    import time

    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    from dlib.utils.utils_reproducibility import set_seed
    from dlib.utils.shared import announce_msg
    from dlib.functional import _functional as dlibf

    from torch.profiler import profile, record_function, ProfilerActivity


    def filtered_2Img(_img):
        tmp = _img.squeeze().cpu().numpy() * 255
        tmp = tmp.astype(np.uint8).transpose(1, 2, 0)
        print(tmp.shape)
        return Image.fromarray(tmp, mode='RGB')


    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        wim, him = _lims[0][0].size
        r = him / float(wim)
        fw = 20
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im)
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()


    seed = 0
    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    set_seed(seed=seed)
    path_imng = join(root_dir, 'data/debug/input',
                     'Black_Footed_Albatross_0002_55.jpg')
    img = Image.open(path_imng, 'r').convert('RGB')
    image_raw = np.array(img, dtype=np.float32)  # h, w, 3
    print(image_raw.shape)
    image_raw = dlibf.to_tensor(image_raw).permute(2, 0, 1)  # 3, h, w.
    totensor = transforms.ToTensor()
    image_normalized = totensor(img)  # 3, h, w
    print(image_normalized.shape, image_raw.shape)
    b = 2
    images_raw = image_raw.unsqueeze(0)
    images_normalized = image_normalized.unsqueeze(0)
    l_imgs = [(img, 'Input'), (filtered_2Img(images_normalized[0]), 'A')]

    sigma_rgb = 1.
    for sigma_xy in [1.]:
        filter = BilateralFiltering(sigma_rgb=sigma_rgb,
                                    sigma_xy=sigma_xy).to(DEVICE)
        announce_msg("testing {}".format(filter))
        set_seed(seed=seed)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with autocast(enabled=False):
            z = filter(images_raw=images_raw,
                       images_normalized=images_normalized)

        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        print('time op: {}'.format(elapsed_time_ms))
        l_imgs.append((filtered_2Img(z[0]), f'xy:{sigma_xy} rgb: {sigma_rgb}'))
    plot_limgs(_lims=l_imgs, title='Variation.')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    test_BilateralFiltering()
