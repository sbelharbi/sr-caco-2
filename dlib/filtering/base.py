import sys
import os
import time
from os.path import dirname, abspath, join
import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from kornia.filters import GaussianBlur2d
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


class GaussianFiltering(nn.Module):
    def __init__(self, blur_ksize: int, blur_sigma: float, device):
        """
        Init. function.
        :param blur_ksize: int. Odd int. the size of squared kernel. height
        or width.
        :param blur_sigma: float. variance of the Gaussian.
        """
        super(GaussianFiltering, self).__init__()

        assert isinstance(blur_ksize, int)
        assert blur_ksize > 0
        assert blur_ksize % 2 == 1
        assert blur_sigma > 0

        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self._device = device

        self.filter = GaussianBlur2d(kernel_size=(blur_ksize, blur_ksize),
                                     sigma=(blur_sigma, blur_sigma),
                                     border_type='reflect').to(device)

    def forward(self, images):
        """
        :param images: torch tensor of the image (values in [0, 255]). shape
        N*C*H*W on the same device as this class.
        :return: filtered images on the same device.
        """
        filtered = self.filter(images)
        assert filtered.shape == images.shape

        return filtered

    def extra_repr(self):
        return 'kernel size={}, sigma={}'.format(self.blur_ksize,
                                                 self.blur_sigma)


def test_GaussianFiltering():
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
    b = 32
    images_normalized = image_normalized.repeat(b, 1, 1, 1)
    l_imgs = [(img, 'Input')]

    blur_ksize = 3
    for blur_sigma in [2.]:
        filter = GaussianFiltering(blur_ksize=blur_ksize,
                                    blur_sigma=blur_sigma,
                                    device=DEVICE).to(DEVICE)
        announce_msg("testing {}".format(filter))
        set_seed(seed=seed)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with autocast(enabled=False):
            z = filter(images_normalized.to(DEVICE))

        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        print('time op: {}'.format(elapsed_time_ms))
        l_imgs.append((filtered_2Img(z[0]), f'ksize:{blur_ksize} sigma:'
                                            f' {blur_sigma}'))
        t = dt.datetime.now()
        z = filter(images_normalized.to(DEVICE))
        print(f'time : {dt.datetime.now() - t}')
    plot_limgs(_lims=l_imgs, title='Variation.')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    test_GaussianFiltering()
