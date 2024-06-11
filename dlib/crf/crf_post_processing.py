import sys
import os
import time
from os.path import dirname, abspath, join
import datetime as dt

import numpy as np
import pydensecrf.densecrf as dcrf

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

__all__ = ['DenseCRFFilter']


class DenseCRFFilter(object):
    def __init__(self, sigma_rgb: float, sigma_xy: float,
                 scale_factor: float, itera: int):
        """
        Init. function.
        :param sigma_rgb: float. sigma for the bilateheral filtering (
        appearance kernel): color similarity.
        :param sigma_xy: float. sigma for the bilateral filtering
        (appearance kernel): proximity.
        :param scale_factor: float. ratio to scale the image and
        segmentation. Helpful to control the computation (speed) / precision.
        :param itera: number of iterations for refinement.
        """
        super(DenseCRFFilter, self).__init__()
        self.sigma_rgb = int(sigma_rgb)
        self.sigma_xy = int(sigma_xy)
        self.scale_factor = scale_factor
        assert isinstance(itera, int)
        assert itera >= 0
        self.itera = itera

    def __call__(self,
                 images: torch.Tensor,
                 segmentations: torch.Tensor) -> torch.Tensor:
        """
        Iterative CRF refinement.

        :param images: torch tensor of the image (values in [0, 255]). shape
        N*C*H*W. DEVICE: CPU. shape:
        :param segmentations: softmaxed logits. float. CPU. shape: bsz, k, h,
        w. k is the number of classes.
        :return: refined segmentations of same shape as 'segmentations'.
        """
        assert isinstance(images, torch.Tensor)
        assert isinstance(segmentations, torch.Tensor)
        assert images.ndim == 4  # bsz, d, h, w
        assert segmentations.ndim == 4  # bsz, k, h, w
        assert images.shape[0] == segmentations.shape[0]
        assert images.shape[2:] == segmentations.shape[2:]

        scaled_images = F.interpolate(images,
                                      scale_factor=self.scale_factor,
                                      mode='nearest',
                                      recompute_scale_factor=False
                                      )
        scaled_segs = F.interpolate(segmentations,
                                    scale_factor=self.scale_factor,
                                    mode='bilinear',
                                    recompute_scale_factor=False,
                                    align_corners=False)

        if self.itera == 0:
            return scaled_segs

        log_prob = - torch.log(scaled_segs)
        v = None
        for i in range(images.shape[0]):
            t = self._refine_one_img(img=scaled_images[i],
                                     seg=log_prob[i])
            if v is None:
                v = t.unsqueeze(0)
            else:
                v = torch.vstack((v, t.unsqueeze(0)))

        return v

    def _refine_one_img(self, img: torch.Tensor,
                        seg: torch.Tensor) -> torch.Tensor:
        n, h, w = img.shape  # n, h, w
        k, h_, w_ = seg.shape
        assert h == h_
        assert w == w_
        assert n == 3

        d = dcrf.DenseCRF2D(w, h, k)  # width, height, nlabels
        u = seg.contiguous().view((k, -1)).numpy()
        d.setUnaryEnergy(u)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        # im is an image-array,
        # e.g. im.dtype == np.uint8 and im.shape == (640,480,3)  (w, h, n)
        rgb = self.sigma_rgb
        xy = int(self.sigma_xy * self.scale_factor)
        d.addPairwiseBilateral(sxy=(xy, xy),
                               srgb=(rgb, rgb, rgb),
                               rgbim=np.ascontiguousarray(
                                   img.contiguous().numpy().astype(
                                   np.uint8).transpose(2, 1, 0)),
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        prob = d.inference(self.itera)
        prob = np.array(prob)  # k, h_ * w_
        prob = torch.from_numpy(prob.reshape((k, h, w)).astype(np.float32))

        return prob

    def __str__(self):
        return '{}: (sigma_rgb={}, sigma_xy={}, scale_factor={})'.format(
            self.__class__.__name__, self.sigma_rgb, self.sigma_xy,
            self.scale_factor)


def test_DenseCRFFilter():
    import time

    from dlib.utils.utils_reproducibility import set_seed
    from dlib.utils.shared import announce_msg

    from torch.profiler import profile, record_function, ProfilerActivity

    seed = 0
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device("cpu")

    set_seed(seed=seed)
    n, h, w = 32, 244, 244
    scale_factor = 1.
    itera = 10
    img = torch.randint(
        low=0, high=256,
        size=(n, 3, h, w), dtype=torch.float, device=DEVICE,
        requires_grad=False).cpu()
    nbr_cl = 2
    segmentations = torch.rand(size=(n, nbr_cl, h, w), dtype=torch.float,
                               device=DEVICE, requires_grad=False)

    filter = DenseCRFFilter(sigma_rgb=15.,
                            sigma_xy=100.,
                            scale_factor=scale_factor,
                            itera=itera
                            )
    announce_msg("testing {}".format(filter))
    set_seed(seed=seed)
    if nbr_cl > 1:
        softmax = nn.Softmax(dim=1)
    else:
        softmax = nn.Sigmoid()

    print(img.sum(), softmax(segmentations).sum())

    t0 = time.perf_counter()
    z = filter(images=img, segmentations=softmax(segmentations))
    print('filtered seg.: {} {} (nbr_cl: {})'.format(z, z.dtype, nbr_cl))
    print(f'seg: {segmentations.shape} filtered seg.: {z.shape}')
    print('time op: {}'.format(time.perf_counter() - t0))
    print('Time ({} x {} : scale: {}: N: {}): TIME_ABOVE'.format(
        h, w, scale_factor, n))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    for i in range(1):
        test_DenseCRFFilter()
