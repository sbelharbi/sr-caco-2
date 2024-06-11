import sys
import os
from os.path import dirname, abspath

import torch
import torch.nn as nn
import numpy as np


from skimage.filters import threshold_otsu

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

__all__ = ['rv1d', 'STOtsu']


def rv1d(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 1
    return torch.flip(t, dims=(0, ))


class STOtsu(nn.Module):
    def __init__(self):
        super(STOtsu, self).__init__()

        self.bad_egg = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.bad_egg = False

        min_x = x.min()
        max_x = x.max()

        if min_x == max_x:
            self.bad_egg = True
            return torch.tensor(min_x)

        bins = int(max_x - min_x + 1)
        bin_centers = torch.arange(min_x, max_x + 1, 1, dtype=torch.float32,
                                   device=x.device)

        hist = torch.histc(x, bins=bins)
        weight1 = torch.cumsum(hist, dim=0)
        _weight2 = torch.cumsum(rv1d(hist), dim=0)
        weight2_r = _weight2
        weight2 = rv1d(_weight2)
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / weight1
        mean2 = rv1d(torch.cumsum(rv1d(hist * bin_centers), dim=0) / weight2_r)
        diff_avg_sq = torch.pow(mean1[:-1] - mean2[1:], 2)
        variance12 = weight1[:-1] * weight2[1:] * diff_avg_sq

        idx = torch.argmax(variance12)
        threshold = bin_centers[:-1][idx]

        return threshold


def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd, exist_ok=True)


def test_stotsu_vs_skiamgeotsu():
    from os.path import join
    import time
    import cProfile
    from dlib.utils.utils_reproducibility import set_seed
    import matplotlib.pyplot as plt
    from torch.cuda.amp import autocast
    from tqdm import tqdm
    from torch.profiler import profile, record_function, ProfilerActivity

    amp = False
    print('amp: {}'.format(amp))
    fdout = join(root_dir, 'data/debug/otsu')
    mkdir(fdout)

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False,
                             sharey=False, squeeze=False)

    cuda = 1
    torch.cuda.set_device(cuda)

    times = []
    ths = []

    def atom(seed):
        set_seed(seed)
        h, w = 224, 224
        img = np.random.rand(h, w) * 100 + np.random.rand(h, w) * 10

        img = img.astype(np.uint8)

        img_torch = torch.from_numpy(img).float().cuda()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        st_otsu = STOtsu().cuda()

        st_th = st_otsu(img_torch)
        start_event.record()
        with autocast(enabled=amp):
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                with record_function("compute_th"):
                    st_otsu(img_torch)
            # cProfile.runctx('st_otsu(img_torch)', globals(), locals())
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms_ours = start_event.elapsed_time(end_event)
        print('')
        print(prof.key_averages().table(sort_by="cuda_time_total",
                                        row_limit=10))

        start_event.record()
        kimh_th = threshold_otsu(img)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms_skim = start_event.elapsed_time(end_event)

        return (st_th, kimh_th), (elapsed_time_ms_ours, elapsed_time_ms_skim)

    n = 2
    for seed_ in tqdm(range(n), ncols=150, total=n):
        th, t = atom(seed_)
        times.append(t)
        ths.append(th)

    axes[0, 0].plot([z[0].item() for z in ths], color='tab:blue',
                    label='Our threshold')
    axes[0, 0].plot([z[1] for z in ths], color='tab:orange',
                    label='SKimage threshold')
    axes[0, 0].set_title('Thresholds')

    axes[0, 1].plot([z[0] for z in times], color='tab:blue',
                    label='Our time')
    axes[0, 2].plot([z[1] for z in times], color='tab:orange',
                    label='SKimage time')
    axes[0, 1].set_title('Time (ms) [AMP: {}]'.format(amp))
    axes[0, 2].set_title('Time (ms')

    fig.suptitle('Otsu: ours vs. Skimage', fontsize=6)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

    fig.savefig(join(fdout, 'otsu-compare-amp-{}'.format(amp)),
                bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    import datetime as dt

    from dlib.utils.shared import announce_msg
    from dlib.utils.utils_reproducibility import set_seed

    set_seed(0)
    test_stotsu_vs_skiamgeotsu()
