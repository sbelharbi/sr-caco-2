import os
import sys
from os.path import dirname, abspath, join
from typing import Optional, List
import math
from copy import deepcopy
import numbers

from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from scipy.signal import medfilt

from dlib.utils.tools import check_box_convention

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

_PRED_COLOR = mcolors.CSS4_COLORS['red']
_GT_COLOR = mcolors.CSS4_COLORS['lime']

_TRUE_POSITIVE = mcolors.CSS4_COLORS['red']
_FALSE_POSITIVE = mcolors.CSS4_COLORS['blue']
_FALSE_NEGATIVE = mcolors.CSS4_COLORS['lime']


def get_bin_mask_colormap_segm():
    col_dict = dict()
    for i in range(256):
        col_dict[i] = 'k'

    col_dict[0] = 'k'  # background. ignored.
    col_dict[1] = _FALSE_NEGATIVE  # FALSE NEGATIVE
    col_dict[2] = _FALSE_POSITIVE  # FALSE POSITIVE
    col_dict[3] = _TRUE_POSITIVE  # TRUE POSITIVE

    colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

    return colormap


def get_bin_mask_colormap_bbx():
    col_dict = dict()
    for i in range(256):
        col_dict[i] = 'k'

    col_dict[0] = 'k'  # background. ignored.
    col_dict[1] = _PRED_COLOR  # PREDICTED
    colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

    return colormap


def get_simple_bin_mask_colormap_mask():
    col_dict = dict()
    for i in range(256):
        col_dict[i] = 'k'

    col_dict[0] = 'k'  # background. ignored.
    col_dict[1] = _GT_COLOR  # GT MASK.
    colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

    return colormap


class Viz_WSOL(object):
    def __init__(self):
        super(Viz_WSOL, self).__init__()

        self.gt_col = _GT_COLOR
        self.pred_col = _PRED_COLOR
        self.dpi = 50
        self.alpha = 128
        self.heatmap_cmap = plt.get_cmap("jet")
        self.mask_cmap_seg = get_bin_mask_colormap_segm()
        self.mask_cmap_bbox = get_bin_mask_colormap_bbx()

        self.top_tag_xy = [1, 7]  # w, h
        self.bottom_tag_x = 1  # w

    @staticmethod
    def tagax(ax, text, xy: list):
        ax.text(xy[0], xy[1],
                text, bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8}
                )

    @staticmethod
    def get_acc(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        _gt_mask = gt_mask.flatten()
        _pred_mask = pred_mask.flatten()
        assert _gt_mask.size == _pred_mask.size
        diff = np.abs(_gt_mask - _pred_mask)
        return (diff == 0).mean()

    @staticmethod
    def convert_bbox(bbox_xyxy: np.ndarray):
        check_box_convention(bbox_xyxy, 'x0y0x1y1')
        assert bbox_xyxy.shape == (1, 4), bbox_xyxy.shape
        x0, y0, x1, y1 = bbox_xyxy.flatten()
        width = x1 - x0
        height = y1 - y0
        anchor = (x0, y1)
        return anchor, width, height

    def _plot_bbox(self,
                   ax,
                   img,
                   gt_matched_bbox: np.ndarray,  # (1, 4)
                   gt_bboxes: np.ndarray,  # (nbr_inst, 4)
                   pred_bbox: Optional[np.ndarray] = None,  # (1, 4)
                   cam: Optional[np.ndarray] = None,
                   tag='',
                   tag_cl=None,
                   camcolormap=None,
                   alpha=None,
                   plot_all_instances=True):
        if camcolormap is None:
            camcolormap = self.heatmap_cmap

        if alpha is None:
            alpha = self.alpha

        if gt_matched_bbox is None:
            assert gt_bboxes is None

        if gt_bboxes is None:
            assert gt_matched_bbox is None

        ax.imshow(img)

        # gt matched bbox.
        if gt_matched_bbox is not None:
            gt_mbo_info = self.convert_bbox(gt_matched_bbox)
            rect_gt_mbo = patches.Rectangle(gt_mbo_info[0], gt_mbo_info[1],
                                            -gt_mbo_info[2],
                                            linewidth=3.,
                                            edgecolor=self.gt_col,
                                            facecolor='none')
            ax.add_patch(rect_gt_mbo)

        # gt bbox
        if gt_bboxes is not None:
            nbr_instances = gt_bboxes.shape[0]
            if plot_all_instances and (nbr_instances > 1):
                for i in range(nbr_instances):
                    gt_info = self.convert_bbox(gt_bboxes[i].reshape((1, 4)))
                    rect_gt = patches.Rectangle(gt_info[0], gt_info[1],
                                                -gt_info[2],
                                                linewidth=1.,
                                                edgecolor=self.gt_col,
                                                facecolor='none')
                    ax.add_patch(rect_gt)

        if cam is not None:
            if cam.dtype in [np.float32, np.float64]:
                ax.imshow(cam, interpolation='bilinear', cmap=camcolormap,
                          alpha=alpha)

            elif cam.dtype == np.bool_:
                cam_ = cam * 1.
                masked_cam = np.ma.masked_where(cam_ == 0, cam_)
                ax.imshow(masked_cam, interpolation=None,
                          cmap=self.mask_cmap_bbox, vmin=0., vmax=255.,
                          alpha=self.alpha)

        if pred_bbox is not None:
            pred_info = self.convert_bbox(pred_bbox)

            rect_pred = patches.Rectangle(pred_info[0], pred_info[1],
                                          -pred_info[2],
                                          linewidth=3,
                                          edgecolor=self.pred_col,
                                          facecolor='none')
            ax.add_patch(rect_pred)

        self.tagax(ax, tag, self.top_tag_xy)
        if tag_cl:
            # img: h, w, ...
            self.tagax(ax, tag_cl,
                       [self.bottom_tag_x,
                        img.shape[0] - self.top_tag_xy[1]])

    def _plot_mask(self, ax, img, gt_mask, cam, tag=''):
        ax.imshow(img)

        if cam.dtype in [np.float32, np.float64]:
            ax.imshow(cam, interpolation='bilinear', cmap=self.heatmap_cmap,
                      alpha=self.alpha)

        elif cam.dtype == np.bool_:
            cam_ = cam * 1.
            _gt_mask = gt_mask.astype(np.float32)
            tmp_gt = np.copy(_gt_mask)
            tmp_cam = np.copy(cam_)
            tmp_cam[tmp_cam == 1] = 2.

            show_mask = tmp_gt + tmp_cam  # tpos: 3. fpos: 2, fng: 1.
            # tmp_gt[_gt_mask == 1.] = 2.
            # show_mask = cam_ + tmp_gt
            # show_mask[show_mask == 3.] = 1.

            show_mask = np.ma.masked_where(show_mask == 0, show_mask)
            ax.imshow(show_mask, interpolation=None, cmap=self.mask_cmap_seg,
                      vmin=0., vmax=255., alpha=self.alpha)

        self.tagax(ax, tag, self.top_tag_xy)

    def cbox_plot_single(self, datum: dict, outf: str):
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)

        assert 'gt_bbox' in datum.keys()

        self._plot_bbox(axes[0, 0],
                        img=datum['img'],
                        gt_bbox=datum['gt_bbox'],
                        pred_bbox=datum['pred_bbox'],
                        cam=None,
                        tag=self.get_tag(datum))
        self.closing(fig, outf)

    def plot_single(self, datum: dict, outf: str,
                    plot_all_instances: bool) -> List[int]:
        nrows = 1
        ncols = 2

        him, wim = datum['img'].shape[:2]
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        if 'gt_bboxes' in datum.keys():
            self._plot_bbox(axes[0, 0],
                            img=datum['img'],
                            gt_matched_bbox=datum['gt_matched_bbox'],
                            gt_bboxes=datum['gt_bboxes'],
                            pred_bbox=datum['pred_bbox'],
                            cam=None,
                            tag=self.get_tag(datum),
                            tag_cl=datum['tag_cl'],
                            plot_all_instances=plot_all_instances)

            self._plot_bbox(axes[0, 1],
                            img=datum['img'],
                            gt_matched_bbox=datum['gt_matched_bbox'],
                            gt_bboxes=datum['gt_bboxes'],
                            pred_bbox=datum['pred_bbox'],
                            cam=datum['cam'],
                            tag=self.get_tag(datum),
                            tag_cl=datum['tag_cl'],
                            plot_all_instances=plot_all_instances)

        elif 'gt_mask' in datum.keys():
            cam = datum['cam']
            pred_mask = (datum['cam'] >= datum['tau'])
            acc = self.get_acc(gt_mask=datum['gt_mask'],
                               pred_mask=pred_mask.astype(np.float32))

            self._plot_mask(axes[0, 0], img=datum['img'],
                            gt_mask=datum['gt_mask'],
                            cam=cam,
                            tag=self.get_tag(datum, acc=acc))

            self._plot_mask(axes[0, 1], img=datum['img'],
                            gt_mask=datum['gt_mask'],
                            cam=pred_mask,
                            tag=self.get_tag(datum, acc=acc))
        else:
            raise NotImplementedError

        size = self.closing(fig, outf)

        return size

    def plot_multiple(self, data: list, outf: str):
        nrows = 2
        ncols = len(data)

        him, wim = data[0]['img'].shape[:2]
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        if 'gt_bbox' in data[0].keys():
            for i, datum in enumerate(data):
                self._plot_bbox(axes[0, i], img=datum['img'],
                                gt_bbox=datum['gt_bbox'],
                                pred_bbox=datum['pred_bbox'], cam=datum['cam'],
                                tag=self.get_tag(datum))
                mask = (datum['cam'] >= datum['tau'])
                self._plot_bbox(axes[1, i], img=datum['img'],
                                gt_bbox=datum['gt_bbox'],
                                pred_bbox=datum['pred_bbox'], cam=mask,
                                tag=self.get_tag(datum))

        elif 'gt_mask' in data[0].keys():
            for i, datum in enumerate(data):
                cam = datum['cam']
                pred_mask = (datum['cam'] >= datum['tau'])
                acc = self.get_acc(gt_mask=datum['gt_mask'],
                                   pred_mask=pred_mask.astype(np.float32))

                self._plot_mask(axes[0, i], img=datum['img'],
                                gt_mask=datum['gt_mask'],
                                cam=cam,
                                tag=self.get_tag(datum, acc=acc))

                self._plot_mask(axes[1, i], img=datum['img'],
                                gt_mask=datum['gt_mask'],
                                cam=pred_mask,
                                tag=self.get_tag(datum, acc=acc))
        else:
            raise NotImplementedError

        self.closing(fig, outf)

    def plot_cam_raw(self, cam: np.ndarray, outf: str, interpolation: str):
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
        ax[0, 0].imshow(cam, interpolation=interpolation,
                        cmap=self.heatmap_cmap,
                        alpha=self.alpha)
        self.closing(fig, outf)

    def plot_map_raw(self, map: np.ndarray, outf: str, interpolation: str):
        nrows = 1
        ncols = 3

        him, wim = map.shape[:2]
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        axes[0, 0].imshow(map[0], interpolation=interpolation,
                          cmap=self.heatmap_cmap,
                          alpha=self.alpha)
        self.tagax(axes[0, 0], 'P1', self.top_tag_xy)
        axes[0, 1].imshow(map[1], interpolation=interpolation,
                          cmap=self.heatmap_cmap,
                          alpha=self.alpha)
        self.tagax(axes[0, 1], 'P2', self.top_tag_xy)
        axes[0, 2].imshow(map[0] + map[1], interpolation=interpolation,
                          cmap=self.heatmap_cmap,
                          alpha=self.alpha)
        self.tagax(axes[0, 2], 'P1/P2', self.top_tag_xy)
        self.closing(fig, outf)

    def get_tag(self, datum, acc=0.0):
        if 'gt_bboxes' in datum.keys():
            if isinstance(datum['tau'], numbers.Number):
                tau_txt = r'@$\tau$={:.2f}'.format(datum['tau'])
            else:
                tau_txt = r'@$\tau$={}'.format(datum['tau'])

            if isinstance(datum['sigma'], numbers.Number):
                sigma_txt = r'@$\sigma$={:.2f}'.format(datum['sigma'])
            else:
                sigma_txt = r'@$\sigma$={}'.format(datum['sigma'])

            tag = r'IoU={:.3f}, {}{}'.format(datum['iou'], tau_txt, sigma_txt)

        elif 'gt_mask' in datum.keys():
            z = '*' if datum['best_tau'] else ''
            tag = r'acc={:.3f}, @$\tau$={:.2f}{}'.format(acc, datum['tau'], z)
        else:
            raise NotImplementedError
        return tag

    def closing(self, fig, outf) -> List[int]:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0,
                            wspace=0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        fig.savefig(outf, pad_inches=0, bbox_inches='tight', dpi=self.dpi,
                    optimize=True)

        size = fig.get_size_inches() * self.dpi  # w, h.
        plt.close(fig)
        size = [int(x) for x in size]

        return size

    def _watch_plot_single(self, datum: dict, datum_filtered: dict, outf: str):
        fig, axes = plt.subplots(nrows=1, ncols=4, squeeze=False)

        if 'gt_bbox' in datum.keys():
            self._plot_bbox(axes[0, 0],
                            img=datum['img'],
                            gt_bbox=datum['gt_bbox'],
                            pred_bbox=datum['pred_bbox'],
                            cam=datum['cam'],
                            tag=self.get_tag(datum))
            mask = (datum['cam'] >= datum['tau'])
            self._plot_bbox(axes[0, 1],
                            img=datum['img'],
                            gt_bbox=datum['gt_bbox'],
                            pred_bbox=datum['pred_bbox'],
                            cam=mask,
                            tag=self.get_tag(datum))

            self._plot_bbox(axes[0, 2],
                            img=datum_filtered['img'],
                            gt_bbox=datum_filtered['gt_bbox'],
                            pred_bbox=datum_filtered['pred_bbox'],
                            cam=datum_filtered['cam'],
                            tag=self.get_tag(datum_filtered))
            mask = (datum_filtered['cam'] >= datum_filtered['tau'])
            self._plot_bbox(axes[0, 3],
                            img=datum_filtered['img'],
                            gt_bbox=datum_filtered['gt_bbox'],
                            pred_bbox=datum_filtered['pred_bbox'],
                            cam=mask,
                            tag=self.get_tag(datum_filtered))

        elif 'gt_mask' in datum.keys():
            raise NotImplementedError
            cam = datum['cam']
            pred_mask = (datum['cam'] >= datum['tau'])
            acc = self.get_acc(gt_mask=datum['gt_mask'],
                               pred_mask=pred_mask.astype(np.float32))

            self._plot_mask(axes[0, 0], img=datum['img'],
                            gt_mask=datum['gt_mask'],
                            cam=cam,
                            tag=self.get_tag(datum, acc=acc))

            self._plot_mask(axes[0, 1], img=datum['img'],
                            gt_mask=datum['gt_mask'],
                            cam=pred_mask,
                            tag=self.get_tag(datum, acc=acc))
        else:
            raise NotImplementedError

        self.closing(fig, outf)

    def _watch_plot_entropy(self, data: dict, outf: str):
        nrows = 1
        ncols = len(list(data['visu'].keys())) + 1

        him, wim = data['raw_img'].shape[:2]
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        if 'gt_bbox' in data.keys():
            self._plot_bbox(axes[0, 0],
                            img=data['raw_img'],
                            gt_bbox=data['gt_bbox'],
                            pred_bbox=None,
                            cam=None,
                            tag='Input')
            for i, datumkey in enumerate(list(data['visu'].keys())):
                self._plot_bbox(axes[0, i + 1],
                                img=data['raw_img'],
                                gt_bbox=data['gt_bbox'],
                                pred_bbox=None,
                                cam=data['visu'][datumkey],
                                tag=data['tags'][datumkey])

        elif 'gt_mask' in data[0].keys():
            for i, datum in enumerate(data):
                cam = datum['cam']
                pred_mask = (datum['cam'] >= datum['tau'])
                acc = self.get_acc(gt_mask=datum['gt_mask'],
                                   pred_mask=pred_mask.astype(np.float32))

                self._plot_mask(axes[0, i], img=datum['img'],
                                gt_mask=datum['gt_mask'],
                                cam=cam,
                                tag=self.get_tag(datum, acc=acc))

                self._plot_mask(axes[1, i], img=datum['img'],
                                gt_mask=datum['gt_mask'],
                                cam=pred_mask,
                                tag=self.get_tag(datum, acc=acc))
        else:
            raise NotImplementedError

        self.closing(fig, outf)

    def _watch_plot_histogram_activations(self, density: np.ndarray,
                                          bins: np.ndarray, outf: str,
                                          split: str):
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)

        widths = bins[:-1] - bins[1:]
        widths = 10.
        # smooth the bars to avoid very large cases. use median filter with
        # kernel 5.
        x = range(density.size)
        axes[0, 0].bar(x, medfilt(volume=density, kernel_size=5), width=widths)
        axes[0, 0].set_xlabel('Normalized CAM activations')
        axes[0, 0].set_ylabel('Percentage from total {} set.'.format(split))

        # scale down the x-ticks.
        scale_x = 1000.
        ticks_x = ticker.FuncFormatter(
            lambda xx, pos: '{0:g}'.format(xx / scale_x))
        axes[0, 0].xaxis.set_major_formatter(ticks_x)

        fig.savefig(outf, bbox_inches='tight', dpi=self.dpi, optimize=True)
        plt.close(fig)

    def _plot_meter(self, metrics: dict, fout: str, perfs_keys: list,
                    title: str = '',
                    xlabel: str = '', best_iter: int = None):

        ncols = 4
        ks = perfs_keys
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = metrics[ks[t]]['value_per_epoch']
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                axes[i, j].set_title(ks[t], fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                # axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                if best_iter is not None:
                    axes[i, j].plot([x[best_iter]], [val[best_iter]],
                                    marker='o',
                                    markersize=5,
                                    color="red")
                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()

        fig.savefig(fout, bbox_inches='tight', dpi=300)

    def _clean_metrics(self, metric: dict) -> dict:
        _metric = deepcopy(metric)
        l = []
        for k in _metric.keys():
            cd = (_metric[k]['value_per_epoch'] == [])
            cd |= (_metric[k]['value_per_epoch'] == [np.inf])
            cd |= (_metric[k]['value_per_epoch'] == [-np.inf])

            if cd:
                l.append(k)

        for k in l:
            _metric.pop(k, None)

        return _metric

    def _watch_plot_perfs_meter(self, meters: dict, split: str, perfs: list,
                                fout: str):
        xlabel = 'epochs'

        # todo: best criterion set to 'localization'. it may change.
        best_epoch = meters[constants.VALIDSET]['localization']['best_epoch']

        title = 'Split: {}. Best iter.: {} {}'.format(split, best_epoch,
                                                      xlabel)
        self._plot_meter(
            self._clean_metrics(meters[split]), fout=fout,
            perfs_keys=perfs,  title=title, xlabel=xlabel, best_iter=best_epoch)

        out = dict()
        out[split] = dict()
        for k in perfs:
            val = self._clean_metrics(meters[split])[k]['value_per_epoch']

            out[split][k] = dict()
            out[split][k] = {'vals': val, 'best_epoch': best_epoch}

        return out

    def _watch_plot_thresh(self, data: dict, outf: str):
        nrows = 1
        ncols = len(list(data['visu'].keys())) + 1

        him, wim = data['raw_img'].shape[:2]
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        if 'gt_bbox' in data.keys():
            self._plot_bbox(axes[0, 0],
                            img=data['raw_img'],
                            gt_bbox=data['gt_bbox'],
                            pred_bbox=None,
                            cam=None,
                            tag='Input')
            for i, datumkey in enumerate(list(data['visu'].keys())):
                if datumkey == 'density':
                    density, bins = data['visu'][datumkey]
                    widths = bins[:-1] - bins[1:]
                    axes[0, i + 1].bar(bins[1:], density, width=widths)
                    axes[0, i + 1].axvline(data['otsu_thresh'],
                                           label='otsu_thresh', color='r')
                    axes[0, i + 1].axvline(data['li_thres'],
                                           label='li_thres', color='b')
                    axes[0, i + 1].legend()
                elif datumkey == 'discrete_cam':
                    axes[0, i + 1].imshow(data['visu'][datumkey], cmap=cm.gray)
                elif datumkey in ['bin_otsu', 'bin_li', 'otsu_bin_eroded',
                                  'li_bin_eroded', 'fg_auto']:
                    gt_info = self.convert_bbox(data['gt_bbox'])
                    rect_gt = patches.Rectangle(gt_info[0], gt_info[1],
                                                -gt_info[2],
                                                linewidth=1.5,
                                                edgecolor=self.gt_col,
                                                facecolor='none')
                    axes[0, i + 1].imshow(data['visu'][datumkey], cmap=cm.gray)
                    axes[0, i + 1].add_patch(rect_gt)
                    self.tagax(axes[0, i + 1], data['tags'][datumkey],
                               self.top_tag_xy)
                else:
                    self._plot_bbox(axes[0, i + 1],
                                    img=data['raw_img'],
                                    gt_bbox=data['gt_bbox'],
                                    pred_bbox=None,
                                    cam=data['visu'][datumkey],
                                    tag=data['tags'][datumkey])

        elif 'gt_mask' in data.keys():
            axes[0, 0].imshow(data['raw_img'])
            show_mask = data['gt_mask']
            show_mask = np.ma.masked_where(data['gt_mask'] == 0, show_mask)
            axes[0, 0].imshow(show_mask, interpolation=None,
                              cmap=get_simple_bin_mask_colormap_mask(),
                              vmin=0., vmax=255., alpha=self.alpha)
            self.tagax(axes[0, 0], 'Input', self.top_tag_xy)

            for i, datumkey in enumerate(list(data['visu'].keys())):
                if datumkey == 'density':
                    density, bins = data['visu'][datumkey]
                    widths = bins[:-1] - bins[1:]
                    axes[0, i + 1].bar(bins[1:], density, width=widths)
                    axes[0, i + 1].axvline(data['otsu_thresh'],
                                           label='otsu_thresh', color='r')
                    axes[0, i + 1].axvline(data['li_thres'],
                                           label='li_thres', color='b')
                    axes[0, i + 1].legend()
                elif datumkey == 'discrete_cam':
                    axes[0, i + 1].imshow(data['visu'][datumkey], cmap=cm.gray)
                elif datumkey in ['bin_otsu', 'bin_li', 'otsu_bin_eroded',
                                  'li_bin_eroded', 'fg_auto']:
                    axes[0, i + 1].imshow(data['visu'][datumkey], cmap=cm.gray)
                    self.tagax(axes[0, i + 1], data['tags'][datumkey],
                               self.top_tag_xy)
                elif datumkey in ['cam', 'cam_normalized']:
                    axes[0, i + 1].imshow(
                        data['visu'][datumkey], interpolation='bilinear',
                        cmap=self.heatmap_cmap, alpha=self.alpha)
                    self.tagax(axes[0, i + 1], data['tags'][datumkey],
                               self.top_tag_xy)
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        self.closing(fig, outf)


def test_Viz_WSOL():
    fdout = join(root_dir, 'data/debug/visualization/wsol')
    if not os.path.isdir(fdout):
        os.makedirs(fdout, exist_ok=True)

    viz = Viz_WSOL()

    debug_fd = join(root_dir, 'data/debug/input')
    img_pil = Image.open(
        join(debug_fd, 'Black_Footed_Albatross_0002_55.jpg'), 'r').convert(
        "RGB")
    img_pil = img_pil.resize((224, 224))
    w, h = img_pil.size
    img = np.asarray(img_pil)
    datum = {'img': img, 'img_id': 123456,
             'gt_bboxes': np.asarray([14, 112, 202, 198]).reshape((1, 4)),
             'gt_matched_bbox': np.asarray([14, 112, 202, 198]).reshape((1, 4)),
             'pred_bbox': np.asarray([14, 80, 100, 200]).reshape((1, 4)),
             'iou': 0.8569632541, 'tau': 0.2533653, 'sigma': 0.2356,
             'cam': np.random.rand(h, w),
             'tag_cl': '[CL] trg: this is a test - pred: this is another test'}

    viz.plot_single(datum=datum, outf=join(fdout, 'single-bbox.png'),
                    plot_all_instances=False)
    sys.exit()

    viz.plot_multiple(
        data=[{'img': img, 'img_id': 123456,
               'gt_bbox':  np.asarray([14, 112, 402, 298]).reshape((1, 4)),
               'pred_bbox': np.asarray([14, 80, 300, 200]).reshape((1, 4)),
               'iou': 0.8569632541, 'tau': 0.125, 'sigma':0.2356,
               'cam': np.random.rand(h, w)} for _ in range(3)],
        outf=join(fdout, 'multilpe-bbox.jpg'))

    datum = {'img': img, 'img_id': 123456,
             'gt_mask': (np.random.rand(h, w) > 0.01).astype(np.float32),
             'tau': 0.12533653,
             'cam': np.random.rand(h, w), 'best_tau': True}

    viz.plot_single(datum=datum, outf=join(fdout, 'single-mask.jpg'))
    viz.plot_multiple(data=[
        {'img': img, 'img_id': 123456,
         'gt_mask': (np.random.rand(h, w) > 0.01).astype(np.float32),
         'tau': 0.12533653,
         'cam': np.random.rand(h, w), 'best_tau': i == 1} for i in range(5)],
        outf=join(fdout, 'multiple-mask.jpg'))


if __name__ == '__main__':
    test_Viz_WSOL()
