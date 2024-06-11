import copy
import os.path
import sys
from os.path import join, dirname, abspath
from typing import Tuple, Any
import math
from textwrap import wrap

import pickle as pkl

import numpy as np
import torch
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import announce_msg

mpl.style.use('seaborn')


__all__ = ['init_tracker',
           'find_last_tracker',
           'save_tracker',
           'update_tracker_train',
           'update_tracker_eval',
           'plot_tracker_eval',
           'plot_tracker_train',
           'reset_tracker_eval',
           'is_last_perf_best_perf',
           'write_current_perf_eval',
           'current_perf_to_str'
           ]


def init_tracker(args: object) -> dict:
    b_inter = args.basic_interpolation

    mtrs = constants.METRICS

    out = {
        constants.TRAINSET: {
            constants.PR_EPOCH: dict(),
            constants.PR_ITER: dict()
        }
    }
    valid_sets = args.valid_dsets.split(constants.SEP)

    out[constants.VALIDSET] = {
        subset: {
            mtr: {'vals': [], 'best_val': 0.0} for mtr in mtrs
        }
        for subset in valid_sets
    }

    for subset in valid_sets:
        out[constants.VALIDSET][f'{subset}_{b_inter}'] = {
            mtr: {'vals': [], 'best_val': 0.0} for mtr in mtrs
        }

    test_sets = args.test_dsets.split(constants.SEP)

    out[constants.TESTSET] = {
        subset: {
            mtr: {'vals': [], 'best_val': 0.0} for mtr in mtrs
        }
        for subset in test_sets
    }

    for subset in test_sets:
        out[constants.TESTSET][f'{subset}_{b_inter}'] = {
            mtr: {'vals': [], 'best_val': 0.0} for mtr in mtrs
        }

    return out


def find_last_tracker(save_dir: str,
                      args: object
                      ) -> Tuple[dict, dict]:

    path = join(save_dir, 'tracker.pkl')
    roi_path = join(save_dir, 'roi_tracker.pkl')

    if os.path.isfile(path):
        try:
            with open(path, 'rb') as f:
                tracker = pkl.load(f)
                DLLogger.log(fmsg(f'Loaded tracker: {path}.'))

            with open(roi_path, 'rb') as f:
                roi_tracker = pkl.load(f)
                DLLogger.log(fmsg(f'Loaded roi tracker: {roi_path}.'))

            return tracker, roi_tracker

        except Exception as e:
            DLLogger.log(fmsg(f'failed to load tracker: {path} or {roi_path}.'
                              f'Error: {e}'))
            return init_tracker(args), init_tracker(args)
    else:
        return init_tracker(args), init_tracker(args)


def is_last_perf_best_perf(tracker: dict,
                           roi_tracker: dict,
                           eval_over_roi_also: bool,
                           eval_over_roi_also_model_select: bool,
                           split: str,
                           ds_name: str,
                           metric: str
                           ) -> bool:
    _tracker = tracker
    if eval_over_roi_also and eval_over_roi_also_model_select:
        _tracker = roi_tracker

    assert split in [constants.VALIDSET, constants.TESTSET]
    assert metric in constants.METRICS

    l = metric
    bv = _tracker[split][ds_name][l]['best_val']
    last_vl = _tracker[split][ds_name][l]['vals'][-1]

    return bv == last_vl


def write_current_perf_eval(tracker: dict,
                            split: str,
                            ds_name: str,
                            save_dir: str,
                            name_f: str,
                            current_step: int,
                            current_epoch: int
                            ) -> dict:
    """
    Write and summarize the tracker.
    if save_dir is None, we do not write.
    """

    assert split in [constants.VALIDSET, constants.TESTSET]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    out = dict()

    for l in tracker[split][ds_name]:
        if tracker[split][ds_name][l]['vals']:
            out[f"last_{l}"] = tracker[split][ds_name][l]['vals'][-1]
            out[f'best_{l}'] = tracker[split][ds_name][l]['best_val']

    out['dataset'] = ds_name
    out['split'] = split
    out['current_step'] = current_step
    out['current_epoch'] = current_epoch

    if save_dir is not None:
        with open(join(save_dir, name_f), 'w') as f:
            yaml.dump(out, f)

    return out


def current_perf_to_str(status: dict,
                        roi_status: dict,
                        master_mtr: str,
                        model_select_roi: bool
                        ) -> str:

    mtrs = constants.METRICS
    msg = f"CURRENT. EPO: {status['current_epoch']}. " \
          f"STEP: {status['current_step']}."
    msg = fmsg(msg)

    msg += fmsg(f"Dataset: {status['dataset']}.  Split: {status['split']}")

    for mtr in mtrs:
        if f"last_{mtr}" in status:
            vl_last = status[f"last_{mtr}"]
            vl_best = status[f"best_{mtr}"]

            if roi_status is None:

                if mtr == master_mtr:
                    msg += f"{mtr}: {vl_best:<.6f} [BEST] | " \
                           f"{vl_last:<.4f} [LAST] ---> MASTER \n"
                else:
                    msg += f"{mtr}: {vl_best:<.6f} [BEST] | " \
                           f"{vl_last:<.4f} [LAST] \n"

            else:
                roi_vl_last = roi_status[f"last_{mtr}"]
                roi_vl_best = roi_status[f"best_{mtr}"]
                a = '*'
                r = ''
                if model_select_roi:
                    a = ''
                    r = '*'

                if mtr == master_mtr:
                    msg += f"{mtr}: {vl_best:<.6f}{a} " \
                           f"(ROI: {roi_vl_best:<.6f}{r}) [BEST] | " \
                           f"{vl_last:<.6f} (ROI: {roi_vl_last:<.6f}) [LAST]" \
                           f" ---> MASTER \n"
                else:
                    msg += f"{mtr}: {vl_best:<.6f}{a} " \
                           f"(ROI: {roi_vl_best:<.6f}{r})[BEST] | " \
                           f"{vl_last:<.6f} (ROI: {roi_vl_last:<.6f})[LAST] \n"


    return msg



def idx_last_occurrence_val_in_list(l: list, v) -> int:
    # v is expected to be in list. error will be thrown otherwise.
    s = len(l)
    ll = l[::-1]
    return s - 1 - ll.index(v)


def update_tracker_eval(tracker: dict,
                        split: str,
                        ds_name: str,
                        metric: str,
                        value,
                        idx_best: int = None
                        ) -> Tuple[dict, int]:

    assert split in [constants.VALIDSET, constants.TESTSET]
    assert metric in constants.METRICS

    v = value
    l = metric
    fn = constants.BEST_MTR[metric]

    if torch.is_tensor(v):
        _v = v.detach().item()
    elif isinstance(v, np.ndarray):
        _v = v.item()
    elif isinstance(v, np.ndarray):
        _v = v.item()
    else:
        _v = v

    _idx_best = None

    if l in tracker[split][ds_name]:
        tracker[split][ds_name][l]['vals'].append(_v)

        if idx_best is None:
            bv = tracker[split][ds_name][l]['best_val']
            tracker[split][ds_name][l]['best_val'] = fn(_v, bv)

            _idx_best = idx_last_occurrence_val_in_list(
                tracker[split][ds_name][l]['vals'],
                tracker[split][ds_name][l]['best_val'])
        else:
            if len(tracker[split][ds_name][l]['vals']) == 0:
                tracker[split][ds_name][l] = {
                    'vals': [_v],
                    'best_val': _v
                }

            else:
                bv = tracker[split][ds_name][l]['vals'][idx_best]
                tracker[split][ds_name][l]['best_val'] = bv
    else:

        tracker[split][ds_name][l] = {
            'vals': [_v],
            'best_val': _v
        }
        if idx_best is None:
            _idx_best = 0

    return tracker, _idx_best


def reset_tracker_eval(tracker: dict, split: str, ds_name: str) -> dict:

    assert split  == constants.TESTSET, split

    mtrs = constants.METRICS

    for l in mtrs:
        tracker[split][ds_name][l] = {
            'vals': [],
            'best_val': 0.0
        }

    return tracker


def update_tracker_train(tracker: dict,
                         n_losses: list,
                         v_losses: list,
                         period: str
                         ) -> dict:
    assert period in constants.PERIODS

    for l, v in zip(n_losses, v_losses):
        if torch.is_tensor(v):
            _v = v.detach().item()
        elif isinstance(v, np.ndarray):
            _v = v.item()
        elif isinstance(v, np.ndarray):
            _v = v.item()
        else:
            _v = v

        if l in tracker[constants.TRAINSET][period]:
            tracker[constants.TRAINSET][period][l]['vals'].append(_v)
            bv = tracker[constants.TRAINSET][period][l]['best_val']
            # todo: weak assumption: best loss is MIN val.
            tracker[constants.TRAINSET][period][l]['best_val'] = min(_v, bv)
        else:

            tracker[constants.TRAINSET][period][l] = {
                'vals': [_v],
                'best_val': _v
            }

    return tracker


def save_bin_pkl(obj: Any, p: str):
    with open(p, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def save_tracker(save_dir: str,
                 tracker: dict,
                 roi_tracker: dict
                 ):
    save_bin_pkl(tracker, join(save_dir, 'tracker.pkl'))
    save_bin_pkl(roi_tracker, join(save_dir, 'roi_tracker.pkl'))


def plot_tracker_eval(tracker: dict,
                      roi_tracker: dict,
                      eval_over_roi_also: bool,
                      eval_over_roi_also_model_select: bool,
                      split: str,
                      path_store_figure: str,
                      args: object
                      ):
        method = args.method

        _sets = list(tracker[split].keys())
        _sets = [s for s in _sets if not s.endswith(
            f'_{args.basic_interpolation}')]
        nbr_sets = len(_sets)
        _mtrs = list(tracker[split][_sets[0]].keys())
        nbr_mtrs = len(tracker[split][_sets[0]])
        ncols = nbr_mtrs
        nrows = nbr_sets

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)

        cnd = eval_over_roi_also and eval_over_roi_also_model_select
        if cnd:
            alpha = 0.4
            roi_alpha = 1.
            ls = 'dashed'
            roi_ls = 'solid'
        else:
            alpha = 1.
            roi_alpha = .4
            ls = 'solid'
            roi_ls = 'dashed'

        t = 0
        _subset = tracker[split]
        for i in range(nrows):
            for j in range(ncols):
                _set = _sets[i]
                _mtr = _mtrs[j]
                # if t >= len(ks):
                #     axes[i, j].set_visible(False)
                #     t += 1
                #     continue

                vals = copy.deepcopy(_subset[_set][_mtr]['vals'])
                b_val = _subset[_set][_mtr]['best_val']

                if vals == []:
                    vals = [b_val]

                x = list(range(len(vals)))
                axes[i, j].plot(x, vals, color='tab:orange', label=method,
                                alpha=alpha, linestyle=ls)

                if eval_over_roi_also:
                    roi_vals = copy.deepcopy(
                        roi_tracker[split][_set][_mtr]['vals'])
                    roi_b_val = roi_tracker[split][_set][_mtr]['best_val']
                    if roi_vals == []:
                        roi_vals = [roi_b_val]

                    axes[i, j].plot(x, roi_vals, color='tab:orange',
                                    label=f"{method}-ROI", linestyle=roi_ls,
                                    alpha=roi_alpha)

                    title = '\n'.join(wrap(f'{_set} / {_mtr}:'
                                           f' best {b_val:.2f}'
                                           f' (ROI: {roi_b_val:.2f})',
                                           30))

                else:
                    title = '\n'.join(wrap(f'{_set} / {_mtr}: '
                                           f'best {b_val:.2f}',
                                           30))

                axes[i, j].set_title(title, fontsize=4)

                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#iter', fontsize=4)
                axes[i, j].grid(True)
                # axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                # best value.
                if cnd:
                    biter = idx_last_occurrence_val_in_list(roi_vals, roi_b_val)

                    axes[i, j].plot([x[biter]], [b_val],
                                    marker='o',
                                    markersize=5,
                                    color="red",
                                    alpha=alpha
                                    )
                    axes[i, j].plot([x[biter]], [roi_b_val],
                                    marker='o',
                                    markersize=5,
                                    color="red",
                                    alpha=roi_alpha
                                    )
                else:
                    biter = idx_last_occurrence_val_in_list(vals, b_val)

                    axes[i, j].plot([x[biter]], [b_val],
                                    marker='o',
                                    markersize=5,
                                    color="red")

                # interpolation perf.
                val_inter = _subset[f'{_set}_{args.basic_interpolation}'][
                    _mtr]['best_val']
                vals_inter = [val_inter for _ in x]
                axes[i, j].plot(x, vals_inter, color='tab:blue',
                                label=args.basic_interpolation,
                                alpha=alpha, linestyle=ls)

                if eval_over_roi_also:
                    roi_val_inter = roi_tracker[split][
                        f'{_set}_{args.basic_interpolation}'][_mtr]['best_val']
                    roi_vals_inter = [roi_val_inter for _ in x]
                    axes[i, j].plot(x, roi_vals_inter, color='tab:blue',
                                    label=f"{args.basic_interpolation}-ROI",
                                    linestyle=roi_ls, alpha=roi_alpha)

                axes[i, j].legend(loc='best', prop={'size': 4})

        fig.suptitle(f'Perf. split: {split}', fontsize=4)
        plt.tight_layout()

        fig.savefig(path_store_figure, bbox_inches='tight', dpi=300)


def plot_tracker_train(tracker: dict, split: str, path_store_figure: str,
                       args: object, period: str):
    assert period in constants.PERIODS

    method = args.method

    _keys = list(tracker[split][period].keys())
    nbr_k = len(_keys)
    ncols = min(5, len(_keys))
    nrows = math.ceil(nbr_k / float(ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                             sharey=False, squeeze=False)
    t = 0
    holder = tracker[split][period]
    str_period = period.replace('period_', '')
    for i in range(nrows):
        for j in range(ncols):
            if t >= nbr_k:
                axes[i, j].set_visible(False)
                t += 1
                continue

            _k = _keys[t]

            vals = copy.deepcopy(holder[_k]['vals'])
            b_val = holder[_k]['best_val']

            if vals == []:
                vals = [b_val]

            x = list(range(len(vals)))
            axes[i, j].plot(x, vals, color='tab:orange', label=method)
            axes[i, j].set_title(f'{split} / {_k}: best {b_val:.2f}',
                                 fontsize=4)

            axes[i, j].xaxis.set_tick_params(labelsize=4)
            axes[i, j].yaxis.set_tick_params(labelsize=4)
            axes[i, j].set_xlabel(f'#{str_period}', fontsize=4)
            axes[i, j].grid(True)
            # axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

            # best value.
            biter = idx_last_occurrence_val_in_list(vals, b_val)
            axes[i, j].plot([x[biter]], [b_val],
                            marker='o',
                            markersize=5,
                            color="red")

            t += 1

    fig.suptitle(f'Stats. split: {split} @Period: {period}', fontsize=4)
    plt.tight_layout()

    fig.savefig(path_store_figure, bbox_inches='tight', dpi=300)
