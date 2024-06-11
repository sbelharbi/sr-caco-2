import sys
from os.path import dirname, abspath, join, basename, normpath
import os
import glob
import shutil
import subprocess
import datetime as dt
import math
from collections.abc import Iterable

import torch
import yaml
from sklearn.metrics import auc
import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger

from dlib.utils.shared import fmsg
from dlib.utils import constants
from dlib.utils.shared import is_cc
from dlib.utils.shared import get_tag_device
from dlib.utils.utils_exps import copy_exp_dir_node_to_scratch


def check_negative_non_float(vls: torch.Tensor, name: str):
    assert isinstance(vls, torch.Tensor), type(vls)

    _inf = torch.isinf(vls).sum()
    _nan = torch.isnan(vls).sum()

    msg = f"inf {name}: {_inf}"
    try:
        assert _inf == 0, msg
    except AssertionError:
        DLLogger.log(f'Terminated due to error: {msg}')
        sys.exit()

    msg = f"nan {name}: {_nan}"

    try:
        assert _nan == 0, msg
    except AssertionError:
        DLLogger.log(f'Terminated due to error: {msg}')
        sys.exit()


def check_model_output_corruption(out: torch.Tensor):
    if out is not None:
        check_negative_non_float(out, 'model output v1')


def check_corruption(model):
    # output
    if model.E is not None:
        check_negative_non_float(model.E, 'model output v0')

    # model weights
    _mdl = model.get_bare_model(model.netG)
    for p in _mdl.parameters():
        check_negative_non_float(p, 'model params v0')


def get_cpu_device():
    """
    Return CPU device.
    :return:
    """
    return torch.device("cpu")


def log_device(args):
    assert torch.cuda.is_available()

    tag = get_tag_device(args=args)

    DLLogger.log(message=tag)


def chunks_into_n(l: Iterable, n: int) -> Iterable:
    """
    Split iterable l into n chunks (iterables) with the same size.

    :param l: iterable.
    :param n: number of chunks.
    :return: iterable of length n.
    """
    chunksize = int(math.ceil(len(l) / n))
    return (l[i * chunksize:i * chunksize + chunksize] for i in range(n))


def chunk_it(l, n):
    """
    Create chunks with the same size (n) from the iterable l.
    :param l: iterable.
    :param n: int, size of the chunk.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of
     the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def count_nb_params(model):
    """
    Count the number of parameters within a model.

    :param model: nn.Module or None.
    :return: int, number of learnable parameters.
    """
    if model is None:
        return 0
    else:
        return sum([p.numel() for p in model.parameters()])


def create_folders_for_exp(exp_folder, name):
    """
    Create a set of folder for the current exp.
    :param exp_folder: str, the path to the current exp.
    :param name: str, name of the dataset (train, validation, test)
    :return: object, where each attribute is a folder.
    There is the following attributes:
        . folder: the name of the folder that will contain everything about
        this dataset.
        . prediction: for the image prediction.
    """
    l_dirs = dict()

    l_dirs["folder"] = join(exp_folder, name)
    l_dirs["prediction"] = join(exp_folder, "{}/prediction".format(name))

    for k in l_dirs:
        if not os.path.exists(l_dirs[k]):
            os.makedirs(l_dirs[k], exist_ok=True)

    return Dict2Obj(l_dirs)


def copy_code(dest,
              compress=False,
              verbose=False
              ):
    """Copy code to the exp folder for reproducibility.
    Input:
        dest: path to the destination folder (the exp folder).
        compress: bool. if true, we compress the destination folder and
        delete it.
        verbose: bool. if true, we show what is going on.
    """
    # extensions to copy.
    exts = tuple(["py", "sh", "yaml"])
    flds_files = ['.']

    for fld in flds_files:
        files = glob.iglob(os.path.join(root_dir, fld, "*"))
        subfd = join(dest, fld) if fld != "." else dest
        if not os.path.exists(subfd):
            os.makedirs(subfd, exist_ok=True)

        for file in files:
            if file.endswith(exts):
                if os.path.isfile(file):
                    shutil.copy(file, subfd)
    # cp dlib
    dirs = ["dlib", "cmds"]
    for dirx in dirs:
        cmds = [
            "cd {} && ".format(root_dir),
            "cp -r {} {} ".format(dirx, dest)
        ]
        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)

    if compress:
        head = dest.split(os.sep)[-1]
        if head == '':  # dest ends with '/'
            head = dest.split(os.sep)[-2]
        cmds = [
            "cd {} && ".format(dest),
            "cd .. && ",
            "tar -cf {}.tar.gz {}  && ".format(head, head),
            "rm -rf {}".format(head)
               ]

        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)


def log_args(args_dict):
    DLLogger.log(fmsg("Configuration"))
    # todo


def save_model(model, args, outfd):
    model.eval()
    cpu_device = get_cpu_device()
    model.to(cpu_device)
    torch.save(model.state_dict(), join(outfd, "best_model.pt"))

    if args.task == constants.STD_CL:
        tag = "{}-{}-{}".format(
            args.dataset, args.model['encoder_name'], args.spatial_pooling)
        path = join(outfd, tag)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        torch.save(model.encoder.state_dict(), join(path, 'encoder.pt'))
        torch.save(model.classification_head.state_dict(),
                   join(path, 'head.pt'))
        DLLogger.log(message="Stored classifier. TAG: {}".format(tag))


def save_config(config_dict, outfd):
    with open(join(outfd, 'config.yml'), 'w') as fout:
        yaml.dump(config_dict, fout)


def get_best_epoch(fyaml):
    with open(fyaml, 'r') as f:
        config = yaml.safe_load(f)
        return config['best_epoch']


def compute_auc(vec: np.ndarray, nbr_p: int):
    """
    Compute the area under a curve.
    :param vec: vector contains values in [0, 100.].
    :param nbr_p: int. number of points in the x-axis. it is expected to be
    the same as the number of values in `vec`.
    :return: float in [0, 100]. percentage of the area from the perfect area.
    """
    if vec.size == 1:
        return float(vec[0])
    else:
        area_under_c = auc(x=np.array(list(range(vec.size))), y=vec)
        area_under_c /= (100. * (nbr_p - 1))
        area_under_c *= 100.  # (%)
        return area_under_c


# WSOL

def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))

def t2n(t):
    return t.detach().cpu().numpy().astype(float)


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def get_tag(args: object) -> str:
    tag = f"{args.task}-{args.train_dsets}-{args.netG['net_type']}"

    return tag


def estimate_var_hx(h: int, w: int, size: int) -> float:
    s = h
    csize = bsize = 1
    g = torch.arange(start=0, end=s, step=1, dtype=torch.float32,
                     requires_grad=False)

    g = g.view(-1, 1)  # s, 1
    g = g.repeat(csize, 1, 1)  # c, s, 1
    grid_h = g.repeat(bsize, 1, 1, 1)  # b, c, s, 1

    maps = torch.zeros((h, w))
    maps[int(h/2) - int(size/2): int(h/2) + int(size/2),
    int(w/2) - int(size/2): int(w/2) + int(size/2)] = 1
    maps = maps + 1e-8
    maps = torch.cat((maps.unsqueeze(0), maps.unsqueeze(0)), dim=0)
    maps = maps.unsqueeze(0)

    z = maps.sum(dim=[-2, -1], keepdim=True)  # b, c, 1, 1
    px = maps.sum(dim=-1, keepdim=True) / z  # b, c, h, 1

    x_hat = px * grid_h  # b, c, h, 1
    x_hat = x_hat.sum(dim=-2, keepdim=True)  # b, c, 1, 1

    x_var = px * ((grid_h - x_hat) ** 2)  # b, c, h, 1
    x_var = x_var.sum(dim=-2, keepdim=True)  # b, c, 1, 1

    x_var = x_var.squeeze()  # 2
    return x_var[0].item()


def estimate_var_wy(h: int, w: int, size: int) -> float:
    s = w
    csize = bsize = 1
    g = torch.arange(start=0, end=s, step=1, dtype=torch.float32,
                     requires_grad=False)

    g = g.view(1, -1)  # 1, s
    g = g.repeat(csize, 1, 1)  # c, 1, s
    grid_w = g.repeat(bsize, 1, 1, 1)  # b, c, 1, s

    maps = torch.zeros((h, w))
    maps[int(h / 2) - int(size / 2): int(h / 2) + int(size / 2),
    int(w / 2) - int(size / 2): int(w / 2) + int(size / 2)] = 1
    maps = maps + 1e-8
    maps = torch.cat((maps.unsqueeze(0), maps.unsqueeze(0)), dim=0)
    maps = maps.unsqueeze(0)

    z = maps.sum(dim=[-2, -1], keepdim=True)  # b, c, 1, 1
    py = maps.sum(dim=-2, keepdim=True) / z  # b, c, 1, w

    y_hat = py * grid_w  # b, c, 1, w
    y_hat = y_hat.sum(dim=-1, keepdim=True)  # b, c, 1, 1

    y_var = py * ((grid_w - y_hat) ** 2)  # b, c, 1, w
    y_var = y_var.sum(dim=-1, keepdim=True)  # b, c, 1, 1

    y_var = y_var.squeeze(-1).squeeze(-1)  # b, c

    y_var = y_var.squeeze()  # 2
    return y_var[0].item()


def bye(args):
    DLLogger.log(fmsg("End time: {}".format(args.tend)))
    DLLogger.log(fmsg("Total time: {}".format(args.tend - args.t0)))

    with open(join(root_dir, 'LOG.txt'), 'a') as f:
        m = f"{dt.datetime.now()}: \t " \
            f"Train Ds: {args.train_dsets} \t " \
            f"Valid Ds: {args.valid_dsets} \t " \
            f"Test Ds: {args.test_dsets} \t " \
            f"Task: {args.task} \t " \
            f"Method: {args.method} \t " \
            f"Net type: {args.netG['net_type']} \t " \
            f"... Passed in [{args.tend - args.t0}]. \n"
        f.write(m)

    with open(join(args.outd, 'passed.txt'), 'w') as fout:
        fout.write('Passed.')

    if is_cc():
        compress_and_delete_fd(args.outd, args.save_dir_imgs)

    DLLogger.log(fmsg('bye.'))

    # clean cc
    if is_cc():
        copy_exp_dir_node_to_scratch(args)


def compress_and_delete_fd(parent_fd, name_fd):
    _dir = join(parent_fd, name_fd)
    assert os.path.isdir(_dir), _dir

    cmdx = [
        "cd {} ".format(parent_fd),
        "tar -cf {}.tar.gz {} ".format(name_fd, name_fd),
        "rm -r {} ".format(name_fd)
    ]
    cmdx = " && ".join(cmdx)
    DLLogger.log("Running: {}".format(cmdx))
    try:
        subprocess.run(cmdx, shell=True, check=True)
    except subprocess.SubprocessError as e:
        DLLogger.log("Failed to run: {}. Error: {}".format(cmdx, e))


if __name__ == '__main__':
    print(root_dir)
