import sys
import os
from os.path import join, dirname, abspath, basename, splitext
import math
import random
from datetime import datetime
from typing import Union


import numpy as np
import torch
import cv2
from torchvision.utils import make_grid
import torch.nn.functional as F

# import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/twhui/SRGAN-pyTorch
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def surf(Z, cmap='rainbow', figsize=None):
    plt.figure(figsize=figsize)
    ax3 = plt.axes(projection='3d')

    w, h = Z.shape[:2]
    xx = np.arange(0, w, 1)
    yy = np.arange(0, h, 1)
    X, Y = np.meshgrid(xx, yy)
    ax3.plot_surface(X, Y, Z, cmap=cmap)
    # ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)
    plt.show()


'''
# --------------------------------------------
# get image pathes
# --------------------------------------------
'''


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


'''
# --------------------------------------------
# split large images into small images 
# --------------------------------------------
'''


def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]
    patches = []
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w - p_size, p_size - p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h - p_size, p_size - p_overlap, dtype=np.int))
        w1.append(w - p_size)
        h1.append(h - p_size)
        # print(w1)
        # print(h1)
        for i in w1:
            for j in h1:
                patches.append(img[i:i + p_size, j:j + p_size, :])
    else:
        patches.append(img)

    return patches


def imssave(imgs, img_path):
    """
    imgs: list, N images of size WxHxC
    """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    for i, img in enumerate(imgs):
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        new_path = os.path.join(os.path.dirname(img_path),
                                img_name + str('_{:04d}'.format(i)) + '.png')
        cv2.imwrite(new_path, img)


def split_imageset(original_dataroot, taget_dataroot, n_channels=3, p_size=512,
                   p_overlap=96, p_max=800):
    """
    split the large images from original_dataroot into small overlapped images with size (p_size)x(p_size),
    and save them into taget_dataroot; only the images with larger size than (p_max)x(p_max)
    will be splitted.

    Args:
        original_dataroot:
        taget_dataroot:
        p_size: size of small images
        p_overlap: patch size in training is a good choice
        p_max: images with smaller size than (p_max)x(p_max) keep unchanged.
    """
    paths = get_image_paths(original_dataroot)
    for img_path in paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img = imread_uint(img_path, n_channels=n_channels)
        patches = patches_from_image(img, p_size, p_overlap, p_max)
        imssave(patches,
                os.path.join(taget_dataroot, os.path.basename(img_path)))
        # if original_dataroot == taget_dataroot:
        # del img_path


'''
# --------------------------------------------
# makedir
# --------------------------------------------
'''


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


'''
# --------------------------------------------
# read image from path
# opencv is fast, but read BGR numpy image
# --------------------------------------------
'''

def _is_caco2(path: str) -> bool:
    return ( ('caco2' in path) and
            # todo: weak test.
            any([constants.CELL0 in path,
                 constants.CELL1 in path,
                 constants.CELL2 in path])
            )


def _is_biosr(path: str) -> bool:
    # todo: weak assumption.
    return 'biosr' in path


def get_cell_type(path: str) -> Union[str, None]:
    # fixit

    return None


def _convert_single_chan_to_rgb(img: np.ndarray) -> np.ndarray:

    assert isinstance(img, np.ndarray), type(img)
    assert img.ndim == 2, img.ndim  # HW
    img = np.expand_dims(img, axis=2)  # HxWx1
    img = np.repeat(img, 3, axis=2)  # HW3: RGB

    return img


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------


def imread_uint(path, n_channels=3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)

    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    else:
        raise NotImplementedError(f'n channels: {n_channels}')

    return img


# --------------------------------------------
# matlab's imwrite
# --------------------------------------------

def cv2_imsave_rgb_in(img: np.ndarray, img_path: str):
    # img: CHW, or HW.
    assert img.ndim in [2, 3], img.ndim

    img = np.squeeze(img)

    if _is_biosr(img_path):
        assert img.ndim == 2, img.ndim
        img = _convert_single_chan_to_rgb(img)  # HW3, RGB


    if img.ndim == 3:  # CHW
        img = img[:, :, [2, 1, 0]]  # BGR
    cv2.imwrite(img_path, img)


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


# --------------------------------------------
# get single image of size HxWxn_channles (BGR)
# --------------------------------------------
def read_img(path):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


'''
# --------------------------------------------
# image format conversion
# --------------------------------------------
# numpy(single) <--->  numpy(uint)
# numpy(single) <--->  tensor
# numpy(uint)   <--->  tensor
# --------------------------------------------
'''


# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(uint)
# --------------------------------------------


def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def uint162single(img):
    return np.float32(img / 65535.)


def single2uint16(img):
    return np.uint16((img.clip(0, 1) * 65535.).round())


# --------------------------------------------
# numpy(uint) (HxWxC or HxW) <--->  tensor
# --------------------------------------------


# convert uint to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0,
                                                               1).float().div(
        255.).unsqueeze(0)


# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0,
                                                               1).float().div(
        255.)


# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def tensor2uint82float(img: torch.Tensor) -> torch.Tensor:
    _img = (img.float().clamp(0, 1) * 255.0).round().clamp(0, 255).float()

    return _img


# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------


# convert single (HxWxC) to 3-dimensional torch tensor:CxHxW.
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0,
                                                               1).float().unsqueeze(
        0)


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img


# convert torch tensor to single
def tensor2single3(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def single2tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1,
                                                               3).float().unsqueeze(
        0)


def single32tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(
        0).unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1,
                                                               3).float()


# from skimage.io import imread, imsave
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array of BGR channel order
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(
        *min_max)  # squeeze first, then clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)),
                           normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(
                n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.uint8() WILL NOT round by default.
    return img_np.astype(out_type)


'''
# --------------------------------------------
# Augmentation, flip and/or rotate
# --------------------------------------------
# The following two are enough.
# (1) augmet_img: numpy image of WxHxC or WxH
# (2) augment_img_tensor4: tensor image 1xCxWxH
# --------------------------------------------
'''


def augment_img(img: np.ndarray, mode: int = 0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def augment_img_tensor4(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])


def augment_img_tensor(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)


def augment_img_np3(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.transpose(1, 0, 2)
    elif mode == 2:
        return img[::-1, :, :]
    elif mode == 3:
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 4:
        return img[:, ::-1, :]
    elif mode == 5:
        img = img[:, ::-1, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 6:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        return img
    elif mode == 7:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img


def augment_imgs(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


'''
# --------------------------------------------
# modcrop and shave
# --------------------------------------------
'''


def modcrop(img_in: np.ndarray, scale) -> np.ndarray:
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def shave(img_in, border=0):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    h, w = img.shape[:2]
    img = img[border:h - border, border:w - border]
    return img


'''
# --------------------------------------------
# image processing process on numpy image
# channel_convert(in_c, tar_type, img_list):
# rgb2ycbcr(img, only_y=True):
# bgr2ycbcr(img, only_y=True):
# ycbcr2rgb(img):
# --------------------------------------------
'''


def mb_gpu_rgb2ycbcr(img: torch.Tensor, only_y: bool = True) -> torch.Tensor:
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    assert torch.is_tensor(img), type(img)
    assert img.ndim == 4, img.ndim
    assert img.shape[1] == 3, img.shape[1]

    in_img_type = img.dtype
    _img = img.float()

    if in_img_type != torch.uint8:
        _img = _img * 255.

    r: torch.Tensor = _img[:, 0, :, :]
    g: torch.Tensor = _img[:, 1, :, :]
    b: torch.Tensor = _img[:, 2, :, :]
    # convert
    y = (65.481 * r + 128.553 * g + 24.966 * b) / 255. + 16.0
    if only_y:
        out = y.unsqueeze(1)
    else:
        cb = (- 37.797 * r - 74.203 * g + 112.0 * b) / 255. + 128.0
        cr = (112.0 * r - 93.786 * g - 18.214 * b) / 255. + 128.0
        out = torch.stack((y, cb, cr), 1)
        assert out.shape == _img.shape

    if in_img_type == torch.uint8:
        out = out.round().clamp(0, 255)
    else:
        out = (out / 255.).clamp(0.0, 1.0)

    return out.type(in_img_type)


def rgb2ycbcr(img: np.ndarray, only_y: bool = True) -> np.ndarray:
    '''same as matlab rgb2ycbcr.
    only_y: only return Y channel
    Input:
        HWC, C = 3.
        uint8, [0, 255]
        float, [0, 1]
    '''
    assert img.ndim == 3, img.ndim
    assert img.shape[2] == 3, img.shape[2]
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img: np.ndarray) -> np.ndarray:
    '''same as matlab ycbcr2rgb
    Input:
        BCHW. C = 3.
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                          [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921,
                                                                    135.576,
                                                                    -276.836]
    rlt = np.clip(rlt, 0, 255)
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


'''
# --------------------------------------------
# metric, MSE, PSNR, SSIM and PSNRB
# --------------------------------------------
'''


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_mse(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return mse

# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)

    # small mse (< 1e45) will lead to psnr of inf.
    # we replace this mse by the smallest mse value that does not lead to
    # psnr of inf. psnr(mse=1e-45) = 496.6655. psnr(mse=1e-46) = inf.
    # We follow pytorch precision.
    # in python:
    # 20. * math.log10(255.0 / math.sqrt(1e-323)) = 3278.1826570831977
    # but in pytorch: pnsr(mse=1e-323) = inf.
    # in pytorch: lowest mse value that does not trigger inf is:
    # pnsr(mse=1e-45) = 496.6655.
    # typically, it is black sample (without cells), that lead to a very low
    # mse since it is easy. Their perforamnce is less relevant. to avoid
    # averaging with inf, we set this conversion, and replace low mse to a
    # constant value.


    if mse < 1e-45:
        mse = 1e-45

    return 20 * math.log10(255.0 / math.sqrt(mse))


def gpu_calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, border: int = 0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError(f'Mismatched shapes: im1 {img1.shape} im2: '
                         f'{img2.shape}.')

    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    h, w = img1.shape[:2]

    img1 = img1.double()
    img2 = img2.double()
    mse = ((img1 - img2) ** 2).mean()

    # small mse (< 1e45) will lead to psnr of inf.
    # we replace this mse by the smallest mse value that does not lead to
    # psnr of inf. psnr(mse=1e-45) = 496.6655. psnr(mse=1e-46) = inf.
    # typically, it is black sample (without cells), that lead to a very low
    # mse since it is easy. Their perforamnce is less relevant. to avoid
    # averaging with inf, we set this conversion, and replace low mse to a
    # constant value.

    if mse < 1e-45:
        mse = torch.tensor([1e-45], device=img1.device).double()

    return 20. * torch.log10(255.0 / torch.sqrt(mse))


def mbatch_gpu_calculate_psnr(img1: torch.Tensor,
                              img2: torch.Tensor,
                              border: int = 0,
                              roi: torch.Tensor = None
                              ) -> torch.Tensor:
    # img1 and img2 have range [0, 255]
    assert img1.ndim == 4, img1.ndim
    assert img2.ndim == 4, img2.ndim
    assert img1.shape == img2.shape, f"{img1.shape} {img2.shape}"

    if roi is not None:
        assert roi.ndim == 4, roi.ndim
        c = roi.shape[1]
        assert c == 1, f"dont support c = {c} > 1."
        assert roi.shape[0] == img1.shape[0], f"{roi.shape[0]} {img1.shape[0]}"
        assert roi.shape[2] == img1.shape[2], f"{roi.shape[2]} {img1.shape[2]}"
        assert roi.shape[3] == img1.shape[3], f"{roi.shape[3]} {img1.shape[3]}"

    b, c, h, w = img1.shape
    img1 = img1[:, :, border:h - border, border:w - border]
    img2 = img2[:, :, border:h - border, border:w - border]

    if roi is not None:
        roi = roi[:, :, border:h - border, border:w - border]

    b, c, h, w = img1.shape

    img1 = img1.double()
    img2 = img2.double()

    if roi is None:
        mse = ((img1 - img2) ** 2).contiguous().view(b, c * h * w).mean(-1)
    else:
        roi = roi.double()
        diff = img1 - img2
        diff = diff * roi
        tt = roi.contiguous().view(b, -1).sum(dim=-1)  # b
        tt[tt == 0] = 1.  # in case of images with empty roi.
        mse = (diff ** 2).contiguous().view(b, c * h * w).sum(dim=-1) / tt

    mse[mse < 1e-45] = 1e-45  # small mse (< 1e45) will lead to psnr of inf.
    # we replace this mse by the smallest mse value that does not lead to
    # psnr of inf. psnr(mse=1e-45) = 496.6655. psnr(mse=1e-46) = inf.
    # typically, it is black sample (without cells), that lead to a very low
    # mse since it is easy. Their perforamnce is less relevant. to avoid
    # averaging with inf, we set this conversion, and replace low mse to a
    # constant value.

    return 20. * torch.log10(255.0 / torch.sqrt(mse)).contiguous().view(b, )


def mbatch_gpu_calculate_mse(img1: torch.Tensor,
                             img2: torch.Tensor,
                             border: int = 0,
                             roi: torch.Tensor = None
                             ) -> torch.Tensor:
    # img1 and img2 have range [0, 255]
    assert img1.ndim == 4, img1.ndim
    assert img2.ndim == 4, img2.ndim
    assert img1.shape == img2.shape, f"{img1.shape} {img2.shape}"

    if roi is not None:
        assert roi.ndim == 4, roi.ndim
        c = roi.shape[1]
        assert c == 1, f"dont support c = {c} > 1."
        assert roi.shape[0] == img1.shape[0], f"{roi.shape[0]} {img1.shape[0]}"
        assert roi.shape[2] == img1.shape[2], f"{roi.shape[2]} {img1.shape[2]}"
        assert roi.shape[3] == img1.shape[3], f"{roi.shape[3]} {img1.shape[3]}"

    b, c, h, w = img1.shape
    img1 = img1[:, :, border:h - border, border:w - border]
    img2 = img2[:, :, border:h - border, border:w - border]

    if roi is not None:
        roi = roi[:, :, border:h - border, border:w - border]

    b, c, h, w = img1.shape

    if roi is None:
        img1 = img1.double()
        img2 = img2.double()
        mse = ((img1 - img2) ** 2).contiguous().view(b, c * h * w).mean(-1)
        # dim: b.
    else:
        roi = roi.double()
        diff = img1 - img2
        diff = diff * roi
        tt = roi.contiguous().view(b, -1).sum(dim=-1)  # b
        tt[tt == 0] = 1.  # in case of images with empty roi.
        mse = (diff ** 2).contiguous().view(b, c * h * w).sum(dim=-1) / tt

    return mse


def mbatch_gpu_calculate_nrmse(img: torch.Tensor,
                               y: torch.Tensor,
                               border: int = 0,
                               roi: torch.Tensor = None
                               ) -> torch.Tensor:
    """
    Compute NRMSE (normalized root mean squared error)
    https://en.wikipedia.org/wiki/Root-mean-square_deviation.
    :param img: prediction image.
    :param y: ground truth image.
    :param border: border.
    :return: nrmse.
    """
    # img and y have range [0, 255]
    assert img.ndim == 4, img.ndim
    assert y.ndim == 4, y.ndim
    assert img.shape == y.shape, f"{img.shape} {y.shape}"

    if roi is not None:
        assert roi.ndim == 4, roi.ndim
        c = roi.shape[1]
        assert c == 1, f"dont support c = {c} > 1."
        assert roi.shape[0] == img.shape[0], f"{roi.shape[0]} {img.shape[0]}"
        assert roi.shape[2] == img.shape[2], f"{roi.shape[2]} {img.shape[2]}"
        assert roi.shape[3] == img.shape[3], f"{roi.shape[3]} {img.shape[3]}"


    b, c, h, w = img.shape
    img = img[:, :, border:h - border, border:w - border]
    y = y[:, :, border:h - border, border:w - border]

    if roi is not None:
        roi = roi[:, :, border:h - border, border:w - border]

    b, c, h, w = img.shape

    img = img.double()
    y = y.double()

    if roi is None:
        mse = ((img - y) ** 2).contiguous().view(b, c * h * w).mean(-1)
        _y = y.contiguous().view(b, c * h * w)
        _min = _y.min(dim=-1)[0]

    else:
        roi = roi.double()
        diff = img - y
        diff = diff * roi

        tt = roi.contiguous().view(b, -1).sum(dim=-1)  # b
        tt[tt == 0] = 1.  # in case of images with empty roi.
        mse = (diff ** 2).contiguous().view(b, c * h * w).sum(dim=-1) / tt

        _min_all = y.contiguous().view(b, c * h * w).min(dim=-1)[0]

        _y = (y * roi).contiguous().view(b, c * h * w)
        _min_roi = _y.min(dim=-1)[0]
        _min = torch.maximum(_min_all, _min_roi)

    rmse = torch.sqrt(mse)
    _max = _y.max(dim=-1)[0]
    denom =  _max - _min
    # todo: denom == 0.
    denom[denom == 0] = 1.
    nrmse = rmse / denom
    nrmse[denom == 0] = 0.0  # discard samples where each pixel has the same
    # value.

    # dim: b.

    return nrmse


def _ssim_per_channel(x: torch.Tensor,
                      y: torch.Tensor,
                      kernel: torch.Tensor,
                      k1: float = 0.01,
                      k2: float = 0.03,
                      roi: torch.Tensor = None
                      ) -> torch.Tensor:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel. (c, 1, kernel, kernel.)
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or
            NaN results.
    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """
    assert x.ndim == 4, x.ndim
    assert y.ndim == 4, y.ndim
    assert x.shape == y.shape, f"{x.shape} {y.shape}"
    assert kernel.ndim == 4, kernel.ndim
    z0, z1, z2, z3 = kernel.shape
    assert z0 == x.shape[1], f"{z0} {x.shape}"

    if roi is not None:
        assert roi.ndim == 4, roi.ndim
        c = roi.shape[1]
        assert c == 1, f"dont support c = {c} > 1."
        assert roi.shape[0] == x.shape[0], f"{roi.shape[0]} {x.shape[0]}"
        assert roi.shape[2] == x.shape[2], f"{roi.shape[2]} {x.shape[2]}"
        assert roi.shape[3] == x.shape[3], f"{roi.shape[3]} {x.shape[3]}"

    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input '
                         f'size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')


    pad_h = (z2 - 1) // 2
    pad_w = (z3 - 1) // 2

    # no padding.
    # x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode="reflect")
    # y = F.pad(y, [pad_w, pad_w, pad_h, pad_h], mode="reflect")

    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0,
                        groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0,
                        groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0,
                        groups=n_channels) - mu_xy

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = ( (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) ) * cs

    # ss: b, c, h, w.
    b, c, h, w = ss.shape

    if roi is None:
        ss = ss.contiguous().view(b, c, h * w)
        ssim_val = ss.mean(dim=-1)  # b, c

    else:
        # remove shirked regions at the borders du to convolution.
        _, _, _h, _w = roi.shape
        roi = roi[:, :, pad_h: _h - pad_h, pad_w: _w - pad_w]

        ss = ss * roi
        tt = roi.contiguous().view(b, -1).sum(dim=-1)  # b
        tt[tt == 0] = 1.  # in case of images with empty roi.
        tt = tt.view(-1, 1)
        ss = ss.contiguous().view(b, c, h * w)
        ssim_val = ss.sum(dim=-1) / tt  # b, c

    return ssim_val


def _gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords = coords - (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def mbatch_gpu_calculate_ssim(x: torch.Tensor,
                              y: torch.Tensor,
                              border: int = 0,
                              roi: torch.Tensor = None
                              ) -> torch.Tensor:
    r"""
    Reference:
    https://github.com/photosynthesis-team/piq/blob/
    53c6b8511a54015067d58d669a242574e8da5077/piq/ssim.py
    Interface of Structural Similarity (SSIM) index.
    Inputs supposed to be in range ``[0, 255]``.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        border: int. dfault: 0.
    Returns:
        Value of Structural Similarity (SSIM) index.
    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: `10.1109/TIP.2003.819861`
    """
    data_range = 255.
    kernel_size = 11
    kernel_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

    assert x.ndim == 4, x.ndim  # b, c, h, w.
    assert y.ndim == 4, y.ndim

    assert x.shape == y.shape, f"x: {x.shape}. y: {y.shape}"

    if roi is not None:
        assert roi.ndim == 4, roi.ndim
        c = roi.shape[1]
        assert c == 1, f"dont support c = {c} > 1."
        assert roi.shape[0] == x.shape[0], f"{roi.shape[0]} {x.shape[0]}"
        assert roi.shape[2] == x.shape[2], f"{roi.shape[2]} {x.shape[2]}"
        assert roi.shape[3] == x.shape[3], f"{roi.shape[3]} {x.shape[3]}"

    _min_x = x.min()
    _max_x = x.max()
    assert 0.0 <= _min_x <= data_range, f"min x: 0 <= {_min_x} <= {data_range}"
    assert 0.0 <= _max_x <= data_range, f"max x: 0 <= {_max_x} <= {data_range}"

    _min_y = y.min()
    _max_y = y.max()
    assert 0.0 <= _min_y <= data_range, f"min y: 0 <= {_min_y} <= {data_range}"
    assert 0.0 <= _max_y <= data_range, f"max y: 0 <= {_max_y} <= {data_range}"

    _, _, h, w = x.shape
    x = x[:, :, border:h - border, border:w - border]
    y = y[:, :, border:h - border, border:w - border]

    if roi is not None:
        roi = roi[:, :, border:h - border, border:w - border]

    x = x / float(data_range)
    y = y / float(data_range)

    kernel = _gaussian_filter(kernel_size, kernel_sigma).repeat(
        x.size(1), 1, 1, 1).to(y.device)  # c, 1, kernel, kernel.
    _compute_ssim_per_channel = _ssim_per_channel
    ssim_p_c = _compute_ssim_per_channel(x=x,
                                         y=y,
                                         kernel=kernel,
                                         k1=k1,
                                         k2=k2,
                                         roi=roi
                                         )
    # b, c

    ssim_val = ssim_p_c.mean(1)  # b

    return ssim_val



# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) *
                (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
            (im[:, :, :, block_horizontal_positions] - im[:, :, :,
                                                       block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
            (im[:, :, block_vertical_positions, :] - im[:, :,
                                                     block_vertical_positions + 1,
                                                     :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(
        torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1),
                                               block_vertical_positions)

    horizontal_nonblock_difference = (
            (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :,
                                                          nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
            (im[:, :, nonblock_vertical_positions, :] - im[:, :,
                                                        nonblock_vertical_positions + 1,
                                                        :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (
                                      horizontal_block_difference + vertical_block_difference) / (
                                  n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (
                                         horizontal_nonblock_difference + vertical_nonblock_difference) / (
                                     n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def calculate_psnrb(img1, img2, border=0):
    """Calculate PSNR-B (Peak Signal-to-Noise Ratio).
    Ref: Quality assessment of deblocked images, for JPEG image deblocking evaluation
    # https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        img1, img2 = np.expand_dims(img1, 2), np.expand_dims(img2, 2)

    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # follow https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

    total = 0
    for c in range(img1.shape[1]):
        mse = torch.nn.functional.mse_loss(img1[:, c:c + 1, :, :],
                                           img2[:, c:c + 1, :, :],
                                           reduction='none')
        bef = _blocking_effect_factor(img1[:, c:c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return float(total) / img1.shape[1]


'''
# --------------------------------------------
# matlab's bicubic imresize (numpy and torch) [0, 1]
# --------------------------------------------
'''


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + \
           (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
               ((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel,
                              kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and
        # antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
        0, P - 1, P).view(1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# --------------------------------------------
# imresize for tensor image [0, 1]
# --------------------------------------------
def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: pytorch tensor, CHW or HW [0,1]
    # output: CHW or HW [0,1] w/o round
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img = img.unsqueeze(0)

    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0,
                                                                             1).mv(
                weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(
                weights_W[i])
    if need_squeeze:
        out_2.squeeze_()
    return out_2


# --------------------------------------------
# imresize for numpy image [0, 1]
# --------------------------------------------
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0,
                                                                             1).mv(
                weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(
                weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()


def test_gpu_cpu_psnr():
    import time
    from dlib.utils.utils_reproducibility import set_seed
    set_seed(0)

    print('time 1 samples ------')
    device = torch.device('cuda:1')
    h, w = 256, 256
    im1 = np.uint8((np.random.rand(h, w) * 255.0).round())
    im2 = np.uint8((np.random.rand(h, w) * 255.0).round())
    t0 = time.perf_counter()
    psnr_np = calculate_psnr(im1, im2)
    print(f'numpy. psnr: {psnr_np}, time: {time.perf_counter() - t0}')

    im1_g = torch.from_numpy(im1.astype(np.float32)).to(device)
    im2_g = torch.from_numpy(im2.astype(np.float32)).to(device)
    for i in range(10):
        t0 = time.perf_counter()
        gpu_psnr = gpu_calculate_psnr(im1_g, im2_g)
        print(f'torch. psnr: {gpu_psnr}, time: {time.perf_counter() - t0}')

    b = 64
    print(f'time {b} samples ------')
    # numpy
    psnr_np = 0.0
    t0 = time.perf_counter()
    for i in range(b):
        psnr_np += calculate_psnr(im1, im2)
    print(f'numpy[batch: {b}]. psnr: {psnr_np / b}, time:'
          f' {time.perf_counter() - t0}')

    # torch
    im1_g = im1_g.unsqueeze(0).repeat(b, 1, 1, 1)
    im2_g = im2_g.unsqueeze(0).repeat(b, 1, 1, 1)
    for i in range(10):
        t0 = time.perf_counter()
        gpu_psnr = mbatch_gpu_calculate_psnr(im1_g, im2_g).sum()
        print(f'torch[batch: {b}]. psnr: {gpu_psnr / b}, time:'
              f' {time.perf_counter() - t0}')


def test_gpu_cpu_rgb2ycbcr():
    import time
    from dlib.utils.utils_reproducibility import set_seed
    set_seed(0)

    device = torch.device('cuda:1')
    b, c, h, w = 1, 3, 256, 256

    im = np.clip((np.random.rand(b, c, h, w) * 255.0).round(), a_min=0.0,
                 a_max=255.).astype(np.uint8)
    gpu_im = torch.from_numpy(im).type(torch.uint8).to(device)

    ybr = rgb2ycbcr(np.transpose(im.squeeze(0), [1, 2, 0]), only_y=True)
    gpu_ybr = mb_gpu_rgb2ycbcr(gpu_im, only_y=True)
    print(f' numpy: {ybr.sum()} torch: {gpu_ybr.sum()}')
    print(ybr)
    print(gpu_ybr)


def test_bicubic_inter():
    import time

    import torch.nn.functional as F
    from dlib.utils.utils_reproducibility import set_seed
    set_seed(0)

    a = torch.rand(224, 224)
    for scale in [2, 4, 8]:
        torch_inter = F.interpolate(input=a.unsqueeze(0).unsqueeze(0),
                                    scale_factor=scale,
                                    mode='bicubic',
                                    antialias=True
                                    ).squeeze()
        our_inter = imresize(a, scale=scale, antialiasing=True)
        diff = (torch_inter - our_inter).abs().sum()
        print(f'scale: {scale} | diff: {diff}')


def test_mbatch_gpu_calculate_ssim():
    import torch.nn.functional as F
    from dlib.utils.utils_reproducibility import set_seed
    set_seed(0)
    device = torch.device('cuda:0')
    b, c, h, w = 1, 1, 512, 512
    img1 = torch.rand(b, c, h, w) * 255.
    img1 = torch.clip(img1, 0., 255.).to(device)

    img2 = torch.rand(b, c, h, w) * 255.
    img2 = torch.clip(img2, 0., 255.).to(device)

    # without roi:
    gpu_ssim = mbatch_gpu_calculate_ssim(img1, img2)
    print(f'gpu ssim img1 vs img2 [no roi]: {gpu_ssim[0]}, {gpu_ssim.shape}')

    gpu_ssim = mbatch_gpu_calculate_ssim(img1, img1)
    print(f'gpu ssim img1 vs img1 [no roi]: {gpu_ssim}, {gpu_ssim.shape}')

    roi = (torch.rand(b, 1, h, w) > 0.01).float().to(device)
    gpu_ssim = mbatch_gpu_calculate_ssim(img1, img2, roi=roi)
    print(f'gpu ssim img1 vs img2 [w/ roi]: {gpu_ssim[0]}, {gpu_ssim.shape}')

    gpu_ssim = mbatch_gpu_calculate_ssim(img1, img1, roi=roi)
    print(f'gpu ssim img1 vs img1 [w/ roi]: {gpu_ssim}, {gpu_ssim.shape}')

    # numpy version
    assert b == 1, b
    assert c == 1, c
    img1_np = img1.cpu().squeeze().numpy()
    img2_np = img2.cpu().squeeze().numpy()
    np_ssim = calculate_ssim(img1_np, img2_np)
    print(f'numpy ssim img1 vs img2 [no roi]: {np_ssim}')

    np_ssim = calculate_ssim(img1_np, img1_np)
    print(f'numpy ssim img1 vs img1: {np_ssim}')


if __name__ == '__main__':
    a = 0
    # img = imread_uint('test.bmp', 3)
    # img = uint2single(img)
    # img_bicubic = imresize_np(img, 1/4)
    # imshow(single2uint(img_bicubic))
    #
    # img_tensor = single2tensor4(img)
    # for i in range(8):
    #     imshow(np.concatenate((augment_img(img, i),
    #         tensor2single(augment_img_tensor4(img_tensor, i))), 1))
    #
    # patches = patches_from_image(img, p_size=128, p_overlap=0, p_max=200)
    # imssave(patches,'a.png')

    # -----------

    # test_gpu_cpu_psnr()
    # test_gpu_cpu_rgb2ycbcr()

    # test bicubic.
    # test_bicubic_inter()

    # test ssim
    test_mbatch_gpu_calculate_ssim()






