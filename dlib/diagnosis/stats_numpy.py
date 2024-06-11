import os
import sys
from os.path import dirname, abspath, join, basename, splitext
from typing import List, Tuple
import fnmatch
import argparse
import pprint

import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


__all__ = ['normed_histogram', 'unnormed_histogram']


def normed_histogram(x: np.ndarray, nbins: int, range = Tuple):
    return np.histogram(x.flatten(), bins=nbins, range=range,
                        weights=(1./x.size) * np.ones(x.size), density=False)

def unnormed_histogram(x: np.ndarray, nbins: int, range = Tuple):
    return np.histogram(x.flatten(), bins=nbins, range=range,
                        weights=np.ones(x.size), density=False)
