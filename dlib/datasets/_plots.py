import os
import sys
from os.path import dirname, abspath, join

import cv2
import matplotlib.pyplot as plt
import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.tools import check_scoremap_validity

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


if __name__ == '__main__':
    from PIL import Image
    from scipy import io
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    from dlib.utils.tools import check_box_convention


    def convert_bbox(bbox_xyxy: np.ndarray):
        check_box_convention(bbox_xyxy, 'x0y0x1y1')
        assert bbox_xyxy.shape == (1, 4)
        x0, y0, x1, y1 = bbox_xyxy.flatten()
        width = x1 - x0
        height = y1 - y0
        anchor = (x0, y1)
        return anchor, width, height


    path_img = 'YouTube-Objects-v1.0/boat/data/0006/shots/026/frame0001.jpg'
    path_mat = 'YouTube-Objects-v1.0/boat/data/0006/shots/026/frame0001.jpg_sticks.mat'
    img = Image.open(path_img).convert('RGB')

    mat = io.loadmat(path_mat)['coor']
    nbr_bx = mat.size
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    axes[0, 0].imshow(img)
    for el in mat.reshape(nbr_bx):

        bbox: np.ndarray = el.squeeze().reshape((1, 4))
        bbox = bbox * (bbox > 0)
        # plotting.
        gt_info = convert_bbox(bbox)
        color = mcolors.CSS4_COLORS['lime']
        rect = patches.Rectangle(gt_info[0], gt_info[1], -gt_info[2],
                                 linewidth=1.5,
                                 edgecolor=color,
                                 facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].axis('off')
        axes[0, 0].margins(0, 0)
        axes[0, 0].xaxis.set_major_locator(plt.NullLocator())
        axes[0, 0].yaxis.set_major_locator(plt.NullLocator())

        fig.show()
        plt.show()










