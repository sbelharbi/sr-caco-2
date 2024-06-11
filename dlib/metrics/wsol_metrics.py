import os
import sys
from os.path import dirname, abspath
from copy import deepcopy
from typing import Union

import cv2
import numpy as np

import torch.utils.data as torchdata
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.datasets.wsol_loader import get_image_ids
from dlib.datasets.wsol_loader import get_bounding_boxes
from dlib.datasets.wsol_loader import get_image_sizes
from dlib.datasets.wsol_loader import get_mask_paths

from dlib.utils.tools import check_scoremap_validity
from dlib.utils.tools import check_box_convention
from dlib.utils.tools import resize_bbox

from dlib.utils import constants
from dlib.parallel import sync_tensor_across_gpus


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = constants.CROP_SIZE  # 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


__all__ = ['BoxEvaluator', 'MaskEvaluator', 'load_mask_image', 'get_mask',
           'calculate_multiple_iou', 'compute_bboxes_from_scoremaps']


def cam2max_bbox(cam: np.ndarray, threshold: float) -> np.ndarray:
    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    check_scoremap_validity(cam)
    height, width = cam.shape
    scoremap_image = np.expand_dims((cam * 255).astype(np.uint8), 2)

    _, thr_gray_cam = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(cam)),
        maxval=255,
        type=cv2.THRESH_BINARY)

    contours = cv2.findContours(
        image=thr_gray_cam,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]])

    contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes)


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def compute_bboxes_from_scoremaps(scoremap: Union[list, None],
                                  scoremap_threshold_list,
                                  multi_contour_eval=False,
                                  bbox: Union[list, None] = None):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

        bbox: a list or None. some methods can predict a SINGLE bounding
            box without using cam. the list contains coordinates of upper
            left and lower right corners: [x1, y1, x2, y2]. x-axis: width,
            y-axi: height.

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    if scoremap is None:
        assert bbox is not None

    if scoremap is not None:
        check_scoremap_validity(scoremap)
        height, width = scoremap.shape
        scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)

        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    if scoremap is not None:
        for threshold in scoremap_threshold_list:
            boxes, number_of_box = scoremap2bbox(threshold)
            estimated_boxes_at_each_thr.append(boxes)
            number_of_box_list.append(number_of_box)
    else:
        boxes = np.array([bbox])
        number_of_box = 1
        for _ in scoremap_threshold_list:
            estimated_boxes_at_each_thr.append(boxes)
            number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, image_ids):
        self.scoremap_path = scoremap_path
        self.image_ids = image_ids

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, image_id + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        return cam, image_id

    def __len__(self):
        return len(self.image_ids)


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self,
                 metadata,
                 dataset_name,
                 split,
                 cam_threshold_list,
                 iou_threshold_list,
                 mask_root,
                 multi_contour_eval,
                 args):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.multi_contour_eval = multi_contour_eval

        self.best_tau_list = []
        self.curve_s = None

        self.top1 = None
        self.top5 = None
        self.curve_top_1_5 = None

        self.args = args

    def accumulate(self, scoremap, image_id, target, preds_ordered, bbox,
                   bbox_status):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def _synch_across_gpus(self):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

        self.num_correct_top1 = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.num_correct_top5 = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self,
                   scoremap: Union[np.ndarray, None],
                   image_id: str,
                   target: int,
                   preds_ordered: np.ndarray,
                   bbox: Union[list, None],
                   bbox_status: Union[float, None]):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float) or None. if
            None, bbox must not be None.
            image_id: string.
            target: int. longint. the true class label. for bbox evaluation (
            top1/5).
            preds_ordered: numpy.ndarray. vector of predicted labels ordered
            from from the most probable the least probable. for evaluation of
            bbox using top1/5.
            bbox: a list or None. some methods can predict a SINGLE bounding
            box without using cam. the list contains coordinates of upper
            left and lower right corners: [x1, y1, x2, y2]. x-axis: width,
            y-axi: height.
            bbox_status: float: 0 or 1. 1: for a valid box (lower right
            corner is located below upper left corner). 0: otherwise.
        """
        if scoremap is None:
            assert bbox is not None
            assert bbox_status in [0, 1]
            assert self.args.task == constants.C_BOX

            if bbox_status == 0:  # bbox that is not valid has 0 performance.
                self.cnt += 1
                return 0
        else:
            assert bbox is None
            assert bbox_status is None

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval,
            bbox=bbox)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))  # (nbr_boxes_in_img, 1)

        idx = 0
        sliced_multiple_iou = []  # length == number tau thresh
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            # pick the maximum score iou among all the boxes.
            idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]

            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1

            if target == preds_ordered[0]:
                self.num_correct_top1[_THRESHOLD][
                    correct_threshold_indices] += 1

            if target in preds_ordered[:5]:
                self.num_correct_top5[_THRESHOLD][
                    correct_threshold_indices] += 1

        self.cnt += 1

    def _synch_across_gpus(self):

        for tracker in [self.num_correct, self.num_correct_top1,
                        self.num_correct_top5]:
            for k in tracker.keys():
                _k_val = torch.from_numpy(tracker[k]).cuda(
                    self.args.c_cudaid).view(1, -1)

                tracker[k] = sync_tensor_across_gpus(
                    _k_val).sum(dim=0).cpu().view(-1).numpy()

        cnt = torch.tensor([self.cnt], dtype=torch.float,
                           requires_grad=False, device=torch.device(
                self.args.c_cudaid))

        cnt = sync_tensor_across_gpus(cnt)
        self.cnt = cnt.sum().cpu().item()

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []
        self.best_tau_list = []
        self.curve_s = {
            'x': self.cam_threshold_list
        }

        self.top1 = []
        self.top5 = []
        self.curve_top_1_5 = {
            'x': self.cam_threshold_list,
            'top1': dict(),
            'top5': dict()
        }

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

            self.curve_s[_THRESHOLD] = localization_accuracies

            self.best_tau_list.append(
                float(self.cam_threshold_list[np.argmax(
                    localization_accuracies)])
            )

            loc_acc = self.num_correct_top1[_THRESHOLD] * 100. / float(self.cnt)
            self.top1.append(loc_acc.max())

            self.curve_top_1_5['top1'][_THRESHOLD] = deepcopy(loc_acc)

            loc_acc = self.num_correct_top5[_THRESHOLD] * 100. / float(self.cnt)
            self.top5.append(loc_acc.max())

            self.curve_top_1_5['top5'][_THRESHOLD] = deepcopy(loc_acc)

        return max_box_acc


def load_mask_image(file_path, resize_size):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask


def get_mask(mask_root, mask_paths, ignore_path):
    """
    Ignore mask is set as the ignore box region \setminus the ground truth
    foreground region.

    Args:
        mask_root: string.
        mask_paths: iterable of strings.
        ignore_path: string.

    Returns:
        mask: numpy.ndarray(size=(224, 224), dtype=np.uint8)
    """
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (_RESIZE_LENGTH, _RESIZE_LENGTH))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

    ignore_file = os.path.join(mask_root, ignore_path)
    ignore_box_mask = load_mask_image(ignore_file,
                                      (_RESIZE_LENGTH, _RESIZE_LENGTH))
    ignore_box_mask = ignore_box_mask > 0.5

    ignore_mask = np.logical_and(ignore_box_mask,
                                 np.logical_not(mask_all_instances))

    if np.logical_and(ignore_mask, mask_all_instances).any():
        raise RuntimeError("Ignore and foreground masks intersect.")

    return (mask_all_instances.astype(np.uint8) +
            255 * ignore_mask.astype(np.uint8))


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        if self.dataset_name != constants.OpenImages:
            raise ValueError("Mask evaluation must be performed on OpenImages.")

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])

        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=float)

    def accumulate(self, scoremap, image_id, target=None, preds_ordered=None,
                   bbox=None, bbox_status=None):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
            target and preds_ordered are not used in this case.
        """
        check_scoremap_validity(scoremap)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(float)

    def get_best_operating_point(self, recall, precision):
        # todo delete.
        tmp = precision[:-3] + recall[:-3]
        idx = np.argmax(tmp[1:]) + 1

        return self.cam_threshold_list[idx]

    def _synch_across_gpus(self):

        for tracker in [self.gt_true_score_hist, self.gt_false_score_hist]:
            _k_val = torch.from_numpy(tracker).cuda(self.args.c_cudaid)
            tracker = sync_tensor_across_gpus(_k_val).sum(dim=0).cpu().numpy()

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        self.curve_s = {
            'x': recall,
            'y': precision
        }

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        self.best_tau_list = [self.get_best_operating_point(recall, precision)]

        # print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc
