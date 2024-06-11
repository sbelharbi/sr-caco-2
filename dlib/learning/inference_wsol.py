import subprocess

import numpy as np
import os
import sys
from os.path import dirname, abspath, join, basename
from typing import Tuple, Union, List

import torch
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F


from torch.cuda.amp import autocast
import torch.distributed as dist

from tqdm import tqdm as tqdm
import cv2

from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.metrics.wsol_metrics import BoxEvaluator
from dlib.metrics.wsol_metrics import MaskEvaluator
from dlib.metrics.wsol_metrics import compute_bboxes_from_scoremaps
from dlib.metrics.wsol_metrics import calculate_multiple_iou
from dlib.metrics.wsol_metrics import get_mask
from dlib.metrics.wsol_metrics import load_mask_image
from dlib.metrics.wsol_metrics import cam2max_bbox

from dlib.cams.core_seeding import STOtsu

from dlib.datasets.wsol_loader import configure_metadata
from dlib.visualization.vision_wsol import Viz_WSOL

from dlib.utils.tools import t2n
from dlib.utils.tools import check_scoremap_validity
from dlib.utils.tools import check_box_convention
from dlib.utils.tools import get_cpu_device
from dlib.utils import constants
from dlib.cams import build_std_cam_extractor
from dlib.cams import build_fcam_extractor
from dlib.cams import build_tcam_extractor
from dlib.utils.shared import reformat_id

from dlib.utils.utils_reproducibility import set_seed
import dlib.dllogger as DLLogger


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = constants.CROP_SIZE  # 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float).
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()

    return cam


def max_normalize(cam):
    max_val = cam.max()
    if max_val == 0.:
        return cam

    return cam / max_val


def entropy_cam(cam: torch.Tensor) -> torch.Tensor:
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == 2

    ops = 1. - cam
    entrop = - cam * torch.log2(cam) - ops * torch.log2(ops)
    assert ((entrop > 1.) + (entrop < 0.)).sum() == 0.

    return entrop


class CAMComputer(object):
    def __init__(self,
                 args,
                 model,
                 loader: DataLoader,
                 metadata_root,
                 mask_root,
                 iou_threshold_list,
                 dataset_name,
                 split,
                 multi_contour_eval,
                 cam_curve_interval: float = .001,
                 out_folder=None,
                 fcam_argmax: bool = False,
                 classifier=None,
                 box_stats=None,
                 blur_op=None
                 ):

        if args.task == constants.C_BOX:
            assert classifier is not None
            assert box_stats is not None
            assert blur_op is not None

        self.args = args
        self.model = model
        self.model.eval()
        self.classifier = classifier
        self.box_stats = box_stats
        self.blur_op = blur_op
        if classifier is not None:
            self.classifier.eval()

        self.loader = loader
        self.dataset_name = dataset_name
        self.split = split
        self.out_folder = out_folder
        self.fcam_argmax = fcam_argmax

        self.multi_contour_eval = multi_contour_eval

        if args.task in [constants.F_CL, constants.TCAM]:
            self.req_grad = False
        elif args.task == constants.C_BOX:
            self.req_grad = False
        elif args.task == constants.STD_CL:
            self.req_grad = constants.METHOD_REQU_GRAD[args.method]
        else:
            raise NotImplementedError

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        # todo: needs to be done in a better way.
        self.evaluator = None  # not needed for trainset.
        if split != constants.TRAINSET:
            self.evaluator = {
                constants.OpenImages: MaskEvaluator,
                constants.CUB: BoxEvaluator,
                constants.ILSVRC: BoxEvaluator,
                constants.YTOV1: BoxEvaluator
            }[dataset_name](metadata=metadata,
                            dataset_name=dataset_name,
                            split=split,
                            cam_threshold_list=cam_threshold_list,
                            iou_threshold_list=iou_threshold_list,
                            mask_root=mask_root,
                            multi_contour_eval=multi_contour_eval,
                            args=args)

        if dataset_name in [constants.CUB, constants.ILSVRC, constants.YTOV1]:
            self.bbox = True
        elif dataset_name in [constants.OpenImages]:
            self.bbox = False
        else:
            raise NotImplementedError

        self.viz = Viz_WSOL()
        self.default_seed = int(os.environ["MYSEED"])

        self.cbox_status = dict()
        self.cbox_status_counter = 0.
        self.cbox_status_total = 0.

        self.std_cam_extractor = None
        self.fcam_extractor = None
        self.tcam_extractor = None

        if args.task == constants.STD_CL:
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=self.model, args=self.args)
        elif args.task == constants.F_CL:
            self.fcam_extractor = self._build_fcam_extractor(
                model=self.model, args=self.args)
            # useful for drawing side-by-side.
            # todo: build classifier from scratch and create its cam extractor.
        elif args.task == constants.TCAM:
            self.tcam_extractor = self._build_tcam_extractor(
                model=self.model, args=self.args)
        elif args.task == constants.C_BOX:
            pass
        else:
            raise NotImplementedError

    def reset_cbox_status(self):
        self.cbox_status = dict()
        self.cbox_status_counter = 0.
        self.cbox_status_total = 0.

    def update_cbox_status(self, image_id: str, status: float):
        self.cbox_status[image_id] = status
        self.cbox_status_counter += status
        self.cbox_status_total += 1.

    def get_failed_boxes_mtr(self) -> float:
        assert self.args.task == constants.C_BOX
        assert self.cbox_status_total > 0
        total = self.cbox_status_total
        correct = self.cbox_status_counter

        return 100. * (total - correct) / float(total)

    def _build_std_cam_extractor(self, classifier, args):
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _build_fcam_extractor(self, model, args):
        return build_fcam_extractor(model=model, args=args)

    def _build_tcam_extractor(self, model, args):
        return build_tcam_extractor(model=model, args=args)

    def get_fg_imgs(self, images: torch.Tensor, blured_imgs: torch.Tensor,
                    mask_fg: torch.Tensor, mask_bg: torch.Tensor):
        assert images.ndim == 4
        assert mask_fg.shape[0] == images.shape[0]
        assert mask_fg.shape[1] == 1
        assert mask_fg.shape[2:] == images.shape[2:]
        assert mask_fg.shape == mask_bg.shape

        return mask_fg * images + mask_bg * blured_imgs

    def get_cam_one_sample(self,
                           image: torch.Tensor,
                           target: int,
                           blured_img: torch.Tensor = None
                           ) -> Tuple[torch.Tensor, torch.Tensor,
                                      Union[dict, None]]:

        cam, cbox = None, None
        task = self.args.task

        with autocast(enabled=self.args.amp_eval):
            output = self.model(image)

        if task == constants.STD_CL:

            if self.args.amp_eval:
                output = output.float()

            cl_logits = output
            cam = self.std_cam_extractor(class_idx=target,
                                         scores=cl_logits,
                                         normalized=True)

            # (h`, w`)

        elif task in [constants.F_CL, constants.TCAM]:

            if self.args.amp_eval:
                tmp = []
                for term in output:
                    tmp.append(term.float() if term is not None else None)
                output = tmp

            cl_logits, fcams, im_recon = output
            if task == constants.F_CL:
                cam = self.fcam_extractor(argmax=self.fcam_argmax)
                # (h`, w`)
            elif task == constants.TCAM:
                cam = self.tcam_extractor(argmax=self.fcam_argmax)
                # (h`, w`)

        elif self.args.task == constants.C_BOX:
            box = output
            assert isinstance(box, torch.Tensor)
            assert box.ndim == 2

            assert blured_img is not None
            assert blured_img.shape == image.shape
            zz = self.box_stats(box=box, eval=True)
            x_hat, y_hat, valid, area, mask_fg, mask_bg = zz
            img_fg = self.get_fg_imgs(images=image, blured_imgs=blured_img,
                                      mask_fg=mask_fg, mask_bg=mask_bg)
            # print(f'area {area} valid {valid} x {x_hat} y {y_hat}')

            with autocast(enabled=self.args.amp_eval):
                # todo: eval on clean, perturbed image.
                cl_logits = self.classifier(img_fg)

            cbox = {
                'raw_scores': box.detach(),
                'x_hat': x_hat.detach(),
                'y_hat': y_hat.detach(),
                'valid': valid.detach(),
                'area': area.detach(),
                'mask_fg': mask_fg.detach(),
                'mask_bg': mask_fg.detach()
            }
        else:
            raise NotImplementedError

        if cam is not None:
            if self.args.amp_eval:
                cam = cam.float()

            # Quick fix: todo...
            cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
            # cl_logits: 1, nc.

        return cam, cl_logits, cbox

    def minibatch_accum(self, images, targets, image_ids, image_size,
                        blured_imgs) -> None:

        i = 0
        for image, target, image_id in zip(images, targets, image_ids):
            with torch.set_grad_enabled(self.req_grad):
                cam, cl_logits, cbox = self.get_cam_one_sample(
                    image=image.unsqueeze(0), target=target.item(),
                    blured_img=None if blured_imgs is None else blured_imgs[
                    i].unsqueeze(0))
                cl_logits = cl_logits.detach()

            with torch.no_grad():
                if cam is not None:
                    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)
                    cam = cam.detach()
                    # todo:
                    # cam = torch.clamp(cam, min=0.0, max=1.)

                    # cam: (h, w)
                    cam = t2n(cam)

                box_public_format = None
                bbox_status = None
                if cbox is not None:
                    assert cam is None
                    self.update_cbox_status_one_sample(cbox=cbox,
                                                       image_id=image_id)
                    box_public_format, _ = self.get_box_cbox_one_sample(
                        x_hat=cbox['x_hat'], y_hat=cbox['y_hat'])
                    bbox_status = cbox['valid'].squeeze().item()
                    assert bbox_status in [0, 1]
                    self.update_cbox_status(image_id=image_id,
                                            status=bbox_status)

                assert cl_logits.ndim == 2
                _, preds_ordered = torch.sort(input=cl_logits.cpu().squeeze(0),
                                              descending=True, stable=True)

                self.evaluator.accumulate(
                    cam, image_id, target.item(), preds_ordered.numpy(),
                    bbox=box_public_format, bbox_status=bbox_status)
            i += 1

    def update_cbox_status_one_sample(self, cbox: dict, image_id: str):
        assert self.args.task == constants.C_BOX
        assert cbox['valid'].ndim == 2
        assert cbox['valid'].shape[0] == 1
        assert cbox['valid'].shape[1] == 1

        if image_id in self.cbox_status:
            raise ValueError

        status = cbox['valid'].squeeze().item()
        self.cbox_status[image_id] = status

    def get_box_cbox_one_sample(self,
                                x_hat: torch.Tensor,
                                y_hat: torch.Tensor) -> Tuple[list, list]:
        assert self.args.task == constants.C_BOX

        for el in [x_hat, y_hat]:
            assert el.ndim == 2
            assert el.shape[0] == 1
            assert el.shape[1] == 2

        x1 = x_hat[0, 0].item()  # float e.g 14.78
        x2 = x_hat[0, 1].item()  # float e.g 56.65

        y1 = y_hat[0, 0].item()  # float e.g 10.45
        y2 = y_hat[0, 1].item()  # float e.g 99.70

        # internal format:
        # (x1, y1): upper left corner.
        # (x2, y2): bottom right corner.
        # x-axis: height.
        # y-axis: width.

        box_internal_format = [x1, y1, x2, y2]  # x-axis: height. y-axis: width.
        box_public_format = [y1, x1, y2, x2]  # x-axis: width. y-axis: height.

        return box_public_format, box_internal_format

    def normalizecam(self, cam):
        if self.args.task == constants.STD_CL:
            cam_normalized = normalize_scoremap(cam)
        elif self.args.task in [constants.F_CL, constants.TCAM]:
            cam_normalized = cam
        elif self.args.task == constants.C_BOX:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return cam_normalized

    def fix_random(self):
        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.deterministic = True

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")

        self.reset_cbox_status()

        for batch_idx, (images, targets, image_ids, _, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):

            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)
            blured_imgs = None
            if self.args.task == constants.C_BOX:
                blured_imgs = self.blur_op(images=images)

            self.minibatch_accum(images=images, targets=targets,
                                 image_ids=image_ids, image_size=image_size,
                                 blured_imgs=blured_imgs)

        if self.args.distributed:
            self.evaluator._synch_across_gpus()
            dist.barrier()

        return self.evaluator.compute()

    def _fast_eval_loc_pre_built_cams(self, cam_paths: dict, cam_cl_pre: dict):
        # Warning: do not call unless you know the work of this function.
        # job: fast evaluation of localization from pre-built cams.
        # no call for the model. requires: pre-built-cams, and class
        # predictions.

        print("Evaluating cams.")

        self.reset_cbox_status()

        for batch_idx, (images, targets, image_ids, _, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):

            self.fix_random()
            assert images.shape[0] == 1
            image_id = image_ids[0]
            target = targets.item()

            image_size = images.shape[2:]
            # images = images.cuda(self.args.c_cudaid)
            std_cam_path = cam_paths[image_id]
            # h', w'
            _cam: torch.Tensor = torch.load(f=std_cam_path,
                                            map_location=torch.device('cpu'))
            assert _cam.ndim == 2

            cam = _cam
            pred_cl = cam_cl_pre[image_id]

            with torch.no_grad():
                if cam is not None:
                    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)
                    cam = cam.detach()
                    cam = torch.clamp(cam, min=0.0, max=1.)

                    # cam: (h, w)
                    cam = t2n(cam)

                box_public_format = None
                bbox_status = None
                # fake top5. irrelevant at this stage.
                # todo: fix in the future.
                preds_ordered = np.array([pred_cl for _ in range(10)])

                self.evaluator.accumulate(cam, image_id, target, preds_ordered,
                                          bbox=box_public_format,
                                          bbox_status=bbox_status)

        if self.args.distributed:
            self.evaluator._synch_across_gpus()
            dist.barrier()

        return self.evaluator.compute()

    @staticmethod
    def is_null_bbox(box: np.ndarray) -> bool:
        assert box.ndim == 2
        assert box.shape[1] == 4

        if box.shape[0] > 1:
            return False

        return all([v == 0.0 for v in box.flatten()])

    @staticmethod
    def compute_area_bboxes(bboxes: np.ndarray) -> np.ndarray:
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        # format: 'x0y0x1y1'
        # todo: constrain this format everywhere.
        check_box_convention(bboxes, 'x0y0x1y1')

        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        return heights * widths

    def build_bbox(self,
                   scoremap,
                   image_id,
                   tau: float,
                   bbox: Union[list, None] = None,
                   bbox_status: Union[float, None] = None
                   ) -> Tuple[np.ndarray, float, np.ndarray]:

        if scoremap is None:
            assert bbox is not None
            assert bbox_status in [0, 1]
            assert self.args.task == constants.C_BOX
        else:
            assert bbox is None
            assert bbox_status is None

        cam_threshold_list = [tau]  # best threshold.
        if self.evaluator is not None:
            multi_contour_eval = self.evaluator.multi_contour_eval
        else:
            multi_contour_eval = self.multi_contour_eval

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=cam_threshold_list,
            multi_contour_eval=multi_contour_eval,
            bbox=bbox
        )

        assert len(boxes_at_thresholds) == 1
        assert len(number_of_box_list) == 1

        # nbrbox, 4
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        # support multi-instances.
        # todo: do it better. self.evaluator is set to None for trainset.
        #  need a flag to tell whether this split has labels or not.
        if self.evaluator is not None:
            gt_bboxes = np.array(
                self.evaluator.gt_bboxes[image_id])  # nbrgtbx, 4

            gt_bbox_unknow = self.is_null_bbox(gt_bboxes)
        else:
            gt_bbox_unknow = True

        if gt_bbox_unknow:  # unknown gt bbox.
            gt_bbox = None
            bbox_iou = 0.0
            # cases for unlabeled frames in videos.
            # how to pick bbox when gt is unknown?
            # pick the largest predicted bbox.
            sizes = self.compute_area_bboxes(boxes_at_thresholds)
            idx = np.argmax(sizes)
            best_bbox = boxes_at_thresholds[idx]  # shape: (4,)

        else:  # known gt bbox.
            nbr_inst = gt_bboxes.shape[0]
            assert gt_bboxes.ndim == 2
            assert nbr_inst > 0

            multiple_iou = calculate_multiple_iou(
                np.array(boxes_at_thresholds),
                gt_bboxes)  # (nbrbox, nbr_inst)

            if nbr_inst == 1:
                multiple_iou = multiple_iou.flatten()
                idx = np.argmax(multiple_iou)
                bbox_iou = multiple_iou[idx]
                best_bbox = boxes_at_thresholds[idx]  # shape: (4,)
                gt_bbox = gt_bboxes.flatten()  # (4,)
            else:
                idx = np.unravel_index(multiple_iou.argmax(),
                                       multiple_iou.shape)
                bbox_iou = multiple_iou[idx]
                best_bbox = boxes_at_thresholds[idx[0]]  # (4,)
                gt_bbox = gt_bboxes[idx[1]]  # (4,)

        return best_bbox, bbox_iou, gt_bbox

    def build_mask(self):
        pass

    def assert_datatset_bbx(self):
        assert self.dataset_name in [constants.CUB,
                                     constants.ILSVRC,
                                     constants.YTOV1]

    def assert_dataset_mask(self):
        assert self.dataset_name == constants.OpenImages

    def assert_tau_list(self, iou_threshold_list, best_tau_list):

        if isinstance(self.evaluator, BoxEvaluator):
            assert len(best_tau_list) == len(iou_threshold_list)
        elif isinstance(self.evaluator, MaskEvaluator):
            assert len(best_tau_list) == 1
        else:
            raise NotImplementedError

    @staticmethod
    def create_folder(fd):
        os.makedirs(fd, exist_ok=True)

    def get_ids_with_zero_ignore_mask(self):
        ids = self.loader.dataset.image_ids

        out = []
        for id in ids:
            ignore_file = os.path.join(self.evaluator.mask_root,
                                       self.evaluator.ignore_paths[id])
            ignore_box_mask = load_mask_image(ignore_file,
                                              (_RESIZE_LENGTH, _RESIZE_LENGTH))
            if ignore_box_mask.sum() == 0:
                out.append(id)

        return out

    def get_ids_with_one_bbx(self):

        ids = self.loader.dataset.image_ids

        out = []
        for id in ids:
            gt_bbx = self.evaluator.gt_bboxes[id]

            if len(gt_bbx) == 1:
                out.append(id)

        return out

    def select_random_ids_to_draw(self, nbr: int) -> list:
        self.fix_random()
        if isinstance(self.evaluator, BoxEvaluator):
            ids: list = self.loader.dataset.image_ids
        elif isinstance(self.evaluator, MaskEvaluator):
            ids: list = self.get_ids_with_zero_ignore_mask()
        elif self.evaluator is None:  # todo: delete. wsol-video.
            ids: list = self.loader.dataset.image_ids
        else:
            raise NotImplementedError

        total_s = len(ids)
        n = min(nbr, total_s)
        idx = np.random.choice(a=total_s, size=n, replace=False).flatten()

        selected_ids = [ids[z] for z in idx]

        if self.args.task == constants.C_BOX:
            selected_ids = [
                iid for iid in selected_ids if self.cbox_status[iid] == 1]

        self.fix_random()

        return selected_ids

    def draw_some_best_pred(self, **kwargs):
        if self.args.task == constants.C_BOX:
            return self.draw_some_best_pred_cbox(**kwargs)
        else:
            return self._draw_some_best_pred(**kwargs)

    @staticmethod
    def get_str_trg_prd_cl(pred_cl: int,
                           trg_cl: int, int_cl: dict = None) -> str:
        if int_cl:
            return f'[CL] Trg: {int_cl[trg_cl]} - Prd: {int_cl[pred_cl]}'
        else:
            return f'[CL] Trg: {trg_cl} - Prd: {pred_cl}'

    @staticmethod
    def switch_key_val_dict(d: dict) -> dict:
        out = dict()
        for k in d:
            assert d[k] not in out, 'more than 1 key with same value. wrong.'
            out[d[k]] = k

        return out

    def _draw_some_best_pred(self,
                             nbr: int = 400,
                             separate: bool = True,
                             compress: bool = True,
                             store_imgs: bool = False,
                             store_cams_alone: bool = False,
                             plot_all_instances: bool = True,
                             pred_cl: dict = None,
                             pred_cams_paths: dict = None,
                             pred_bbox: dict = None,
                             cl_int: dict = None,
                             show_cl: bool = False,
                             best_tau_list_: list = None,
                             iou_threshold_list_: list = None,
                             rename_ordered: bool = False,
                             convert_to_video: bool = False,
                             prepare_to_convert_to_video: bool = False
                             ):

        print(f'Drawing {nbr} pictures.')

        assert not (convert_to_video and rename_ordered)
        if convert_to_video:
            assert not store_cams_alone
            assert not store_imgs

        if best_tau_list_ is not None:
            assert iou_threshold_list_ is not None

        if iou_threshold_list_ is not None:
            assert best_tau_list_ is not None

        if best_tau_list_ is None:
            assert self.evaluator.best_tau_list != []
            best_tau_list = self.evaluator.best_tau_list
        else:
            assert isinstance(best_tau_list_, list)
            best_tau_list = best_tau_list_

        if iou_threshold_list_ is None:
            iou_threshold_list = self.evaluator.iou_threshold_list
        else:
            assert isinstance(iou_threshold_list_, list)
            iou_threshold_list = iou_threshold_list_

        iou_thresh = 50  # CorLoc (threshold= .5).
        assert iou_thresh in iou_threshold_list
        _iou_threshold_list = [iou_thresh]  # todo: delete. only WSOL-VIDEO

        DLLogger.log(f'We are plotting visu only for iou thresh: {iou_thresh}')

        if self.evaluator is None:  # todo: delete. wsol-video.
            assert len(best_tau_list) == len(iou_threshold_list)
        else:
            self.assert_tau_list(iou_threshold_list, best_tau_list)

        int_cl = self.switch_key_val_dict(cl_int) if cl_int else None

        iou_tracker = {
            iou: dict() for iou in iou_threshold_list
        }

        shots_paths = dict()  # todo: del. only for wsol-video.

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        for _image_id in tqdm(ids_to_draw, ncols=constants.NCOLS, total=len(
                ids_to_draw)):

            img_idx = self.loader.dataset.index_id[_image_id]
            image, target, image_id, raw_img, _ = self.loader.dataset[
                img_idx]

            if self.loader.dataset.dataset_mode != constants.DS_SHOTS:
                assert image_id == _image_id

            self.fix_random()

            image = image.cuda(self.args.c_cudaid)  # 3, h, w.
            image_size = image.shape[1:]
            # raw_img: 3, h, w
            raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
            raw_img = raw_img.astype(np.uint8)

            if store_imgs:
                img_fd = join(self.out_folder, 'vizu/imgs')
                self.create_folder(img_fd)
                Image.fromarray(raw_img).save(join(img_fd, '{}.png'.format(
                    reformat_id(image_id))))

            with torch.set_grad_enabled(self.req_grad):
                # todo: this is NOT low res.
                low_cam, cl_logits, cbox = self.get_cam_one_sample(
                    image=image.unsqueeze(0), target=target)

                if pred_cams_paths:  # todo: improve. avoid calling model.
                    low_cam = torch.load(pred_cams_paths[image_id],
                                         map_location=get_cpu_device())

                tag_cl = None
                if show_cl:
                    if pred_cl:
                        p_cl = pred_cl[image_id]
                    else:
                        assert cl_logits.ndim == 2
                        p_cl = cl_logits.argmax(dim=1).item()

                    tag_cl = self.get_str_trg_prd_cl(
                        pred_cl=p_cl, trg_cl=target, int_cl=int_cl)

            with torch.no_grad():
                cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False
                                    ).squeeze(0).squeeze(0)

                cam = torch.clamp(cam, min=0.0, max=1.)

            if store_cams_alone:
                calone_fd = join(self.out_folder, 'vizu/cams_alone/low_res')
                self.create_folder(calone_fd)

                self.viz.plot_cam_raw(t2n(low_cam), outf=join(
                    calone_fd, '{}.png'.format(reformat_id(
                        image_id))), interpolation='none')

                calone_fd = join(self.out_folder,
                                 'vizu/cams_alone/high_res')
                self.create_folder(calone_fd)

                self.viz.plot_cam_raw(t2n(cam), outf=join(
                    calone_fd, '{}.png'.format(reformat_id(
                        image_id))), interpolation='bilinear')

            cam = torch.clamp(cam, min=0.0, max=1.)
            cam = t2n(cam)

            # cams shape (h, w).
            assert cam.shape == image_size

            cam_resized = cam
            cam_normalized = cam_resized
            check_scoremap_validity(cam_normalized)

            # todo: fix or None. wsol-video.
            if isinstance(self.evaluator, BoxEvaluator) or self.evaluator is \
                    None:
                self.assert_datatset_bbx()
                l_datum = []
                for k, _THRESHOLD in enumerate(_iou_threshold_list):
                    th_fd = join(self.out_folder, 'vizu', str(_THRESHOLD))
                    self.create_folder(th_fd)

                    tau = best_tau_list[k]
                    best_bbox, bbox_iou, matched_gtbox = self.build_bbox(
                        scoremap=cam_normalized, image_id=image_id,
                        tau=tau
                    )
                    assert image_id not in iou_tracker[_THRESHOLD]
                    iou_tracker[_THRESHOLD][image_id] = bbox_iou

                    if pred_bbox:
                        best_bbox = pred_bbox[image_id]
                        assert isinstance(best_bbox, np.ndarray)
                        assert best_bbox.ndim == 1
                        assert best_bbox.size == 4

                    # frames without bbox.
                    if matched_gtbox is not None:
                        matched_gtbox = matched_gtbox.reshape((1, 4))
                        gt_bbxes = self.evaluator.gt_bboxes[image_id]
                        gt_bbxes = np.array(gt_bbxes)  # (nbr_inst, 4)
                    else:
                        gt_bbxes = None

                    datum = {'img': raw_img,
                             'img_id': image_id,
                             'gt_bboxes': gt_bbxes,
                             'gt_matched_bbox': matched_gtbox,
                             'pred_bbox': best_bbox.reshape((1, 4)),
                             'iou': bbox_iou,
                             'tau': tau,
                             'sigma': _THRESHOLD,
                             'cam': cam_normalized,
                             'tag_cl': tag_cl
                             }

                    if separate:
                        if convert_to_video or prepare_to_convert_to_video:
                            outf = join(th_fd, '{}.png'.format(image_id))
                            fdx = dirname(outf)

                            if fdx in shots_paths:
                                shots_paths[fdx]['frames'].append(
                                    (outf, basename(outf))
                                )
                            else:
                                shots_paths[fdx] = {
                                    'frames': [(outf, basename(outf))],
                                    'sizes': []
                                }

                            os.makedirs(fdx, exist_ok=True)

                            _size: List[int] = self.viz.plot_single(
                                datum=datum, outf=outf,
                                plot_all_instances=plot_all_instances)
                            shots_paths[fdx]['sizes'].append(_size)  # w, h.

                        else:
                            outf = join(th_fd, '{}.png'.format(reformat_id(
                                image_id)))

                            self.viz.plot_single(
                                datum=datum, outf=outf,
                                plot_all_instances=plot_all_instances)

                    l_datum.append(datum)

                # not necessary to plot multilple.
                # th_fd = join(self.out_folder, 'vizu', 'all_taux')
                # self.create_folder(th_fd)
                # outf = join(th_fd, '{}.png'.format(reformat_id(
                #     image_id)))
                # self.viz.plot_multiple(data=l_datum, outf=outf)

            elif isinstance(self.evaluator, MaskEvaluator):
                self.assert_dataset_mask()
                tau = best_tau_list[0]
                taux = sorted(list({0.5, 0.6, 0.7, 0.8, 0.9}))
                gt_mask = get_mask(self.evaluator.mask_root,
                                   self.evaluator.mask_paths[image_id],
                                   self.evaluator.ignore_paths[image_id])
                # gt_mask numpy.ndarray(size=(224, 224), dtype=np.uint8)

                l_datum = []
                for tau in taux:
                    th_fd = join(self.out_folder, 'vizu', str(tau))
                    self.create_folder(th_fd)
                    l_datum.append(
                        {'img': raw_img, 'img_id': image_id,
                         'gt_mask': gt_mask, 'tau': tau,
                         'best_tau': tau == best_tau_list[0],
                         'cam': cam_normalized}
                    )
                    # todo: plotting singles is not necessary for now.
                    # todo: control it latter for standalone inference.
                    if separate:
                        outf = join(th_fd, '{}.png'.format(reformat_id(
                            image_id)))
                        self.viz.plot_single(datum=l_datum[-1], outf=outf)

                th_fd = join(self.out_folder, 'vizu', 'some_taux')
                self.create_folder(th_fd)
                outf = join(th_fd, '{}.png'.format(reformat_id(
                    image_id)))
                self.viz.plot_multiple(data=l_datum, outf=outf)
            else:
                raise NotImplementedError

        # ordered perf.
        if not convert_to_video:
            for iou_thresh in iou_tracker:
                l = self.build_desc_ordered_list(iou_tracker[iou_thresh])
                with open(join(self.out_folder,
                               f'ordered_iou_{iou_thresh}.yaml'), 'w') as fx:
                    yaml.dump(iou_tracker[iou_thresh], fx)

                with open(join(self.out_folder,
                               f'ordered_iou_{iou_thresh}.txt'), 'w') as fx:

                    cc = 0
                    for zz, (_id, _iou) in enumerate(l):
                        name = reformat_id(_id)

                        fx.write(f'{name}: {_iou} \n')

                        pfile = join(self.out_folder,
                                     f'vizu/{iou_thresh}/{name}.png')

                        if rename_ordered and os.path.isfile(pfile):
                            new_pfile = join(
                                self.out_folder,
                                f'vizu/{iou_thresh}/{cc}_{name}.png')
                            os.rename(pfile, new_pfile)
                            cc += 1
        else:
            print('Building videos...')
            lshots = list(shots_paths.keys())

            for shot in tqdm(lshots, ncols=80, total=len(lshots)):
                lframes = shots_paths[shot]['frames']
                _sizes = shots_paths[shot]['sizes']
                w, h = self.smallest_size(_sizes)
                # sort frames: 0, 1, ...
                lframes = sorted(lframes, key=lambda tup: tup[1], reverse=False)
                _lframes = [v[0] for v in lframes]

                video_path = shot.rstrip(os.sep)
                for sep in ['/', '\\']:
                    video_path = video_path.rstrip(sep)  # depends where
                    # folds where generated.

                self.build_video_from_frames(
                    lframes=_lframes, shot_folder=shot, w=w, h=h, fps=15,
                    output_v_path=video_path,
                    delete_frames=False, delete_folder_shot=False)

        if compress:
            self.compress_fdout(self.out_folder, 'vizu')

    @staticmethod
    def smallest_size(l: list) -> Tuple[int, int]:
        # l=[(w, h), (w, h), ...]
        w = min([v[0] for v in l])
        h = min([v[1] for v in l])
        return w, h

    @staticmethod
    def build_video_from_frames(lframes: list,
                                shot_folder: str,
                                fps: int,
                                w: int, h: int,
                                output_v_path: str,  # no extension.
                                delete_frames: bool,
                                delete_folder_shot: bool):
        ext = '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            f'{output_v_path}{ext}', fourcc, fps, (w, h))

        for frame in lframes:
            video.write(cv2.resize(cv2.imread(frame), (w, h),
                                   interpolation=cv2.INTER_AREA))

            if delete_frames:
                os.system(f'rm {frame}')

        cv2.destroyAllWindows()
        video.release()

        if delete_folder_shot and os.path.isdir(shot_folder):
            os.system(f'rm -r {shot_folder}')

    @staticmethod
    def build_desc_ordered_list(d: dict) -> List:
        assert isinstance(d, dict)
        l = [(k, d[k]) for k in d]
        return sorted(l, key=lambda tup: tup[1], reverse=True)

    def _build_store_std_cam_low(self, fdout: str, cams_roi_file: str = None):
        print('Building low res. cam and storing them.')
        roifx = None
        otsu = None
        s = constants.CROP_SIZE
        if cams_roi_file:
            roifx = open(cams_roi_file, 'w')
            otsu = STOtsu()

        for idx, (images, targets, image_ids, raw_imgs, _) in tqdm(enumerate(
                self.loader), ncols=constants.NCOLS, total=len(self.loader)):
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):
                with torch.set_grad_enabled(self.req_grad):
                    output = self.model(image.unsqueeze(0))

                    assert self.args.task == constants.STD_CL
                    cl_logits = output
                    cam = self.std_cam_extractor(class_idx=target.item(),
                                                 scores=cl_logits,
                                                 normalized=True)
                    # (h`, w`)

                    # Quick fix: todo...
                    cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)

                    if roifx:
                        _full_cam = F.interpolate(
                            cam.unsqueeze(0).unsqueeze(0),
                            size=(s, s), mode='bilinear', align_corners=True
                        )
                        thresh: float = otsu(
                            torch.floor(_full_cam * 255)).item()
                        roi = cam2max_bbox(
                            _full_cam.detach().cpu().squeeze().numpy().astype(
                            float), thresh)
                        # roi: x0y0x1y1.
                        assert roi.ndim == 2, roi.ndim
                        assert roi.shape == (1, 4), roi.shape

                        thresh = thresh / 255.  # [0., 1.]

                        info = [image_id, thresh]
                        roifx.write(','.join([str(zz) for zz in info]) + '\n')

                    cam = cam.detach().cpu()
                torch.save(cam, join(fdout, f'{reformat_id(image_id)}.pt'))
        if cams_roi_file:
            roifx.close()

    def _fast_plot_prebuilt_cams(self, fdout: str, cams_roi_file: str = None):
        print('Fast plot pre-built cams.')
        # todo
        raise NotImplementedError

    @staticmethod
    def _build_roi_from_cams(fdcams: str, out_roi_file: str, s_ids: list):
        print('Building ROI from CAMs...')
        s = constants.CROP_SIZE
        otsu = STOtsu()
        roifx = open(out_roi_file, 'w')
        n = len(list(s_ids))

        for _id in tqdm(s_ids, ncols=80, total=n):
            f = join(fdcams, f'{reformat_id(_id)}.pt')
            assert os.path.isfile(f), f
            cam: torch.Tensor = torch.load(f, map_location=get_cpu_device())
            assert cam.ndim == 2, cam.ndim

            full_cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                     size=(s, s), mode='bilinear',
                                     align_corners=True)
            thresh: float = otsu(
                torch.floor(full_cam * 255)).item()
            roi = cam2max_bbox(
                full_cam.detach().cpu().squeeze().numpy().astype(float),
                thresh)
            # roi: x0y0x1y1.
            assert roi.ndim == 2, roi.ndim
            assert roi.shape == (1, 4), roi.shape

            thresh = thresh / 255.  # [0., 1.]

            info = [_id, thresh]
            roifx.write(','.join([str(zz) for zz in info]) + '\n')

        roifx.close()

    def draw_some_best_pred_cbox(self, nbr=200, separate=False, compress=True,
                                 store_imgs=False, store_cams_alone=False):

        assert isinstance(self.evaluator, BoxEvaluator)
        assert self.args.task == constants.C_BOX
        self.assert_datatset_bbx()

        print(f'Drawing {nbr} pictures...')

        viz_fd = join(self.out_folder, 'vizu', 'image-bbox')
        img_fd = join(self.out_folder, 'vizu/imgs')
        for fdx in [viz_fd, img_fd]:
            self.create_folder(fdx)

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        for _image_id in tqdm(ids_to_draw, ncols=constants.NCOLS, total=len(
                ids_to_draw)):
            img_idx = self.loader.dataset.index_id[_image_id]
            image, target, image_id, raw_img, _ = self.loader.dataset[img_idx]
            assert image_id == _image_id

            self.fix_random()
            image = image.cuda(self.args.c_cudaid)  # 3, h, w.
            image_size = image.shape[1:]
            blured_img = self.blur_op(image.unsqueeze(0))  # 1, 3, h, w.
            # raw_img: 3, h, w
            raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
            raw_img = raw_img.astype(np.uint8)

            if store_imgs:
                Image.fromarray(raw_img).save(join(img_fd, '{}.png'.format(
                    reformat_id(image_id))))

            with torch.set_grad_enabled(self.req_grad):
                _, _, cbox = self.get_cam_one_sample(
                    image=image.unsqueeze(0),
                    target=target,
                    blured_img=blured_img
                )
                bbox_status = cbox['valid'].squeeze().item()  # float: 0, 1
                assert bbox_status == 1

            with torch.no_grad():

                bbox, _ = self.get_box_cbox_one_sample(
                    x_hat=cbox['x_hat'], y_hat=cbox['y_hat'])

            # tau threshold is irrelevant.
            best_bbox, bbox_iou, matched_gtbox = self.build_bbox(
                scoremap=None, image_id=image_id, tau=0.0, bbox=bbox,
                bbox_status=bbox_status
            )
            gt_bbx = self.evaluator.gt_bboxes[image_id]
            gt_bbx = np.array(gt_bbx)

            datum = {'img': raw_img,
                     'img_id': image_id,
                     'gt_bbox': gt_bbx,
                     'pred_bbox': best_bbox.reshape((1, 4)),
                     'iou': bbox_iou,
                     'tau': None,
                     'sigma': None,
                     'cam': None
                     }

            outf = join(viz_fd, '{}.png'.format(reformat_id(
                image_id)))
            self.viz.cbox_plot_single(datum=datum, outf=outf)

        if compress:
            self.compress_fdout(self.out_folder, 'vizu')

    @staticmethod
    def compress_fdout(parent_fd, fd_trg):
        assert os.path.isdir(join(parent_fd, fd_trg))

        cmdx = [
            "cd {} ".format(parent_fd),
            "tar -cf {}.tar.gz {} ".format(fd_trg, fd_trg),
            "rm -r {} ".format(fd_trg)
        ]
        cmdx = " && ".join(cmdx)
        DLLogger.log("Running: {}".format(cmdx))
        try:
            subprocess.run(cmdx, shell=True, check=True)
        except subprocess.SubprocessError as e:
            DLLogger.log("Failed to run: {}. Error: {}".format(cmdx, e))
