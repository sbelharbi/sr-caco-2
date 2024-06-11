import sys
from os.path import join, dirname, abspath
import random
from typing import Tuple, Optional, Dict, List
from typing import Sequence as TSequence
import numbers
from collections.abc import Sequence
import fnmatch
import copy
import os

from torch import Tensor
import torch
import munch
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.distributed import DistributedSampler

PROB_THRESHOLD = 0.5  # probability threshold.

"Credit: https://github.com/clovaai/wsolevaluation/blob/master/data_loaders.py"

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.functional import _functional as dlibf
from dlib.utils import constants

from dlib.utils.shared import reformat_id
from dlib.utils.tools import chunk_it
from dlib.utils.tools import resize_bbox

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = join(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = join(metadata_root, 'image_sizes.txt')
    metadata.localization = join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')

            x0, x1, y0, y1 = float(x0s), float(x1s), float(y0s), float(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes


def get_mask_paths(metadata):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


def get_cams_paths(root_data_cams: str, image_ids: list) -> dict:
    paths = dict()
    for idx_ in image_ids:
        paths[idx_] = join(root_data_cams, '{}.pt'.format(reformat_id(idx_)))

    return paths


def list_file_names_extension(fd_path: str, pattern_ext: str) -> List[str]:
    out = []
    content = next(os.walk(fd_path))[2]
    for item in content:
        path = join(fd_path, item)
        if os.path.isfile(path) and fnmatch.fnmatch(path, pattern_ext):
            out.append(item)

    out = sorted(out, reverse=False)
    return out


def convert_abs_path_2_rel_p(root: str, path: str) -> str:
    return path.replace(root, '').lstrip(os.sep)


class WSOLImageLabelDataset(Dataset):
    def __init__(self,
                 data_root,
                 metadata_root,
                 transform,
                 proxy,
                 resize_size,
                 crop_size,
                 dataset: str,
                 num_sample_per_class=0,
                 root_data_cams='',
                 image_ids: Optional[list] = None):

        self.dataset = dataset
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform

        if image_ids is not None:
            self.image_ids: list = image_ids
        else:
            self.image_ids: list = get_image_ids(self.metadata, proxy=proxy)

        self.index_id: dict = {
            id_: idx for id_, idx in zip(self.image_ids,
                                         range(len(self.image_ids)))
        }

        self.image_labels: dict = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class

        self.dataset_mode = self.get_dataset_mode()
        self.index_of_frames: dict = dict()

        if self.dataset_mode == constants.DS_SHOTS:
            self.index_frames_from_shots()

        self.cams_paths: dict = None

        if os.path.isdir(root_data_cams):
            ims_id = self.image_ids
            if self.dataset_mode == constants.DS_SHOTS:
                ims_id = []
                for shot in self.index_of_frames:
                    ims_id += self.index_of_frames[shot]
                assert len(set(ims_id)) == len(ims_id)

            self.cams_paths: dict = get_cams_paths(
                root_data_cams=root_data_cams, image_ids=ims_id)

        self.resize_size = resize_size
        self.crop_size = crop_size

        # priors
        self.original_bboxes = None
        self.image_sizes = None
        self.gt_bboxes = None
        self.size_priors: dict = dict()

        self._adjust_samples_per_class()

    def _switch_to_frames_mode(self):
        # todo: weak change. could break the code. used only to build stuff
        #  over trainset directly over frames such as visualization.
        assert self.dataset_mode == constants.DS_SHOTS
        print(f'Warning: Switching dataset into {constants.DS_FRAMES} mode.')
        print('Indexing frames...')

        img_ids = []
        img_l = dict()
        for shot in self.image_ids:
            lframes = self.index_of_frames[shot]
            img_ids += lframes

            for f in lframes:
                img_l[f] = self.image_labels[shot]

        self.image_ids: list = img_ids
        self.index_id: dict = {
            id_: idx for id_, idx in zip(self.image_ids,
                                         range(len(self.image_ids)))
        }
        self.image_labels: dict = img_l

        self.set_dataset_mode(constants.DS_FRAMES)

    def get_dataset_mode(self):

        if self.dataset not in [constants.YTOV1]:
            return constants.DS_FRAMES

        image_id = self.image_ids[0]
        path = join(self.data_root, image_id)

        mode = None

        if os.path.isfile(path):
            mode = constants.DS_FRAMES
        elif os.path.isdir(path):
            mode = constants.DS_SHOTS
        else:
            raise ValueError(f'path {path} not recognized as dir/file.')

        assert mode in constants.DS_MODES

        return mode

    def set_dataset_mode(self, dsmode: str):
        assert dsmode in constants.DS_MODES
        self.dataset_mode = dsmode

    def index_frames_from_shots(self):
        assert self.dataset in [constants.YTOV1]
        assert self.get_dataset_mode() == constants.DS_SHOTS
        print('Indexing frames from shots.')

        for shot in tqdm.tqdm(self.image_ids, ncols=80,
                              total=len(self.image_ids)):
            path_shot = join(self.data_root, shot)
            # ordered frames: 0, 1, 2, ....
            l_frames = list_file_names_extension(path_shot,
                                                 pattern_ext='*.jpg')

            assert len(l_frames) > 0, 'Empty shots should not be used.'

            # change ids to frames id.
            l_frames = [join(path_shot, frame) for frame in l_frames]
            l_frames = [convert_abs_path_2_rel_p(self.data_root, f) for f in
                        l_frames]
            self.index_of_frames[shot] = copy.deepcopy(l_frames)

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]

        if self.dataset_mode == constants.DS_SHOTS:
            # for datasets that we indexed by shots [not frames] such as
            # trainset of ytov1, the dataloader sees only shots. at this
            # section, we randomly sample a frame from the selected shot.
            # -------------
            # implications: for one epoch, the dataloader will see exactly
            # one frame only per every shot. the rest of frames are missed.
            # we set it this way to class diversity per mini-batch. consider
            # that: every class has different number of videos; every video
            # has different number of shots; every shot has different number
            # of frames.

            l_frames = self.index_of_frames[image_id]
            fr_idx = np.random.randint(low=0, high=len(l_frames), size=1).item()
            # switch image_id from shot id to frame id.
            image_id = l_frames[fr_idx]
            # todo: grab left-right neighbors.

        image = Image.open(join(self.data_root, image_id))
        image = image.convert('RGB')
        raw_img = image.copy()

        std_cam = None
        if self.cams_paths is not None:
            # todo: fix this to deal with shots/frames.
            std_cam_path = self.cams_paths[image_id]
            # h', w'
            std_cam: torch.Tensor = torch.load(f=std_cam_path,
                                               map_location=torch.device('cpu'))
            assert std_cam.ndim == 2
            std_cam = std_cam.unsqueeze(0)  # 1, h', w'

        image, raw_img, std_cam = self.transform(image, raw_img, std_cam)

        raw_img = np.array(raw_img, dtype=np.float32)  # h, w, 3
        raw_img = dlibf.to_tensor(raw_img).permute(2, 0, 1)  # 3, h, w.

        if std_cam is None:
            std_cam = 0

        return image, image_label, image_id, raw_img, std_cam

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.crop_size, self.crop_size))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def _get_stats_box(self, box: TSequence[int]) -> TSequence[float]:
        x0, y0, x1, y1 = box
        assert x1 > x0
        assert y1 > y0
        w = (x1 - x0) / float(self.crop_size)
        h = (y1 - y0) / float(self.crop_size)
        s = h * w
        assert 0 < h <= 1.
        assert 0 < w <= 1.
        assert 0 < s <= 1.

        return h, w, s

    def build_size_priors(self) -> Dict[str, float]:
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)
        for idimg in self.image_labels:
            label: int = self.image_labels[idimg]

            for box in self.gt_bboxes[idimg]:
                h, w, s = self._get_stats_box(box)

                if label in self.size_priors:
                    self.size_priors[label] = {
                        'min_h': min(h, self.size_priors[label]['min_h']),
                        'max_h': max(h, self.size_priors[label]['max_h']),

                        'min_w': min(w, self.size_priors[label]['min_w']),
                        'max_w': max(w, self.size_priors[label]['max_w']),

                        'min_s': min(s, self.size_priors[label]['min_s']),
                        'max_s': max(s, self.size_priors[label]['max_s']),
                    }
                else:
                    self.size_priors[label] = {
                        'min_h': h,
                        'max_h': h,

                        'min_w': w,
                        'max_w': w,

                        'min_s': s,
                        'max_s': s,
                    }

        return self.size_priors

    def __len__(self):
        return len(self.image_ids)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Compose(object):
    def __init__(self, mytransforms: list):
        self.transforms = mytransforms

        for t in mytransforms:
            assert any([isinstance(t, Resize), isinstance(t, RandomCrop),
                       isinstance(t, RandomHorizontalFlip),
                       isinstance(t, transforms.ToTensor),
                        isinstance(t, transforms.Normalize)]
                       )

    def chec_if_random(self, transf):
        # todo.
        if isinstance(transf, RandomCrop):
            return True

    def __call__(self, img, raw_img, std_cam):
        for t in self.transforms:
            if isinstance(t, (RandomHorizontalFlip, RandomCrop, Resize)):
                img, raw_img, std_cam = t(img, raw_img, std_cam)
            else:
                img = t(img)

        return img, raw_img, std_cam

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class _BasicTransform(object):
    def __call__(self, img, raw_img):
        raise NotImplementedError


class RandomHorizontalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, std_cam):
        if random.random() < self.p:
            std_cam_ = std_cam
            if std_cam_ is not None:
                std_cam_ = TF.hflip(std_cam)
            return TF.hflip(img), TF.hflip(raw_img), std_cam_

        return img, raw_img, std_cam

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(_BasicTransform):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:

        w, h = TF.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image "
                "size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two "
                            "dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        return img

    def __call__(self, img, raw_img, std_cam):
        img_ = self.forward(img)
        raw_img_ = self.forward(raw_img)
        assert img_.size == raw_img_.size

        i, j, h, w = self.get_params(img_, self.size)
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = self.forward(std_cam)
            std_cam_ = TF.crop(std_cam_, i, j, h, w)

        return TF.crop(img_, i, j, h, w), TF.crop(
            raw_img_, i, j, h, w), std_cam_

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)


class Resize(_BasicTransform):
    def __init__(self, size,
                 interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. "
                            "Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, "
                             "it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, raw_img, std_cam):
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = TF.resize(std_cam_, self.size, self.interpolation)

        return TF.resize(img, self.size, self.interpolation), TF.resize(
            raw_img, self.size, self.interpolation), std_cam_

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


def get_image_ids_bucket(args, tr_bucket: int, split: str,
                         metadata_root: str) -> list:
    assert split == constants.TRAINSET
    chunks = list(range(constants.NBR_CHUNKS_TR[args.dataset]))
    buckets = list(chunk_it(chunks, constants.BUCKET_SZ))
    assert tr_bucket < len(buckets)

    _image_ids = []
    for i in buckets[tr_bucket]:
        metadata = {'image_ids': join(metadata_root, split,
                                      f'train_chunk_{i}.txt')}
        _image_ids.extend(get_image_ids(metadata, proxy=False))

    return _image_ids


def get_data_loader(args,
                    data_roots,
                    metadata_root,
                    batch_size,
                    workers,
                    resize_size,
                    crop_size,
                    proxy_training_set,
                    dataset: str,
                    num_val_sample_per_class=0,
                    std_cams_folder=None,
                    get_splits_eval=None,
                    tr_bucket: Optional[int] = None,
                    isdistributed=True
                    ):
    train_sampler = None

    def get_eval_tranforms():
        return Compose([
            Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ])

    if isinstance(get_splits_eval, list):
        assert len(get_splits_eval) > 0
        eval_datasets = {
            split: WSOLImageLabelDataset(
                    data_root=data_roots[split],
                    metadata_root=join(metadata_root, split),
                    transform=get_eval_tranforms(),
                    proxy=False,
                    resize_size=resize_size,
                    crop_size=crop_size,
                    dataset=dataset,
                    num_sample_per_class=0,
                    root_data_cams=''
                )
            for split in get_splits_eval
        }

        loaders = {
            split: DataLoader(
                eval_datasets[split],
                batch_size=batch_size,
                shuffle=False,
                sampler=DistributedSampler(
                    dataset=eval_datasets[split], shuffle=False) if
                isdistributed else None,
                num_workers=workers
            )
            for split in get_splits_eval
        }

        return loaders, train_sampler

    dataset_transforms = dict(
        train=Compose([
            Resize((resize_size, resize_size)),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=get_eval_tranforms(),
        test=get_eval_tranforms()
    )

    image_ids = {
        split: None for split in _SPLITS
    }

    if not args.ds_chunkable:
        assert tr_bucket in [0, None]
    elif tr_bucket is not None:
        assert args.dataset == constants.ILSVRC
        image_ids[constants.TRAINSET] = get_image_ids_bucket(
            args=args, tr_bucket=tr_bucket, split=constants.TRAINSET,
            metadata_root=metadata_root)

    datasets = {
        split: WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == constants.TRAINSET,
                resize_size=resize_size,
                crop_size=crop_size,
                dataset=dataset,
                num_sample_per_class=(num_val_sample_per_class
                                      if split == constants.VALIDSET else 0),
                root_data_cams=std_cams_folder[split],
                image_ids=image_ids[split]
            )
        for split in _SPLITS
    }

    samplers = {
        split: DistributedSampler(dataset=datasets[split],
                                  shuffle=split == constants.TRAINSET)
        for split in _SPLITS
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            sampler=samplers[split],
            num_workers=workers)
        for split in _SPLITS
    }

    if constants.TRAINSET in _SPLITS:
        train_sampler = samplers[constants.TRAINSET]

    return loaders, train_sampler
