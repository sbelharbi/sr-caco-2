import os
import random
import sys
from os.path import dirname, abspath, join
from typing import List, Tuple
import shutil
import pprint
from copy import deepcopy
import fnmatch

from scipy import io
import yaml
import tqdm

from PIL import Image
import numpy as np

from dlib.utils.tools import check_box_convention

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
from dlib.utils.utils_config import get_root_datasets
from dlib.utils.utils_reproducibility import set_seed

_NBR_VIDS = 'NBR_VIDS'
_NBR_SHOTS = 'NBR_SHOTS'
_NBR_FRAMES = 'NBR_FRAMES'
_NBR_FRAMES_W_BBOX = 'NBR_FRAMES_W_BBOX'
_NBR_BBOXES = 'NBR_BBOXes'
_NBR_VIDS_W_BBOX = 'NBR_VIDS_W_BBOX'
_NBR_SHOTS_W_BBOX = 'NBR_SHOTS_W_BBOX'

_NBRS = [_NBR_VIDS,
         _NBR_SHOTS,
         _NBR_FRAMES,
         _NBR_VIDS_W_BBOX,
         _NBR_SHOTS_W_BBOX,
         _NBR_FRAMES_W_BBOX,
         _NBR_BBOXES]

_TRAIN = 'TRAIN'
_TEST = 'TEST'

_SUBSETS = [_TRAIN, _TEST]

# nbr of videos to take per class for validset.
_NBR_VIDS_CL_VL = 5


def print_total(stats: dict, filelog=None):
    pp = pprint.PrettyPrinter(depth=1, stream=filelog)
    for nbr in _NBRS:
        track_subsets = {k: 0 for k in _SUBSETS}

        for cl in stats:
            for subset in stats[cl]:
                track_subsets[subset] += stats[cl][subset][nbr]

        print(f'{nbr}:', file=filelog)
        pp.pprint(track_subsets)
        print(f'Total {sum(track_subsets.values())}', file=filelog)


def get_train_test_name_vids(path_file: str) -> List[str]:
    with open(path_file, 'r') as fin:
        content = fin.readlines()

    out = [l.strip() for l in content]
    for e in out:
        assert e.startswith('0'), f'{path_file}, line: {e}'

    return out


def convert_list_videos_to_dict(lpath_fd: list) -> dict:
    out = dict()
    for fd in lpath_fd:
        out[fd] = fd

    return out


def list_name_fds(path: str) -> List[str]:
    out = []
    for item in os.listdir(path):
        if os.path.isdir(join(path, item)):
            out.append(item)

    return out


def check_if_video_has_bbox(pathvideo: str) -> bool:
    assert os.path.isdir(pathvideo), pathvideo
    out = find_files_pattern(pathvideo, '*_sticks.mat')

    return len(out) > 0


def random_select_k_vids(lvids: List[str]) -> Tuple[List[str], List[str]]:
    assert isinstance(lvids, list)

    for i in range(1000):
        random.shuffle(lvids)

    sel, leftover = lvids[:_NBR_VIDS_CL_VL], lvids[_NBR_VIDS_CL_VL:]
    return sel, leftover


def list_file_names_extension(fd_path: str, pattern_ext: str) -> List[str]:
    out = []
    content = next(os.walk(fd_path))[2]
    for item in content:
        path = join(fd_path, item)
        if os.path.isfile(path) and fnmatch.fnmatch(path, pattern_ext):
            out.append(item)

    out = sorted(out, reverse=False)
    return out


def get_file_names_sticks_mat(shot_path: str) -> [List[str], List[int]]:
    out = []
    content = next(os.walk(shot_path))[2]
    for item in content:
        path = join(shot_path, item)
        if os.path.isfile(path) and fnmatch.fnmatch(path, '*_sticks.mat'):
            out.append(item)

    out = sorted(out, reverse=False)
    nbr_boxs = []
    for item in out:
        file = join(shot_path, item)
        mat = io.loadmat(file)['coor']
        nbr_boxs.append(mat.size)
    return out, nbr_boxs


def fix_bbox(bbox: List[float], img_path: str) -> List[float]:
    # bbox: [x0, y0, x1, y1]. x@with. y@height.
    _w = bbox[2] - bbox[0]
    _h = bbox[3] - bbox[1]

    out = [v for v in bbox]
    # case 1: 'horse/data/0004/shots/055/frame0020.jpg'
    # swap x0 and x1.
    if 'horse/data/0004/shots/055/frame0020.jpg' in img_path:
        print(f'w: {_w} h: {_h} @ {img_path}')
        assert _w < 0
        out = [bbox[2], bbox[1], bbox[0], bbox[3]]

    # case 2: 'boat/data/0006/shots/026/frame0001.jpg'
    # bbox of large ship.
    # width box is larger than image. x1 > width.
    if 'boat/data/0006/shots/026/frame0001.jpg' in img_path:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if bbox[2] > w:  # bbox of large ship.
            print(f'w: {w} x1: {bbox[2]} @ {img_path}')
            # x0, y0, x1, y1. set x1 tp w - 1.
            out = [bbox[0], bbox[1], w - 1, bbox[3]]

    return out


def get_list_bbox(matfile: str) -> List[list]:
    img_path = matfile.replace('_sticks.mat', '')
    assert os.path.isfile(img_path)
    w, h = Image.open(img_path).convert('RGB').size

    mat = io.loadmat(matfile)['coor']
    out = []
    nbr_bx = mat.size
    for el in mat.reshape((nbr_bx)):
        el = numpy_relu(el)  # drop negative values.
        el = el.squeeze().tolist()
        el = fix_bbox(el, img_path)

        # double check.
        _w = el[2] - el[0]
        _h = el[3] - el[1]
        assert 0 < _w < w, f'{el}: {w}, {_w} @ {img_path}'
        assert 0 < _h < h, f'{el}: {h}, {_h} @ {img_path}'

        check_box_convention(np.array(el).reshape((1, 4)), 'x0y0x1y1')

        out.append(el)  # [x0, y0, x1, y1]. x@with. y@height. floats.

    return out


def convert_abs_path_2_rel_p(root: str, path: str) -> str:
    return path.replace(root, '').lstrip(os.sep)


def numpy_relu(v: np.ndarray) -> np.ndarray:
    return np.abs(v * (v > 0))


def check_original_ytov1_0():
    ds = constants.YTOV1
    root = get_root_datasets()
    root_ds = join(root, ds)
    announce_msg(f'Processing {ds} @ {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el not in ['vo-release', 'code']]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    classes_id = {
        k: v for v, k in enumerate(ldirs_root)
    }

    with open(join(root_ds, "class_id.yaml"), 'w') as f:
        yaml.dump(classes_id, f)

    # get train and test vids folders.
    nbr_vids_tr, nbr_vids_tst = 0, 0
    tplt: dict = {
        k: 0 for k in _NBRS
    }
    stats = dict()

    holder = dict()
    empty_shots = 0

    for cl in tqdm.tqdm(classes_id, ncols=80,
                        total=len(list(classes_id.keys()))):
        # 1. vids.
        h = ['train.txt', 'test.txt']
        paths = []
        for el in h:
            pathx = join(root_ds, cl, f'sets/{el}')
            assert os.path.isfile(pathx)

            lfds = get_train_test_name_vids(pathx)
            lfds = [join(root_ds, cl, 'data', fd) for fd in lfds]

            for e in lfds:
                assert os.path.isdir(e), e

            paths.append(lfds)

        holder[cl] = {
            _TRAIN: dict(),
            _TEST: dict()
        }

        train_vids_fds, test_vids_fds = paths
        holder[cl][_TRAIN]: dict = convert_list_videos_to_dict(train_vids_fds)
        holder[cl][_TEST]: dict = convert_list_videos_to_dict(test_vids_fds)

        stats[cl] = deepcopy(tplt)
        stats[cl] = {
            _TRAIN: deepcopy(tplt),
            _TEST: deepcopy(tplt)
        }

        stats[cl][_TRAIN][_NBR_VIDS] = len(train_vids_fds)
        stats[cl][_TEST][_NBR_VIDS] = len(test_vids_fds)

        # 2. shots + frames.

        for subset in _SUBSETS:
            for vid in holder[cl][subset]:
                vids_has_bbox = False
                # shots
                list_dirs_shots = list_name_fds(join(vid, 'shots'))
                stats[cl][subset][_NBR_SHOTS] += len(list_dirs_shots)

                shots_dict = dict()

                # frames
                for shot in list_dirs_shots:
                    shot = join(vid, 'shots', shot)
                    l_frames = list_file_names_extension(shot,
                                                         pattern_ext='*.jpg')

                    empty_shots += (len(l_frames) == 0)
                    if len(l_frames) > 0:
                        stats[cl][subset][_NBR_FRAMES] += len(l_frames)

                        l_frames = [join(shot, frame) for frame in l_frames]
                        shots_dict[shot] = deepcopy(l_frames)

                    # frames with BBOX
                    l_fw_bbox, nbr_boxes = get_file_names_sticks_mat(shot)
                    stats[cl][subset][_NBR_FRAMES_W_BBOX] += len(l_fw_bbox)
                    stats[cl][subset][_NBR_BBOXES] += sum(nbr_boxes)
                    stats[cl][subset][_NBR_SHOTS_W_BBOX] += (sum(nbr_boxes) > 0)

                    if (not vids_has_bbox) and (sum(nbr_boxes) > 0):
                        vids_has_bbox = True

                stats[cl][subset][_NBR_VIDS_W_BBOX] += int(vids_has_bbox)

                holder[cl][subset][vid] = deepcopy(shots_dict)

    # log stats.
    with open(join(root_ds, 'original-stats.txt'), 'w') as fstats:
        pp = pprint.PrettyPrinter(depth=3, stream=fstats)
        pp.pprint(stats)
        print_total(stats, filelog=fstats)
        print(f'Nbr empty shots: {empty_shots}')


def build_test_ytov1_0():
    """
    Build test set for YTO v1.0 only over frames with bbox. used for evaluated.
    """
    ds = constants.YTOV1
    root = get_root_datasets()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TESTSET} set for {ds} @ {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el not in ['vo-release', 'code']]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TESTSET)
    os.makedirs(outfolds, exist_ok=True)
    with open(join(root_ds, "class_id.yaml"), 'r') as f:
        classes_id = yaml.safe_load(f)

    holder = dict()
    info = []
    for cl in tqdm.tqdm(classes_id, ncols=80,
                        total=len(list(classes_id.keys()))):
        # 1. vids.
        h = ['test.txt']
        paths = []
        for el in h:
            pathx = join(root_ds, cl, f'sets/{el}')
            assert os.path.isfile(pathx)

            lfds = get_train_test_name_vids(pathx)
            lfds = [join(root_ds, cl, 'data', fd) for fd in lfds]

            for e in lfds:
                assert os.path.isdir(e), e

            paths.append(lfds)

        holder[cl] = {
            _TEST: dict()
        }

        test_vids_fds = paths[0]
        holder[cl][_TEST]: dict = convert_list_videos_to_dict(test_vids_fds)

        # 2. shots + frames.

        subset = _TEST
        for vid in holder[cl][subset]:
            vids_has_bbox = False
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            # frames
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                content = next(os.walk(shot))[2]
                for item in content:
                    path = join(shot, item)
                    if os.path.isfile(path) and fnmatch.fnmatch(path,
                                                                '*_sticks.mat'):

                        img_file = path.replace('_sticks.mat', '')
                        assert os.path.isfile(img_file), img_file

                        identifier = convert_abs_path_2_rel_p(root_ds, img_file)
                        l_bbox = get_list_bbox(path)
                        for bb in l_bbox:
                            info.append([identifier, classes_id[cl], bb])

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')


def build_video_demo_test_ytov1_0():
    """
    Build test set over YTO v1.0 using all frames. useful for demos.
    """
    ds = constants.YTOV1
    root = get_root_datasets()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TESTSET_VIDEO_DEMO} set for {ds} @'
                 f' {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el not in ['vo-release', 'code']]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TESTSET_VIDEO_DEMO)
    os.makedirs(outfolds, exist_ok=True)
    with open(join(root_ds, "class_id.yaml"), 'r') as f:
        classes_id = yaml.safe_load(f)

    holder = dict()
    info = []
    for cl in tqdm.tqdm(classes_id, ncols=80,
                        total=len(list(classes_id.keys()))):
        # 1. vids.
        h = ['test.txt']
        paths = []
        for el in h:
            pathx = join(root_ds, cl, f'sets/{el}')
            assert os.path.isfile(pathx)

            lfds = get_train_test_name_vids(pathx)
            lfds = [join(root_ds, cl, 'data', fd) for fd in lfds]

            for e in lfds:
                assert os.path.isdir(e), e

            paths.append(lfds)

        holder[cl] = {
            _TEST: dict()
        }

        test_vids_fds = paths[0]
        holder[cl][_TEST]: dict = convert_list_videos_to_dict(test_vids_fds)

        # 2. shots + frames.

        subset = _TEST
        for vid in holder[cl][subset]:
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            # frames
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                content = next(os.walk(shot))[2]
                for item in content:
                    path = join(shot, item)

                    if os.path.isfile(path) and fnmatch.fnmatch(path, '*.jpg'):
                        identifier = convert_abs_path_2_rel_p(root_ds, path)
                        path_bbox = path + '_sticks.mat'
                        if os.path.isfile(path_bbox):
                            l_bbox = get_list_bbox(path_bbox)
                            for bb in l_bbox:
                                info.append([identifier, classes_id[cl], bb])
                        else:
                            zero_bbox = [0., 0., 0., 0.]  # null bbox.
                            info.append([identifier, classes_id[cl], zero_bbox])

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')


def build_train_valid_ytov1_0():
    """
    Build train and valid set for YTO v1.0.
    the original trainset videos has some videos with some shots that have
    bounding boxes.

    the original trainset of VIDEOS is split into two sets: videos for train,
    and videos for validation. videos in validset has bounding boxes
    localization.

    for each class, we take _NBR_VIDS_CL_VL videos that have bbox for
    validset. from these selected videos for validset, ONLY frames that have
    bbox are used. frames without bbox labels are discarded [not used].


    all other videos that are not considered for validset are used for
    trainset. we index shots instead of frames. when using dataloader,
    it will sample over shots. internally, we randomly select frame from the
    selected shot.
    """
    ds = constants.YTOV1
    root = get_root_datasets()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TRAINSET} and {constants.VALIDSET} '
                 f'set for {ds} @ {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el not in ['vo-release', 'code']]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    with open(join(root_ds, "class_id.yaml"), 'r') as f:
        classes_id = yaml.safe_load(f)

    holder = dict()
    info_valid = []
    info_train = []
    nbr_vids = {
        constants.TRAINSET: 0,
        constants.VALIDSET: 0
    }
    seed = 0

    for cl in tqdm.tqdm(classes_id, ncols=80,
                        total=len(list(classes_id.keys()))):
        # 1. vids.
        pathx = join(root_ds, cl, 'sets/train.txt')
        assert os.path.isfile(pathx)

        lfds = get_train_test_name_vids(pathx)
        lfds = [join(root_ds, cl, 'data', fd) for fd in lfds]

        labeld = []
        unlabeled = []
        for e in lfds:
            assert os.path.isdir(e), e
            if check_if_video_has_bbox(e):
                labeld.append(e)
            else:
                unlabeled.append(e)

        labeld = sorted(labeld, reverse=False)
        unlabeled = sorted(unlabeled, reverse=False)

        set_seed(seed)

        selected, leftover = random_select_k_vids(labeld)
        unlabeled += leftover

        nbr_vids[constants.VALIDSET] += len(selected)
        nbr_vids[constants.TRAINSET] += len(unlabeled)

        seed += 1

        holder[cl] = {
            constants.TRAINSET: dict(),
            constants.VALIDSET: dict()
        }

        holder[cl][constants.TRAINSET]: dict = convert_list_videos_to_dict(
            unlabeled)
        holder[cl][constants.VALIDSET]: dict = convert_list_videos_to_dict(
            selected)

        # 2. shots + frames.
        subset = constants.VALIDSET
        for vid in holder[cl][subset]:
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            # frames
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                content = next(os.walk(shot))[2]
                for item in content:
                    path = join(shot, item)
                    if os.path.isfile(path) and fnmatch.fnmatch(path,
                                                                '*_sticks.mat'):

                        img_file = path.replace('_sticks.mat', '')
                        assert os.path.isfile(img_file), img_file

                        identifier = convert_abs_path_2_rel_p(root_ds, img_file)
                        l_bbox = get_list_bbox(path)
                        for bb in l_bbox:
                            info_valid.append([identifier, classes_id[cl], bb])

        subset = constants.TRAINSET
        for vid in holder[cl][subset]:
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                l_frames = list_file_names_extension(shot,
                                                     pattern_ext='*.jpg')
                # add only shots that have frames.
                # some shots are empty.
                if len(l_frames):
                    identifier = convert_abs_path_2_rel_p(root_ds, shot)
                    info_train.append([identifier, classes_id[cl], None])

    # validset -----------------------------------------------------------------
    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.VALIDSET)
    os.makedirs(outfolds, exist_ok=True)

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info_valid:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')

    # trainset -----------------------------------------------------------------
    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TRAINSET)
    os.makedirs(outfolds, exist_ok=True)

    # shuffle train shots.
    set_seed(0)
    for i in range(10000):
        random.shuffle(info_train)

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_train:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_train:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w'):
        pass

    # localization.txt  #empty.
    with open(join(outfolds, 'localization.txt'), 'w'):
        pass

    with open(join(root_ds, 'stats-train-valid.txt'), 'w') as fstats:
        pp = pprint.PrettyPrinter(depth=3, stream=fstats)
        pp.pprint(nbr_vids)


def compress_only_needed_frames_ytov1_0():
    """
    In a different COPY of the dataset, we keep ONLY the NEEDED frames.
    For the trainset, we keep ALL frames from every included shot.
    For valid and test sets, we only keep used frames.
    This yields a COPY of the dataset with only necessary files.
    This could be useful on servers with limit quota in terms of number of
    files.

    COPY ONLY JPG FILES.
    """
    ds = constants.YTOV1
    root = get_root_datasets()
    root_ds = join(root, ds)
    outputfd = join(root, 'compressed', ds)
    os.makedirs(outputfd, exist_ok=True)
    announce_msg(f'Compressing {ds} dataset @ {root_ds}')

    # get all train frames.
    folds = {
        constants.TRAINSET: join(root_dir, constants.RELATIVE_META_ROOT, ds,
                                 constants.TRAINSET),
        constants.VALIDSET: join(root_dir, constants.RELATIVE_META_ROOT, ds,
                                 constants.VALIDSET),
        constants.TESTSET: join(root_dir, constants.RELATIVE_META_ROOT, ds,
                                constants.TESTSET)
    }

    with open(join(folds[constants.TRAINSET], 'image_ids.txt'), 'r') as fin:
        train_shots = fin.readlines()

    train_frames = []
    for shot in train_shots:
        shot = join(root_ds, shot.strip())
        assert os.path.isdir(shot)
        l_frames = list_file_names_extension(shot,
                                             pattern_ext='*.jpg')
        # add only shots that have frames.
        # some shots are empty.
        if len(l_frames):
            l_frames = [join(shot, frame) for frame in l_frames]
            train_frames += l_frames

    # train and test
    with open(join(folds[constants.VALIDSET], 'image_ids.txt'), 'r') as fin:
        valid_frames = fin.readlines()
        valid_frames = [join(root_ds, f.strip()) for f in valid_frames]
        assert all([os.path.isfile(f) for f in valid_frames])

    with open(join(folds[constants.TESTSET], 'image_ids.txt'), 'r') as fin:
        test_frames = fin.readlines()
        test_frames = [join(root_ds, f.strip()) for f in test_frames]
        assert all([os.path.isfile(f) for f in test_frames])

    necessary_frames = train_frames + valid_frames + test_frames
    print(f'@{ds}: Number of necessary frames {len(necessary_frames)}')

    print('Making a copy...')
    for frame in tqdm.tqdm(necessary_frames, ncols=80, total=len(
            necessary_frames)):
        source = frame
        dest = source.replace(root_ds, outputfd)
        os.makedirs(dirname(dest), exist_ok=True)
        shutil.copy(source, dest)


def process_ytov2_2():
    # todo
    pass


if __name__ == '__main__':

    # ==========================================================================
    #                                  YTO V1.0
    # ==========================================================================

    # 1. get stats from original dataset.
    # check_original_ytov1_0()

    # 2. build testset only over frames with bbox.
    # build_test_ytov1_0()

    # 3. build testset for all frames. w/wo bbox.
    # build_video_demo_test_ytov1_0()

    # 3. build train and valid set.
    # build_train_valid_ytov1_0()

    # 4- [optional] compress dataset into a COPY and keep only necessary frames
    compress_only_needed_frames_ytov1_0()

    print('Done.')

