"""
Create csv file of camelyon16 dataset.
"""
import random
import csv
import sys
import os
from os.path import join, dirname, abspath

import yaml

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.utils.tools import Dict2Obj
from dlib.datasets.tools import get_rootpath_2_dataset

from dlib.utils.shared import announce_msg

from dlib.utils import constants

from dlib.utils.utils_reproducibility import set_default_seed


__all__ = ["do_camelyon16"]


def dump_fold_into_csv_CAM16(lsamples, outpath, tag):
    """
    for camelyon16 dataset.
    Write a list of list of information about each sample into a csv
    file where each row represent a sample in the following format:
    row = "id": 0, "img": 1, "mask": 2, "label": 3, "tag": 4
    Possible tags:
    0: labeled
    1: unlabeled
    2: labeled but came from unlabeled set. [not possible at this level]

    Relative paths allow running the code an any device.
    The absolute path within the device will be determined at the running
    time.

    :param lsamples: list of tuple (str: relative path to the image,
    float: id, str: label).
    :param outpath: str, output file name.
    :param tag: int, tag in constants.samples_tags for ALL the samples.
    :return:
    """
    msg = "'tag' must be an integer. Found {}.".format(tag)
    assert isinstance(tag, int), msg
    msg = "'tag' = {} is unknown. Please see " \
          "constants.samples_tags = {}.".format(tag, constants.samples_tags)
    assert tag in constants.samples_tags, msg

    msg = "It is weird that you are tagging a sample as {} while we" \
          "are still in the splitting phase. This is not expected." \
          "We're out.".format(constants.PL)
    assert tag != constants.PL, msg

    with open(outpath, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for idcnt, img_path, mask_path, img_label in lsamples:
            filewriter.writerow(
                [str(int(idcnt)),
                 img_path,
                 mask_path,
                 img_label,
                 tag]
            )


def split_camelyon16(args):
    """
    Use the provided split:
    https://github.com/jeromerony/survey_wsl_histology/blob/master/
    datasets-split/README.md

    :param args:
    :return:
    """

    def csv_loader(fname):
        """
        Read a *.csv file. Each line contains:
         1. img: str
         2. mask: str or '' or None
         3. label: str

        :param fname: Path to the *.csv file.
        :param rootpath: The root path to the folders of the images.
        :return: List of elements.
        Each element is the path to an image: image path, mask path [optional],
        class name.
        """
        with open(fname, 'r') as f:
            out = [
                [row[0],
                 row[1] if row[1] else None,
                 row[2]
                 ]
                for row in csv.reader(f)
            ]

        return out

    csv_df = 'folds/camelyon16-split-0-fold-0-512-512-survey'
    # load survey csv files.
    trainset = csv_loader(join(csv_df, 'train_s_0_f_0.csv'))
    validset = csv_loader(join(csv_df, 'valid_s_0_f_0.csv'))
    testset = csv_loader(join(csv_df, 'test_s_0_f_0.csv'))

    baseurl = args.baseurl

    # find all the files
    fdimg = join(baseurl, 'jpg')
    tr_set, vl_set, ts_set = [], [], []
    idcnt = 0.  # count the unique id for each sample

    stats = {
        'train': {
            'normal': 0.,
            'tumor': 0.
        },
        'valid': {
            'normal': 0.,
            'tumor': 0.
        },
        'test': {
            'normal': 0.,
            'tumor': 0.
        }
    }

    # train
    for f in trainset:
        img = f[0]
        mask = f[1]
        label = f[2]
        tr_set.append((idcnt, img, mask, label))
        idcnt += 1.
        if label == 'normal':
            stats['train']['normal'] += 1.
        else:
            stats['train']['tumor'] += 1.

    # valid
    for f in validset:
        img = f[0]
        mask = f[1]
        label = f[2]
        vl_set.append((idcnt, img, mask, label))
        idcnt += 1.

        if label == 'normal':
            stats['valid']['normal'] += 1.
        else:
            stats['valid']['tumor'] += 1.

    # test
    for f in testset:
        img = f[0]
        mask = f[1]
        label = f[2]
        ts_set.append((idcnt, img, mask, label))
        idcnt += 1.

        if label == 'normal':
            stats['test']['normal'] += 1.
        else:
            stats['test']['tumor'] += 1.

    dict_classes_names = {"normal": 0, "tumor": 1}

    outd = args.fold_folder
    out_fold = join(outd, "split_{}/fold_{}".format(0, 0))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold, exist_ok=True)

    readme = "Format: float `id`: 0, str `img`: 1, None `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."

    # shuffle train
    for t in range(1000):
        random.shuffle(tr_set)

    dump_fold_into_csv_CAM16(tr_set,
                             join(out_fold, "train_s_{}_f_{}.csv".format(0, 0)),
                             constants.L
                             )
    dump_fold_into_csv_CAM16(vl_set,
                             join(out_fold, "valid_s_{}_f_{}.csv".format(0, 0)),
                             constants.L
                             )
    dump_fold_into_csv_CAM16(ts_set,
                             join(out_fold, "test_s_{}_f_{}.csv".format(0, 0)),
                             constants.L
                             )

    # current fold.

    # dump the coding.
    with open(join(out_fold, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    # dump the seed
    with open(join(out_fold, "seed.txt"), 'w') as fx:
        fx.write("MYSEED: " + os.environ["MYSEED"])

    with open(join(out_fold, "readme.md"), 'w') as fx:
        fx.write(readme)

    with open(join(out_fold, "stats-sets.yaml"), 'w') as fx:
        total = sum([stats[el]['normal'] + stats[el]['tumor'] for el in
                   list(stats.keys())])
        stats['total_samples'] = total
        yaml.dump(stats, fx)
        print("Stats:", stats)

    # folder of folds

    # readme
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    print("camelyon16 splitting (`{}`) ended with success .... [OK]".format(0))


def do_camelyon16(root_main):
    """
    camelyon16.
    The train/valid/test sets are already provided.

    :param root_main: str. absolute path to folder containing main.py.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    set_default_seed()

    # ===========================

    ds = constants.CAM16
    announce_msg("Processing dataset: {}".format(ds))
    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': ds})),
            "dataset": ds,
            "fold_folder": join(root_main, "folds/{}".format(ds)),
            "img_extension": "jpg",
            "path_encoding": join(root_main,
                                  "folds/{}/encoding-origine.yaml".format(ds)
                                  )
            }

    set_default_seed()
    split_camelyon16(Dict2Obj(args))
