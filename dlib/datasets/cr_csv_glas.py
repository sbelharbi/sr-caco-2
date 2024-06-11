"""
Create csv file of glas dataset.
"""
import math
import random
import copy
import csv
import sys
import os
from os.path import join, dirname, abspath

import matplotlib.pyplot as plt
import numpy as np
import yaml


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.csv_tools import get_stats

from dlib.utils.tools import chunk_it
from dlib.utils.tools import Dict2Obj
from dlib.datasets.tools import get_rootpath_2_dataset

from dlib.utils.shared import announce_msg

from dlib.utils import constants

from dlib.utils.utils_reproducibility import set_default_seed


__all__ = ["do_glas"]


def dump_fold_into_csv_glas(lsamples, outpath, tag):
    """
    For glas dataset.

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

    assert tag == constants.L

    with open(outpath, 'w') as fcsv:
        filewriter = csv.writer(
            fcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, clas, id_ in lsamples:
            filewriter.writerow(
                [str(int(id_)),
                 name + ".bmp",
                 name + "_anno.bmp",
                 clas,
                 tag]
            )


def split_glas(args):
    """
    Splits Glas dataset.
    It creates a validation/train sets in GlaS dataset.

    :param args:
    :return:
    """
    classes = ["benign", "malignant"]
    all_samples = []
    # Read the file Grade.csv
    baseurl = args.baseurl
    idcnt = 0.  # count the unique id for each sample
    with open(join(baseurl, "Grade.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # get rid of the header.
        for row in reader:
            # not sure why they thought it is a good idea to put a space
            # before the class. Now, I have to get rid of
            # it and possibly other hidden spaces ...
            row = [r.replace(" ", "") for r in row]
            msg = "The class `{}` is not within the predefined " \
                  "classes `{}`".format(row[2], classes)
            assert row[2] in classes, msg
            # name file, patient id, label
            all_samples.append([row[0], int(row[1]), row[2], idcnt])
            idcnt += 1.

    msg = "The number of samples {} do not match what they said " \
          "(165) .... [NOT OK]".format(len(all_samples))
    assert len(all_samples) == 165, msg

    # Take test samples aside. They are fix.
    test_samples = [[s[0], s[2], s[3]] for s in all_samples if s[0].startswith(
        "test")]
    msg = "The number of test samples {} is not 80 as they " \
          "said .... [NOT OK]".format(len(test_samples))
    assert len(test_samples) == 80, msg

    all_train_samples = [s for s in all_samples if s[0].startswith("train")]

    msg = "The number of train samples {} is not 85 " \
          "as they said .... [NOT OK]".format(len(all_train_samples))
    assert len(all_train_samples) == 85, msg

    patients_id = np.array([el[1] for el in all_train_samples])
    fig = plt.figure()
    plt.hist(patients_id, bins=np.unique(patients_id))
    plt.title("histogram-glas-train.")
    plt.xlabel("patient_id")
    plt.ylabel("number of samples")
    fig.savefig(join(root_dir, "tmp/glas-train.jpeg"))
    # the number of samples per patient are highly unbalanced. so, we do not
    # split patients, but classes. --> we allow that samples from same
    # patient end up in train and valid. it is not that bad. it is just the
    # validation. plus, they are histology images. only the stain is more
    # likely to be relatively similar.

    all_train_samples = [[s[0], s[2], s[3]] for s in all_samples if s[
        0].startswith(
        "train")]

    benign = [s for s in all_train_samples if s[1] == "benign"]
    malignant = [s for s in all_train_samples if s[1] == "malignant"]

    # encode class name into int.
    dict_classes_names = {'benign': 0, 'malignant': 1}

    if not os.path.exists(args.fold_folder):
        os.makedirs(args.fold_folder, exist_ok=True)

    readme = "Format: float `id`: 0, str `img`: 1, str `mask`: 2, " \
             "str `label`: 3, int `tag`: 4 \n" \
             "Possible tags: \n" \
             "0: labeled\n" \
             "1: unlabeled\n" \
             "2: labeled but came from unlabeled set. " \
             "[not possible at this level]."
    # dump the readme
    with open(join(args.fold_folder, "readme.md"), 'w') as fx:
        fx.write(readme)
    # dump the coding.
    with open(join(args.fold_folder, "encoding.yaml"), 'w') as f:
        yaml.dump(dict_classes_names, f)

    # Split
    splits = []
    for i in range(args.nbr_splits):
        for _ in range(1000):
            random.shuffle(benign)
            random.shuffle(malignant)
        splits.append({"benign": copy.deepcopy(benign),
                       "malignant": copy.deepcopy(malignant)}
                      )

    # Create the folds.
    def create_folds_of_one_class(lsamps, s_tr, s_vl):
        """
        Create k folds from a list of samples of the same class, each fold
        contains a train, and valid set with a
        predefined size.

        Note: Samples are expected to be shuffled beforehand.

        :param lsamps: list of paths to samples of the same class.
        :param s_tr: int, number of samples in the train set.
        :param s_vl: int, number of samples in the valid set.
        :return: list_folds: list of k tuples (tr_set, vl_set, ts_set): w
        here each element is the list (str paths)
                 of the samples of each set: train, valid, and test,
                 respectively.
        """
        msg = "Something wrong with the provided sizes."
        assert len(lsamps) == s_tr + s_vl, msg

        # chunk the data into chunks of size ts
        # (the size of the test set), so we can rotate the test set.
        list_chunks = list(chunk_it(lsamps, s_vl))
        list_folds = []

        for i in range(len(list_chunks)):
            vl_set = list_chunks[i]

            right, left = [], []
            if i < len(list_chunks) - 1:
                right = list_chunks[i + 1:]
            if i > 0:
                left = list_chunks[:i]

            leftoverchunks = right + left

            leftoversamples = []
            for e in leftoverchunks:
                leftoversamples += e

            tr_set = leftoversamples
            list_folds.append((tr_set, vl_set))

        return list_folds

    def create_one_split(split_i, test_samples, benign, malignant, nbr_folds):
        """
        Create one split of k-folds.

        :param split_i: int, the id of the split.
        :param test_samples: list, list of test samples.
        :param benign: list, list of train benign samples.
        :param malignant: list, list of train maligant samples.
        :param nbr_folds: int, number of folds [the k value in k-folds].
        :return:
        """
        vl_size_benign = math.ceil(len(benign) * args.folding["vl"] / 100.)
        vl_size_malignant = math.ceil(
            len(malignant) * args.folding["vl"] / 100.)

        list_folds_benign = create_folds_of_one_class(
            benign, len(benign) - vl_size_benign, vl_size_benign)
        list_folds_malignant = create_folds_of_one_class(
            malignant, len(malignant) - vl_size_malignant, vl_size_malignant)

        msg = "We didn't obtain the same number of fold .... [NOT OK]"
        assert len(list_folds_benign) == len(list_folds_malignant), msg

        print("We found {} folds .... [OK]".format(len(list_folds_malignant)))

        outd = args.fold_folder
        for i in range(nbr_folds):
            out_fold = join(outd, "split_" + str(split_i) + "/fold_" + str(i))
            if not os.path.exists(out_fold):
                os.makedirs(out_fold, exist_ok=True)

            # dump the test set
            dump_fold_into_csv_glas(
                test_samples,
                join(out_fold, "test_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the train set
            train = list_folds_malignant[i][0] + list_folds_benign[i][0]
            # shuffle
            for t in range(1000):
                random.shuffle(train)

            dump_fold_into_csv_glas(
                train,
                join(out_fold, "train_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the valid set
            valid = list_folds_malignant[i][1] + list_folds_benign[i][1]
            dump_fold_into_csv_glas(
                valid,
                join(out_fold, "valid_s_{}_f_{}.csv".format(split_i, i)),
                constants.L
            )

            # dump the seed
            with open(join(out_fold, "seed.txt"), 'w') as fx:
                fx.write("MYSEED: " + os.environ["MYSEED"])
            # dump the coding.
            with open(join(out_fold, "encoding.yaml"), 'w') as f:
                yaml.dump(dict_classes_names, f)
            # dump the readme
            with open(join(out_fold, "readme.md"), 'w') as fx:
                fx.write(readme)

    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder, exist_ok=True)

    # Creates the splits
    for i in range(args.nbr_splits):
        create_one_split(
            i, test_samples, splits[i]["benign"], splits[i]["malignant"],
            args.nbr_folds)

    print("All GlaS splitting (`{}`) ended with success .... [OK]".format(
        args.nbr_splits))


def do_glas(root_main):
    """
    GlaS.

    :param root_main: str. absolute path to folder containing main.py.
    :return:
    """
    set_default_seed()

    announce_msg("Processing dataset: {}".format(constants.GLAS))

    args = {"baseurl": get_rootpath_2_dataset(
            Dict2Obj({'dataset': constants.GLAS})),
            "folding": {"vl": 20},  # 80 % for train, 20% for validation.
            "dataset": "glas",
            "fold_folder": join(root_main, "folds/glas"),
            "img_extension": "bmp",
            # nbr_splits: how many times to perform the k-folds over
            # the available train samples.
            "nbr_splits": 1
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])

    set_default_seed()
    split_glas(Dict2Obj(args))
    get_stats(Dict2Obj(args), split=0, fold=0, subset='train')
