"""
Create cvs folds for all the datasets.
- glas
- Caltech_UCSD_Birds_200_2011
- Cityscapes
"""
import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.cr_csv_glas import do_glas
from dlib.datasets.cr_csv_Caltech_UCSD_Birds_200_2011 import do_Caltech_UCSD_Birds_200_2011
from dlib.datasets.cr_csv_camelyon16 import do_camelyon16
from dlib.datasets.cr_csv_cityscapes import do_cityscapes

from dlib.utils.utils_reproducibility import set_default_seed

if __name__ == "__main__":
    set_default_seed()

    # ==========================================================================
    #                             START
    # ==========================================================================
    do_glas(root_main=root_dir)
    do_Caltech_UCSD_Birds_200_2011(root_main=root_dir)
    do_cityscapes(root_main=root_dir)
    # ==========================================================================
    #                             END
    # ==========================================================================
