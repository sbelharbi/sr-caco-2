import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.crf.crf_post_processing import DenseCRFFilter
from dlib.crf.dense_crf_loss import DenseCRFLoss
