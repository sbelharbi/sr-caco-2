import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.loss.master import MasterLoss

from dlib.loss.main import L1
from dlib.loss.main import L2
from dlib.loss.main import L2Sum
from dlib.loss.main import Charbonnier
from dlib.loss.main import NegativeSsim
from dlib.loss.main import BoundedPrediction
from dlib.loss.main import LocalMoments
from dlib.loss.main import ImageGradientLoss
from dlib.loss.main import LaplacianFilterLoss
from dlib.loss.main import LocalVariationLoss
from dlib.loss.main import NormImageGradientLoss
from dlib.loss.main import NormLaplacianFilterLoss
from dlib.loss.main import NormLocalVariationLoss
from dlib.loss.main import HistogramMatch
from dlib.loss.main import KDEMatch
from dlib.loss.main import CrossEntropyL
from dlib.loss.main import WeightsSparsityLoss
