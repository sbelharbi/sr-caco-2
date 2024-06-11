import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.distmatching.base import ColorDistDisentangle
from dlib.distmatching.base import MaskColorKDE
from dlib.distmatching.base import BhattacharyyaCoeffs
from dlib.distmatching.base import KDE4Loss