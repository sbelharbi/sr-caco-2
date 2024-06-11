import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from dlib.boxnet.core import DenseBoxNet
