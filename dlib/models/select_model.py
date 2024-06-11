import sys
from os.path import dirname, abspath


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


__all__ = ['define_model']


def define_model(args):
    task = args.task

    if task in [constants.SUPER_RES, constants.RECONSTRUCT]:
        from dlib.models.model_plain import ModelPlain as Model

    else:
        raise NotImplementedError(f'Unknown task: {task}')

    m = Model(args)

    DLLogger.log(fmsg(f'Model {m.__class__.__name__} has been created.'))

    return m
