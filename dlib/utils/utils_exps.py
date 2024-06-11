import os.path
import sys
from os.path import join, dirname, abspath, normpath, basename
import subprocess

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc

__all__ = ['copy_exp_dir_node_to_scratch']


def copy_exp_dir_node_to_scratch(args):
    assert is_cc()

    scratch_exp_fd = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER,
                          args.subpath)
    scratch_tmp = dirname(normpath(scratch_exp_fd))  # parent
    _tag = basename(normpath(args.outd))
    cmdx = [
        "cd {} ".format(args.outd),
        "cd .. ",
        "tar -cf {}.tar.gz {}".format(_tag, _tag),
        'cp {}.tar.gz {}'.format(_tag, scratch_tmp),
        'cd {}'.format(scratch_tmp),
        'tar -xf {}.tar.gz -C {} --strip-components=1'.format(
            _tag, basename(normpath(scratch_exp_fd))),
        "rm {}.tar.gz".format(_tag)
    ]
    cmdx = " && ".join(cmdx)
    DLLogger.log("Running bash-cmds: \n{}".format(cmdx.replace("&& ", "\n")))
    subprocess.run(cmdx, shell=True, check=True)
