import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.cams.normalizers import CamStandardizer
from dlib.cams.fcam_seeding import GetFastSeederSLFCAMS
from dlib.cams.fcam_seeding import MBSeederSLFCAMS
from dlib.cams.fcam_seeding import SeederCBOX


from dlib.cams.seeds_eval import AccSeeds
from dlib.cams.seeds_eval import AccSeedsmeter
from dlib.cams.seeds_eval import BasicAccSeedsMeter


from dlib.cams.cam import CAM
from dlib.cams.cam import ScoreCAM
from dlib.cams.cam import SSCAM
from dlib.cams.cam import ISCAM

from dlib.cams.gradcam import GradCAM
from dlib.cams.gradcam import GradCAMpp
from dlib.cams.gradcam import SmoothGradCAMpp
from dlib.cams.gradcam import XGradCAM
from dlib.cams.gradcam import LayerCAM

from dlib.cams.builtincam import BuiltinCam
from dlib.cams.builtincam import SegmentationCam

from dlib.utils import constants


def build_fcam_extractor(model, args):
    assert args.task == constants.F_CL
    model.eval()
    return SegmentationCam(model=model)


def build_tcam_extractor(model, args):
    assert args.task == constants.TCAM
    model.eval()
    return SegmentationCam(model=model)


def build_std_cam_extractor(classifier, args):
    p1 = [constants.GAP, constants.MAXPOOL, constants.WILDCATHEAD,
          constants.LSEPOOL]
    mbin1 = [constants.METHOD_WILDCAT, constants.METHOD_GAP,
             constants.METHOD_MAXPOOL, constants.METHOD_LSE]

    classifier.eval()

    method = args.method

    # builtin
    if method in mbin1:
        assert args.spatial_pooling in p1
        return BuiltinCam(model=classifier)

    # cam
    mcam = [constants.METHOD_CAM, constants.METHOD_SCORECAM,
            constants.METHOD_SSCAM, constants.METHOD_ISCAM]

    encoder_name = args.model['encoder_name']
    trg_layer = constants.TRG_LAYERS[encoder_name]
    fc_layer = constants.FC_LAYERS[encoder_name]
    if method in mcam:
        assert args.spatial_pooling == constants.WGAP

        if method == constants.METHOD_CAM:
            return CAM(model=classifier, target_layer=trg_layer,
                       fc_layer=fc_layer)

        if method == constants.METHOD_SCORECAM:
            return ScoreCAM(model=classifier, target_layer=trg_layer)

        if method == constants.METHOD_SSCAM:
            return SSCAM(model=classifier, target_layer=trg_layer)

        if method == constants.METHOD_ISCAM:
            return ISCAM(model=classifier, target_layer=trg_layer)

        raise NotImplementedError

    # gradcam
    mgradc = [constants.METHOD_GRADCAM, constants.METHOD_GRADCAMPP,
              constants.METHOD_SMOOTHGRADCAMPP,
              constants.METHOD_XGRADCAM, constants.METHOD_LAYERCAM]
    if method in mgradc:
        assert args.spatial_pooling == constants.WGAP

        if method == constants.METHOD_GRADCAM:
            return GradCAM(model=classifier, target_layer=trg_layer)

        if method == constants.METHOD_GRADCAMPP:
            return GradCAMpp(model=classifier, target_layer=trg_layer)

        if method == constants.METHOD_SMOOTHGRADCAMPP:
            return SmoothGradCAMpp(model=classifier, target_layer=trg_layer)

        if method == constants.METHOD_XGRADCAM:
            return XGradCAM(model=classifier, target_layer=trg_layer)

        if method == constants.METHOD_LAYERCAM:
            return LayerCAM(model=classifier, target_layer=trg_layer)

        raise NotImplementedError

    raise NotImplementedError

