import sys
from os.path import dirname, abspath
from typing import Optional

import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.base import DenseBoxModel
from dlib.base import BboxHead

from dlib.utils import constants


class DenseBoxNet(DenseBoxModel):
    """
    Dense box model.

    Encoder + a localization head (for one box).
    """

    def __init__(
        self,
        task: str,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        h: int = 224,
        w: int = 224,
        scale_in: float = 1.,
        freeze_enc=False
    ):
        super().__init__()

        self.freeze_enc = freeze_enc
        self.task = constants.C_BOX
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert h > 0
        assert w > 0
        assert isinstance(h, int)
        assert isinstance(w, int)

        assert h == float(h)
        assert w == float(w)

        self.h = float(h)
        self.w = float(w)

        self.x_in = None

        self.encoder = get_encoder(
            task,
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.box_head = BboxHead(
            in_channels=self.encoder.out_channels[-1],
            h=h,
            w=w
        )

        self.coords = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


def test_DenseBoxNet():
    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    encoders = [constants.INCEPTIONV3, constants.VGG16, constants.RESNET50]
    SZ = 224
    CL = int(SZ * 1.)
    in_channels = 3
    bsz = 8
    sample = torch.rand((bsz, in_channels, SZ, SZ)).to(DEVICE)
    nmaps = 2

    seg_h_out_channels = nmaps

    for encoder_name in encoders:

        vgg_encoders = dlib.encoders.vgg_encoders

        if encoder_name == constants.VGG16:
            decoder_channels = (256, 128, 64)
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            decoder_channels = (256, 128, 64, 32, 16)
            encoder_depth = 5

        announce_msg("Testing backbone {}".format(encoder_name))

        target = torch.zeros(size=(bsz, nmaps, SZ, SZ), dtype=torch.long,
                             device=DEVICE)
        target[:, :, int(SZ/2): int(SZ/2) + 10, int(SZ/2): int(SZ/2) + 10] = 1

        model = DenseBoxNet(
            task=constants.C_BOX,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=constants.IMAGENET,
            in_channels=in_channels,
            h=CL,
            w=CL
        ).to(DEVICE)

        announce_msg("TESTING: {} -- ]n {}".format(
            model, model.get_info_nbr_params()))

        # glabel = torch.randint(low=0, high=classes, size=(bsz,),
        #                        dtype=torch.long, device=DEVICE)
        box = model(sample)
        print(f'input {sample.shape}')
        print(f'box: {box.shape}')


if __name__ == "__main__":
    import dlib
    from dlib.utils.shared import announce_msg
    from dlib.utils.utils_reproducibility import set_seed

    set_seed(0)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = False

    test_DenseBoxNet()



