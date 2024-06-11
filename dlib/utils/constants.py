# tasks:
SUPER_RES = 'super-resolution'  # upscale by some factor an input image.
RECONSTRUCT = 'reconstruct'  # takes an input and output an image with the
# same size.

TASKS = [SUPER_RES, RECONSTRUCT]

LOW_RES = 'low_res'
HIGH_RES = 'high_res'
RECONSTRUCT_TYPE = [LOW_RES, HIGH_RES]

RECON_IN_FAKE = 'fake'
RECON_IN_REAL = 'real'
RECON_IN_HR = 'high_res'
RECON_IN_L_TO_HR = 'low_to_high_res'

RECON_INPUTS = [RECON_IN_FAKE, RECON_IN_REAL, RECON_IN_HR, RECON_IN_L_TO_HR]

# models

#
DSRSPLINES = 'DSR-Splines'
CSRCNN = 'CSR-CNN'

# Pre-sampling.
SRCNN = 'SRCNN'  # https://arxiv.org/pdf/1501.00092.pdf
VDSR = 'VDSR'  # https://arxiv.org/pdf/1511.04587.pdf
MEMNET = 'MemNet'  # https://arxiv.org/pdf/1708.02209.pdf
DRRN = 'DRRN'  # https://ieeexplore.ieee.org/document/8099781

# Post-sampling
SWINIR = 'swinir'  # https://arxiv.org/pdf/2108.10257.pdf
DFCAN = 'DFCAN'  # https://www.nature.com/articles/s41592-020-01048-5
OMNISR = 'OmniSR'  # https://arxiv.org/pdf/2304.10244.pdf
GRL = 'GRL'  # https://arxiv.org/pdf/2303.00748.pdf
ENLCN = 'ENLCN'  # https://arxiv.org/pdf/2201.03794.pdf
ACT = 'ACT'  # https://arxiv.org/pdf/2203.07682.pdf
NLSN = 'NLSN'  # https://openaccess.thecvf.com/content/CVPR2021/papers

# Iterative sampling
SRFBN = 'SRFBN'  # https://arxiv.org/pdf/1903.09814.pdf
DBPN = 'DBPN'  # https://arxiv.org/pdf/1803.02735.pdf


# Progressive sampling
MSLAPSR = 'MSLapSRN'  # https://arxiv.org/pdf/1710.01992.pdf
PROSR = 'ProSR'  # https://arxiv.org/pdf/1804.02900.pdf


MODELS = [SWINIR,
          DSRSPLINES,
          CSRCNN,
          DFCAN,
          SRCNN,
          VDSR,
          MEMNET,
          DRRN,
          OMNISR,
          GRL,
          ENLCN,
          ACT,
          NLSN,
          SRFBN,
          DBPN,
          MSLAPSR,
          PROSR
          ]

# init weights types
INIT_W_NORMAL = 'init_w_normal'
INIT_W_UNIFORM = 'init_w_uniform'
INIT_W_XAVIER_NORMAL = 'init_w_xavier_normal'
INIT_W_XAVIER_UNIFORM = 'init_w_xavier_uniform'
INIT_W_KAIMING_NORMAL = 'init_w_kaiming_normal'
INIT_W_KAIMING_UNIFORM = 'init_w_kaiming_uniform'
INIT_W_ORTHOGONAL = 'init_w_orthogonal'
INIT_W_DEFAULT = 'init_w_default'

INIT_W = [INIT_W_NORMAL, INIT_W_UNIFORM, INIT_W_XAVIER_NORMAL,
          INIT_W_XAVIER_UNIFORM, INIT_W_KAIMING_NORMAL,
          INIT_W_KAIMING_UNIFORM, INIT_W_ORTHOGONAL, INIT_W_DEFAULT]

INIT_BN_UNIFORM = 'init_bn_uniform'
INIT_BN_CONSTANT = 'init_bn_constant'

INIT_BN = [INIT_BN_CONSTANT, INIT_BN_UNIFORM]

# swinir upsampler
US_PIXEL_SHUFFLE = 'pixelshuffle'
US_PIXEL_SHUFFLE_DIRECT = 'pixelshuffledirect'
US_NEAREST_CONV = 'nearest_conv'

R_CONNECTION_1CONV = '1conv'
R_CONNECTION_3CONV = '3conv'


# phases:
TRAIN_PHASE = 'train'
EVAL_PHASE = 'eval'
PHASES = [TRAIN_PHASE, EVAL_PHASE]

# datasets:
TRAINSET = 'train'
VALIDSET = 'val'
TESTSET = 'test'

SPLITS = [TRAINSET, VALIDSET, TESTSET]

# Metrics
PSNR_MTR = 'psnr'
SSIM_MTR = 'ssim'
MSE_MTR = 'mse'
NRMSE_MTR = 'nrmse'

PSNR_Y_MTR = 'psnr_y'
SSIM_Y_MTR = 'ssim_y'


METRICS = [PSNR_MTR, SSIM_MTR, MSE_MTR, NRMSE_MTR, PSNR_Y_MTR, SSIM_Y_MTR]

BEST_MTR = {
    PSNR_MTR: max,
    SSIM_MTR: max,
    MSE_MTR: min,
    NRMSE_MTR: min,
    PSNR_Y_MTR: max,
    SSIM_Y_MTR: max
}

# tracker periods
PR_EPOCH = 'period_epoch'
PR_ITER = 'period_iter'

PERIODS = [PR_ITER, PR_EPOCH]

# METHODs
SWINIR_MTH = 'SWINIR'
DSRSPLINES_MTH = 'DSR-SPLINES'
CSRCNN_MTH = 'CSR-CNN'
DFCAN_MTH = 'DFCAN'
SRCNN_MTH = 'SRCNN'
VDSR_MTH = 'VDSR'
MEMNET_MTH = 'MemNet'
DRRN_MTH = 'DRRN'

OMNISR_MTH = 'OmniSR'
GRL_MTH = 'GRL'
ENLCN_MTH = 'ENLCN'
ACT_MTH = 'ACT'
NLSN_MTH = 'NLSN'

SRFBN_MTH = 'SRFBN'
DBPN_MTH = 'DBPN'

MSLAPSR_MTH = 'MSLAPSR'
PROSR_MTH = 'PROSR'


METHODS = [DSRSPLINES_MTH, CSRCNN_MTH, SRCNN_MTH,
           VDSR_MTH, MEMNET_MTH, DRRN_MTH,
           DFCAN_MTH, SWINIR_MTH, OMNISR_MTH, GRL_MTH, ENLCN_MTH,
           ACT_MTH, NLSN_MTH,
           SRFBN_MTH, DBPN_MTH,
           MSLAPSR_MTH, PROSR_MTH
           ]

# NET-TYPE to METHOD
NETTYPE_METHOD = {
    SWINIR: SWINIR_MTH,
    DSRSPLINES: DSRSPLINES_MTH,
    CSRCNN: CSRCNN_MTH,
    DFCAN: DFCAN_MTH,
    SRCNN: SRCNN_MTH,
    VDSR: VDSR_MTH,
    MEMNET: MEMNET_MTH,
    DRRN: DRRN_MTH,
    OMNISR: OMNISR_MTH,
    GRL: GRL_MTH,
    ENLCN: ENLCN_MTH,
    ACT: ACT_MTH,
    NLSN: NLSN_MTH,
    SRFBN: SRFBN_MTH,
    DBPN: DBPN_MTH,
    MSLAPSR: MSLAPSR_MTH,
    PROSR: PROSR_MTH
}

# Separator
SEP = '+'

CACO2 = 'CACO2'  # new dataset.
# its cells: from top to down.
CELL0 = 'CELL0'
CELL1 = 'CELL1'
CELL2 = 'CELL2'

# index in tile file.
CACO2_CELL_INDEX = {
    CELL0: 0,
    CELL1: 1,
    CELL2: 2
}

# Specimen names BIOSR.
BIOSR = 'BIOSR'
# cells
CCPS = 'CCPs'
ER = 'ER'
F_ACTIN = 'F-actin'
F_ACTIN_NONLINEAR = 'F-actin-Nonlinear'
MICROTUBULES = 'Microtubules'

# Interpolcation modes
INTER_BICUBIC = 'bicubic'

INTERPOLATION_MODES = [INTER_BICUBIC]

# datasets mode
DS_TRAIN = "TRAIN"
DS_EVAL = "EVAL"

dataset_modes = [DS_TRAIN, DS_EVAL]

# Tags for samples
L = 0  # Labeled samples

samples_tags = [L]  # list of possible sample tags.

# pixel-wise supervision:
ORACLE = "ORACLE"  # provided by an oracle.
SELF_LEARNED = "SELF-LEARNED"  # self-learned.
VOID = "VOID"  # None

# segmentation modes.
#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"


# pretraining
IMAGENET = "imagenet"

# archs
STDCLASSIFIER = "STDClassifier"

UNETFCAM = 'UnetFCAM'  # USED
# UNETCBOX = 'UnetCBox'  #
DENSEBOXNET = 'DenseBoxNet'  # used

UNETTCAM = 'UnetTCAM'  # used

UNET = "Unet"
UNETPLUPLUS = "UnetPlusPlus"
MANET = "MAnet"
LINKNET = "Linknet"
FPN = "FPN"
PSPNET = "PSPNet"
DEEPLABV3 = "DeepLabV3"
DEEPLABV3PLUS = "DeepLabV3Plus"
PAN = "PAN"

ARCHS = [STDCLASSIFIER, UNETFCAM, UNETTCAM, DENSEBOXNET]

# ecnoders

#  resnet
RESNET50 = 'resnet50'

# vgg
VGG16 = 'vgg16'

# inceptionv3
INCEPTIONV3 = 'inceptionv3'

BACKBONES = [RESNET50,
             VGG16,
             INCEPTIONV3
             ]

# ------------------------------------------------------------------------------

# datasets
DEBUG = False


ILSVRC = "ILSVRC"
CUB = "CUB"
OpenImages = 'OpenImages'
# wsol in videos.
YTOV1 = "YouTube-Objects-v1.0"
YTOV22 = "YouTube-Objects-v2.2"

# CACO2 ========================================================================
# scale 2
CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL0 = \
    'caco2_train_X_2_in_256_out_512_cell_CELL0'
CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL0 = \
    'caco2_val_X_2_in_256_out_512_cell_CELL0'
CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL0 = \
    'caco2_test_X_2_in_256_out_512_cell_CELL0'

CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL1 = \
    'caco2_train_X_2_in_256_out_512_cell_CELL1'
CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL1 = \
    'caco2_val_X_2_in_256_out_512_cell_CELL1'
CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL1 = \
    'caco2_test_X_2_in_256_out_512_cell_CELL1'

CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL2 = \
    'caco2_train_X_2_in_256_out_512_cell_CELL2'
CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL2 = \
    'caco2_val_X_2_in_256_out_512_cell_CELL2'
CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL2 = \
    'caco2_test_X_2_in_256_out_512_cell_CELL2'


# scale 4
CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL0 = \
    'caco2_train_X_4_in_128_out_512_cell_CELL0'
CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL0 = \
    'caco2_val_X_4_in_128_out_512_cell_CELL0'
CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL0 = \
    'caco2_test_X_4_in_128_out_512_cell_CELL0'

CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL1 = \
    'caco2_train_X_4_in_128_out_512_cell_CELL1'
CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL1 = \
    'caco2_val_X_4_in_128_out_512_cell_CELL1'
CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL1 = \
    'caco2_test_X_4_in_128_out_512_cell_CELL1'

CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL2 = \
    'caco2_train_X_4_in_128_out_512_cell_CELL2'
CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL2 = \
    'caco2_val_X_4_in_128_out_512_cell_CELL2'
CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL2 = \
    'caco2_test_X_4_in_128_out_512_cell_CELL2'


# scale 8
CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL0 = \
    'caco2_train_X_8_in_64_out_512_cell_CELL0'
CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL0 = \
    'caco2_val_X_8_in_64_out_512_cell_CELL0'
CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL0 = \
    'caco2_test_X_8_in_64_out_512_cell_CELL0'

CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL1 = \
    'caco2_train_X_8_in_64_out_512_cell_CELL1'
CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL1 = \
    'caco2_val_X_8_in_64_out_512_cell_CELL1'
CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL1 = \
    'caco2_test_X_8_in_64_out_512_cell_CELL1'

CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL2 = \
    'caco2_train_X_8_in_64_out_512_cell_CELL2'
CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL2 = \
    'caco2_val_X_8_in_64_out_512_cell_CELL2'
CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL2 = \
    'caco2_test_X_8_in_64_out_512_cell_CELL2'

# ==============================================================================

# BIOSR v1: v1 few samples

BIOSRV1_CCPS_TRAIN_X2 = 'biosrv1-ccps-train-X-2'
BIOSRV1_CCPS_VALID_X2 = 'biosrv1-ccps-val-X-2'
BIOSRV1_CCPS_TEST_X2 = 'biosrv1-ccps-test-X-2'

BIOSRV1_ER_TRAIN_X2 = 'biosrv1-er-train-X-2'
BIOSRV1_ER_VALID_X2 = 'biosrv1-er-val-X-2'
BIOSRV1_ER_TEST_X2 = 'biosrv1-er-test-X-2'

BIOSRV1_F_ACTIN_TRAIN_X2 = 'biosrv1-f-actin-train-X-2'
BIOSRV1_F_ACTIN_VALID_X2 = 'biosrv1-f-actin-val-X-2'
BIOSRV1_F_ACTIN_TEST_X2 = 'biosrv1-f-actin-test-X-2'

BIOSRV1_MICROTUBULES_TRAIN_X2 = 'biosrv2-microtubules-train-X-2'
BIOSRV1_MICROTUBULES_VALID_X2 = 'biosrv2-microtubules-val-X-2'
BIOSRV1_MICROTUBULES_TEST_X2 = 'biosrv2-microtubules-test-X-2'

# todo: x3 f-actin-nonlinear


FORMAT_DEBUG = 'DEBUG_{}'
if DEBUG:
    CUB = FORMAT_DEBUG.format(CUB)
    ILSVRC = FORMAT_DEBUG.format(ILSVRC)
    OpenImages = FORMAT_DEBUG.format(OpenImages)
    YTOV1 = FORMAT_DEBUG.format(YTOV1)
    YTOV22 = FORMAT_DEBUG.format(YTOV22)

datasets = [
    # CACO2
    # x2
    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL0,
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL0,
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL0,

    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL1,
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL1,
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL1,

    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL2,
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL2,
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL2,

    # x4
    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL0,
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL0,
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL0,

    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL1,
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL1,
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL1,

    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL2,
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL2,
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL2,

    # x8
    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL0,
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL0,
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL0,

    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL1,
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL1,
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL1,

    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL2,
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL2,
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL2,

    # BIOSR v1
    BIOSRV1_CCPS_TRAIN_X2,
    BIOSRV1_CCPS_VALID_X2,
    BIOSRV1_CCPS_TEST_X2,

    BIOSRV1_ER_TRAIN_X2,
    BIOSRV1_ER_VALID_X2,
    BIOSRV1_ER_TEST_X2,

    BIOSRV1_F_ACTIN_TRAIN_X2,
    BIOSRV1_F_ACTIN_VALID_X2,
    BIOSRV1_F_ACTIN_TEST_X2,

    BIOSRV1_MICROTUBULES_TRAIN_X2,
    BIOSRV1_MICROTUBULES_VALID_X2,
    BIOSRV1_MICROTUBULES_TEST_X2
]

# todo
DS_DIR = {
    # CACO2
    # x2
    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL0: 'caco2',
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL0: 'caco2',
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL0: 'caco2',

    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL1: 'caco2',
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL1: 'caco2',
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL1: 'caco2',

    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL2: 'caco2',
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL2: 'caco2',
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL2: 'caco2',

    # x4
    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL0: 'caco2',
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL0: 'caco2',
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL0: 'caco2',

    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL1: 'caco2',
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL1: 'caco2',
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL1: 'caco2',

    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL2: 'caco2',
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL2: 'caco2',
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL2: 'caco2',

    # x8
    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL0: 'caco2',
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL0: 'caco2',
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL0: 'caco2',

    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL1: 'caco2',
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL1: 'caco2',
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL1: 'caco2',

    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL2: 'caco2',
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL2: 'caco2',
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL2: 'caco2',

    # BIOSR v1
    BIOSRV1_CCPS_TRAIN_X2: 'biosr',
    BIOSRV1_CCPS_VALID_X2: 'biosr',
    BIOSRV1_CCPS_TEST_X2: 'biosr',

    BIOSRV1_ER_TRAIN_X2: 'biosr',
    BIOSRV1_ER_VALID_X2: 'biosr',
    BIOSRV1_ER_TEST_X2: 'biosr',

    BIOSRV1_F_ACTIN_TRAIN_X2: 'biosr',
    BIOSRV1_F_ACTIN_VALID_X2: 'biosr',
    BIOSRV1_F_ACTIN_TEST_X2: 'biosr',

    BIOSRV1_MICROTUBULES_TRAIN_X2: 'biosr',
    BIOSRV1_MICROTUBULES_VALID_X2: 'biosr',
    BIOSRV1_MICROTUBULES_TEST_X2: 'biosr'
}

DS_N_CHANNELS = {
    # CACO2
    # x2
    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL0: 1,
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL0: 1,
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL0: 1,

    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL1: 1,
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL1: 1,
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL1: 1,

    CACO2_TRAIN_X2_IN_256_OUT_512_CELL_CELL2: 1,
    CACO2_VALID_X2_IN_256_OUT_512_CELL_CELL2: 1,
    CACO2_TEST_X2_IN_256_OUT_512_CELL_CELL2: 1,

    # x4
    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL0: 1,
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL0: 1,
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL0: 1,

    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL1: 1,
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL1: 1,
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL1: 1,

    CACO2_TRAIN_X4_IN_128_OUT_512_CELL_CELL2: 1,
    CACO2_VALID_X4_IN_128_OUT_512_CELL_CELL2: 1,
    CACO2_TEST_X4_IN_128_OUT_512_CELL_CELL2: 1,

    # x8
    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL0: 1,
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL0: 1,
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL0: 1,

    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL1: 1,
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL1: 1,
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL1: 1,

    CACO2_TRAIN_X8_IN_64_OUT_512_CELL_CELL2: 1,
    CACO2_VALID_X8_IN_64_OUT_512_CELL_CELL2: 1,
    CACO2_TEST_X8_IN_64_OUT_512_CELL_CELL2: 1,

    # biosr v1
    BIOSRV1_CCPS_TRAIN_X2: 1,
    BIOSRV1_CCPS_VALID_X2: 1,
    BIOSRV1_CCPS_TEST_X2: 1,

    BIOSRV1_ER_TRAIN_X2: 1,
    BIOSRV1_ER_VALID_X2: 1,
    BIOSRV1_ER_TEST_X2: 1,

    BIOSRV1_F_ACTIN_TRAIN_X2: 1,
    BIOSRV1_F_ACTIN_VALID_X2: 1,
    BIOSRV1_F_ACTIN_TEST_X2: 1,

    BIOSRV1_MICROTUBULES_TRAIN_X2: 1,
    BIOSRV1_MICROTUBULES_VALID_X2: 1,
    BIOSRV1_MICROTUBULES_TEST_X2: 1
}

PLAN_IMG = {

}
BUCKET_SZ = 8

NBR_CHUNKS_TR = {
    'ILSVRC': 30 * 8,  # 30 *8: ~5k per chunk
    'DEBUG_ILSVRC': 3 * 8,  # 3 *8: ~5k per chunk.
    # no chunking:
    'CUB': -1,
    'DEBUG_CUB': -1,
    'OpenImages': -1,
    'DEBUG_OpenImages': -1,
    'YouTube-Objects-v1.0': -1,
    'DEBUG_YouTube-Objects-v1.0': -1
}

RELATIVE_META_ROOT = 'folds'

NUMBER_CLASSES = {
    ILSVRC: 1000,
    CUB: 200,
    OpenImages: 100,
    YTOV1: 10,
    YTOV22: 10
}

CROP_SIZE = 224
RESIZE_SIZE = 256

DS_SHOTS = 'SHOTS'
DS_FRAMES = 'FRAMES'

DS_MODES = [DS_SHOTS, DS_FRAMES]

# ================= check points
BEST_CL = 'best_classification'
BEST_LOC = 'best_localization'

# ==============================================================================

# Colours
COLOR_WHITE = "white"
COLOR_BLACK = "black"

# backbones.

# =================================================
NCOLS = 80  # tqdm ncols.

# stages:
STGS_TR = "TRAIN"
STGS_EV = "EVAL"


# image range: [0, 1] --> Sigmoid. [-1, 1]: TANH
RANGE_TANH = "tanh"
RANGE_SIGMOID = 'sigmoid'

# ==============================================================================
# cams extractor
TRG_LAYERS = {
            RESNET50: 'encoder.layer4.2.relu3',
            VGG16: 'encoder.relu',
            INCEPTIONV3: 'encoder.SPG_A3_2b.2'
        }
FC_LAYERS = {
    RESNET50: 'classification_head.fc',
    VGG16: 'classification_head.fc',
    INCEPTIONV3: 'classification_head.fc'
}

# EXPs
OVERRUN = False

# cam_curve_interval: for bbox. use high interval for validation (not test).
# high number of threshold slows down the validation because of
# `cv2.findContours`. this gets when cams are bad leading to >1k contours per
# threshold. default evaluation: .001.
VALID_FAST_CAM_CURVE_INTERVAL = .004

# data: name of the folder where cams will be stored.
DATA_CAMS = 'data_cams'

FULL_BEST_EXPS = 'full_best_exps'
PERTURBATIONS_FD = 'perturbations_analysis'

# DDP
NCCL = 'nccl'# phases:
TRAIN_PHASE = 'train'
EVAL_PHASE = 'eval'
PHASES = [TRAIN_PHASE, EVAL_PHASE]
GLOO = 'gloo'
MPI = 'mpi'

# CC: communitation folder
SCRATCH_COMM = 'super-resolution/communication'
SCRATCH_FOLDER = 'super-resolution'

# metrics names
LOCALIZATION_MTR = 'localization'
CLASSIFICATION_MTR = 'classification'
FAILD_BOXES_MTR = 'failed boxes'


# norms
NORM1 = '1'
NORM2 = '2'
NORM0EXP = '0EXP'
KL = 'KL'
BH = 'BHATTACHARYYA'

NORMS = [NORM1, NORM2, NORM0EXP]
LPNORMS = [NORM1, NORM2]

# size estimation.
SIZE_DATA = 'size_data'
SIZE_CONST = 'size_constant'

# virtual env name: local.
_ENV_NAME = 'sr.micro'

# string id.
CODE_IDENTIFIER = 'CODEXXXXXXXIDENTIFIER'

# splines
SPLINE_NET_TYPE1 = 'snet_type1'  # 1 hidden layer
SPLINE_NET_TYPE2 = 'snet_type2'  # 2 hidden layers
SPLINE_NET_TYPE3 = 'snet_type3'  # 3 hidden layers
SPLINE_NET_TYPE4 = 'snet_type4'  # 4 hidden layers
SPLINE_NET_TYPE5 = 'snet_type5'  # 5 hidden layers
SPLINE_NET_TYPE6 = 'snet_type6'  # 6 hidden layers
SPLINE_NET_TYPE7 = 'snet_type7'  # 7 hidden layers
SPLINE_NET_TYPE8 = 'snet_type8'  # 8 hidden layers

SPLINE_NET_TYPES = [
    SPLINE_NET_TYPE1,
    SPLINE_NET_TYPE2,
    SPLINE_NET_TYPE3,
    SPLINE_NET_TYPE4,
    SPLINE_NET_TYPE5,
    SPLINE_NET_TYPE6,
    SPLINE_NET_TYPE7,
    SPLINE_NET_TYPE8
]

# splines hidden layers
SPLINEHIDDEN = {
    SPLINE_NET_TYPE1: [16],
    SPLINE_NET_TYPE2: [32, 16],
    SPLINE_NET_TYPE3: [32, 32, 16],
    SPLINE_NET_TYPE4: [32, 32, 32, 16],
    SPLINE_NET_TYPE5: [32, 32, 32, 32, 16],
    SPLINE_NET_TYPE6: [32, 32, 32, 32, 32, 16],
    SPLINE_NET_TYPE7: [32, 32, 32, 32, 32, 32, 16],
    SPLINE_NET_TYPE8: [32, 32, 32, 32, 32, 32, 32, 16]
}


# splines
NET_TYPE1 = 'snet_type1'  # 1 hidden layer
NET_TYPE2 = 'snet_type2'  # 2 hidden layers
NET_TYPE3 = 'snet_type3'  # 3 hidden layers
NET_TYPE4 = 'snet_type4'  # 4 hidden layers
NET_TYPE5 = 'snet_type5'  # 5 hidden layers
NET_TYPE6 = 'snet_type6'  # 6 hidden layers
NET_TYPE7 = 'snet_type7'  # 7 hidden layers
NET_TYPE8 = 'snet_type8'  # 8 hidden layers


NET_TYPES = [
    NET_TYPE1,
    NET_TYPE2,
    NET_TYPE3,
    NET_TYPE4,
    NET_TYPE5,
    NET_TYPE6,
    NET_TYPE7,
    NET_TYPE8
]

# hidden layers
NETS_CNN = {
    NET_TYPE1: [32],
    NET_TYPE2: [32, 32],
    NET_TYPE3: [256, 256, 256],
    NET_TYPE4: [32, 32, 32, 32],
    NET_TYPE5: [32, 32, 32, 32, 32],
    NET_TYPE6: [32, 32, 32, 32, 32, 32],
    NET_TYPE7: [32, 32, 32, 32, 32, 32, 32],
    NET_TYPE8: [32, 32, 32, 32, 32, 32, 32, 32]
}

NET_TYPE_UNET = 'unet'
NET_TYPE_PYRAMID = 'pyramid'

# activation
RELU = 'RELU'
TANH = 'TANH'
NONE_ACTIV = 'None'
ACTIVATIONS = [RELU, TANH, NONE_ACTIV]

# sampling tech. of training patches.
SAMPLE_UNIF = 'uniform'  # uniformly.
SAMPLE_ROI = 'roi'  # sample from binary ROI.
SAMPLE_EDT = 'edt'  # sample from Euclidean distance transform.
SAMPLE_EDTXROI = 'edt*roi'  # sample from EDT * ROI.

SAMPLE_PATCHES = [SAMPLE_UNIF, SAMPLE_ROI, SAMPLE_EDT, SAMPLE_EDTXROI]


TH_AUTO = 'automatic_threshold'
TH_FIX = 'fix_threshold'
ROI_STYLE_TH = [TH_AUTO, TH_FIX]

# variance type: global, or adaptive.
VAR_GLOBAL = 'var-global'
VAR_ADAPTIVE = 'var-adaptive'
VAR_TYPES = [VAR_GLOBAL, VAR_ADAPTIVE]

# task net:
REGRESSION = 'regression'
SEGMENTATION = 'segmentation'



# ROI thresholds
ROI_THRESH = [4, 5, 6, 7, 8, 9, 10]

# Optimizers:
SGD = 'sgd'
ADAM = 'adam'
OPTIMIZERS = [SGD, ADAM]

# LR step
MULTISTEPLR = 'MultiStepLR'  # pytorch multi-step lr.
MYSTEPLR = 'MyStepLR'  # single step with minimum learning rate.

STEPSLR = [MULTISTEPLR, MYSTEPLR]