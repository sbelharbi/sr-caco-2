import sys
from os.path import join, dirname, abspath
from copy import deepcopy


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
from dlib.utils.shared import safe_str_var


def init_net_g(netG: dict, args: dict) -> dict:
    out = deepcopy(netG)
    nt = netG['net_type']
    nt = safe_str_var(nt)

    if netG['net_type'] == constants.SWINIR:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']
        out[f'{nt}_img_size'] = args['h_size'] // args['scale']

        out[f'{nt}_window_size'] = 8
        out[f'{nt}_img_range'] = 1.0
        out[f'{nt}_depths'] = [6, 6, 6, 6, 6, 6]
        out[f'{nt}_embed_dim'] = 180
        out[f'{nt}_num_heads'] = [6, 6, 6, 6, 6, 6]
        out[f'{nt}_mlp_ratio'] = 2
        out[f'{nt}_upsampler'] = constants.US_PIXEL_SHUFFLE
        out[f'{nt}_resi_connection'] = constants.R_CONNECTION_1CONV

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.ACT:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']


        out[f'{nt}_n_feats'] = 64
        out[f'{nt}_img_range'] = 1.0
        out[f'{nt}_n_resgroups'] = 4
        out[f'{nt}_n_resblocks'] = 12
        out[f'{nt}_reduction'] = 16
        out[f'{nt}_n_heads'] = 8
        out[f'{nt}_n_layers'] = 8
        out[f'{nt}_n_fusionblocks'] = 4
        out[f'{nt}_dropout_rate'] = 0.0
        out[f'{nt}_token_size'] = 3
        out[f'{nt}_expansion_ratio'] = 4

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.ENLCN:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_n_resblock'] = 32
        out[f'{nt}_n_feats'] = 256
        out[f'{nt}_res_scale'] = 0.1
        out[f'{nt}_img_range'] = 1.0

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.NLSN:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_n_resblocks'] = 32
        out[f'{nt}_n_feats'] = 256
        out[f'{nt}_n_hashes'] = 4
        out[f'{nt}_chunk_size'] = 144
        out[f'{nt}_res_scale'] = 0.1
        out[f'{nt}_img_range'] = 1.0

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.SRFBN:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_num_features'] = 64
        out[f'{nt}_num_steps'] = 4
        out[f'{nt}_num_groups'] = 6
        out[f'{nt}_use_cl'] = True  # use curriculum learning strategy.

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.MSLAPSR:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.PROSR:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_residual_denseblock'] = True
        out[f'{nt}_num_init_features'] = 160
        out[f'{nt}_bn_size'] = 4
        out[f'{nt}_growth_rate'] = 40
        out[f'{nt}_ps_woReLU'] = False
        out[f'{nt}_level_compression'] = -1
        out[f'{nt}_res_factor'] = 0.2
        out[f'{nt}_max_num_feature'] = 312
        out[f'{nt}_block_compression'] = 0.4

        # hard-coded: level_config
        out[f'{nt}_level_config'] = {
            2: [[8, 8, 8, 8, 8, 8, 8, 8, 8]],
            4: [[8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8]],
            8: [[8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8], [8]]
        }

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.DBPN:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_base_filter'] = 64
        out[f'{nt}_feat'] = 256
        out[f'{nt}_num_stages'] = 3

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.GRL:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']
        out[f'{nt}_img_size'] = args['h_size'] // args['scale']

        # Big.
        out[f'{nt}_window_size'] = 8
        out[f'{nt}_embed_dim'] = 180
        out[f'{nt}_mlp_ratio'] = 2
        out[f'{nt}_img_range'] = 1.0

        out[f'{nt}_depths'] = [4, 4, 8, 8, 8, 4, 4]
        out[f'{nt}_num_heads_window'] = [3, 3, 3, 3, 3, 3, 3]
        out[f'{nt}_num_heads_stripe'] = [3, 3, 3, 3, 3, 3, 3]

        out[f'{nt}_upsampler'] = constants.US_PIXEL_SHUFFLE
        out[f'{nt}_conv_type'] = '1conv'
        out[f'{nt}_out_proj_type'] = 'linear'
        out[f'{nt}_anchor_window_down_factor'] = 2
        out[f'{nt}_qkv_proj_type'] = 'linear'
        out[f'{nt}_anchor_proj_type'] = 'avgpool'
        out[f'{nt}_local_connection'] = True

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.DFCAN:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.OMNISR:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_num_feat'] = 64
        out[f'{nt}_res_num'] = 5
        out[f'{nt}_bias'] = True
        out[f'{nt}_window_size'] = 8
        out[f'{nt}_block_num'] = 4
        out[f'{nt}_pe'] = True
        out[f'{nt}_ffn_bias'] = True

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.MEMNET:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']
        out[f'{nt}_num_memory_blocks'] = 6
        out[f'{nt}_num_residual_blocks'] = 6

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.DRRN:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']
        out[f'{nt}_num_residual_units'] = 25

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.VDSR:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.SRCNN:

        out[f'{nt}_in_chans'] = args['n_channels']

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.DSRSPLINES:


        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_planes'] = args['n_channels']
        out[f'{nt}_color_min'] = args['color_min']
        out[f'{nt}_color_max'] = args['color_max']

        out[f'{nt}_in_ksz'] = 3
        out[f'{nt}_splinenet_type'] = constants.SPLINE_NET_TYPE1
        out[f'{nt}_n_splines_per_color'] = 16
        out[f'{nt}_use_local_residual'] = False
        out[f'{nt}_use_global_residual'] = False

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.

    elif netG['net_type'] == constants.CSRCNN:

        out[f'{nt}_upscale'] = args['scale']
        out[f'{nt}_in_planes'] = args['n_channels']

        # only small cnn
        out[f'{nt}_in_ksz'] = 3
        out[f'{nt}_ngroups'] = 16
        out[f'{nt}_use_local_residual'] = False

        # only unet type.
        out[f'{nt}_norm_groups'] = 16
        out[f'{nt}_channel_mults'] = '1_2_4_8_16_32_32_32'
        out[f'{nt}_dropout'] = 0.0
        out[f'{nt}_outksz'] = 3

        out[f'{nt}_inner_channel'] = 32
        out[f'{nt}_res_blocks'] = 3


        # small cnn or unet.
        out[f'{nt}_net_type'] = constants.NET_TYPE_UNET
        out[f'{nt}_use_global_residual'] = True

        out[f'{nt}_init_type'] = constants.INIT_W_DEFAULT
        out[f'{nt}_init_bn_type'] = constants.INIT_BN_CONSTANT
        out[f'{nt}_init_gain'] = 1.


    return out
