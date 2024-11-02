import sys
from os.path import dirname, abspath

import functools
import torch
from torch.nn import init

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import safe_str_var


__all__ = ['define_G']


def define_G(args):
    opt_net = args.netG
    net_type = opt_net['net_type']
    nt = safe_str_var(net_type)

    if net_type == constants.SWINIR:
        from dlib.models.network_swinir import SwinIR as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_chans=opt_net[f'{nt}_in_chans'],
                   img_size=opt_net[f'{nt}_img_size'],
                   window_size=opt_net[f'{nt}_window_size'],
                   img_range=opt_net[f'{nt}_img_range'],
                   depths=opt_net[f'{nt}_depths'],
                   embed_dim=opt_net[f'{nt}_embed_dim'],
                   num_heads=opt_net[f'{nt}_num_heads'],
                   mlp_ratio=opt_net[f'{nt}_mlp_ratio'],
                   upsampler=opt_net[f'{nt}_upsampler'],
                   resi_connection=opt_net[f'{nt}_resi_connection'])

    elif net_type == constants.EDSR_LIIF:
        from dlib.models.network_edsr_liif import EDSR_LIIF as net

        netG = net(in_chans=opt_net[f'{nt}_in_chans'],
                   n_resblocks=opt_net[f'{nt}_n_resblocks'],
                   n_feats=opt_net[f'{nt}_n_feats'],
                   scale=opt_net[f'{nt}_upscale'],
                   rgb_range=opt_net[f'{nt}_img_range'],
                   local_ensemble=True,
                   feat_unfold=True,
                   cell_decode=True
                   )

    elif net_type == constants.ACT:
        from dlib.models.network_act import ACT as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_chans=opt_net[f'{nt}_in_chans'],
                   img_range=opt_net[f'{nt}_img_range'],
                   n_feats=opt_net[f'{nt}_n_feats'],
                   n_resgroups=opt_net[f'{nt}_n_resgroups'],
                   n_resblocks=opt_net[f'{nt}_n_resblocks'],
                   reduction=opt_net[f'{nt}_reduction'],
                   n_heads=opt_net[f'{nt}_n_heads'],
                   n_layers=opt_net[f'{nt}_n_layers'],
                   dropout_rate=opt_net[f'{nt}_dropout_rate'],
                   n_fusionblocks=opt_net[f'{nt}_n_fusionblocks'],
                   token_size=opt_net[f'{nt}_token_size'],
                   expansion_ratio=opt_net[f'{nt}_expansion_ratio']
                   )

    elif net_type == constants.GRL:
        from dlib.models.network_grl import GRL as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_chans=opt_net[f'{nt}_in_chans'],
                   img_size=opt_net[f'{nt}_img_size'],
                   window_size=opt_net[f'{nt}_window_size'],
                   img_range=opt_net[f'{nt}_img_range'],
                   depths=opt_net[f'{nt}_depths'],
                   embed_dim=opt_net[f'{nt}_embed_dim'],
                   num_heads_window=opt_net[f'{nt}_num_heads_window'],
                   num_heads_stripe=opt_net[f'{nt}_num_heads_stripe'],
                   mlp_ratio=opt_net[f'{nt}_mlp_ratio'],
                   upsampler=opt_net[f'{nt}_upsampler'],
                   qkv_proj_type=opt_net[f'{nt}_qkv_proj_type'],
                   anchor_proj_type=opt_net[f'{nt}_anchor_proj_type'],
                   anchor_window_down_factor=opt_net[f'{nt}_anchor_window_down_factor'],
                   out_proj_type=opt_net[f'{nt}_out_proj_type'],
                   conv_type=opt_net[f'{nt}_conv_type'],
                   local_connection=opt_net[f'{nt}_local_connection']
                   )

    elif net_type == constants.ENLCN:
        from dlib.models.network_enlcn import ENLCN as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   n_resblock=opt_net[f'{nt}_n_resblock'],
                   n_feats=opt_net[f'{nt}_n_feats'],
                   res_scale=opt_net[f'{nt}_res_scale'],
                   img_range=opt_net[f'{nt}_img_range'],
                   in_chans=opt_net[f'{nt}_in_chans']
                   )

    elif net_type == constants.MSLAPSR:
        from dlib.models.network_mslapsr import MSLapSRN as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_chans=opt_net[f'{nt}_in_chans']
                   )

    elif net_type == constants.PROSR:
        from dlib.models.network_prosr import ProSR as net

        upscale = opt_net[f'{nt}_upscale']

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_chans=opt_net[f'{nt}_in_chans'],
                   residual_denseblock=opt_net[f'{nt}_residual_denseblock'],
                   num_init_features=opt_net[f'{nt}_num_init_features'],
                   bn_size=opt_net[f'{nt}_bn_size'],
                   growth_rate=opt_net[f'{nt}_growth_rate'],
                   ps_woReLU=opt_net[f'{nt}_ps_woReLU'],
                   level_config=opt_net[f'{nt}_level_config'][upscale],
                   level_compression=opt_net[f'{nt}_level_compression'],
                   res_factor=opt_net[f'{nt}_res_factor'],
                   max_num_feature=opt_net[f'{nt}_max_num_feature'],
                   block_compression=opt_net[f'{nt}_block_compression']
                   )

    elif net_type == constants.SRFBN:
        from dlib.models.network_srfbn import SRFBN as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_chans=opt_net[f'{nt}_in_chans'],
                   num_features=opt_net[f'{nt}_num_features'],
                   num_steps=opt_net[f'{nt}_num_steps'],
                   num_groups=opt_net[f'{nt}_num_groups']
                   )

    elif net_type == constants.DBPN:
        from dlib.models.network_dbpn import DBPN as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_chans=opt_net[f'{nt}_in_chans'],
                   base_filter=opt_net[f'{nt}_base_filter'],
                   feat=opt_net[f'{nt}_feat'],
                   num_stages=opt_net[f'{nt}_num_stages']
                   )

    elif net_type == constants.NLSN:
        from dlib.models.network_nlsn import NLSN as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   n_resblocks=opt_net[f'{nt}_n_resblocks'],
                   n_feats=opt_net[f'{nt}_n_feats'],
                   n_hashes=opt_net[f'{nt}_n_hashes'],
                   chunk_size=opt_net[f'{nt}_chunk_size'],
                   res_scale=opt_net[f'{nt}_res_scale'],
                   img_range=opt_net[f'{nt}_img_range'],
                   in_chans=opt_net[f'{nt}_in_chans']
                   )

    elif net_type == constants.DFCAN:
        from dlib.models.network_dfcan import DFCAN as net

        netG = net(input_shape=opt_net[f'{nt}_in_chans'],
                   upscale=opt_net[f'{nt}_upscale']
                   )

    elif net_type == constants.OMNISR:
        from dlib.models.network_omni_sr import OmniSR as net

        netG = net(input_shape=opt_net[f'{nt}_in_chans'],
                   upscale=opt_net[f'{nt}_upscale'],
                   num_feat=opt_net[f'{nt}_num_feat'],
                   res_num=opt_net[f'{nt}_res_num'],
                   bias=opt_net[f'{nt}_bias'],
                   window_size=opt_net[f'{nt}_window_size'],
                   block_num=opt_net[f'{nt}_block_num'],
                   pe=opt_net[f'{nt}_pe'],
                   ffn_bias=opt_net[f'{nt}_ffn_bias']
                   )

    elif net_type == constants.MEMNET:
        from dlib.models.network_memnet import MemNet as net

        netG = net(in_chans=opt_net[f'{nt}_in_chans'],
                   upscale=opt_net[f'{nt}_upscale'],
                   num_memory_blocks=opt_net[f'{nt}_num_memory_blocks'],
                   num_residual_blocks=opt_net[f'{nt}_num_residual_blocks']
                   )

    elif net_type == constants.DRRN:
        from dlib.models.network_drrn import DRRN as net

        netG = net(in_chans=opt_net[f'{nt}_in_chans'],
                   upscale=opt_net[f'{nt}_upscale'],
                   num_residual_units=opt_net[f'{nt}_num_residual_units']
                   )

    elif net_type == constants.VDSR:
        from dlib.models.network_vdsr import VDSR as net

        netG = net(in_chans=opt_net[f'{nt}_in_chans'],
                   upscale=opt_net[f'{nt}_upscale']
                   )

    elif net_type == constants.SRCNN:
        from dlib.models.network_srcnn import SRCNN as net

        netG = net(in_chans=opt_net[f'{nt}_in_chans'])

    elif net_type == constants.DSRSPLINES:
        from dlib.models.network_dsr_splines import DsrSplines as net

        netG = net(upscale=opt_net[f'{nt}_upscale'],
                   in_planes=opt_net[f'{nt}_in_planes'],
                   in_ksz=opt_net[f'{nt}_in_ksz'],
                   splinenet_type=opt_net[f'{nt}_splinenet_type'],
                   n_splines_per_color=opt_net[f'{nt}_n_splines_per_color'],
                   use_local_residual=opt_net[f'{nt}_use_local_residual'],
                   use_global_residual=opt_net[f'{nt}_use_global_residual'],
                   color_min=opt_net[f'{nt}_color_min'],
                   color_max=opt_net[f'{nt}_color_max'])

    elif net_type == constants.CSRCNN:

        if opt_net[f'{nt}_net_type']  == constants.NET_TYPE_UNET:

            # todo: add in parser that only this type supports segmentation.

            from dlib.models.network_unet2 import UNet as net

            channel_mults = opt_net[f'{nt}_channel_mults'].split('_')
            channel_mults = [int(z) for z in channel_mults]

            # netG = net(upscale=opt_net[f'{nt}_upscale'],
            #            in_channel=opt_net[f'{nt}_in_planes'],
            #            out_channel=opt_net[f'{nt}_in_planes'],
            #            outksz=opt_net[f'{nt}_outksz'],
            #            inner_channel=opt_net[f'{nt}_inner_channel'],
            #            norm_groups=opt_net[f'{nt}_norm_groups'],
            #            channel_mults=channel_mults,
            #            res_blocks=opt_net[f'{nt}_res_blocks'],
            #            dropout=opt_net[f'{nt}_dropout'],
            #            use_global_residual=opt_net[f'{nt}_use_global_residual'])

            netG = net(upscale=opt_net[f'{nt}_upscale'],
                       in_channel=opt_net[f'{nt}_in_planes'],
                       out_channel=opt_net[f'{nt}_in_planes'],
                       outksz=opt_net[f'{nt}_outksz'],
                       inner_channel=opt_net[f'{nt}_inner_channel'],
                       res_blocks=opt_net[f'{nt}_res_blocks'],
                       use_global_residual=opt_net[
                           f'{nt}_use_global_residual'],
                       task=opt_net[f'net_task'],
                       color_min=args.color_min,
                       color_max=args.color_max)

        elif opt_net[f'{nt}_net_type']  == constants.NET_TYPE_PYRAMID:

            from dlib.models.network_deconv import Pyramid as net

            netG = net(upscale=opt_net[f'{nt}_upscale'],
                       in_channel=opt_net[f'{nt}_in_planes'],
                       out_channel=opt_net[f'{nt}_in_planes'],
                       outksz=opt_net[f'{nt}_outksz'],
                       inner_channel=opt_net[f'{nt}_inner_channel'],
                       res_blocks=opt_net[f'{nt}_res_blocks'],
                       use_global_residual=opt_net[f'{nt}_use_global_residual'])

        else:
            from dlib.models.network_csr_cnn import ConstrainedSupResCnn as net

            netG = net(upscale=opt_net[f'{nt}_upscale'],
                       in_planes=opt_net[f'{nt}_in_planes'],
                       h_layers=opt_net[f'{nt}_net_type'],
                       in_ksz=opt_net[f'{nt}_in_ksz'],
                       ngroups=opt_net[f'{nt}_ngroups'],
                       use_local_residual=opt_net[f'{nt}_use_local_residual'],
                       use_global_residual=opt_net[f'{nt}_use_global_residual'])

    else:
        raise NotImplementedError(f'netG [{net_type}] is not found.')

    if args.is_train:
        init_weights(netG,
                     init_type=opt_net[f'{nt}_init_type'],
                     init_bn_type=opt_net[f'{nt}_init_bn_type'],
                     gain=opt_net[f'{nt}_init_gain'])

    return netG


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform',
                 gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            assert init_type in constants.INIT_W, init_type

            if init_type == constants.INIT_W_NORMAL:
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == constants.INIT_W_UNIFORM:
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == constants.INIT_W_XAVIER_NORMAL:
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == constants.INIT_W_UNIFORM:
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == constants.INIT_W_KAIMING_NORMAL:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in',
                                     nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == constants.INIT_W_KAIMING_UNIFORM:
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in',
                                      nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == constants.INIT_W_ORTHOGONAL:
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError(
                    f'Initialization method [{init_type}] is not implemented')

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:
            assert init_bn_type in constants.INIT_BN, init_bn_type

            if init_bn_type == constants.INIT_BN_UNIFORM:  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == constants.INIT_BN_CONSTANT:
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(f'Init. method [{init_bn_type}] is '
                                          f'not implemented')

    if init_type not in [constants.INIT_W_DEFAULT, 'none']:
        DLLogger.log(f'Initialization method [{init_type} + {init_bn_type}], '
                     f''f'gain is [{gain:.2f}]')
        fn = functools.partial(init_fn, init_type=init_type,
                               init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        DLLogger.log(
            'Pass this init.! Init. was done during network definition!')
