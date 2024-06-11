import sys
from os.path import join, dirname, abspath
import math
from abc import ABC
from math import prod
from typing import Tuple

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat

# Credit: https://github.com/Zj-BinXia/ENLCA
# Paper: https://arxiv.org/pdf/2201.03794.pdf


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['ENLCN']

# Paper: "Efficient Non-Local Contrastive Attention for Image Super-Resolution",
# AAAI, 2022, Bin Xia, Yucheng Hang, Yapeng Tian, Wenming Yang, Qingmin Liao,
# Jie Zhou.


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=True
                 ):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), stride=stride, bias=bias
                     )

def generalized_kernel(data,
                       *,
                       projection_matrix,
                       kernel_fn=nn.ReLU(),
                       kernel_epsilon=0.001,
                       normalize_data=True,
                       device=None
                       ):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data),
                             projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    # q, r = torch.qr(unstructured_block.cpu(), some=True)
    # changed in torch 2.0:
    # https://pytorch.org/docs/2.0/generated/torch.qr.html?highlight=torch+qr#torch.qr
    q, r = torch.linalg.qr(unstructured_block.cpu())
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows,
                                      nb_columns,
                                      scaling=0,
                                      device=None
                                      ):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device
                                 ).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,),
                                                                 device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
            sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size,
                 stride=1, bias=True, bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(in_channels=n_feats,
                              out_channels=4 * n_feats,
                              kernel_size=3,
                              bias=bias)
                         )
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))


        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def softmax_kernel(data,
                   *,
                   projection_matrix,
                   is_query,
                   normalize_data=False,
                   eps=1e-4,
                   device=None
                   ):
    b, h, *_ = data.shape

    # data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    #data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data),
    # projection)
    data_dash = torch.einsum('...id,...jd->...ij',  data, projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data ) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data ) + eps)

    return data_dash.type_as(data)


def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


class ENLA(nn.Module):
    def __init__(self,
                 dim_heads,
                 nb_features=None,
                 ortho_scaling=0,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(),
                 no_projection=False
                 ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         nb_rows=self.nb_features,
                                         nb_columns=dim_heads,
                                         scaling=ortho_scaling
                                         )
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient
        # attention paper
        self.no_projection = no_projection


    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        #q[b,h,n,d],b is batch ,h is multi head, n is number of batch, d is
        # feature
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel,
                                    kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix,
                                    device=device
                                    )
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel,
                                    projection_matrix=self.projection_matrix,
                                    device=device
                                    )
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        return out


class ENLCA(nn.Module):
    def __init__(self,
                 channel=128,
                 reduction=2,
                 ksize=3,
                 scale=3,
                 stride=1,
                 softmax_scale=10,
                 average=True,
                 conv=default_conv,
                 res_scale=0.1
                 ):
        super(ENLCA, self).__init__()
        self.conv_match1 = BasicBlock(conv, channel, channel // reduction, 1,
                                      bn=False, act=None)
        self.conv_match2 = BasicBlock(conv, channel, channel // reduction, 1,
                                      bn=False, act=None)
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False,
                                        act=None)
        self.res_scale = res_scale
        self.attn_fn = ENLA(dim_heads=channel // reduction,
                            nb_features=128,
                            )
        self.k = math.sqrt(6)

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input) #[B,C,H,W]
        x_embed_2 = F.normalize(x_embed_2, p=2, dim=1,eps=5e-5) * self.k
        x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5) * self.k
        N, C, H, W = x_embed_1.shape
        loss = 0
        if self.training:
            score = torch.matmul(x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C)),
                                 x_embed_2.view(N, C, H * W))  # [N,H*W,H*W]
            score = torch.exp(score)
            score = torch.sort(score, dim=2, descending=True)[0]
            positive = torch.mean(score[:, :, :15], dim=2)
            negative = torch.mean(score[:, :, 50:65], dim=2)  # [N,H*W]
            loss = F.relu(-1 * torch.log(positive / (negative + 1e-6))+1)
            loss = torch.mean(loss)

        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(N,1, H*W, -1 )
        x_final = self.attn_fn(x_embed_1, x_embed_2, x_assembly).squeeze(1)# (1, H*W, C)
        return x_final.permute(
            0, 2, 1).view(N, -1, H, W) * self.res_scale + input, loss


class ENLCN(nn.Module):
    def __init__(self,
                 upscale: int = 2,
                 n_resblock: int = 32,
                 n_feats: int = 256,
                 res_scale: float = 0.1,
                 img_range: float = 1.,
                 in_chans: int = 3,
                 conv=default_conv
                 ):
        super(ENLCN, self).__init__()

        # n_resblock = args.n_resblocks
        # n_feats = args.n_feats
        kernel_size = 3
        scale = upscale
        act = nn.ReLU(True)
        rgb_range = img_range
        n_colors = in_chans

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        m_head = [conv(n_colors, n_feats, kernel_size)]

        m_body = [ENLCA(channel=n_feats, reduction=4, res_scale=res_scale)]
        for i in range(n_resblock):
            m_body.append(ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ))
            if (i + 1) % 8 == 0:
                m_body.append(
                    ENLCA(channel=n_feats, reduction=4, res_scale=res_scale)
                              )
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, n_colors, kernel_size,
                      padding=(kernel_size // 2)
            )
        ]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = x
        comparative_loss = []
        for i in range(len(self.body)):
            if i % 9 == 0:
                res, loss = self.body[i](res)
                comparative_loss.append(loss)
            else:
                res = self.body[i](res)

        # drop loss: does not add much improvement.
        # https://arxiv.org/pdf/2201.03794.pdf
        comparative_loss = []

        res = res + x

        x = self.tail(res)
        # x = self.add_mean(x)

        # if self.training:
        #     return x, comparative_loss
        # else:
        #     return x

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))



if __name__ == '__main__':

    device = torch.device('cuda:0')
    scale = 2
    c = 1
    b = 32
    x = torch.rand(b, c, 64, 64).to(device)

    model = ENLCN(upscale=scale,
                  n_resblock=32,
                  n_feats=256,
                  res_scale=0.1,
                  img_range=1.,
                  in_chans=c
                  ).to(device)

    with torch.no_grad():
        y = model(x)
    print(f'scale: {scale}', x.shape, y.shape)