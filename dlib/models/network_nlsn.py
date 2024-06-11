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

# Credit: https://github.com/HarukiYqM/Non-Local-Sparse-Attention
# Paper: https://openaccess.thecvf.com/content/CVPR2021/papers
# /Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['NLSN']

# Paper: "Image Super-Resolution With Non-Local Sparse Attention",
# CVPR, 2021, Yiqun Mei, Yuchen Fan, Yuqian Zhou.


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


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
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1,
            bias=True,
            bn=False, act=nn.PReLU()):

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
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
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


class NonLocalSparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4,
                 chunk_size=144, conv=default_conv, res_scale=1.):
        super(NonLocalSparseAttention, self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = BasicBlock(conv, channels,
                                     channels // reduction, k_size,
                                     bn=False, act=None)
        self.conv_assembly = BasicBlock(conv, channels, channels, 1,
                                        bn=False, act=None)

    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes,
                           hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype,
                                       device=device).expand(N, -1, -1,
                                                             -1)  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x,
                                    random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs],
                                 dim=-1)  # [N, n_hashes, H*W, hash_buckets]

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]

        # add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets,
                                   (N, -1,))  # [N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):

        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1,
                                              H * W).contiguous().permute(0, 2,
                                                                          1)
        y_embed = self.conv_assembly(input).view(N, -1,
                                                 H * W).contiguous().permute(0,
                                                                             2,
                                                                             1)
        L, C = x_embed.shape[-2:]

        # number of hash buckets/hash bits
        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2,
                           128)

        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, x_embed)  # [N,n_hashes*H*W]
        hash_codes = hash_codes.detach()

        # group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1)  # [N,n_hashes*H*W]
        _, undo_sort = indices.sort(
            dim=-1)  # undo_sort to recover original order
        mod_indices = (indices % L)  # now range from (0->H*W)
        x_embed_sorted = batched_index_select(x_embed,
                                              mod_indices)  # [N,n_hashes*H*W,C]
        y_embed_sorted = batched_index_select(y_embed,
                                              mod_indices)  # [N,n_hashes*H*W,C]

        # pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (
        N, self.n_hashes, -1, C))  # [N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (
        N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(x_att_buckets, (
        N, self.n_hashes, -1, self.chunk_size,
        C))  # [N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (
        N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

        # allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  # [N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        # attention
        ret = torch.einsum('bukij,bukje->bukie', score,
                           y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret,
                            (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (
        N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H,
                                        W).contiguous() * self.res_scale + input
        return ret


class NonLocalAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1,
                 softmax_scale=10, average=True, res_scale=1.,
                 conv=default_conv):
        super(NonLocalAttention, self).__init__()
        self.res_scale = res_scale
        self.conv_match1 = BasicBlock(conv, channel,
                                             channel // reduction, 1, bn=False,
                                             act=nn.PReLU())
        self.conv_match2 = BasicBlock(conv, channel,
                                             channel // reduction, 1, bn=False,
                                             act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1,
                                               bn=False, act=nn.PReLU())

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        return x_final.permute(
            0, 2, 1).view(N, -1, H, W) + self.res_scale * input

class NLSN(nn.Module):
    def __init__(self,
                 upscale: int = 2,
                 n_resblocks: int = 32,
                 n_feats: int = 256,
                 n_hashes: int = 4,
                 res_scale: float = 0.1,
                 img_range: float = 1.,
                 in_chans: int = 3,
                 chunk_size: int = 144,
                 conv=default_conv
                 ):
        super(NLSN, self).__init__()

        n_resblock = n_resblocks
        kernel_size = 3
        scale = upscale
        rgb_range = img_range
        n_colors = in_chans
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [NonLocalSparseAttention(
            channels=n_feats, chunk_size=chunk_size,
            n_hashes=n_hashes, reduction=4, res_scale=res_scale)]

        for i in range(n_resblock):
            m_body.append(ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ))
            if (i+1)%8==0:
                m_body.append(NonLocalSparseAttention(
                    channels=n_feats, chunk_size=chunk_size,
                    n_hashes=n_hashes, reduction=4,
                    res_scale=res_scale)
                )
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res = res + x

        x = self.tail(res)
        # x = self.add_mean(x)

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
    scale = 8
    c = 1
    b = 1
    x = torch.rand(b, c, 12, 12).to(device)

    model = NLSN(upscale=scale,
                 n_resblocks=32,
                 n_feats=256,
                 n_hashes=4,
                 res_scale=0.1,
                 img_range=1.,
                 in_chans=c,
                 chunk_size=144
                 ).to(device)

    with torch.no_grad():
        y = model(x)
    print(f'scale: x{scale}', x.shape, y.shape)