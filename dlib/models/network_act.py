import sys
from os.path import join, dirname, abspath
import math

import torch
from torch import nn
import torch.nn.functional as F


from torch import einsum
from einops import rearrange

# Credit: https://github.com/ofsoundof/GRL-Image-Restoration
# Paper: https://arxiv.org/pdf/2203.07682.pdf


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['ACT']

# Paper: "Enriched CNN-Transformer Feature Aggregation Networks for
# Super-Resolution ", WACV, 2023, Jinsu Yoo, Taehoon Kim, Sihaeng Lee,
# Seung Hwan Kim, Honglak Lee, Tae Hyun Kim.


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2),
        bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(self,
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self,
                 conv,
                 n_feats,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

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
    def __init__(self,
                 conv,
                 scale,
                 n_feats,
                 bn=False,
                 act=False,
                 bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x_q, x_kv):
        _, _, dim, heads = *x_q.shape, self.heads
        _, _, dim_large = x_kv.shape

        assert dim == dim_large

        q = self.to_q(x_q)

        q = rearrange(q, 'b n (h d) -> b h n d', h=heads)

        kv = self.to_kv(x_kv).chunk(2, dim=-1)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), kv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self,
                 conv,
                 n_feat,
                 kernel_size,
                 reduction,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv,
                 n_feat,
                 kernel_size,
                 reduction,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1)
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False,
                 act=nn.ReLU(True)):
        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ACT(nn.Module):
    def __init__(self,
                 upscale: int = 2,
                 in_chans: int = 3,
                 img_range: float = 1.0,
                 n_feats: int = 64,
                 n_resgroups: int = 4,
                 n_resblocks: int = 12,
                 reduction: int = 16,
                 n_heads: int = 8,
                 n_layers: int = 8,
                 dropout_rate: float = 0.0,
                 n_fusionblocks: int = 4,
                 token_size: int = 3,
                 expansion_ratio: int = 4
                 ):

        super(ACT, self).__init__()

        conv = default_conv

        task = 'sr'  # super resolution.
        scale = upscale
        rgb_range = img_range
        n_colors = in_chans
        self.n_feats = n_feats

        self.token_size = token_size
        self.n_fusionblocks = n_fusionblocks
        embedding_dim = n_feats * (token_size ** 2)
        self.embedding_dim = embedding_dim

        flatten_dim = embedding_dim
        hidden_dim = embedding_dim * expansion_ratio
        dim_head = embedding_dim // n_heads

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # head module includes two residual blocks
        self.head = nn.Sequential(
            conv(n_colors, n_feats, 3),
            ResBlock(conv, n_feats, 5, act=nn.ReLU(True)),
            ResBlock(conv, n_feats, 5, act=nn.ReLU(True)),
        )

        # linear encoding after tokenization
        self.linear_encoding = nn.Linear(flatten_dim, embedding_dim)

        # conventional self-attention block inside Transformer Block
        self.mhsa_block = nn.ModuleList([
            nn.ModuleList([
                PreNorm(
                    embedding_dim,
                    SelfAttention(embedding_dim, n_heads, dim_head,
                                  dropout_rate)
                ),
                PreNorm(
                    embedding_dim,
                    FeedForward(embedding_dim, hidden_dim, dropout_rate)
                ),
            ]) for _ in range(n_layers // 2)
        ])

        # cross-scale token attention block inside Transformer Block
        self.csta_block = nn.ModuleList([
            nn.ModuleList([
                # FFN for large tokens before the cross-attention
                nn.Sequential(
                    nn.LayerNorm(embedding_dim * 2),
                    nn.Linear(embedding_dim * 2, embedding_dim // 2),
                    nn.GELU(),
                    nn.Linear(embedding_dim // 2, embedding_dim // 2)
                ),
                # Two cross-attentions
                PreNorm2(
                    embedding_dim // 2,
                    CrossAttention(embedding_dim // 2, n_heads // 2, dim_head,
                                   dropout_rate)
                ),
                PreNorm2(
                    embedding_dim // 2,
                    CrossAttention(embedding_dim // 2, n_heads // 2, dim_head,
                                   dropout_rate)
                ),
                # FFN for large tokens after the cross-attention
                nn.Sequential(
                    nn.LayerNorm(embedding_dim // 2),
                    nn.Linear(embedding_dim // 2, embedding_dim // 2),
                    nn.GELU(),
                    nn.Linear(embedding_dim // 2, embedding_dim * 2)
                ),
                # conventional FFN after the attention
                PreNorm(
                    embedding_dim,
                    FeedForward(embedding_dim, hidden_dim, dropout_rate)
                )
            ]) for _ in range(n_layers // 2)
        ])

        # CNN Branch borrowed from RCAN
        modules_body = [
            ResidualGroup(conv, n_feats, 3, reduction, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, 3))
        self.cnn_branch = nn.Sequential(*modules_body)

        # Fusion Blocks
        self.fusion_block = nn.ModuleList([
            nn.Sequential(
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
                FB(conv, n_feats * 2, 1, act=nn.ReLU(True)),
            ) for _ in range(n_fusionblocks)
        ])

        self.fusion_mlp = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            ) for _ in range(n_fusionblocks - 1)
        ])

        self.fusion_cnn = nn.ModuleList([
            nn.Sequential(
                conv(n_feats, n_feats, 3), nn.ReLU(True), conv(
                    n_feats, n_feats, 3)
            ) for _ in range(n_fusionblocks - 1)
        ])

        # single convolution to lessen dimension after body module
        self.conv_last = conv(n_feats * 2, n_feats, 3)

        # tail module
        if task == 'sr':
            self.tail = nn.Sequential(
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, n_colors, 3),
            )
        elif task == 'car':
            self.tail = conv(n_feats, n_colors, 3)

    def forward(self, x):
        h, w = x.shape[-2:]

        # x = self.sub_mean(x)
        x = self.head(x)

        identity = x

        x_tkn = F.unfold(x, self.token_size, stride=self.token_size)
        x_tkn = rearrange(x_tkn, 'b d t -> b t d')

        x_tkn = self.linear_encoding(x_tkn) + x_tkn

        for i in range(self.n_fusionblocks):
            x_tkn = self.mhsa_block[i][0](x_tkn) + x_tkn
            x_tkn = self.mhsa_block[i][1](x_tkn) + x_tkn

            x_tkn_a, x_tkn_b = torch.split(x_tkn, self.embedding_dim // 2, -1)

            x_tkn_b = rearrange(x_tkn_b, 'b t d -> b d t')
            x_tkn_b = F.fold(x_tkn_b, (h, w), self.token_size,
                             stride=self.token_size)

            x_tkn_b = F.unfold(x_tkn_b, self.token_size * 2,
                               stride=self.token_size)
            x_tkn_b = rearrange(x_tkn_b, 'b d t -> b t d')

            x_tkn_b = self.csta_block[i][0](x_tkn_b)
            _x_tkn_a, _x_tkn_b = x_tkn_a, x_tkn_b
            x_tkn_a = self.csta_block[i][1](x_tkn_a, _x_tkn_b) + x_tkn_a
            x_tkn_b = self.csta_block[i][2](x_tkn_b, _x_tkn_a) + x_tkn_b
            x_tkn_b = self.csta_block[i][3](x_tkn_b)

            x_tkn_b = rearrange(x_tkn_b, 'b t d -> b d t')
            x_tkn_b = F.fold(x_tkn_b, (h, w), self.token_size * 2,
                             stride=self.token_size)

            x_tkn_b = F.unfold(x_tkn_b, self.token_size, stride=self.token_size)
            x_tkn_b = rearrange(x_tkn_b, 'b d t -> b t d')

            x_tkn = torch.cat((x_tkn_a, x_tkn_b), -1)
            x_tkn = self.csta_block[i][4](x_tkn) + x_tkn

            x = self.cnn_branch[i](x)

            x_tkn_res, x_res = x_tkn, x

            x_tkn = rearrange(x_tkn, 'b t d -> b d t')
            x_tkn = F.fold(x_tkn, (h, w), self.token_size,
                           stride=self.token_size)

            f = torch.cat((x, x_tkn), 1)
            f = f + self.fusion_block[i](f)

            if i != (self.n_fusionblocks - 1):
                x_tkn, x = torch.split(f, self.n_feats, 1)

                x_tkn = F.unfold(x_tkn, self.token_size, stride=self.token_size)
                x_tkn = rearrange(x_tkn, 'b d t -> b t d')
                x_tkn = self.fusion_mlp[i](x_tkn)+ x_tkn_res

                x = self.fusion_cnn[i](x) + x_res

        x = self.conv_last(f)

        x = x + identity

        x = self.tail(x)
        # x = self.add_mean(x)

        return x


if __name__ == '__main__':

    device = torch.device('cuda:0')
    scale = 8
    c = 3
    b = 1
    x = torch.rand(b, c, 8, 8).to(device)

    # Large, 20.13 M
    model = ACT(upscale=scale,
                in_chans=c,
                img_range=1.,
                n_feats=64,
                n_resgroups=4,
                n_resblocks=12,
                reduction=16,
                n_heads=8,
                n_layers=8,
                dropout_rate=0.0,
                n_fusionblocks=4,
                token_size=3,
                expansion_ratio=4
                ).to(device)

    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)
