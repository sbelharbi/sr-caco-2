import sys
from os.path import join, dirname, abspath
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat

# Credit: https://github.com/Francis0625/Omni-SR


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

__all__ = ['OmniSR']

# Paper: "Omni Aggregation Networks for Lightweight Image Super-Resolution",
# CVPR, 2023, Hang Wang, Xuanhong Chen, Bingbing Ni, Yutian Liu, Jinfan Liu.


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,
                       bias=False):
    """
    Upsample features according to `upscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(in_channels,
                     out_channels * (upscale_factor ** 2),
                     kernel_size,
                     padding=1,
                     bias=bias)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1),
                                      device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                  groups=hidden_dim),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        with_pe = True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible ' \
                                      'by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2,
                                             self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(
                grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])
                               ).sum(dim = -1)

            self.register_buffer('rel_pos_indices', rel_pos_indices,
                                 persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, \
        device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h),
                      (q, k, v)
                      )

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height,
                        w2=window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1,
                                    bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1,
                                     bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Channel_Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t,
                                          'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)',
                                          ph=self.ps, pw=self.ps,
                                          head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out,
                        'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)',
                        h=h // self.ps, w=w // self.ps, ph=self.ps, pw=self.ps,
                        head=self.heads)

        out = self.project_out(out)

        return out


class Channel_Attention_grid(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t,
                                          'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)',
                                          ph=self.ps, pw=self.ps,
                                          head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out,
                        'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)',
                        h=h // self.ps, w=w // self.ps, ph=self.ps, pw=self.ps,
                        head=self.heads)

        out = self.project_out(out)

        return out


class OSA_Block(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8,
                 with_pe=False, dropout=0.0):
        super(OSA_Block, self).__init__()

        w = window_size

        self.layer = nn.Sequential(
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25
            ),

            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),
            # block-like attention
            PreNormResidual(channel_num, Attention(dim=channel_num,
                                                   dim_head=channel_num // 4,
                                                   dropout=dropout,
                                                   window_size=window_size,
                                                   with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num,
                                 Gated_Conv_FeedForward(dim=channel_num,
                                                        dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention(dim=channel_num, heads=4,
                                                   dropout=dropout,
                                                   window_size=window_size)),
            Conv_PreNormResidual(channel_num,
                                 Gated_Conv_FeedForward(dim=channel_num,
                                                        dropout=dropout)),

            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
            # grid-like attention
            PreNormResidual(channel_num, Attention(dim=channel_num,
                                                   dim_head=channel_num // 4,
                                                   dropout=dropout,
                                                   window_size=window_size,
                                                   with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(channel_num,
                                 Gated_Conv_FeedForward(dim=channel_num,
                                                        dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention_grid(dim=channel_num,
                                                        heads=4,
                                                        dropout=dropout,
                                                        window_size=window_size)),
            Conv_PreNormResidual(channel_num,
                                 Gated_Conv_FeedForward(dim=channel_num,
                                                        dropout=dropout)),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class OSAG(nn.Module):
    def __init__(self,
                 channel_num=64,
                 bias=True,
                 block_num=4,
                 ffn_bias: bool = False,
                 window_size: int = 8,
                 pe: bool = False
                 ):
        super(OSAG, self).__init__()


        # script_name = "ops." + block_script_name
        # package = __import__(script_name, fromlist=True)
        # block_class = getattr(package, block_class_name)
        group_list = []
        for _ in range(block_num):
            temp_res = OSA_Block(channel_num, bias, ffn_bias=ffn_bias,
                                   window_size=window_size, with_pe=pe)
            group_list.append(temp_res)
        group_list.append(
            nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)


class OmniSR(nn.Module):
    def __init__(self,
                 input_shape: int = 3,
                 upscale: int = 2,
                 num_feat: int = 64,
                 res_num: int = 5,
                 bias: bool = True,
                 window_size: int = 8,
                 block_num: int = 4,
                 pe: bool = True,
                 ffn_bias: bool = True
                 ):
        super(OmniSR, self).__init__()

        output_shape = input_shape

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, bias=bias,
                            block_num=block_num, ffn_bias=ffn_bias,
                            window_size=window_size, pe=pe)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(in_channels=input_shape, out_channels=num_feat,
                               kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat,
                                kernel_size=3, stride=1, padding=1, bias=bias)
        self.up = pixelshuffle_block(num_feat, output_shape, upscale, bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,output_shape,upscale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size = window_size
        self.upscale = upscale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, :H * self.upscale, :W * self.upscale]
        return out


if __name__ == '__main__':

    device = torch.device('cuda:0')
    scale = 8
    c = 3
    x = torch.rand(1, c, 12, 12).to(device)

    model = OmniSR(input_shape=c,
                   upscale=scale,
                   num_feat=64,
                   res_num=5,
                   bias=True,
                   window_size=8,
                   block_num=4,
                   pe=True,
                   ffn_bias=True
                   ).to(device)
    with torch.no_grad():
        y = model(x)
    print(x.shape, y.shape)