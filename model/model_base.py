import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from .model import *


class StyleVectorizer(nn.Module):
    """z => w"""
    def __init__(self, emb, depth, lr_mul=0.1, normalize_p: int = 2):
        super().__init__()
        self.normalize_p = normalize_p

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1, p=self.normalize_p)
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, emb, depth, last_act: bool):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(emb, emb))
            if i < depth - 1 or last_act:
                layers.append(leaky_relu())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class EqualLinear(nn.Module):
    """a linear layer."""
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input,
                        self.weight * self.lr_mul,
                        bias=self.bias * self.lr_mul)


class ResBlockMod(nn.Module):
    """
    modulated resblock
    """
    def __init__(self,
                 in_ch,
                 out_ch,
                 style_ch,
                 tdim=None,
                 dropout=None,
                 attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )

        if tdim is not None:
            self.temb_proj = nn.Sequential(
                Swish(),
                nn.Linear(tdim, out_ch),
            )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
        )
        self.block2style = nn.Linear(style_ch, out_ch)
        self.block2conv = EqualizedConv2DMod(out_ch,
                                             out_ch,
                                             3,
                                             stride=1,
                                             padding=1)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
        init.xavier_uniform_(self.block1[-1].weight, gain=1e-5)
        init.xavier_uniform_(self.block2conv.weight, gain=1e-5)

    def forward(self, x, temb=None, style=None):
        """
        Args:
            x: (n, c, h, w)
            style: (n, c)
        """
        h = self.block1(x)

        if temb is not None:
            h += self.temb_proj(temb)[:, :, None, None]

        h = self.block2(h)
        style2 = self.block2style(style)
        h = self.block2conv.forward(h, style2)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


##############################################################


class EqualizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        weight_mode: str = 'default',
    ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)

        if weight_mode == 'equalized':
            fan_in = init._calculate_correct_fan(self.weight, 'fan_in')
            gain = init.calculate_gain('leaky_relu', param=0)
            init.normal_(self.weight)
            if bias:
                init.zeros_(self.bias)
            self.scale = gain / math.sqrt(fan_in)
        elif weight_mode == 'kaiming':
            init.kaiming_normal_(self.weight,
                                 a=0,
                                 mode='fan_in',
                                 nonlinearity='leaky_relu')
            if bias:
                init.zeros_(self.bias)
            self.scale = 1
        elif weight_mode == 'default':
            self.scale = 1
        else:
            raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight * self.scale, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class EqualizedConv2DMod(EqualizedConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        weight_mode: str = 'default',
        demod=True,
        eps=1e-8,
    ):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=False,
                         weight_mode=weight_mode)
        self.eps = eps
        self.demod = demod
        self.out_channels = out_channels

    def forward(self, x, cond):
        """
        Args:
            x: (n, c, h, w)
            cond: (n, c)
        """
        b, c, h, w = x.shape
        # print('x:', x.shape, self.in_channels)
        assert c == self.in_channels, f'{c} != {self.in_channels}'
        assert len(x) == len(cond)

        # (n, 1, c, 1, 1)
        w1 = cond[:, None, :, None, None]
        # multiplied by scale at runtime
        # (1, c_out, c_in, kh, kw)
        w2 = self.weight[None, :, :, :, :] * self.scale
        # (n, c_out, c_in, kh, kw)
        weights = w2 * (w1 + 1)

        if self.demod:
            # this takes a lot of memory
            d = torch.rsqrt((weights**2).sum(dim=(2, 3, 4), keepdim=True) +
                            self.eps)
            weights = weights * d
        # (1, n * c, h, w)
        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        # (n * c_out, c_in, kh, kw)
        weights = weights.reshape(b * self.out_channels, *ws)
        # print('x:', x.shape)
        # print('cond:', cond.shape)
        # print('weights:', weights.shape)
        x = F.conv2d(x,
                     weights,
                     padding=self.padding,
                     groups=b,
                     stride=self.stride,
                     dilation=self.dilation)
        _, _, h, w = x.shape
        x = x.reshape(-1, self.out_channels, h, w)
        return x


class EqualizedLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 weight_mode: str = 'default',
                 extra_scale: float = 1.) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight_mode = weight_mode
        self.extra_scale = extra_scale
        self.init_weights()

    def init_weights(self):
        if self.weight_mode == 'equalized':
            fan_in = init._calculate_correct_fan(self.weight, 'fan_in')
            gain = init.calculate_gain('leaky_relu', param=0)
            init.normal_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
            self.scale = gain / math.sqrt(fan_in)
        elif self.weight_mode == 'kaiming':
            init.kaiming_normal_(self.weight,
                                 a=0,
                                 mode='fan_in',
                                 nonlinearity='leaky_relu')
            if self.bias is not None:
                init.zeros_(self.bias)
            self.scale = 1
        elif self.weight_mode == 'default':
            self.scale = 1
            self.reset_parameters()
        elif self.weight_mode == 'zero':
            init.zeros_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
            self.scale = 1
        else:
            raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(
            input, self.weight * self.scale * self.extra_scale,
            self.bias * self.extra_scale if self.bias is not None else None)
