import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch import Tensor
import math


class StyleConvSequential(nn.Sequential):
    def forward(self, x, cond=None):
        for layer in self:
            if isinstance(layer, EqualizedConv2d):
                x = layer(x, cond=cond)
            else:
                x = layer(x)
        return x


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

    def forward(self, x, cond, **kwargs):
        """
        Args:
            x: (n, c, h, w)
            cond: (n, c)
        """
        assert cond.dim() == 2
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
            # this needs to be performed in fp32 to keep the precision ?
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
