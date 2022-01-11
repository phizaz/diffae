import torch
from torch import nn
import torch.nn.functional as F
import math


class BeatGANsStyleVectorizer(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layer,
        lr_mul=0.01,
        pixel_norm: bool = False,
        # this makes the first layer undoing the normalization effect, allowing later layers to be kaiming normal
        no_fan_in_first_layer: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.pixel_norm = pixel_norm

        layers = []
        for i in range(num_layer):
            a = in_dim if i == 0 else hid_dim
            fan_in = 1 if i == 0 and no_fan_in_first_layer else None
            layers.append(
                EqualLinear(a, hid_dim, lr_mul, activation=True,
                            fan_in=fan_in))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.pixel_norm:
            x = PixelNorm().forward(x)
        else:
            x = F.normalize(x, dim=1, p=2)
        return self.layers(x)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # NOTE: it is "mean" not "sum" (as in F.normalize)
        return input * torch.rsqrt(
            torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """
    from: https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        lr_mul=1,
        activation=False,
        fan_in=None,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        # allow for not normalizing with fan-in
        fan_in = fan_in or in_dim

        self.scale = (1 / math.sqrt(fan_in)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input,
                       self.weight * self.scale,
                       bias=self.bias * self.lr_mul)
        if self.activation:
            out = F.leaky_relu(out, negative_slope=0.2)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )