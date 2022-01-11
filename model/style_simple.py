import math

import torch
import torch.nn.functional as F
from choices import *
from torch import nn
from torch.nn import init


class SimpleVectorizer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_layers,
                 use_pixel_norm: bool,
                 activation: Activation,
                 initialization: str = 'default'):
        super().__init__()
        self.use_pixel_norm = use_pixel_norm

        self.layers = []
        for i in range(num_layers):
            if i == 0:
                a, b = in_dim, out_dim
            else:
                a, b = out_dim, out_dim
            self.layers.append(nn.Linear(a, b))
            self.layers.append(activation.get_act())
        self.layers = nn.Sequential(*self.layers)

        if initialization == 'default':
            pass
        elif initialization == 'kaiming':
            for each in self.modules():
                if isinstance(each, nn.Linear):
                    init.kaiming_uniform_(each.weight,
                                          a=0,
                                          nonlinearity='relu')
        else:
            raise NotImplementedError()

    def forward(self, x):
        if self.use_pixel_norm:
            x = PixelNorm().forward(x)
        return self.layers(x)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # NOTE: it is "mean" not "sum" (as in F.normalize)
        return input * torch.rsqrt(
            torch.mean(input**2, dim=1, keepdim=True) + 1e-8)
