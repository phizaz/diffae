import torch
import torch.nn.functional as F
from torch import nn


class OldVectorizer(nn.Module):
    """z => w"""
    def __init__(self, in_dim, hid_dim, num_layer, lr_mul=0.1):
        super().__init__()
        layers = []
        for i in range(num_layer):
            a = in_dim if i == 0 else hid_dim
            # forgot to put the acitvations!!!
            layers.append(EqualLinear(a, hid_dim, lr_mul))
        self.layers = nn.Sequential(*layers)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1, p=2)
        return self.net(x)


class OldVectorizer2(nn.Module):
    """z => w"""
    def __init__(self, in_dim, hid_dim, num_layer, lr_mul=0.1):
        super().__init__()
        layers = []
        for i in range(num_layer):
            a = in_dim if i == 0 else hid_dim
            layers.append(EqualLinear(a, hid_dim, lr_mul))
            layers.append(leaky_relu())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1, p=2)
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
