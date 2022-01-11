from dataclasses import dataclass
from enum import Enum

import torch
from choices import *
from config_base import BaseConfig
from torch import nn
from torch.nn import init

__all__ = ['NoiseNetConfig', 'NoiseNet', 'NoiseNetType']


class NoiseNetType(Enum):
    """
    choices for network mapping normal distribution into non-trivial distribution
    """
    identity = 'identity'
    vanilla = 'vanilla'


@dataclass
class NoiseNetConfig(BaseConfig):
    type: NoiseNetType
    num_channels: int = None
    num_hid_channels: int = None
    num_layers: int = None
    activation: Activation = None
    use_norm: bool = None
    dropout: float = None
    last_act: Activation = None

    @property
    def name(self):
        name = 'noisenet'
        if self.type == NoiseNetType.identity:
            name += '-identity'
        else:
            name += f'{self.type.value}-ch{self.num_channels}-hid{self.num_hid_channels}-{self.num_layers}layers-act{self.activation.value}'
            if self.use_norm:
                name += '-norm'
            if self.dropout > 0:
                name += f'-dropout{self.dropout}'
            if self.last_act != Activation.none:
                name += f'-lastact{self.last_act.value}'
        return name

    def make_model(self):
        return NoiseNet(self)


class NoiseNet(nn.Module):
    """
    used for latent diffusion process
    """
    def __init__(self, conf: NoiseNetConfig):
        super().__init__()
        self.conf = conf

        if conf.type == NoiseNetType.identity:
            self.layers = nn.Identity()
        else:
            self.layers = []
            for i in range(conf.num_layers):
                if i == 0:
                    act = conf.activation
                    norm = conf.use_norm
                    a, b = conf.num_channels, conf.num_hid_channels
                    dropout = conf.dropout
                elif i == conf.num_layers - 1:
                    act = Activation.none
                    norm = False
                    a, b = conf.num_hid_channels, conf.num_channels
                    dropout = 0
                else:
                    act = conf.activation
                    norm = conf.use_norm
                    a, b = conf.num_hid_channels, conf.num_hid_channels
                    dropout = conf.dropout

                self.layers.append(
                    MLPLNAct(
                        a,
                        b,
                        norm=norm,
                        activation=act,
                        dropout=dropout,
                    ))
            self.layers.append(conf.last_act.get_act())
            self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class MLPLNAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm: bool,
                 activation: Activation,
                 dropout: float = 0):
        super().__init__()
        self.activation = activation

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
