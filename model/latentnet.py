import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import torch
from choices import *
from config_base import BaseConfig
from torch import nn
from torch.nn import init

from .blocks import *
from .nn import timestep_embedding
from .unet import *


class LatentNetType(Enum):
    none = 'none'
    # mlpnet
    vanilla = 'vanilla'
    # mlpresnet
    resnet = 'resnet'
    # concat with time at the first layer
    concat = 'concat'
    # injecting inputs into the hidden layers
    skip = 'skip'
    # for 2d latent
    conv = 'conv'
    # project a vector into a spatial form and apply a conv
    projected_conv = 'projconv'
    projected_unet = 'projunet'
    projected_inv_unet = 'projinvunet'
    projected_half_unet = 'projhalfunet'
    mlpmixer = 'mlpmixer'
    prenormskip = 'prenormskip'


class LatentNetReturn(NamedTuple):
    pred: torch.Tensor = None


@dataclass
class MLPSkipNetConfig(BaseConfig):
    """
    default MLP for the latent DPM in the paper!
    """
    num_channels: int
    skip_layers: Tuple[int]
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int = 64
    activation: Activation = Activation.silu
    use_norm: bool = True
    condition_2x: bool = False
    condition_bias: float = 1
    dropout: float = 0
    last_act: Activation = Activation.none
    num_time_layers: int = 2
    time_layer_init: bool = False
    time_is_int: bool = True
    residual: bool = False
    time_last_act: bool = False

    @property
    def name(self):
        name = f'mlp-ch{self.num_channels}-hid{self.num_hid_channels}-{self.num_layers}layers'
        name += '-skip(' + ','.join(str(x) for x in self.skip_layers) + ')'
        name += f'-act{self.activation.value}'
        if self.use_norm:
            name += '-norm'
        name += f'-emb{self.num_time_emb_channels}normmod'
        if self.num_time_layers != 2:
            name += f'-timel{self.num_time_layers}'
        if self.time_layer_init:
            name += '-timinit'
        if self.condition_2x:
            name += '2x'
        if self.condition_bias > 0:
            name += f'-bias{self.condition_bias}'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        if self.last_act != Activation.none:
            name += f'-lastact{self.last_act.value}'
        if self.residual:
            name += '-res'
        if self.time_last_act:
            name += '-tlastact'
        return name

    def make_model(self):
        return MLPSkipNet(self)


class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, conf: MLPSkipNetConfig):
        super().__init__()
        self.conf = conf

        # self.time_embed = nn.Sequential(
        #     nn.Linear(conf.num_time_emb_channels, conf.num_channels),
        #     conf.activation.get_act(),
        #     nn.Linear(conf.num_channels, conf.num_channels),
        # )
        layers = []
        for i in range(conf.num_time_layers):
            if i == 0:
                a = conf.num_time_emb_channels
                b = conf.num_channels
            else:
                a = conf.num_channels
                b = conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(conf.activation.get_act())
        self.time_embed = nn.Sequential(*layers)

        if conf.time_layer_init:
            for each in self.time_embed.modules():
                if isinstance(each, nn.Linear):
                    init.kaiming_normal_(each.weight)

        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                cond = conf.condition_type
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
                residual = False
            elif i == conf.num_layers - 1:
                act = Activation.none
                norm = False
                cond = ConditionType.no
                a, b = conf.num_hid_channels, conf.num_channels
                dropout = 0
                residual = False
            else:
                act = conf.activation
                norm = conf.use_norm
                cond = conf.condition_type
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout
                residual = conf.residual

            if i in conf.skip_layers:
                a += conf.num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=conf.num_channels,
                    condition_type=cond,
                    condition_2x=conf.condition_2x,
                    condition_bias=conf.condition_bias,
                    dropout=dropout,
                    residual=residual,
                ))
        self.last_act = conf.last_act.get_act()

    def forward(self, x, t, **kwargs):
        if self.conf.time_is_int:
            t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            res = h
            if i in self.conf.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond, res=res)
        h = self.last_act(h)
        return LatentNetReturn(h)


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation: Activation,
        cond_channels: int,
        condition_2x: bool,
        condition_bias: float = 0,
        dropout: float = 0,
        residual: bool = False,
    ):
        super().__init__()
        self.condition_2x = condition_2x
        self.activation = activation
        self.condition_bias = condition_bias
        self.residual = residual

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        self.linear_emb = nn.Linear(
            cond_channels,
            (out_channels * 2 if condition_2x else out_channels))
        self.cond_layers = nn.Sequential(self.act, self.linear_emb)
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
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None, res=None):
        x = self.linear(x)
        if self.condition_type != ConditionType.no:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            if self.condition_2x:
                cond = torch.chunk(cond, 2, dim=1)
            else:
                cond = (cond, None)

            if self.condition_type == ConditionType.add:
                x = x + cond[0]
                x = self.norm(x)
            elif self.condition_type == ConditionType.scale_shift_norm:
                # scale shift first
                x = x * (self.condition_bias + cond[0])
                if cond[1] is not None:
                    x = x + cond[1]
                # then norm
                x = self.norm(x)
            elif self.condition_type == ConditionType.norm_scale_shift:
                # norm first
                x = self.norm(x)
                # scale shift first
                x = x * (self.condition_bias + cond[0])
                if cond[1] is not None:
                    x = x + cond[1]
            else:
                raise NotImplementedError()
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.residual:
            x = (x + res) / math.sqrt(2)
        return x
