from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import init

from .model import *
from .model_base import *
from config_base import *


@dataclass
class StyleEncoder2Config(BaseConfig):
    img_size: int
    style_ch: int
    ch: int
    ch_mult: int
    attn: Tuple[int]
    num_res_blocks: int
    dropout: float
    tail_depth: int
    pooling: str
    k: int

    @property
    def name(self):
        name = f'enc2-ch{self.ch}-mult('
        name += ','.join(str(x) for x in self.ch_mult) + ')'
        name += '-attn('
        name += ','.join(str(x) for x in self.attn) + ')'
        name += f'-blk{self.num_res_blocks}-dropout{self.dropout}'
        name += f'-tail{self.tail_depth}'
        name += f'-pool{self.pooling}'
        return name

    def make_model(self):
        return StyleEncoder2(self)


class StyleEncoder2(nn.Module):
    def __init__(self, conf: StyleEncoder2Config):
        super().__init__()
        self.conf = conf
        assert all([i < len(conf.ch_mult)
                    for i in conf.attn]), 'attn index out of bound'

        self.head = nn.Conv2d(3, conf.ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [conf.ch]  # record output channel when dowmsample for upsample

        now_ch = conf.ch
        resolution = conf.img_size
        for i, mult in enumerate(conf.ch_mult):
            out_ch = conf.ch * mult
            for _ in range(conf.num_res_blocks):
                self.downblocks.append(
                    ResBlock(in_ch=now_ch,
                             out_ch=out_ch,
                             tdim=None,
                             dropout=conf.dropout,
                             attn=(resolution in conf.attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(conf.ch_mult) - 1:
                resolution //= 2
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        if conf.pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif conf.pooling == 'conv':
            self.pool = nn.Sequential(
                nn.Conv2d(now_ch, conf.style_ch, kernel_size=4),
                leaky_relu(),
            )
        elif conf.pooling == 'depthconv':
            self.pool = nn.Sequential(
                nn.Conv2d(now_ch,
                          conf.style_ch * conf.k,
                          kernel_size=4,
                          groups=now_ch),
                nn.Conv2d(conf.style_ch * conf.k, conf.style_ch,
                          kernel_size=1),
                leaky_relu(),
            )
        else:
            raise NotImplementedError()

        self.tail = MLP(conf.style_ch, conf.tail_depth, last_act=False)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Args:
            x: (n, c, h, w)

        Returns: (n, c)
        """
        # Downsampling
        h = self.head(x)
        for layer in self.downblocks:
            h = layer(h)

        h = self.pool(h).flatten(start_dim=1)
        # (n, c)
        h = self.tail(h)
        return h


@dataclass
class StyleEncoderConfig(BaseConfig):
    img_size: int
    style_ch: int
    ch: int
    ch_mult: int
    attn: Tuple[int]
    num_res_blocks: int
    dropout: float

    @property
    def name(self):
        name = f'enc-ch{self.ch}-mult('
        name += ','.join(str(x) for x in self.ch_mult) + ')'
        name += '-attn('
        name += ','.join(str(x) for x in self.attn) + ')'
        name += f'-blk{self.num_res_blocks}-dropout{self.dropout}'
        return name

    def make_model(self):
        return StyleEncoder(self)


class StyleEncoder(nn.Module):
    """
    encode and image into a style vector
    """
    def __init__(self, conf: StyleEncoderConfig):
        super().__init__()
        self.conf = conf
        assert all([i < len(conf.ch_mult)
                    for i in conf.attn]), 'attn index out of bound'

        self.head = nn.Conv2d(3, conf.ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [conf.ch]  # record output channel when dowmsample for upsample

        now_ch = conf.ch

        resolution = conf.img_size
        for i, mult in enumerate(conf.ch_mult):
            out_ch = conf.ch * mult
            for _ in range(conf.num_res_blocks):
                self.downblocks.append(
                    ResBlock(in_ch=now_ch,
                             out_ch=out_ch,
                             tdim=None,
                             dropout=conf.dropout,
                             attn=(resolution in conf.attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(conf.ch_mult) - 1:
                resolution //= 2
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Conv2d(now_ch, conf.style_ch, kernel_size=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Args:
            x: (n, c, h, w)

        Returns: (n, c)
        """
        # Downsampling
        h = self.head(x)
        for layer in self.downblocks:
            h = layer(h)

        h = self.pool(h)
        # (n, c)
        h = self.tail(h).flatten(start_dim=1)
        return h
