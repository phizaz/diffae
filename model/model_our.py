import math
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from .model import *
from .model_base import *
from .style_encoder import *


@dataclass
class StyleUNetConfig(BaseConfig):
    img_size: int
    T: int
    ch: int
    ch_mult: Tuple[int]
    attn: Tuple[int]
    num_res_blocks: int
    dropout: float
    style_ch: int
    mid_attn: bool
    style_enc_conf: Union[StyleEncoderConfig, StyleEncoder2Config]

    @property
    def name(self):
        name = f'defaultautoenc-ch{self.ch}-mult('
        name += ','.join(str(x) for x in self.ch_mult) + ')'
        name += '-attn('
        name += ','.join(str(x) for x in self.attn) + ')'
        name += f'-blk{self.num_res_blocks}-dropout{self.dropout}'
        name += f'-style{self.style_ch}'
        if self.mid_attn:
            name += '-midattn'

        name += f'_{self.style_enc_conf.name}'

        return name

    def make_model(self):
        return StyleUNet(self)


class StyleUNet(nn.Module):
    """
    unet with style encoder and decoder
    """
    def __init__(self, conf: StyleUNetConfig):
        super().__init__()
        assert all([i < len(conf.ch_mult)
                    for i in conf.attn]), 'attn index out of bound'
        self.conf = conf
        tdim = conf.ch * 4
        self.time_embedding = TimeEmbedding(conf.T, conf.ch, tdim)

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
                             tdim=tdim,
                             dropout=conf.dropout,
                             attn=(resolution in conf.attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(conf.ch_mult) - 1:
                resolution //= 2
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlockMod(now_ch,
                        now_ch,
                        style_ch=conf.style_ch,
                        tdim=tdim,
                        dropout=conf.dropout,
                        attn=conf.mid_attn),
            ResBlockMod(now_ch,
                        now_ch,
                        style_ch=conf.style_ch,
                        tdim=tdim,
                        dropout=conf.dropout,
                        attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(conf.ch_mult))):
            out_ch = conf.ch * mult
            for _ in range(conf.num_res_blocks + 1):
                self.upblocks.append(
                    ResBlockMod(in_ch=chs.pop() + now_ch,
                                out_ch=out_ch,
                                style_ch=conf.style_ch,
                                tdim=tdim,
                                dropout=conf.dropout,
                                attn=(resolution in conf.attn)))
                now_ch = out_ch
            if i != 0:
                resolution *= 2
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(nn.GroupNorm(32, now_ch), Swish(),
                                  nn.Conv2d(now_ch, 3, 3, stride=1, padding=1))

        ######
        self.encoder = conf.style_enc_conf.make_model()
        self.style = StyleVectorizer(conf.style_ch, depth=8, lr_mul=0.1)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    @property
    def num_style_layers(self):
        return len(self.middleblocks) + len(self.upblocks)

    def forward(self,
                x,
                t,
                x_0=None,
                cond=None,
                style=None,
                return_interm=False,
                **kwargs):
        """
        Args:
            x: (n, c, h, w) x_t
            t: 
            x0: (n, c, h, w) x_0 for encoding
            style: (n, style_ch) or (n, layers * style_ch)
        """
        style_layers = self.num_style_layers
        style_full_length = style_layers * self.conf.style_ch

        # encoder
        if style is None:
            if cond is None:
                # (n, c)
                cond = self.encoder.forward(x_0)

            if cond.shape[1] == self.conf.style_ch:
                # (n, c)
                style = self.style.forward(cond)
                # (n, layers * c)
                style = style.repeat(1, style_layers)
            else:
                assert cond.shape[1] == style_full_length
                n = len(cond)
                # (n * layer, c)
                cond = cond.reshape(-1, self.conf.style_ch)
                style = self.style.forward(cond)
                # (n, layer * c)
                style = style.reshape(n, -1)
        else:
            if style.shape[1] == self.conf.style_ch:
                style = style.repeat(1, style_layers)
            else:
                # full length
                assert style.shape[1] == style_full_length

        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb=temb)
            hs.append(h)

        # Middle
        style_offset = 0
        for layer in self.middleblocks:
            # print('mid h:', h.shape)
            h = layer(h,
                      temb=temb,
                      style=style[:, style_offset:style_offset +
                                  self.conf.style_ch])
            style_offset += self.conf.style_ch

        interm = []
        if return_interm:
            interm.append(h)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlockMod):
                h = torch.cat([h, hs.pop()], dim=1)
            # print('h:', h.shape)
            new_h = layer(h,
                          temb=temb,
                          style=style[:, style_offset:style_offset +
                                      self.conf.style_ch])
            style_offset += self.conf.style_ch
            _, _, HH, WW = new_h.shape
            _, _, H, W = h.shape

            if return_interm and (HH, WW) != (H, W):
                # print((HH, WW), (H, W))
                interm.append(h)

            h = new_h

        if return_interm:
            interm.append(h)

        h = self.tail(h)

        if return_interm:
            interm.append(h)

        assert len(hs) == 0
        return Return(pred=h, interm=interm)


@dataclass
class VAEStyleUNetConfig(BaseConfig):
    img_size: int
    T: int
    ch: int
    ch_mult: int
    attn: int
    num_res_blocks: int
    dropout: float
    style_ch: int

    def make_model(self):
        return VAEStyleUNet(self)


class VAEReturn(NamedTuple):
    pred: Tensor
    interm: List[Tensor]
    cond: Tensor = None
    cond_mu: Tensor = None
    cond_logvar: Tensor = None


class VAEStyleUNet(StyleUNet):
    def __init__(self, conf: VAEStyleUNetConfig):
        super().__init__(conf)
        self.conf = conf

        # encoder outputs mean and log_var
        self.encoder = StyleEncoder(conf.style_ch * 2, conf.ch, conf.ch_mult,
                                    conf.attn, conf.num_res_blocks,
                                    conf.dropout)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        return torch.randn(n, self.conf.style_ch, device=device)

    def forward(self,
                x,
                t,
                x_0=None,
                cond=None,
                return_interm=False,
                **kwargs):
        """
        Args:
            x: (n, c, h, w) x_t
            t: 
            x0: (n, c, h, w) x_0 for encoding
        """
        # encoder
        if cond is None:
            # (n, c)
            tmp = self.encoder.forward(x_0)
            mu, logvar = (tmp[:, :self.conf.style_ch],
                          tmp[:, self.conf.style_ch:])
            cond = self.reparameterize(mu, logvar)
        else:
            mu, logvar = None, None

        # (n, c)
        style = self.style.forward(cond)

        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb=temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            # print('mid h:', h.shape)
            h = layer(h, temb=temb, style=style)

        interm = []
        if return_interm:
            interm.append(h)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlockMod):
                h = torch.cat([h, hs.pop()], dim=1)
            # print('h:', h.shape)
            new_h = layer(h, temb=temb, style=style)
            _, _, HH, WW = new_h.shape
            _, _, H, W = h.shape

            if return_interm and (HH, WW) != (H, W):
                # print((HH, WW), (H, W))
                interm.append(h)

            h = new_h

        if return_interm:
            interm.append(h)

        h = self.tail(h)

        if return_interm:
            interm.append(h)

        assert len(hs) == 0
        return VAEReturn(
            pred=h,
            interm=interm,
            cond=cond,
            cond_mu=mu,
            cond_logvar=logvar,
        )
