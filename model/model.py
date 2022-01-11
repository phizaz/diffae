from config_base import BaseConfig
from dataclasses import dataclass
import math
from typing import NamedTuple, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, **kwargs):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, **kwargs):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C)**(-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim=None, dropout=None, attn=False):
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
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
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
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb=None):
        h = self.block1(x)
        if temb is not None:
            h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


@dataclass
class UNetConfig(BaseConfig):
    img_size: int
    T: int
    ch: int
    ch_mult: int
    attn: int
    num_res_blocks: int
    dropout: float

    def make_model(self):
        return UNet(self)


class UNet(nn.Module):
    def __init__(self, conf: UNetConfig):
        super().__init__()
        assert all([i < len(conf.ch_mult)
                    for i in conf.attn]), 'attn index out of bound'
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
            ResBlock(now_ch, now_ch, tdim, conf.dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, conf.dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(conf.ch_mult))):
            out_ch = conf.ch * mult
            for _ in range(conf.num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch,
                             out_ch=out_ch,
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
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, return_interm=False, **kwargs):
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
            h = layer(h, temb=temb)

        interm = []
        if return_interm:
            interm.append(h)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            new_h = layer(h, temb=temb)
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


class Return(NamedTuple):
    pred: Tensor
    interm: List[Tensor]


if __name__ == '__main__':
    batch_size = 8
    model = UNet(T=1000,
                 ch=128,
                 ch_mult=[1, 2, 2, 2],
                 attn=[1],
                 num_res_blocks=2,
                 dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
