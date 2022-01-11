import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple
from functools import partial

import torch
from choices import *
from choices import ConditionType
from config_base import BaseConfig
from torch import nn
from torch.nn import init

from .unet import *
from .blocks import *

from .nn import timestep_embedding


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
class LatentConvConfig(BaseConfig):
    num_channels: int = 512
    num_hid_channels: int = 512
    num_time_emb_channels: int = 64
    dropout: float = 0
    num_layers: int = 3
    use_zero_module: bool = True

    @property
    def name(self):
        name = f'latentconv-ch{self.num_channels}-hid{self.num_hid_channels}-{self.num_layers}layers'
        name += f'-emb{self.num_time_emb_channels}'
        if self.use_zero_module:
            name += '-zero'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        return name

    def make_model(self):
        return LatentConvNet(self)


class LatentConvNet(nn.Module):
    def __init__(self, conf: LatentConvConfig):
        super().__init__()
        self.conf = conf

        self.time_embed = nn.Sequential(
            nn.Linear(conf.num_time_emb_channels, conf.num_channels),
            nn.SiLU(),
            nn.Linear(conf.num_channels, conf.num_channels),
        )

        blocks = []
        for i in range(conf.num_layers):
            if i == 0:
                a = conf.num_channels
                b = conf.num_hid_channels
            else:
                a = conf.num_hid_channels
                b = conf.num_hid_channels

            block = ResBlockConfig(
                channels=a,
                emb_channels=conf.num_channels,
                dropout=conf.dropout,
                out_channels=b,
                use_condition=True,
                use_conv=False,
                dims=2,
                up=False,
                down=False,
                condition_type=ConditionType.scale_shift_norm,
                condition_scale_bias=1,
                two_cond=False,
                time_first=True,
                time_emb_2xwidth=True,
                cond_emb_2xwidth=False,
                use_zero_module=conf.use_zero_module,
            ).make_model()
            blocks.append(block)

            # block = AttentionBlock(
            #     b,
            #     use_checkpoint=False,
            #     num_heads=1,
            #     num_head_channels=-1,
            #     use_new_attention_order=False,
            # )
            # blocks.append(block)

        self.blocks = TimestepEmbedSequential(*blocks)
        # self.out = nn.Conv2d(conf.num_hid_channels,
        #                      conf.num_channels,
        #                      kernel_size=3,
        #                      padding=1)
        self.out = nn.Sequential(
            normalization(conf.num_hid_channels),
            nn.SiLU(),
            zero_module(
                conv_nd(2,
                        conf.num_hid_channels,
                        conf.num_channels,
                        3,
                        padding=1)),
        )

    def forward(self, x, t, **kwargs):
        t_emb = self.time_embed(
            timestep_embedding(t, dim=self.conf.num_time_emb_channels))
        x = self.blocks.forward(x, emb=t_emb)
        x = self.out.forward(x)
        return LatentNetReturn(pred=x)


@dataclass
class ProjectedConvLatentConfig(LatentConvConfig):
    project_size: int = 4
    unpool: str = 'conv'

    @property
    def name(self):
        name = f'projectedconv-size{self.project_size}-ch{self.num_channels}-hid{self.num_hid_channels}-{self.num_layers}layers'
        name += f'-{self.unpool}'
        name += f'-emb{self.num_time_emb_channels}'
        if self.use_zero_module:
            name += '-zero'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        return name

    def make_model(self):
        return ProjectedConvLatentNet(self)


class ProjectedConvLatentNet(nn.Module):
    def __init__(self, conf: ProjectedConvLatentConfig) -> None:
        super().__init__()
        self.conf = conf

        self.time_embed = nn.Sequential(
            nn.Linear(conf.num_time_emb_channels, conf.num_channels),
            nn.SiLU(),
            nn.Linear(conf.num_channels, conf.num_channels),
        )

        if conf.unpool == 'conv':
            self.proj = nn.ConvTranspose2d(conf.num_channels,
                                           conf.num_hid_channels,
                                           kernel_size=conf.project_size)
        elif conf.unpool == 'avg':
            self.proj = nn.Sequential(
                nn.Conv2d(conf.num_channels, conf.num_hid_channels, 1),
                nn.Upsample(conf.project_size, mode='nearest'),
            )
        else:
            raise NotImplementedError()

        blocks = []
        for i in range(conf.num_layers):
            block = ResBlockConfig(
                channels=conf.num_hid_channels,
                emb_channels=conf.num_channels,
                dropout=conf.dropout,
                out_channels=conf.num_hid_channels,
                use_condition=True,
                use_conv=False,
                dims=2,
                up=False,
                down=False,
                condition_type=ConditionType.scale_shift_norm,
                condition_scale_bias=1,
                two_cond=False,
                time_first=True,
                time_emb_2xwidth=True,
                cond_emb_2xwidth=False,
                use_zero_module=conf.use_zero_module,
            ).make_model()
            blocks.append(block)

        self.blocks = TimestepEmbedSequential(*blocks)

        self.out = nn.Sequential(
            normalization(conf.num_hid_channels),
            nn.SiLU(),
            zero_module(
                conv_nd(2, conf.num_hid_channels, conf.num_channels,
                        conf.project_size)),
            nn.Flatten(),
        )
        # self.out = nn.Sequential(
        #     normalization(conf.num_hid_channels),
        #     nn.SiLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     zero_module(conv_nd(2, conf.num_hid_channels, conf.num_channels,
        #                         1)),
        #     nn.Flatten(),
        # )

    def forward(self, x, t, **kwargs):
        t_emb = self.time_embed(
            timestep_embedding(t, dim=self.conf.num_time_emb_channels))
        x = x[:, :, None, None]
        x = self.proj.forward(x)
        x = self.blocks.forward(x, emb=t_emb)
        x = self.out.forward(x)
        return LatentNetReturn(pred=x)


@dataclass
class ProjectedUnetLatentConfig(BaseConfig):
    project_size: int = 8
    num_channels: int = 512
    num_hid_channels: int = 32
    attn_resolutions: Tuple[int] = tuple()
    use_mid_attn: bool = True
    channel_mult: Tuple[int] = (1, 2, 4)
    num_time_emb_channels: int = 64
    num_res_blocks: int = 2
    dropout: float = 0.1

    @property
    def name(self):
        name = f'projectedunet-size{self.project_size}-ch{self.num_channels}-hid{self.num_hid_channels}'
        name += '-ch(' + ','.join(str(x) for x in self.channel_mult) + ')'
        if self.num_res_blocks != 2:
            name += f'-resblk{self.num_res_blocks}'
        name += '-attn(' + ','.join(str(x)
                                    for x in self.attn_resolutions) + ')'
        if not self.use_mid_attn:
            name += '-nomidattn'
        name += f'-emb{self.num_time_emb_channels}'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        return name

    def make_model(self):
        return ProjectedUnetLatentNet(self)


class ProjectedUnetLatentNet(nn.Module):
    def __init__(self, conf: ProjectedUnetLatentConfig) -> None:
        super().__init__()
        self.conf = conf

        self.proj = nn.ConvTranspose2d(conf.num_channels,
                                       conf.num_hid_channels,
                                       kernel_size=conf.project_size)

        self.unet = BeatGANsUNetConfig(
            image_size=conf.project_size,
            in_channels=conf.num_hid_channels,
            model_channels=conf.num_hid_channels,
            out_channels=conf.num_hid_channels,
            num_res_blocks=conf.num_res_blocks,
            time_embed_channels=conf.num_time_emb_channels,
            embed_channels=conf.num_hid_channels,
            attention_resolutions=conf.attn_resolutions,
            dropout=conf.dropout,
            channel_mult=conf.channel_mult,
            use_mid_attn=conf.use_mid_attn,
            resblock_updown=True,
            resnet_condition_type=ConditionType.scale_shift_norm,
            resnet_condition_scale_bias=1,
            resnet_use_zero_module=False,
        ).make_model()

        self.out = nn.Sequential(
            zero_module(
                conv_nd(2, conf.num_hid_channels, conf.num_channels,
                        conf.project_size)),
            nn.Flatten(),
        )

    def forward(self, x, t, **kwargs):
        x = x[:, :, None, None]
        x = self.proj.forward(x)
        x = self.unet.forward(x, t=t).pred
        x = self.out.forward(x)
        return LatentNetReturn(pred=x)


@dataclass
class ProjectedInvertUnetLatentConfig(BaseConfig):
    project_size: int = 4
    num_channels: int = 512
    num_hid_channels: int = 256
    attn_resolutions: Tuple[int] = tuple()
    channel_mult: Tuple[int] = (4, 2, 1)
    num_time_emb_channels: int = 64
    dropout: float = 0.1
    num_res_blocks: int = 1

    @property
    def name(self):
        name = f'projectedinvunet-size{self.project_size}-ch{self.num_channels}-hid{self.num_hid_channels}'
        name += f'-resblk{self.num_res_blocks}'
        name += '-ch(' + ','.join(str(x) for x in self.channel_mult) + ')'
        name += '-attn(' + ','.join(str(x)
                                    for x in self.attn_resolutions) + ')'
        name += f'-emb{self.num_time_emb_channels}'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        return name

    def make_model(self):
        return ProjectedInvertUnetLatentNet(self)


class ProjectedInvertUnetLatentNet(nn.Module):
    def __init__(self, conf: ProjectedInvertUnetLatentConfig) -> None:
        super().__init__()
        self.conf = conf

        self.time_embed = nn.Sequential(
            nn.Linear(conf.num_time_emb_channels, conf.num_channels),
            nn.SiLU(),
            nn.Linear(conf.num_channels, conf.num_channels),
        )

        self.proj = nn.ConvTranspose2d(conf.num_channels,
                                       conf.num_hid_channels,
                                       kernel_size=conf.project_size)

        kwargs = dict(
            condition_type=ConditionType.scale_shift_norm,
            condition_scale_bias=1,
            two_cond=False,
            time_first=True,
            time_emb_2xwidth=True,
            use_zero_module=True,
        )

        expand = []
        max_mul = max(conf.channel_mult)
        ch = conf.num_hid_channels
        for level, mult in enumerate(conf.channel_mult):
            for i in range(conf.num_res_blocks):
                ch_next = int(conf.num_hid_channels * mult / max_mul)
                expand.append(
                    ResBlockConfig(
                        channels=ch,
                        emb_channels=conf.num_channels,
                        dropout=conf.dropout,
                        out_channels=ch_next,
                        **kwargs,
                    ).make_model())
                ch = ch_next
            if level < len(conf.channel_mult) - 1:
                # expand.append(Upsample(ch, use_conv=True))
                expand.append(
                    ResBlockConfig(
                        ch,
                        conf.num_channels,
                        conf.dropout,
                        out_channels=ch,
                        up=True,
                        **kwargs,
                    ).make_model())
        self.expand = TimestepEmbedSequential(*expand)

        contract = []
        for level, mult in enumerate(conf.channel_mult[::-1]):
            for _ in range(conf.num_res_blocks):
                ch_next = int(conf.num_hid_channels * mult / max_mul)
                contract.append(
                    ResBlockConfig(
                        ch,
                        emb_channels=conf.num_channels,
                        dropout=conf.dropout,
                        out_channels=ch_next,
                        **kwargs,
                    ).make_model())
                ch = ch_next
            if level < len(conf.channel_mult) - 1:
                contract.append(
                    ResBlockConfig(
                        ch,
                        emb_channels=conf.num_channels,
                        dropout=conf.dropout,
                        out_channels=ch,
                        down=True,
                        **kwargs,
                    ).make_model())
        contract += [
            ResBlockConfig(
                ch,
                emb_channels=conf.num_channels,
                dropout=conf.dropout,
                **kwargs,
            ).make_model(),
            ResBlockConfig(
                ch,
                emb_channels=conf.num_channels,
                dropout=conf.dropout,
                **kwargs,
            ).make_model(),
        ]
        self.contract = TimestepEmbedSequential(*contract)

        self.out = nn.Sequential(
            zero_module(
                conv_nd(2, conf.num_hid_channels, conf.num_channels,
                        conf.project_size)),
            nn.Flatten(),
        )

    def forward(self, x, t, **kwargs):
        t_emb = self.time_embed(
            timestep_embedding(t, dim=self.conf.num_time_emb_channels))
        x = x[:, :, None, None]
        x = self.proj.forward(x)
        x = self.expand.forward(x, emb=t_emb)
        x = self.contract.forward(x, emb=t_emb)
        x = self.out.forward(x)
        return LatentNetReturn(pred=x)


@dataclass
class ProjectedHalfUnetLatentConfig(BaseConfig):
    project_size: int = 8
    num_channels: int = 512
    num_hid_channels: int = 128
    attn_resolutions: Tuple[int] = tuple()
    channel_mult: Tuple[int] = (1, 2, 4)
    num_time_emb_channels: int = 64
    dropout: float = 0.1
    num_res_blocks: int = 1

    @property
    def name(self):
        name = f'projectedhalfunet-size{self.project_size}-ch{self.num_channels}-hid{self.num_hid_channels}'
        name += f'-resblk{self.num_res_blocks}'
        name += '-ch(' + ','.join(str(x) for x in self.channel_mult) + ')'
        name += '-attn(' + ','.join(str(x)
                                    for x in self.attn_resolutions) + ')'
        name += f'-emb{self.num_time_emb_channels}'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        return name

    def make_model(self):
        return ProjectedHalfUnetLatentNet(self)


class ProjectedHalfUnetLatentNet(nn.Module):
    def __init__(self, conf: ProjectedHalfUnetLatentConfig) -> None:
        super().__init__()
        self.conf = conf

        self.time_embed = nn.Sequential(
            nn.Linear(conf.num_time_emb_channels, conf.num_channels),
            nn.SiLU(),
            nn.Linear(conf.num_channels, conf.num_channels),
        )

        self.proj = nn.ConvTranspose2d(conf.num_channels,
                                       conf.num_hid_channels,
                                       kernel_size=conf.project_size)

        kwargs = dict(
            condition_type=ConditionType.scale_shift_norm,
            condition_scale_bias=1,
            two_cond=False,
            time_first=True,
            time_emb_2xwidth=True,
            use_zero_module=True,
        )

        contract = []
        ch = conf.num_hid_channels
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                ch_next = int(conf.num_hid_channels * mult)
                # print(ch)
                contract.append(
                    ResBlockConfig(
                        ch,
                        emb_channels=conf.num_channels,
                        dropout=conf.dropout,
                        out_channels=ch_next,
                        **kwargs,
                    ).make_model())
                ch = ch_next
            if level < len(conf.channel_mult) - 1:
                contract.append(
                    ResBlockConfig(
                        ch,
                        emb_channels=conf.num_channels,
                        dropout=conf.dropout,
                        out_channels=ch,
                        down=True,
                        **kwargs,
                    ).make_model())
        contract += [
            ResBlockConfig(
                ch,
                emb_channels=conf.num_channels,
                dropout=conf.dropout,
                **kwargs,
            ).make_model(),
            ResBlockConfig(
                ch,
                emb_channels=conf.num_channels,
                dropout=conf.dropout,
                **kwargs,
            ).make_model(),
        ]
        self.contract = TimestepEmbedSequential(*contract)

        self.out = nn.Sequential(
            zero_module(
                conv_nd(2, ch, conf.num_channels, conf.project_size //
                        (2**(len(conf.channel_mult) - 1)))),
            nn.Flatten(),
        )

    def forward(self, x, t, **kwargs):
        t_emb = self.time_embed(
            timestep_embedding(t, dim=self.conf.num_time_emb_channels))
        x = x[:, :, None, None]
        x = self.proj.forward(x)
        x = self.contract.forward(x, emb=t_emb)
        x = self.out.forward(x)
        return LatentNetReturn(pred=x)


@dataclass
class MLPNetConfig(BaseConfig):
    num_channels: int
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int
    activation: Activation
    use_norm: bool
    condition_type: ConditionType
    condition_2x: bool
    condition_bias: float
    dropout: float = 0
    last_act: Activation = Activation.none
    time_is_int: bool = True

    @property
    def name(self):
        name = f'mlp-ch{self.num_channels}-hid{self.num_hid_channels}-{self.num_layers}layers-act{self.activation.value}'
        if self.use_norm:
            name += '-norm'
        name += f'-emb{self.num_time_emb_channels}{self.condition_type.value}'
        if self.condition_2x:
            name += '2x'
        if self.condition_bias > 0:
            name += f'-bias{self.condition_bias}'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        if self.last_act != Activation.none:
            name += f'-lastact{self.last_act.value}'
        return name

    def make_model(self):
        return MLPNet(self)


class MLPNet(nn.Module):
    """
    used for latent diffusion process
    """
    def __init__(self, conf: MLPNetConfig):
        super().__init__()
        self.conf = conf

        self.time_embed = nn.Sequential(
            nn.Linear(conf.num_time_emb_channels, conf.num_channels),
            conf.activation.get_act(),
            nn.Linear(conf.num_channels, conf.num_channels),
        )

        self.layers = []
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                cond = conf.condition_type
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
            elif i == conf.num_layers - 1:
                act = conf.last_act
                norm = False
                cond = ConditionType.no
                a, b = conf.num_hid_channels, conf.num_channels
                dropout = 0
            else:
                act = conf.activation
                norm = conf.use_norm
                cond = conf.condition_type
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout

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
                ))
        self.layers = CondSequential(*self.layers)

    def forward(self, x, t, **kwargs):
        if self.conf.time_is_int:
            t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        x = self.layers.forward(x=x, cond=cond)
        return LatentNetReturn(x)


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
    condition_type: ConditionType = ConditionType.scale_shift_norm
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
        name += f'-emb{self.num_time_emb_channels}{self.condition_type.value}'
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


@dataclass
class MLPConcatConfig(BaseConfig):
    num_channels: int
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int
    activation: Activation
    use_norm: bool
    cond_type: ConditionType = ConditionType.scale_shift_norm

    @property
    def name(self):
        name = f'mlpconcat-ch{self.num_channels}-hid{self.num_hid_channels}-{self.num_layers}layers-act{self.activation.value}'
        if self.use_norm:
            name += '-norm'
        name += f'-emb{self.num_time_emb_channels}'
        if self.cond_type != ConditionType.scale_shift_norm:
            name += f'-cond{self.cond_type.value}'
        return name

    def make_model(self):
        return MLPConcat(self)


class MLPConcat(nn.Module):
    """
    concat x with the time
    """
    def __init__(self, conf: MLPConcatConfig):
        super().__init__()
        self.conf = conf

        self.layers = []
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                a, b = (conf.num_channels + conf.num_time_emb_channels,
                        conf.num_hid_channels)
            elif i == conf.num_layers - 1:
                act = Activation.none
                norm = False
                a, b = conf.num_hid_channels, conf.num_channels
            else:
                act = conf.activation
                norm = conf.use_norm
                a, b = conf.num_hid_channels, conf.num_hid_channels

            self.layers.append(
                MLPLNAct(a,
                         b,
                         norm=norm,
                         activation=act,
                         cond_channels=conf.num_channels,
                         condition_type=ConditionType.no,
                         condition_2x=False))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, t, **kwargs):
        cond = timestep_embedding(t, self.conf.num_time_emb_channels)
        x = torch.cat([x, cond], dim=1)
        x = self.layers(x)
        return LatentNetReturn(x)


@dataclass
class MLPResNetConfig(BaseConfig):
    num_channels: int
    num_hid_channels: int
    num_blocks: int
    num_layers_per_block: int
    num_time_emb_channels: int
    activation: Activation
    use_norm: bool
    condition_type: ConditionType
    condition_2x: bool

    @property
    def name(self):
        name = f'mlpresnet-ch{self.num_channels}-hid{self.num_hid_channels}-{self.num_blocks}blocks-{self.num_layers_per_block}layers-act{self.activation.value}'
        if self.use_norm:
            name += '-norm'
        name += f'-emb{self.num_time_emb_channels}{self.condition_type.value}'
        if self.condition_2x:
            name += '2x'
        return name

    def make_model(self):
        return MLPResNet(self)


class MLPResNet(nn.Module):
    def __init__(self, conf: MLPResNetConfig):
        super().__init__()
        self.conf = conf

        self.time_embed = nn.Sequential(
            nn.Linear(conf.num_time_emb_channels, conf.num_channels),
            conf.activation.get_act(),
            nn.Linear(conf.num_channels, conf.num_channels),
        )

        self.blocks = []
        for i in range(conf.num_blocks):
            self.blocks.append(
                MLPResBlock(num_channels=conf.num_channels,
                            num_hid_channels=conf.num_hid_channels,
                            num_layers=conf.num_layers_per_block,
                            activation=conf.activation,
                            norm=conf.use_norm,
                            condition_type=conf.condition_type,
                            condition_2x=conf.condition_2x))
        self.blocks = CondSequential(*self.blocks)

        self.out = nn.Linear(conf.num_channels, conf.num_channels)

        self.init_weights()

    def init_weights(self):
        for module in self.out.modules():
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, t, **kwargs):
        cond = self.time_embed(
            timestep_embedding(t, self.conf.num_time_emb_channels))
        x = self.blocks.forward(x=x, cond=cond)
        x = self.out(x)
        return LatentNetReturn(x)


class MLPResBlock(nn.Module):
    def __init__(self, num_channels: int, num_hid_channels: int,
                 num_layers: int, norm: bool, activation: Activation,
                 condition_type: ConditionType, condition_2x: bool):
        super().__init__()
        assert num_layers >= 2
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                a, b = num_channels, num_hid_channels
            elif i == num_layers - 1:
                a, b = num_hid_channels, num_channels
            else:
                a, b = num_hid_channels, num_hid_channels

            self.layers.append(
                MLPLNAct(a,
                         b,
                         norm=norm,
                         activation=activation,
                         cond_channels=num_channels,
                         condition_type=condition_type,
                         condition_2x=condition_2x))

        self.layers = CondSequential(*self.layers)
        self.merge = MLPLNAct(num_channels * 2,
                              num_channels,
                              norm=norm,
                              activation=activation,
                              cond_channels=num_channels,
                              condition_type=condition_type,
                              condition_2x=condition_2x)

    def forward(self, x, cond):
        h = self.layers.forward(x, cond)
        x = self.merge.forward(torch.cat([x, h], dim=1), cond)
        return x


class CondSequential(nn.Sequential):
    def forward(self, x, cond):
        for module in self:
            x = module(x, cond)
        return x


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation: Activation,
        cond_channels: int,
        condition_type: ConditionType,
        condition_2x: bool,
        condition_bias: float = 0,
        dropout: float = 0,
        residual: bool = False,
    ):
        super().__init__()
        self.condition_type = condition_type
        self.condition_2x = condition_2x
        self.activation = activation
        self.condition_bias = condition_bias
        self.residual = residual

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if condition_type != ConditionType.no:
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


@dataclass
class MLPMixerConfig(BaseConfig):
    num_channels: int
    num_patches: int
    num_layers: int
    num_time_emb_channels: int = 64
    expansion_factor: int = 4
    dropout: float = 0
    num_time_layers: int = 2
    pooling: str = 'avg'
    time_is_int: bool = True
    cond_both: bool = True

    @property
    def name(self):
        name = f'mixer-ch{self.num_channels}-patch{self.num_patches}-exp{self.expansion_factor}-{self.num_layers}layers'
        name += f'-emb{self.num_time_emb_channels}'
        name += f'-pool{self.pooling}'
        if self.num_time_layers != 2:
            name += f'-timel{self.num_time_layers}'
        if not self.cond_both:
            name += '-condfirst'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        return name

    def make_model(self):
        return MLPMixer(self)


class MLPMixer(nn.Module):
    def __init__(self, conf: MLPMixerConfig) -> None:
        super().__init__()
        self.conf = conf

        layers = []
        for i in range(conf.num_time_layers):
            if i == 0:
                a = conf.num_time_emb_channels
                b = conf.num_channels
            else:
                a = conf.num_channels
                b = conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1:
                layers.append(nn.SiLU())
        self.time_embed = nn.Sequential(*layers)

        self.first = nn.Linear(conf.num_channels,
                               conf.num_channels * conf.num_patches)
        layers = []
        for i in range(conf.num_layers):
            layers.append(
                MixerLayer(
                    conf.num_channels,
                    conf.expansion_factor,
                    conf.dropout,
                    conf.num_channels,
                    cond_both=conf.cond_both,
                ))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(conf.num_channels)
        if conf.pooling == 'linear':
            self.pool = nn.Linear(conf.num_channels * conf.num_patches,
                                  conf.num_channels)
        self.last = nn.Linear(conf.num_channels, conf.num_channels)
        # print(self)

    def forward(self, x, t, **kwargs):
        if self.conf.time_is_int:
            t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        n, c = x.shape
        # (n, c) => (n, c * t)
        x = self.first(x)
        # (n, t, c)
        x = x.reshape(n, -1, c)
        for layer in self.layers:
            x = layer(x, cond=cond)
        # (n, t, c)
        x = self.norm(x)
        if self.conf.pooling == 'avg':
            # (n, c)
            x = x.mean(dim=1)
        elif self.conf.pooling == 'max':
            # (n, c)
            x, _ = x.max(dim=1)
        elif self.conf.pooling == 'linear':
            # (n, t*c)
            x = x.flatten(1)
            # (n, c)
            x = self.pool(x)
        else:
            raise NotImplementedError()
        x = self.last(x)
        return LatentNetReturn(x)


class MixerLayer(nn.Module):
    def __init__(self, dim, expansion_factor, dropout, cond_dim,
                 cond_both) -> None:
        super().__init__()
        self.cond_both = cond_both
        self.mlp1 = PreNormCondResidual(dim,
                                        PermutationLayer(dim),
                                        cond_dim=cond_dim)
        mlp = FeedForward(dim, expansion_factor, dropout, nn.Linear)
        if cond_both:
            self.mlp2 = PreNormCondResidual(dim, mlp, cond_dim=cond_dim)
        else:
            self.mlp2 = PreNormResidual(dim, mlp)

    def forward(self, x, cond):
        """
        Args:
            x: (n, t, c)
        
        Returns: (n, t, c)
        """
        x = self.mlp1(x, cond=cond)
        if self.cond_both:
            x = self.mlp2(x, cond=cond)
        else:
            x = self.mlp2(x)
        return x


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PreNormCondResidual(nn.Module):
    def __init__(self, dim, fn, cond_dim):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        # (n, 1, c)
        scale = self.linear(cond[:, None, :]) + 1
        return self.fn(scale * self.norm(x)) + x


class PermutationLayer(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: (n, t, c)
        """

        n, t, c = x.shape

        # (n, t, c)
        x1 = self.linear1(x)

        # mixing t and c channels
        # (n, t, t, c // t)
        x2 = x.reshape(n, t, t, -1)
        x2 = x2.permute([0, 2, 1, 3])
        # (n, t, c)
        x2 = x2.reshape(n, t, c)
        x2 = self.linear2(x2)

        x = self.proj(x1 + x2)
        return x


def FeedForward(dim,
                expansion_factor=4,
                dropout=0.,
                dense=nn.Linear,
                activation=nn.GELU):
    return nn.Sequential(dense(dim, dim * expansion_factor), activation(),
                         nn.Dropout(dropout), dense(dim * expansion_factor,
                                                    dim), nn.Dropout(dropout))


@dataclass
class MLPPreNormSkipNetConfig(BaseConfig):
    num_channels: int
    skip_layers: Tuple[int]
    num_layers: int
    expansion_factor: int = 4
    num_time_emb_channels: int = 64
    num_time_layers: int = 2
    time_layer_init: bool = False
    layer_init: bool = False
    time_is_int: bool = True
    dropout: float = 0

    @property
    def name(self):
        name = f'mlpprenorm-ch{self.num_channels}-exp{self.expansion_factor}-{self.num_layers}layers'
        name += '-skip(' + ','.join(str(x) for x in self.skip_layers) + ')'
        name += f'-emb{self.num_time_emb_channels}'
        if self.num_time_layers != 2:
            name += f'-timel{self.num_time_layers}'
        if self.time_layer_init:
            name += '-timinit'
        if self.layer_init:
            name += '-layerinit'
        if self.dropout > 0:
            name += f'-dropout{self.dropout}'
        return name

    def make_model(self):
        return MLPPreNormSkipNet(self)


class MLPPreNormSkipNet(nn.Module):
    """
    concat x to hidden layers
    """
    def __init__(self, conf: MLPPreNormSkipNetConfig):
        super().__init__()
        self.conf = conf

        layers = []
        for i in range(conf.num_time_layers):
            if i == 0:
                a = conf.num_time_emb_channels
                b = conf.num_channels
            else:
                a = conf.num_channels
                b = conf.num_channels
            layers.append(nn.Linear(a, b))
            layers.append(nn.SiLU())
        self.time_embed = nn.Sequential(*layers)

        if conf.time_layer_init:
            for each in self.time_embed.modules():
                if isinstance(each, nn.Linear):
                    init.kaiming_normal_(each.weight)

        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                a = conf.num_channels
                b = conf.num_channels * conf.expansion_factor
            else:
                a = conf.num_channels * conf.expansion_factor
                b = conf.num_channels * conf.expansion_factor

            if i in conf.skip_layers:
                c = conf.num_channels
            else:
                c = 0

            self.layers.append(
                PreNormCondResidualWithSkip(
                    a + c,
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Dropout(conf.dropout),
                        nn.Linear(a + c, b),
                    ),
                    conf.num_channels,
                ))

        if conf.layer_init:
            for each in self.layers.modules():
                if isinstance(each, nn.Linear):
                    init.kaiming_normal_(each.weight)

        self.tail = nn.Sequential(
            nn.LayerNorm(b),
            nn.Linear(b, conf.num_channels),
        )
        # print(self)

    def forward(self, x, t, **kwargs):
        if self.conf.time_is_int:
            t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i == 0:
                res = None
            else:
                res = h
            if i in self.conf.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond, res=res)
        h = self.tail(h)
        return LatentNetReturn(h)


def FeedForward2(dim,
                 skip_dim,
                 expansion_factor=4,
                 dropout=0.,
                 dense=nn.Linear,
                 activation=nn.GELU):
    return nn.Sequential(dense(dim + skip_dim, dim * expansion_factor),
                         activation(), nn.Dropout(dropout),
                         dense(dim * expansion_factor, dim),
                         nn.Dropout(dropout))


class PreNormCondResidualWithSkip(nn.Module):
    def __init__(self, dim, fn, cond_dim):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(cond_dim, dim)

    def forward(self, x, cond, res=None):
        # (n, c)
        scale = self.linear(cond) + 1
        if res is None:
            res = 0
        return self.fn(scale * self.norm(x)) + res