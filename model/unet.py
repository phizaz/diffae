import math
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from .blocks import *

from .nn import (CheckpointGNShiftScaleSiLU, GatedConv, GateType, avg_pool_nd,
                 checkpoint, conv_nd, linear, normalization,
                 timestep_embedding, torch_checkpoint, zero_module)


@dataclass
class BeatGANsUNetConfig(BaseConfig):
    image_size: int = 64
    in_channels: int = 3
    # base channels, will be multiplied
    model_channels: int = 64
    # output of the unet
    # suggest: 3
    # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
    out_channels: int = 3
    # how many repeating resblocks per resolution
    # the decoding side would have "one more" resblock
    # default: 2
    num_res_blocks: int = 2
    # you can also set the number of resblocks specifically for the input blocks
    # default: None = above
    num_input_res_blocks: int = None
    # number of time embed channels and style channels
    embed_channels: int = 512
    # at what resolutions you want to do self-attention of the feature maps
    # attentions generally improve performance
    # default: [16]
    # beatgans: [32, 16, 8]
    attention_resolutions: Tuple[int] = (16, )
    # number of time embed channels
    time_embed_channels: int = None
    # dropout applies to the resblocks (on feature maps)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    input_channel_mult: Tuple[int] = None
    conv_resample: bool = True
    # always 2 = 2d conv
    dims: int = 2
    # don't use this, legacy from BeatGANs
    num_classes: int = None
    use_checkpoint: bool = False
    use_fp16: bool = False
    # number of attention heads
    num_heads: int = 1
    # or specify the number of channels per attention head
    num_head_channels: int = -1
    # what's this?
    num_heads_upsample: int = -1
    # use resblock for upscale/downscale blocks (expensive)
    # default: True (BeatGANs)
    resblock_updown: bool = True
    # never tried
    use_new_attention_order: bool = False
    # does the middle_blocks have attention as well?
    # default: True
    use_mid_attn: bool = True
    # also apply conditions to the input_layers of the resblock
    # default: False (as in BeatGANs)
    # hypothesis: enabling more conditioning improves performance
    resnet_use_inlayers_condition: bool = False
    resnet_condition_type: ConditionType = ConditionType.add
    resnet_condition_scale_bias: Union[float, Tuple[float]] = 1
    resnet_two_cond: bool = False
    resnet_three_cond: bool = False
    resnet_cond_channels: int = None
    resnet_time_first: bool = False
    # whether to scale and shift, or just scale only
    resnet_time_emb_2xwidth: bool = True
    # default: False (scale + shift doesn't improve)
    resnet_cond_emb_2xwidth: bool = True
    ###
    resnet_gate_type: GateType = None
    resnet_gate_init: float = None
    # additional normalization layer after the last convolution of each resblock
    # hypothesis: this is to mimic the StyleGAN2 normalization order
    # deprecated due to no improvement
    resnet_use_after_norm: bool = False
    # init the decoding conv layers with zero weights, this speeds up training
    # default: True (BeattGANs)
    resnet_use_zero_module: bool = True
    resnet_scale_at: ScaleAt = ScaleAt.after_norm
    # whether the replace convolutions with Modulated Conv as in StyleGAN2 (slow, no improvement)
    resnet_use_style_conv: bool = False
    # checkpoint the scale & shift operations (this is cheap). This can save as much as 20% of the memory on large models
    resnet_use_checkpoint_gnscalesilu: bool = False
    # gradient checkpoint the attention operation
    attn_checkpoint: bool = False

    @property
    def name(self):
        name = f'netbeatgans-ch{self.model_channels}('
        name += ','.join(str(x) for x in self.channel_mult) + ')'
        if self.input_channel_mult is not None:
            name += f'-inpch{self.model_channels}('
            name += ','.join(str(x) for x in self.input_channel_mult) + ')'
        if self.embed_channels > 0:
            name += f'-emb{self.embed_channels}'
        name += f'-blk{self.num_res_blocks}'
        if self.num_input_res_blocks is not None:
            name += f'-inpblk{self.num_input_res_blocks}'
        name += f'-attn{self.num_heads}(' + ','.join(
            str(x) for x in self.attention_resolutions) + ')'
        if not self.use_mid_attn:
            name += '-nomidattn'
        name += f'-dropout{self.dropout}'

        name += f'-{self.resnet_condition_type.value}'
        if self.resnet_condition_type in [
                ConditionType.scale_shift_norm,
                ConditionType.scale_shift_hybrid
        ]:
            if isinstance(self.resnet_condition_scale_bias, Number):
                name += f'-bias{self.resnet_condition_scale_bias}'
            else:
                biases = self.resnet_condition_scale_bias
                name += f'-bias({biases[0]},{biases[1]})'
        if self.resnet_use_inlayers_condition:
            name += '-incond'
        if self.resnet_two_cond:
            name += '-twocond'
            if self.resnet_time_first:
                name += '-timefirst'
            if self.resnet_three_cond:
                name += '-threecond'
                name += f'-cond{self.resnet_cond_channels}'
        if self.resnet_time_emb_2xwidth:
            name += '-time2x'
        if self.resnet_cond_emb_2xwidth:
            name += '-cond2x'
        if self.resnet_scale_at != ScaleAt.after_norm:
            name += f'-scaleat{self.resnet_scale_at.value}'
        if self.resnet_use_after_norm:
            name += '-afternorm'
        if self.resnet_use_style_conv:
            name += '-styleconv'
        if not self.resnet_use_zero_module:
            name += '-nonzero'
        if self.resnet_use_checkpoint_gnscalesilu:
            name += '-ckptgnnormsilu'

        if self.resblock_updown:
            name += '-residue'
        return name

    def make_model(self):
        return BeatGANsUNetModel(self)


class BeatGANsUNetModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf

        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads

        self.dtype = th.float16 if conf.use_fp16 else th.float32

        self.time_emb_channels = conf.time_embed_channels or conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, conf.embed_channels),
            nn.SiLU(),
            linear(conf.embed_channels, conf.embed_channels),
        )

        if conf.num_classes is not None:
            self.label_emb = nn.Embedding(conf.num_classes,
                                          conf.embed_channels)

        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])

        # a hack!
        style_at_input = False
        style_at_mid = False
        style_at_dec = False
        if hasattr(conf, 'cond_at') and conf.resnet_use_style_conv:
            assert isinstance(conf.cond_at, CondAt)
            if conf.cond_at == CondAt.all:
                style_at_input = True
                style_at_mid = True
                style_at_dec = True
            elif conf.cond_at == CondAt.dec:
                style_at_dec = True
            elif conf.cond_at == CondAt.enc:
                style_at_input = True
            elif conf.cond_at == CondAt.mid_dec:
                style_at_mid = True
                style_at_dec = True
            else:
                raise NotImplementedError()

        kwargs = dict(
            use_condition=True,
            use_inlayers_condition=conf.resnet_use_inlayers_condition,
            condition_type=conf.resnet_condition_type,
            condition_scale_bias=conf.resnet_condition_scale_bias,
            two_cond=conf.resnet_two_cond,
            three_cond=conf.resnet_three_cond,
            time_first=conf.resnet_time_first,
            time_emb_2xwidth=conf.resnet_time_emb_2xwidth,
            cond_emb_2xwidth=conf.resnet_cond_emb_2xwidth,
            gate_type=conf.resnet_gate_type,
            gate_init=conf.resnet_gate_init,
            use_after_norm=conf.resnet_use_after_norm,
            use_zero_module=conf.resnet_use_zero_module,
            scale_at=conf.resnet_scale_at,
            use_checkpoint_gnscalesilu=conf.resnet_use_checkpoint_gnscalesilu,
            # style channels for the resnet block
            cond_emb_channels=conf.resnet_cond_channels,
        )

        if conf.resnet_three_cond:
            kwargs[
                'cond2_emb_channels'] = conf.embed_channels - conf.resnet_cond_channels

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.input_channel_mult
                                     or conf.channel_mult):
            for _ in range(conf.num_input_res_blocks or conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        use_styleconv=style_at_input,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                            use_styleconv=style_at_input,
                            **kwargs,
                        ).make_model() if conf.
                        resblock_updown else Downsample(ch,
                                                        conf.conv_resample,
                                                        dims=conf.dims,
                                                        out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                use_styleconv=style_at_mid,
                **kwargs,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ) if conf.use_mid_attn else nn.Identity(),
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                use_styleconv=style_at_mid,
                **kwargs,
            ).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        use_styleconv=style_at_dec,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True,
                            use_styleconv=style_at_dec,
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        if conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    conv_nd(conf.dims,
                            input_ch,
                            conf.out_channels,
                            3,
                            padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(conf.dims, input_ch, conf.out_channels, 3, padding=1),
            )

    def forward(self, x, t, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred=pred)


class Return(NamedTuple):
    pred: th.Tensor


@dataclass
class BeatGANsEncoderConfig(BaseConfig):
    image_size: int
    in_channels: int
    model_channels: int
    out_hid_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int]
    dropout: float = 0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    use_time_condition: bool = True
    conv_resample: bool = True
    dims: int = 2
    use_checkpoint: bool = False
    use_fp16: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = "adaptive"
    pool_tail_layer: int = None
    last_act: Activation = Activation.none

    @property
    def name(self):
        name = f'encoder-ch{self.model_channels}('
        name += ','.join(str(x) for x in self.channel_mult) + ')'
        name += f'-blk{self.num_res_blocks}'
        name += f'-attn{self.num_heads}(' + ','.join(
            str(x) for x in self.attention_resolutions) + ')'
        name += f'-dropout{self.dropout}'
        if self.use_time_condition:
            name += '-time'
        if self.resblock_updown:
            name += '-residue'
        return name

    def make_model(self):
        return BeatGANsEncoderModel(self)


class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """
    def __init__(self, conf: BeatGANsEncoderConfig):
        super().__init__()
        self.conf = conf

        self.dtype = th.float16 if conf.use_fp16 else th.float32

        if conf.use_time_condition:
            time_embed_dim = conf.model_channels * 4
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_condition=conf.use_time_condition,
                        use_checkpoint=conf.use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_condition=conf.use_time_condition,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Downsample(ch,
                                          conf.conv_resample,
                                          dims=conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch
        if conf.pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(conf.dims, ch, conf.out_channels, 1)),
                nn.Flatten(),
            )
        elif conf.pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
            )
        elif conf.pool == "adaptivenonzerotail":
            tail = []
            for i in range(conf.pool_tail_layer):
                tail.append(normalization(conf.out_channels))
                tail.append(nn.SiLU())
                tail.append(nn.Linear(conf.out_channels, conf.out_channels))

            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
                *tail,
            )
        elif conf.pool == 'depthconv':
            assert conf.out_channels >= ch
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.Conv2d(ch,
                          conf.out_hid_channels,
                          kernel_size=resolution,
                          groups=ch),
                nn.Conv2d(conf.out_hid_channels,
                          conf.out_channels,
                          kernel_size=1),
                nn.Flatten(),
            )
        elif conf.pool == 'depthconv2048':
            assert conf.out_channels >= ch
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.Conv2d(ch, 2048, kernel_size=resolution, groups=ch),
                nn.Conv2d(2048, conf.out_channels, kernel_size=1),
                nn.Flatten(),
            )
        elif conf.pool == "attention":
            assert conf.num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d((conf.image_size // ds), ch,
                                conf.num_head_channels, conf.out_channels),
            )
        elif conf.pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, conf.out_channels),
            )
        elif conf.pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, conf.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")

        self.last_act = self.conf.last_act.get_act()

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, t=None, return_2d_feature=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        if self.conf.use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else:
            emb = None

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb=emb)
            if self.conf.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb=emb)
        if self.conf.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)

        h_2d = h
        h = self.out(h)
        h = self.last_act(h)

        if return_2d_feature:
            return h, h_2d
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        h = self.last_act(h)
        return h


class SuperResModel(BeatGANsUNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width),
                                  mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)
