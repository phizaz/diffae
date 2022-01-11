import math
from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number

import torch as th
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from torch import nn

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (CheckpointGNShiftScaleSiLU, GatedConv, GateType, avg_pool_nd,
                 checkpoint, conv_nd, linear, normalization,
                 timestep_embedding, torch_checkpoint, zero_module)
from .styleconv import EqualizedConv2DMod, StyleConvSequential


class ScaleAt(Enum):
    after_norm = 'afternorm'
    before_conv = 'beforeconv'


class RunningNormalizer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.eps = 1e-5
        self.register_buffer('mean', th.zeros(num_channels))
        self.register_buffer('sqmean', th.zeros(num_channels))
        self.register_buffer('num_batches', th.tensor(0))

    @property
    def std(self):
        var = self.sqmean - self.mean.pow(2)
        return var.sqrt() + self.eps

    def mean_std(self, x):
        if x.dim() == 4:
            mean = self.mean[None, :, None, None]
            std = self.std[None, :, None, None]
        elif x.dim() == 2:
            mean = self.mean[None, :]
            std = self.std[None, :]
        else:
            raise NotImplementedError()

        return mean, std

    def forward(self, x):
        if self.training:
            with th.no_grad():
                if x.dim() == 4:
                    # (c, )
                    first = x.mean(dim=[0, 2, 3])
                    second = x.pow(2).mean(dim=[0, 2, 3])
                elif x.dim() == 2:
                    first = x.mean(dim=0)
                    second = x.pow(2).mean(dim=0)
                else:
                    raise NotImplementedError()
                self.num_batches += 1
                self.mean += 1 / self.num_batches * (first - self.mean)
                self.sqmean += 1 / self.num_batches * (second - self.sqmean)

        mean, std = self.mean_std(x)
        # print('mean:', mean)
        # print('var:', var)
        return (x - mean) / std

    def denormalize(self, x):
        mean, std = self.mean_std(x)
        return x * std + mean


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self,
                x,
                emb=None,
                cond=None,
                lateral=None,
                cond2=None,
                stylespace_cond=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self,
                x,
                emb=None,
                cond=None,
                lateral=None,
                cond2=None,
                stylespace_cond=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x,
                          emb=emb,
                          cond=cond,
                          lateral=lateral,
                          cond2=cond2,
                          stylespace_cond=stylespace_cond)
            else:
                x = layer(x)
        return x


@dataclass
class ResBlockConfig(BaseConfig):
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    # condition the resblock with time (and encoder's output)
    use_condition: bool = True
    # also condition on the in_layers pipeline (default: False)
    # hypothesis: stylegan has conditions on all conv layers, while the default UNET has conditions only on the out_layers pipeline
    use_inlayers_condition: bool = False
    # whether to use 3x3 conv for skip path when the channels aren't matched
    use_conv: bool = False
    # dimension of conv (always 2 = 2d)
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    # how to condition the feature maps:
    # - add
    # - mult + shift
    condition_type: ConditionType = ConditionType.scale_shift_norm
    # the shift should start from 1
    condition_scale_bias: float = 1
    # whether to condition with both time & encoder's output
    two_cond: bool = False
    # this is a test whether separating the encoder's output into many latents would help learn disentagled representations
    # in this case, one time + two encoder's output = 3 total
    three_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int = None
    cond2_emb_channels: int = None
    # whether to condition with time before encoder's output latent
    # suggest: True
    time_first: bool = False
    # whether to use both scale & shift for time condition
    time_emb_2xwidth: bool = True
    # whether to use both scale & shift for encoder's output condition
    cond_emb_2xwidth: bool = True
    # suggest: False
    has_lateral: bool = False
    # deprecated experiment
    gated: bool = False
    lateral_channels: int = None
    gate_type: GateType = None
    gate_init: float = None
    # whether to use modulated Conv like in StyleGAN2 (slower + uses more memory)
    use_styleconv: bool = False
    # whether to apply another normalization after the last convolution
    # suggest: False, it doesn't improve
    use_after_norm: bool = False
    # whether to init the convolution with zero weights
    # this is default from BeatGANs and seems to help learning
    use_zero_module: bool = True
    # where to scale & shift the feature maps
    # default is after normalization, but before the activation
    # it's also possible to do after activation
    # suggestion: after_norm, there is no difference
    scale_at: ScaleAt = ScaleAt.after_norm
    # only apply gradient chcekpoint to the scale & shift calculations
    # this is very cheap but reduces memory footprint by as much as 20% (for large models)
    use_checkpoint_gnscalesilu: bool = False

    def __post_init__(self):
        if self.gated:
            assert self.has_lateral
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels
        self.cond2_emb_channels = self.cond2_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """
    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf

        #############################
        # IN LAYERS
        #############################
        if conf.gated:
            assert not conf.use_inlayers_condition
            self.in_a = nn.Sequential(
                normalization(conf.channels),
                nn.SiLU(),
            )
            self.in_b = nn.Sequential(
                normalization(conf.lateral_channels),
                nn.SiLU(),
            )
            self.in_ab = GatedConv(conf.channels,
                                   conf.lateral_channels,
                                   conf.out_channels,
                                   3,
                                   padding=1,
                                   gate_type=conf.gate_type,
                                   gate_init=conf.gate_init)
        else:
            assert conf.lateral_channels is None
            layers = []
            if self.conf.use_checkpoint_gnscalesilu:
                # more memory efficient, this could be a lot (20%) for large models
                assert self.conf.condition_type == ConditionType.scale_shift_norm
                layers.append(CheckpointGNShiftScaleSiLU(32, conf.channels))
            else:
                layers += [
                    normalization(conf.channels),
                    nn.SiLU(),
                ]
            layers.append(
                conv_nd(conf.dims,
                        conf.channels,
                        conf.out_channels,
                        3,
                        padding=1))
            self.in_layers = nn.Sequential(*layers)
            # self.in_layers = nn.Sequential(
            #     normalization(conf.channels),
            #     nn.SiLU(),
            #     conv_nd(conf.dims,
            #             conf.channels,
            #             conf.out_channels,
            #             3,
            #             padding=1),
            # )

        self.updown = conf.up or conf.down

        if conf.up:
            self.h_upd = Upsample(conf.channels, False, conf.dims)
            self.x_upd = Upsample(conf.channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.channels, False, conf.dims)
            self.x_upd = Downsample(conf.channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # IN LAYERS CONDITIONS
        #############################
        if conf.use_inlayers_condition:
            self.emb_inlayers = nn.Sequential(
                nn.SiLU(),
                linear(
                    conf.emb_channels,
                    2 *
                    conf.channels if conf.time_emb_2xwidth else conf.channels,
                ),
            )

            if conf.two_cond:
                self.cond_emb_inlayers = nn.Sequential(
                    nn.SiLU(),
                    linear(
                        conf.cond_emb_channels,
                        2 * conf.channels
                        if conf.cond_emb_2xwidth else conf.channels,
                    ),
                )

            if conf.three_cond:
                self.cond2_emb_inlayers = nn.Sequential(
                    nn.SiLU(),
                    linear(
                        conf.cond2_emb_channels,
                        2 * conf.channels
                        if conf.cond_emb_2xwidth else conf.channels,
                    ),
                )

        #############################
        # OUT LAYERS CONDITIONS
        #############################
        if conf.use_condition:
            # condition layers for the out_layers
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    conf.emb_channels,
                    2 * conf.out_channels
                    if conf.time_emb_2xwidth else conf.out_channels,
                ),
            )

            if conf.two_cond:
                self.cond_emb_layers = nn.Sequential(
                    nn.SiLU(),
                    linear(
                        conf.cond_emb_channels,
                        2 * conf.out_channels
                        if conf.cond_emb_2xwidth else conf.out_channels,
                    ),
                )

            if conf.three_cond:
                self.cond2_emb_layers = nn.Sequential(
                    nn.SiLU(),
                    linear(
                        conf.cond2_emb_channels,
                        2 * conf.out_channels
                        if conf.cond_emb_2xwidth else conf.out_channels,
                    ),
                )

            #############################
            # OUT LAYERS (ignored when there is no condition)
            #############################
            if conf.use_styleconv:
                assert conf.use_styleconv
                assert conf.time_first
                # assert not conf.use_zero_module
                assert conf.condition_type in [
                    ConditionType.scale_shift_hybrid,
                    ConditionType.scale_shift_norm
                ]
                assert conf.scale_at == ScaleAt.before_conv
                assert not conf.cond_emb_2xwidth
                assert not conf.use_after_norm
                # use style conv
                conv = EqualizedConv2DMod(conf.out_channels,
                                          conf.out_channels,
                                          3,
                                          padding=1)
                if conf.use_zero_module:
                    conv = zero_module(conv)
                layers = [
                    normalization(conf.out_channels),
                    nn.SiLU(),
                    nn.Dropout(p=conf.dropout),
                    conv,
                ]
                self.out_layers = StyleConvSequential(*layers)
            else:
                # original version
                conv = conv_nd(conf.dims,
                               conf.out_channels,
                               conf.out_channels,
                               3,
                               padding=1)
                if conf.use_zero_module:
                    # zere out the weights
                    # it seems to help training
                    conv = zero_module(conv)

                # construct the layers
                # - norm
                # - (modulation)
                # - act
                # - dropout
                # - conv
                layers = []
                if self.conf.use_checkpoint_gnscalesilu:
                    # more memory efficient, this could be a lot (20%) for large models
                    assert self.conf.condition_type == ConditionType.scale_shift_norm
                    assert not self.conf.use_styleconv
                    layers.append(
                        CheckpointGNShiftScaleSiLU(32, conf.out_channels))
                else:
                    layers += [
                        normalization(conf.out_channels),
                        nn.SiLU(),
                    ]
                layers += [
                    nn.Dropout(p=conf.dropout),
                    conv,
                ]
                # self.out_layers = nn.Sequential(*layers)
                self.out_layers = StyleConvSequential(*layers)

        if conf.use_after_norm:
            # same as stylegan1 demodulation
            self.after_norm = nn.GroupNorm(conf.out_channels,
                                           conf.out_channels)
        else:
            self.after_norm = None

        #############################
        # SKIP LAYERS
        #############################
        if not conf.gated and conf.out_channels == conf.channels:
            # cannot be used with gatedconv, also gatedconv is alsways used as the first block
            self.skip_connection = nn.Identity()
        else:
            if conf.use_conv:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            if conf.gated:
                self.skip_connection = GatedConv(conf.channels,
                                                 conf.lateral_channels,
                                                 conf.out_channels,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 gate_type=conf.gate_type,
                                                 gate_init=conf.gate_init)
            else:
                self.skip_connection = conv_nd(conf.dims,
                                               conf.channels,
                                               conf.out_channels,
                                               kernel_size,
                                               padding=padding)

    def forward(self,
                x,
                emb=None,
                cond=None,
                lateral=None,
                cond2=None,
                stylespace_cond=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return torch_checkpoint(
            self._forward, (x, emb, cond, lateral, cond2, stylespace_cond),
            self.conf.use_checkpoint)
        # return checkpoint(self._forward, (x, emb), self.parameters(),
        #                   self.use_checkpoint)

    def _forward(
        self,
        x,
        emb=None,
        cond=None,
        lateral=None,
        cond2=None,
        stylespace_cond=None,
        # not used yet, just in case
        stylespace_cond_in=None,
    ):
        """
        Args:
            lateral: required if "has_lateral" and non-gated, with gated, it can be supplied optionally    
        """
        if self.conf.has_lateral:
            # lateral may be supplied even if it doesn't require
            # the model will take the lateral only if "has_lateral"
            if not self.conf.gated:
                assert lateral is not None
                x = th.cat([x, lateral], dim=1)

        if self.conf.gated:
            # deprecated not used in the final version
            assert not self.updown
            assert not self.conf.use_inlayers_condition, 'not support the condition at the gated layers yet'
            x = self.in_a(x)
            if lateral is not None:
                # lateral is not mandatory for gated
                # this allows separately training autoencoder from ddpm
                lateral = self.in_b(lateral)
            h = self.in_ab.forward(x, lateral)
        else:
            if self.conf.use_inlayers_condition:
                # apply condition to the input blocks as well
                # this is new to BeatGANs model

                # it's possible that the network may not receieve the time emb
                # this happens with autoenc and setting the time_at
                if emb is not None:
                    emb_in = self.emb_inlayers(emb).type(x.dtype)
                else:
                    emb_in = None

                if self.conf.two_cond:
                    # it's possible that the network is two_cond
                    # but it doesn't get the second condition
                    # in which case, we ignore the second condition
                    # and treat as if the network has one condition
                    if stylespace_cond_in is not None:
                        cond_in = stylespace_cond_in
                    else:
                        if cond is None:
                            cond_in = None
                        else:
                            cond_in = self.cond_emb_inlayers(cond).type(
                                x.dtype)

                    if cond_in is not None:
                        while len(cond_in.shape) < len(x.shape):
                            cond_in = cond_in[..., None]
                else:
                    cond_in = None

                if self.conf.three_cond:
                    cond2_in = self.cond2_emb_inlayers(cond2).type(x.dtype)
                    while len(cond2_in.shape) < len(x.shape):
                        cond2_in = cond2_in[..., None]
                else:
                    cond2_in = None

                h = apply_conditions(
                    h=x,
                    emb=emb_in,
                    cond=cond_in,
                    cond2=cond2_in,
                    layers=self.in_layers,
                    time_first=self.conf.time_first,
                    scale_bias=self.conf.condition_scale_bias,
                    condition_type=self.conf.condition_type,
                    scale_at=self.conf.scale_at,
                    in_channels=self.conf.channels,
                    use_styleconv=False,
                    use_checkpoint_gnscalesilu=self.conf.
                    use_checkpoint_gnscalesilu,
                    use_after_norm=False,
                    after_norm=None,
                    up_down_layer=self.h_upd,
                )
                # for the skip connection
                x = self.x_upd(x)
            else:
                if self.updown:
                    assert not self.conf.gated, 'not yet supported with gated'
                    in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                    h = in_rest(x)
                    h = self.h_upd(h)
                    x = self.x_upd(x)
                    h = in_conv(h)
                else:
                    h = self.in_layers(x)

        # legacy code
        # if self.updown:
        #     assert not self.conf.gated, 'not yet supported with gated'
        #     in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        #     h = in_rest(x)
        #     h = self.h_upd(h)
        #     x = self.x_upd(x)
        #     h = in_conv(h)
        # else:
        #     if self.conf.gated:
        #         assert not self.conf.use_inlayers_condition, 'not support the condition at the gated layers yet'
        #         x = self.in_a(x)
        #         if lateral is not None:
        #             # lateral is not mandatory for gated
        #             # this allows separately training autoencoder from ddpm
        #             lateral = self.in_b(lateral)
        #         h = self.in_ab.forward(x, lateral)
        #     else:
        #         h = self.in_layers(x)

        if self.conf.use_condition:
            # it's possible that the network may not receieve the time emb
            # this happens with autoenc and setting the time_at
            if emb is not None:
                emb_out = self.emb_layers(emb).type(h.dtype)
            else:
                emb_out = None

            if self.conf.two_cond:
                # it's possible that the network is two_cond
                # but it doesn't get the second condition
                # in which case, we ignore the second condition
                # and treat as if the network has one condition
                if stylespace_cond is not None:
                    cond_out = stylespace_cond
                else:
                    if cond is None:
                        cond_out = None
                    else:
                        cond_out = self.cond_emb_layers(cond).type(h.dtype)

                if cond_out is not None:
                    while len(cond_out.shape) < len(h.shape):
                        cond_out = cond_out[..., None]
            else:
                cond_out = None

            if self.conf.three_cond:
                cond2_out = self.cond2_emb_layers(cond2).type(h.dtype)
                while len(cond2_out.shape) < len(h.shape):
                    cond2_out = cond2_out[..., None]
            else:
                cond2_out = None

            # this is the new refactored code
            h = apply_conditions(
                h=h,
                emb=emb_out,
                cond=cond_out,
                cond2=cond2_out,
                layers=self.out_layers,
                time_first=self.conf.time_first,
                scale_bias=self.conf.condition_scale_bias,
                condition_type=self.conf.condition_type,
                scale_at=self.conf.scale_at,
                in_channels=self.conf.out_channels,
                use_styleconv=self.conf.use_styleconv,
                use_checkpoint_gnscalesilu=self.conf.
                use_checkpoint_gnscalesilu,
                use_after_norm=self.conf.use_after_norm,
                after_norm=self.after_norm,
                up_down_layer=None,
            )

        if self.conf.gated:
            # there is no identity connection with gated
            return self.skip_connection.forward(x, lateral) + h
        else:
            return self.skip_connection(x) + h


def apply_conditions(
    h,
    emb=None,
    cond=None,
    cond2=None,
    layers: nn.Sequential = None,
    time_first: bool = True,
    scale_bias: float = 1,
    condition_type: ConditionType = ConditionType.scale_shift_norm,
    scale_at: ScaleAt = ScaleAt.after_norm,
    in_channels: int = 512,
    use_styleconv: bool = False,
    use_checkpoint_gnscalesilu: bool = False,
    use_after_norm: bool = False,
    after_norm: nn.Module = None,
    up_down_layer: nn.Module = None,
):
    """
    apply conditions on the feature maps

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
        cond2: second encoder's conditional (ready to scale + shift)
    """
    two_cond = emb is not None and cond is not None
    three_cond = two_cond and cond2 is not None

    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]

    if two_cond:
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]

        if time_first:
            scale_shifts = [emb, cond]
        else:
            scale_shifts = [cond, emb]
    else:
        # "cond" is not used with single cond mode
        scale_shifts = [emb]

    if three_cond:
        while len(cond2.shape) < len(h.shape):
            cond2 = cond2[..., None]
        scale_shifts.append(cond2)

    # support scale, shift or shift only
    for i, each in enumerate(scale_shifts):
        if each is None:
            # special case: the condition is not provided
            a = None
            b = None
        else:
            if each.shape[1] == in_channels * 2:
                a, b = th.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
        scale_shifts[i] = (a, b)

    # condition scale bias could be a list
    if isinstance(scale_bias, Number):
        biases = [scale_bias] * len(scale_shifts)
    else:
        # a list
        biases = scale_bias

    # split layers at the point of conditon usually after norm
    if scale_at == ScaleAt.after_norm:
        # default, the scale & shift are applied after the group norm but BEFORE SiLU
        pre_layers, post_layers = layers[0], layers[1:]
    elif scale_at == ScaleAt.before_conv:
        n_layers = len(layers)
        pre_layers, post_layers = layers[:n_layers - 1], layers[n_layers - 1:]
    else:
        raise NotImplementedError()

    # spilt the post layer to be able to scale up or down before conv
    # post layers will contain only the conv
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    # if not used the styleconv, it will remain None
    style_cond = None

    if condition_type == ConditionType.scale_shift_norm:
        # default conditioning method!

        if use_checkpoint_gnscalesilu:
            assert not three_cond
            assert after_norm is None, 'does not support after norm'
            scale1, shift1 = scale_shifts[0]
            if len(scale_shifts) > 1:
                scale2, shift2 = scale_shifts[1]
            else:
                scale2, shift2 = None, None
            h = pre_layers(h, scale1, shift1, scale2, shift2)
        else:
            h = pre_layers(h)

        if use_styleconv:
            # styleconv = using the modulated convolution as in StyleGAN
            # it's slow, not used in the final paper
            assert not three_cond
            # the first cond (time) multiplies
            scale, shift = scale_shifts[0]
            if scale is not None:
                h = h * (biases[0] + scale)
                if shift is not None:
                    h = h + shift
            # the second (style) use styleconv which means also demod
            # supply this to the styleconv
            style_cond, _ = scale_shifts[1] + biases[1]
        else:
            if not use_checkpoint_gnscalesilu:
                # ignore this if use the checkpoint version
                # scale and shift for each condition
                for i, (scale, shift) in enumerate(scale_shifts):
                    # if scale is None, it indicates that the condition is not provided
                    if scale is not None:
                        h = h * (biases[i] + scale)
                        if shift is not None:
                            h = h + shift
        h = mid_layers(h)
    elif condition_type == ConditionType.scale_shift_hybrid:
        # time condition adds
        # encoder's condition scales
        assert not three_cond
        shift, _ = scale_shifts[0]
        # if shift is None, the condition is not provided, ignore
        if shift is not None:
            h = h + shift
        h = pre_layers(h)
        # only works with two conditions
        # scale and shift for each condition
        # scale (C, 1, 1), shift (C, 1, 1) <=== z (512,) <== encoder
        scale, shift = scale_shifts[1]
        # if scale is None, the condition is not provided
        if scale is not None:
            if use_styleconv:
                # supply this to the styleconv
                style_cond = (biases[1] + scale).flatten(1)
            else:
                h = h * (biases[1] + scale)
                if shift is not None:
                    h = h + shift
        h = mid_layers(h)
    elif condition_type == ConditionType.add:
        for (shift, _) in scale_shifts:
            # if shift is None, the condition is not provided
            if shift is not None:
                h = h + shift
        h = pre_layers(h)
        h = mid_layers(h)
    else:
        raise NotImplementedError()

    # upscale or downscale if any just before the last conv
    if up_down_layer is not None:
        h = up_down_layer(h)
    # last conv layer
    if style_cond is not None:
        # when the conv layer is modulated conv
        h = post_layers(h, cond=style_cond)
    else:
        h = post_layers(h)
    if use_after_norm and scale is not None:
        # only applies when there is a scaling operation
        h = after_norm(h)
    return h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x, ), self.use_checkpoint)
        # return checkpoint(self._forward, (x, ), self.parameters(),
        #                   self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]
