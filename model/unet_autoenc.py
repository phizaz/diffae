from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .noisenet import *
from .unet import *
from choices import *


class TimeMode(Enum):
    # style only
    time_style_separate = 'timestylesep'


class VectorizerType(Enum):
    identity = 'identity'


class MergerType(Enum):
    conv1 = 'conv1'
    conv3 = 'conv3'
    block = 'block'


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    use_external_encoder: bool = False
    is_stochastic: bool = False
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_pool_tail_layer: int = None
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    enc_tanh: bool = False
    style_time_mode: TimeMode = None 
    # unconditioned style layers
    style_layer: int = 8
    # film-layers conditioned on time
    time_style_layer: int = 2
    style_lr_mul: float = 0.1
    vectorizer_type: VectorizerType = VectorizerType.identity
    time_at: CondAt = CondAt.all
    cond_at: CondAt = CondAt.all
    has_init: bool = False
    merger_type: MergerType = MergerType.conv1
    latent_net_conf: MLPSkipNetConfig = None
    noise_net_conf: NoiseNetConfig = None

    @property
    def name(self):
        name = super().name
        if self.is_stochastic:
            name = name.replace('netbeatgans', 'vaebeatgans')
        else:
            name = name.replace('netbeatgans', 'autoencbeatgans')

        if not self.use_external_encoder:
            name += f'-pool{self.enc_pool}'
            if self.enc_pool == 'adaptivenonzerotail':
                name += f'-tail{self.enc_pool_tail_layer}'
            name += f'-ch{self.enc_out_channels}'
            if self.enc_num_res_block != 2:
                name += f'-resblk{self.enc_num_res_block}'
            if self.enc_channel_mult is not None:
                name += '-encch(' + ','.join(
                    str(x) for x in self.enc_channel_mult) + ')'
            if self.enc_attn_resolutions is not None:
                name += '-encatt(' + ','.join(
                    str(x) for x in self.enc_attn_resolutions) + ')'
            if self.enc_tanh:
                name += '-tanh'
        else:
            name += '-extenc'

        if self.time_at != CondAt.all:
            name += f'-timeat{self.time_at.value}'
        if self.cond_at != CondAt.all:
            name += f'-at{self.cond_at.value}'
        if self.has_init:
            name += f'-init-merge{self.merger_type.value}'

        name += f'/{self.style_time_mode.value}'
        name += f'-identity'

        # elif self.style_time_mode == TimeMode.time_and_style:
        #     name += f'-layer{self.style_layer}lr{self.style_lr_mul}'
        # elif self.style_time_mode == TimeMode.time_varying_style:
        #     name += f'-layer{self.time_style_layer}+{self.style_layer}lr{self.style_lr_mul}'
        # elif self.style_time_mode == TimeMode.time_cond_is_style:
        #     name += f'-layer{self.style_layer}lr{self.style_lr_mul}'
        # elif self.style_time_mode == TimeMode.time_cond_is_style_concat:
        #     name += f'-layer{self.style_layer}lr{self.style_lr_mul}'
        # else:
        #     raise NotImplementedError()

        if self.latent_net_conf is not None:
            name += f'-latent{self.latent_net_conf.name}'
        if self.noise_net_conf is not None:
            name += f'-{self.noise_net_conf.name}'

        return name

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        if conf.style_time_mode == TimeMode.time_style_separate:
            self.time_embed = TimeStyleSeperateEmbed(
                time_channels=conf.model_channels,
                time_out_channels=conf.embed_channels,
                cond_channels=conf.enc_out_channels,
                out_channels=conf.embed_channels,
                num_layer=conf.style_layer,
                lr_mul=conf.style_lr_mul,
                vectorizer_type=conf.vectorizer_type,
            )
        elif conf.style_time_mode == TimeMode.time_style_time_separate:
            self.time_embed = TimeStyleTimeEmbed(
                time_channels=conf.model_channels,
                time_out_channels=conf.embed_channels,
                cond_channels=conf.enc_out_channels,
                num_layer=conf.style_layer,
            )
        elif conf.style_time_mode == TimeMode.time_style_time_residual_separate:
            self.time_embed = TimeStyleTimeEmbed(
                time_channels=conf.model_channels,
                time_out_channels=conf.embed_channels,
                cond_channels=conf.enc_out_channels,
                num_layer=conf.style_layer,
            )
        elif conf.style_time_mode == TimeMode.time_and_style:
            self.time_embed = TimeAndStyleEmbed(
                time_channels=conf.model_channels,
                time_out_channels=conf.embed_channels,
                cond_channels=conf.enc_out_channels,
                out_channels=conf.embed_channels,
                num_layer=conf.style_layer,
                lr_mul=conf.style_lr_mul,
                vectorizer_type=conf.vectorizer_type,
            )
        elif conf.style_time_mode == TimeMode.time_varying_style:
            self.time_embed = TimeVaryingStyleEmbed(
                time_channels=conf.model_channels,
                time_out_channels=conf.embed_channels,
                cond_channels=conf.enc_out_channels,
                out_channels=conf.embed_channels,
                num_big_layer=conf.time_style_layer,
                num_layer=conf.style_layer,
                lr_mul=conf.style_lr_mul,
                vectorizer_type=conf.vectorizer_type,
            )
        elif conf.style_time_mode == TimeMode.time_cond_is_style:
            self.time_embed = TimeCondIsStyleEmbed(
                time_channels=conf.model_channels,
                time_out_channels=conf.embed_channels,
                cond_channels=conf.enc_out_channels,
                out_channels=conf.embed_channels,
                num_layer=conf.style_layer,
                lr_mul=conf.style_lr_mul,
                vectorizer_type=conf.vectorizer_type,
            )
        elif conf.style_time_mode == TimeMode.time_cond_is_style_concat:
            self.time_embed = TimeCondIsStyleConcatEmbed(
                time_channels=conf.model_channels,
                time_out_channels=conf.embed_channels,
                cond_channels=conf.enc_out_channels,
                out_channels=conf.embed_channels,
                num_layer=conf.style_layer,
                lr_mul=conf.style_lr_mul,
                vectorizer_type=conf.vectorizer_type,
            )
        else:
            raise NotImplementedError()

        if not conf.use_external_encoder:
            self.encoder = BeatGANsEncoderConfig(
                image_size=conf.image_size,
                in_channels=conf.in_channels,
                model_channels=conf.model_channels,
                out_hid_channels=conf.enc_out_channels,
                out_channels=(conf.enc_out_channels * 2 if conf.is_stochastic
                              else conf.enc_out_channels),
                num_res_blocks=conf.enc_num_res_block,
                attention_resolutions=(conf.enc_attn_resolutions
                                       or conf.attention_resolutions),
                dropout=conf.dropout,
                channel_mult=conf.enc_channel_mult or conf.channel_mult,
                use_time_condition=False,
                conv_resample=conf.conv_resample,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
                use_fp16=conf.use_fp16,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                resblock_updown=conf.resblock_updown,
                use_new_attention_order=conf.use_new_attention_order,
                pool=conf.enc_pool,
                pool_tail_layer=conf.enc_pool_tail_layer,
                last_act=Activation.tanh if conf.enc_tanh else Activation.none,
            ).make_model()

        if conf.has_init:
            # init block this allows the autoencoder to be standalone
            # not used in the final version
            highest_ch = conf.channel_mult[-1] * conf.model_channels
            self.initial = nn.Parameter(torch.randn((1, highest_ch, 4, 4)))
            self.initial_conv = nn.Conv2d(highest_ch, highest_ch, 3, padding=1)

            # start from 4x4 upto the the lowest resolution of the unet
            # these layers do not have lateral connections
            # layers before the connections of Unet's encoder
            lowest_resolution = conf.image_size // 2**(len(conf.channel_mult) -
                                                       1)
            print('lowest resolution:', lowest_resolution)
            resolution = 4

            kwargs = dict(
                condition_type=conf.resnet_condition_type,
                condition_scale_bias=conf.resnet_condition_scale_bias,
                two_cond=conf.resnet_two_cond,
                time_first=conf.resnet_time_first,
                time_emb_2xwidth=conf.resnet_time_emb_2xwidth,
                cond_emb_2xwidth=conf.resnet_cond_emb_2xwidth,
                # gates are used in the merger
                gate_type=conf.resnet_gate_type,
                gate_init=conf.resnet_gate_init,
            )

            # to merge the Unet's encoder trunk with the pre blocks
            if conf.merger_type == MergerType.block:
                self.merger = ResBlockConfig(
                    # only direct channels when gated
                    channels=highest_ch,
                    emb_channels=conf.embed_channels,
                    dropout=conf.dropout,
                    out_channels=highest_ch,
                    dims=conf.dims,
                    use_checkpoint=conf.use_checkpoint,
                    # lateral is from the main trunk of the Unet's encoder
                    has_lateral=True,
                    lateral_channels=highest_ch,
                    gated=True,
                    **kwargs,
                ).make_model()
            else:
                if conf.merger_type == MergerType.conv1:
                    kernel = 1
                    padding = 0
                elif conf.merger_type == MergerType.conv3:
                    kernel = 3
                    padding = 1
                else:
                    raise NotImplementedError()

                self.merger = GatedConv(
                    highest_ch,
                    highest_ch,
                    highest_ch,
                    kernel_size=kernel,
                    padding=padding,
                    gate_type=conf.resnet_gate_type,
                    gate_init=conf.resnet_gate_init,
                    has_conv_a=False,
                )

            self.pre_blocks = nn.ModuleList([])
            while resolution < lowest_resolution:
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=highest_ch,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=highest_ch,
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            highest_ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                # increase the resolution
                resolution *= 2
                layers.append(
                    ResBlockConfig(
                        highest_ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=highest_ch,
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        up=True,
                        **kwargs,
                    ).make_model() if (
                        conf.resblock_updown
                    ) else Upsample(highest_ch,
                                    conf.conv_resample,
                                    dims=conf.dims,
                                    out_channels=highest_ch))
                self.pre_blocks.append(TimestepEmbedSequential(*layers))

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

        if conf.noise_net_conf is not None:
            self.noise_net = conf.noise_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        assert not self.conf.use_external_encoder
        cond = self.encoder.forward(x)
        return {'cond': cond}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                stylespace_cond=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        if t_cond is None:
            t_cond = t

        mu, logvar = None, None

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if stylespace_cond is not None:
            stylespace_cond = list(
                torch.split(stylespace_cond, self.stylespace_sizes, dim=1))
            cond = None
        else:
            if cond is None:
                if self.conf.is_stochastic:
                    if x_start is None:
                        # sample if don't have either x_start and cond
                        cond = self.sample_z(len(x), device=x.device)
                    else:
                        # (n, c*2)
                        tmp = self.encoder.forward(x_start)
                        # (n, c), (n, c)
                        mu, logvar = torch.chunk(tmp, 2, dim=1)
                        cond = self.reparameterize(mu, logvar)
                else:
                    if x is not None:
                        assert len(x) == len(
                            x_start), f'{len(x)} != {len(x_start)}'

                    tmp = self.encode(x_start)
                    cond = tmp['cond']

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        if self.conf.time_at == CondAt.all:
            enc_time_emb = emb
            mid_time_emb = emb
            dec_time_emb = emb
        elif self.conf.time_at == CondAt.enc:
            enc_time_emb = emb
            mid_time_emb = None
            dec_time_emb = None
        else:
            raise NotImplementedError()

        if self.conf.cond_at == CondAt.all:
            enc_cond_emb = cond_emb
            mid_cond_emb = cond_emb
            dec_cond_emb = cond_emb
        elif self.conf.cond_at == CondAt.mid_dec:
            enc_cond_emb = None
            mid_cond_emb = cond_emb
            dec_cond_emb = cond_emb
        elif self.conf.cond_at == CondAt.dec:
            enc_cond_emb = None
            mid_cond_emb = None
            dec_cond_emb = cond_emb
        else:
            raise NotImplementedError()

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    if stylespace_cond is None:
                        h = self.input_blocks[k](h,
                                                 emb=enc_time_emb,
                                                 cond=enc_cond_emb)
                    else:
                        if i == 0 and j == 0:
                            # the first block is just a conv not resblock
                            h = self.input_blocks[k](h)
                        else:
                            h = self.input_blocks[k](
                                h,
                                emb=enc_time_emb,
                                stylespace_cond=stylespace_cond.pop(0))

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            if stylespace_cond is None:
                h = self.middle_block(h,
                                      emb=mid_time_emb,
                                      cond=mid_cond_emb)
            else:
                for each in self.middle_block:
                    if isinstance(each, ResBlock):
                        h = each(h,
                                 emb=mid_time_emb,
                                 stylespace_cond=stylespace_cond.pop(0))
                    else:
                        h = each(h)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        if self.conf.has_init:
            n = len(h) if h is not None else len(x_start)
            # (n, c, 4, 4)
            init = self.initial.repeat(n, 1, 1, 1)
            init = self.initial_conv(init)
            # (n, c, h, w)
            for module in self.pre_blocks:
                init = module(init, emb=dec_time_emb, cond=dec_cond_emb)
            # merge with the trunk (can work with None h)
            if self.conf.merger_type == MergerType.block:
                h = self.merger.forward(init,
                                        emb=dec_time_emb,
                                        cond=dec_cond_emb,
                                        lateral=h)
            else:
                h = self.merger.forward(a=init, b=h)

        # output blocks
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

                if stylespace_cond is None:
                    h = self.output_blocks[k](h,
                                              emb=dec_time_emb,
                                              cond=dec_cond_emb,
                                              lateral=lateral)
                else:
                    if isinstance(each, ResBlock):
                        h = self.output_blocks[k](
                            h,
                            emb=dec_time_emb,
                            lateral=lateral,
                            stylespace_cond=stylespace_cond.pop(0))
                    else:
                        h = self.output_blocks[k](h)
                k += 1

        if stylespace_cond is not None:
            assert len(stylespace_cond) == 0

        # h = h.type(x.dtype)
        pred = self.out(h)

        if self.conf.is_stochastic:
            return AutoencReturn(pred=pred,
                                 cond=cond,
                                 cond_mu=mu,
                                 cond_logvar=logvar)
        else:
            return AutoencReturn(pred=pred, cond=cond)


@dataclass
class LatentGenerativeModelConfig(BeatGANsAutoencConfig):
    num_vec_layer: int = 8
    vec_lr: float = 1

    @property
    def name(self):
        name = super().name
        name += f'_latentgen-layer{self.num_vec_layer}lr{self.vec_lr}'
        return name

    def make_model(self):
        return LatentGenerativeModel(self)


class LatentGenerativeModel(BeatGANsAutoencModel):
    def __init__(self, conf: LatentGenerativeModelConfig):
        super().__init__(conf)
        self.conf = conf
        self.vectorizer = BeatGANsStyleVectorizer(conf.enc_out_channels,
                                                  conf.enc_out_channels,
                                                  conf.num_vec_layer,
                                                  lr_mul=conf.vec_lr,
                                                  pixel_norm=True)

    def noise_to_cond(self, noise: Tensor):
        return self.vectorizer.forward(noise)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None
    cond_mu: Tensor = None
    cond_logvar: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None
    style2: Tensor = None


class TimeEmbed(nn.Module):
    # no condition
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(in_channels, out_channels),
            nn.SiLU(),
            linear(out_channels, out_channels),
        )

    def forward(self, time_emb, cond=None):
        return EmbedReturn(emb=self.time_embed(time_emb))


class TimeTwoStyleSeperateEmbed(nn.Module):
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        return EmbedReturn(emb=cond,
                           time_emb=time_emb,
                           style=cond)


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels, cond_channels,
                 out_channels, num_layer, lr_mul,
                 vectorizer_type: VectorizerType):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

        if vectorizer_type == VectorizerType.identity:
            self.style = nn.Identity()
        else:
            raise NotImplementedError()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)


class TimeStyleTimeEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels, cond_channels,
                 num_layer):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

        self.style = MLPNetConfig(
            num_channels=cond_channels,
            num_hid_channels=cond_channels,
            num_layers=num_layer,
            num_time_emb_channels=time_channels,
            activation=Activation.silu,
            use_norm=False,
            condition_type=ConditionType.scale_shift_norm,
            condition_2x=False,
            condition_bias=1,
            dropout=0,
            last_act=Activation.silu,
            time_is_int=False).make_model()

    def forward(self, time_emb=None, cond=None, time_cond_emb=None):
        """
        Args:
            time_cond_emb: embedding time for the condition (it may not be the same as time_emb)
        """
        style = self.style.forward(cond, t=time_cond_emb).pred
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)


class TimeStyleTimeResidualEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels, cond_channels,
                 num_layer):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

        self.style = MLPSkipNetConfig(
            num_channels=cond_channels,
            num_hid_channels=cond_channels,
            num_layers=num_layer,
            num_time_emb_channels=time_channels,
            activation=Activation.silu,
            use_norm=False,
            condition_type=ConditionType.scale_shift_norm,
            condition_2x=False,
            condition_bias=1,
            dropout=0,
            last_act=Activation.silu,
            time_is_int=False).make_model()

    def forward(self, time_emb=None, cond=None, time_cond_emb=None):
        """
        Args:
            time_cond_emb: embedding time for the condition (it may not be the same as time_emb)
        """
        style = self.style.forward(cond, t=time_cond_emb).pred
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)


class TimeAndStyleEmbed(nn.Module):
    # time and style are independent only concat and project to the right width
    def __init__(self, time_channels, time_out_channels, cond_channels,
                 out_channels, num_layer, lr_mul,
                 vectorizer_type: VectorizerType):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        if vectorizer_type == VectorizerType.new:
            self.style = BeatGANsStyleVectorizer(cond_channels, out_channels,
                                                 num_layer, lr_mul)
        elif vectorizer_type == VectorizerType.old:
            self.style = OldVectorizer(cond_channels, out_channels, num_layer,
                                       lr_mul)
        else:
            raise NotImplementedError()
        self.out = nn.Linear(out_channels + time_out_channels, out_channels)

    def forward(self, time_emb, cond=None):
        time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        emb = self.out(torch.cat([time_emb, style], dim=1))
        return EmbedReturn(emb=emb, time_emb=time_emb, style=style)


class TimeVaryingStyleEmbed(nn.Module):
    # the condition is modulated with time vector, cond_t
    # cond_t is used to derive style_t
    def __init__(self, time_channels, time_out_channels, cond_channels,
                 out_channels, num_big_layer, num_layer, lr_mul,
                 vectorizer_type: VectorizerType):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

        big_mlp = []
        for i in range(num_big_layer):
            big_mlp.append(
                FiLMLinear(in_channels=cond_channels,
                           out_channels=cond_channels,
                           cond_channels=time_out_channels,
                           activation=True,
                           norm=True))
        self.big_mlp = FiLMSequential(*big_mlp)

        if vectorizer_type == VectorizerType.new:
            self.style = BeatGANsStyleVectorizer(cond_channels, out_channels,
                                                 num_layer, lr_mul)
        elif vectorizer_type == VectorizerType.old:
            self.style = OldVectorizer(cond_channels, out_channels, num_layer,
                                       lr_mul)
        else:
            raise NotImplementedError()

        self.out = nn.Linear(out_channels + time_out_channels, out_channels)

    def forward(self, time_emb, cond=None):
        time_emb = self.time_embed(time_emb)
        cond_t = self.big_mlp.forward(cond, time_emb)
        style_t = self.style(cond_t)
        emb = self.out(torch.cat([time_emb, style_t], dim=1))
        return EmbedReturn(emb=emb, time_emb=time_emb, style=style_t)


class TimeCondIsStyleEmbed(nn.Module):
    # concat(time + cond) => style = emb
    def __init__(self, time_channels, time_out_channels, cond_channels,
                 out_channels, num_layer, lr_mul,
                 vectorizer_type: VectorizerType):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

        if vectorizer_type == VectorizerType.new:
            self.style = BeatGANsStyleVectorizer(
                cond_channels + time_out_channels, out_channels, num_layer,
                lr_mul)
        elif vectorizer_type == VectorizerType.old:
            self.style = OldVectorizer(cond_channels + time_out_channels,
                                       out_channels, num_layer, lr_mul)
        else:
            raise NotImplementedError()

    def forward(self, time_emb, cond=None):
        time_emb = self.time_embed(time_emb)
        style = self.style(torch.cat([time_emb, cond], dim=1))
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)


class TimeCondIsStyleConcatEmbed(nn.Module):
    # seems to perform best!
    # concat(time + cond) => style => concat(style + time) => emb
    def __init__(self, time_channels, time_out_channels, cond_channels,
                 out_channels, num_layer, lr_mul,
                 vectorizer_type: VectorizerType):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

        if vectorizer_type == VectorizerType.new:
            self.style = BeatGANsStyleVectorizer(
                cond_channels + time_out_channels, out_channels, num_layer,
                lr_mul)
        elif vectorizer_type == VectorizerType.old:
            self.style = OldVectorizer(cond_channels + time_out_channels,
                                       out_channels, num_layer, lr_mul)
        else:
            raise NotImplementedError()

        self.out = nn.Linear(out_channels + time_out_channels, out_channels)

    def forward(self, time_emb, cond=None):
        time_emb = self.time_embed(time_emb)
        style = self.style(torch.cat([time_emb, cond], dim=1))
        emb = self.out(torch.cat([time_emb, style], dim=1))
        return EmbedReturn(emb=emb, time_emb=time_emb, style=style)


class FiLMSequential(nn.Sequential):
    def forward(self, input, cond):
        for module in self:
            input = module(input, cond)
        return input


class FiLMLinear(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, activation,
                 norm):
        super().__init__()
        self.norm = norm

        self.linear = nn.Linear(in_channels, out_channels)
        self.cond_linear = nn.Linear(cond_channels, out_channels * 2)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x, cond):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        tmp = self.cond_linear(cond)
        scale, shift = th.chunk(tmp, 2, dim=1)
        x = x * scale + shift
        x = self.activation(x)
        return x
