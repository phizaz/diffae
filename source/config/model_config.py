from dataclasses import dataclass
from typing import Tuple, Optional

from model.unet import ScaleAt
from model.latentnet import *
from choices import *

@dataclass
class ModelConfigBase:
    """Base class for all model configurations"""
    pass

@dataclass
class BeatGANsUNetConfig(ModelConfigBase):
    attention_resolutions: Tuple[int]
    channel_mult: Tuple[int]
    conv_resample: bool
    dims: int
    dropout: float
    embed_channels: int
    image_size: int
    in_channels: int
    model_channels: int
    num_classes: Optional[int]
    num_head_channels: int
    num_heads_upsample: int
    num_heads: int
    num_res_blocks: int
    num_input_res_blocks: Optional[int]
    out_channels: int
    resblock_updown: bool
    use_checkpoint: bool
    use_new_attention_order: bool
    resnet_two_cond: bool
    resnet_use_zero_module: bool
    resnet_cond_channels: Optional[int] = None

@dataclass
class MLPSkipNetConfig(ModelConfigBase):
    num_channels: int
    skip_layers: Tuple[int]
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int
    activation: Activation
    use_norm: bool
    condition_bias: float
    dropout: float
    last_act: Activation
    num_time_layers: int
    time_last_act: bool

@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    enc_out_channels: int
    enc_pool: str
    enc_num_res_block: int
    enc_channel_mult: Tuple[int]
    enc_grad_checkpoint: bool
    enc_attn_resolutions: Tuple[int]
    latent_net_conf: Optional[MLPSkipNetConfig] = None
