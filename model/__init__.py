from typing import Union
from .model import UNet, UNetConfig
from .model_our import StyleUNet, StyleUNetConfig
from .unet import BeatGANsUNetModel, BeatGANsUNetConfig
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel

Model = Union[StyleUNet, UNet, BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[StyleUNetConfig, UNetConfig, BeatGANsUNetConfig,
                    BeatGANsAutoencConfig]
