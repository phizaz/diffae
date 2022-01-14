from experiment import *


@dataclass
class ddpm(TrainConfig):
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beta_scheduler: str = 'linear'
    batch_size: int = 32
    sample_size: int = 32
    data_name: str = 'ffhq'
    diffusion_type: str = 'beatgans'
    fp16: bool = True
    lr: float = 1e-4
    model_name: ModelName = ModelName.beatgans_ddpm
    net_attn: Tuple[int] = (16, )
    net_beatgans_attn_head: int = 1
    net_beatgans_scale_shift_norm: bool = True
    net_beatgans_embed_channels: int = 512
    net_ch_mult: Tuple[int] = (1, 2, 4, 8)
    net_ch: int = 64
    T_eval: int = 20
    T: int = 1000


@dataclass
class autoenc_base(TrainConfig):
    postfix: str = '_lightning'
    batch_size: int = 32
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beta_scheduler: str = 'linear'
    data_name: str = 'ffhq'
    diffusion_type: str = 'beatgans'
    eval_ema_every_samples: int = 200_000
    eval_every_samples: int = 200_000
    fp16: bool = True
    lr: float = 1e-4
    model_name: ModelName = ModelName.beatgans_autoenc
    net_attn: Tuple[int] = (16, )
    net_beatgans_attn_head: int = 1
    net_beatgans_embed_channels: int = 512
    net_beatgans_enc_out_channels: int = 2048
    net_beatgans_enc_out_channels: int = 512
    net_beatgans_resnet_cond_emb_2xwidth: bool = False
    net_beatgans_resnet_condition_scale_bias: float = 0
    net_beatgans_resnet_time_emb_2xwidth: bool = False
    net_beatgans_resnet_time_first: bool = True
    net_beatgans_resnet_two_cond: bool = True
    net_beatgans_scale_shift_norm: bool = True
    net_beatgans_style_layer: int = 8
    net_beatgans_style_lr_mul: float = 0.1
    net_beatgans_style_time_mode: TimeMode = TimeMode.time_style_separate
    net_ch_mult: Tuple[int] = (1, 2, 4, 8)
    net_ch: int = 64
    net_enc_channel_mult: Tuple[int] = (1, 2, 4, 8)
    net_enc_pool: str = 'adaptivenonzero'
    net_enc_vectorizer_type: VectorizerType = None
    sample_size: int = 32
    T_eval: int = 20
    T: int = 1000



def encoder128(conf: TrainConfig):
    conf.model_name = ModelName.beatgans_encoder
    conf.net_ch = 128
    conf.net_enc_channel_mult = [1, 1, 2, 3, 4]
    conf.net_enc_num_res_blocks = 2
    conf.net_enc_attn = [16]
    conf.dropout = 0.1
    conf.net_enc_pool = 'attention'
    return conf


def ffhq64_autoenc_12M():
    conf = autoenc_base()
    conf.postfix = '_fixseed'
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 12_000_000
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_vectorizer_type = VectorizerType.identity
    conf.net_beatgans_resnet_condition_scale_bias = 1
    conf.net_beatgans_resnet_time_emb_2xwidth = True
    conf.net_beatgans_resnet_cond_emb_2xwidth = False
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    return conf


def ffhq64_autoenc_48M():
    conf = ffhq64_autoenc_12M()
    conf.total_samples = 48_000_000
    conf.continue_from = PretrainConfig(
        name='12M',
        path=f'logs/{ffhq64_autoenc_12M().name}/last.ckpt',
    )
    return conf


def at64_default(conf: TrainConfig):
    conf.total_samples = 48_000_000
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    return conf


def no_attn(conf: TrainConfig):
    conf.net_attn = []
    conf.net_enc_attn = []
    conf.net_beatgans_use_mid_attn = False
    return conf


def mid_attn(conf: TrainConfig):
    conf.net_attn = []
    conf.net_enc_attn = []
    return conf


def celeba64_autoenc():
    conf = ffhq64_autoenc_12M()
    conf.postfix = ''
    conf.data_name = 'celebaalignlmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    return conf


def celeba64d2c_autoenc():
    conf = celeba64_autoenc()
    conf.data_name = 'celebalmdb'
    return conf


def celeba64_ddpm():
    conf = ffhq64_ddpm_12M()
    conf.postfix = ''
    conf.data_name = 'celebaalignlmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    return conf


def celeba64d2c_ddpm():
    conf = celeba64_ddpm()
    conf.data_name = 'celebalmdb'
    return conf


def ffhq_heldout(conf: TrainConfig):
    conf.postfix = ''
    conf.continue_from = None
    conf.data_name = 'ffhqlmdbsplit256'
    return conf


def set_style_size(conf: TrainConfig, size):
    conf.style_ch = size
    conf.net_beatgans_resnet_cond_channels = size
    return conf


def ffhq64_autoenc_heldout():
    conf = ffhq64_autoenc_48M()
    conf = ffhq_heldout(conf)
    return conf


def ffhq64_autoenc_heldout_style256():
    conf = ffhq64_autoenc_heldout()
    conf = set_style_size(conf, 256)
    return conf


def ffhq64_autoenc_heldout_style128():
    conf = ffhq64_autoenc_heldout()
    conf = set_style_size(conf, 128)
    return conf


def ffhq64_autoenc_three_cond():
    conf = ffhq64_autoenc_48M()
    conf.postfix = ''
    conf.net_beatgans_resnet_cond_channels = 512 - 128
    conf.net_beatgans_three_cond = True
    conf.continue_from = None
    return conf


def ffhq64_ddpm_12M():
    conf = ddpm_adain_bias()
    conf.postfix = '_fixseed'
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 12_000_000
    conf.scale_up_gpus(4)
    return conf


def ffhq64_ddpm_48M():
    conf = ffhq64_ddpm_12M()
    conf.total_samples = 48_000_000
    conf.continue_from = PretrainConfig(
        name='12M',
        path=f'logs/{ffhq64_ddpm_12M().name}/last.ckpt',
    )
    return conf


def ffhq128_autoenc():
    conf = ffhq64_autoenc_12M()
    conf.postfix = '_fixseed'
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    return conf


def ffhq256_autoenc():
    conf = ffhq128_autoenc()
    conf.postfix = ''
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    return conf


def ffhq128_autoenc_heldout():
    conf = ffhq128_autoenc()
    conf = ffhq_heldout(conf)
    return conf


def ffhq128_autoenc_heldout_style256():
    conf = ffhq128_autoenc_heldout()
    conf = set_style_size(conf, 256)
    return conf


def ffhq128_autoenc_heldout_style128():
    conf = ffhq128_autoenc_heldout()
    conf = set_style_size(conf, 128)
    return conf


def ffhq128_autoenc_heldout_style64():
    conf = ffhq128_autoenc_heldout()
    conf = set_style_size(conf, 64)
    return conf


def ffhq128_autoenc_style256():
    conf = ffhq128_autoenc()
    conf = set_style_size(conf, 256)
    conf.eval_every_samples = 48_000_000
    conf.eval_ema_every_samples = 48_000_000
    return conf


def ffhq128_autoenc_style128():
    conf = ffhq128_autoenc()
    conf = set_style_size(conf, 128)
    conf.eval_every_samples = 48_000_000
    conf.eval_ema_every_samples = 48_000_000
    return conf


def ffhq128_autoenc_style64():
    conf = ffhq128_autoenc()
    conf = set_style_size(conf, 64)
    conf.eval_every_samples = 48_000_000
    conf.eval_ema_every_samples = 48_000_000
    return conf


def ffhq128_ddpm():
    conf = ddpm_adain_bias()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    return conf


def ffhq128_autoenc_72M():
    conf = ffhq128_autoenc()
    conf.total_samples = 48_000_000 + 24_000_000
    conf.continue_from = PretrainConfig(
        name='48M', path=f'logs/{ffhq128_autoenc().name}/last.ckpt')
    return conf


def ffhq128_ddpm_72M():
    conf = ffhq128_ddpm()
    conf.total_samples = 48_000_000 + 24_000_000
    conf.continue_from = PretrainConfig(
        name='48M', path=f'logs/{ffhq128_ddpm().name}/last.ckpt')
    return conf


def ffhq128_autoenc_200M():
    conf = ffhq128_autoenc()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.continue_from = PretrainConfig(
        name='72M', path=f'logs/{ffhq128_autoenc_72M().name}/last.ckpt')
    return conf


def ffhq128_ddpm_200M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.continue_from = PretrainConfig(
        name='72M', path=f'logs/{ffhq128_ddpm_72M().name}/last.ckpt')
    return conf


def horse128_autoenc():
    conf = ffhq128_autoenc()
    conf.postfix = '_fixseed'
    conf.data_name = 'horse256'
    conf.total_samples = 82_000_000
    return conf


def horse128_autoenc_thick():
    conf = horse128_autoenc()
    conf.net_ch = 192
    # saves 20% memory
    conf.net_beatgans_resnet_use_checkpoint_gnscalesilu = True
    conf.net_enc_grad_checkpoint = True
    return conf


def horse128_autoenc_200M():
    conf = horse128_autoenc()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.continue_from = PretrainConfig(
        name='82M', path=f'logs/{horse128_autoenc().name}/last.ckpt')
    return conf


def horse128_cosine_autoenc_thinner_deep():
    conf = ffhq128_autoenc()
    conf.postfix = ''
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf = thinner_encthin(conf)
    conf.beta_scheduler = 'cosine'
    conf.net_ch_mult = (1, 1, 2, 3, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    return conf


def horse128_cosine_autoenc_thinner_deep_morech():
    conf = horse128_cosine_autoenc_thinner_deep()
    conf.net_ch_mult = (1, 2, 4, 4, 4, 4)
    conf.net_enc_channel_mult = (1, 2, 4, 4, 4, 4)
    return conf


def horse128_cosine_autoenc_thinner_deep_morech2():
    conf = horse128_cosine_autoenc_thinner_deep()
    conf.net_ch_mult = (2, 4, 4, 4, 4, 4)
    conf.net_enc_channel_mult = (2, 4, 4, 4, 4, 4)
    return conf


def horse128_cosine_autoenc_thinner_deep_incond():
    conf = horse128_cosine_autoenc_thinner_deep()
    conf.net_beatgans_resnet_use_inlayers_cond = True
    return conf


def horse128_ddpm_200M():
    conf = horse128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.continue_from = PretrainConfig(
        name='82M', path=f'logs/{horse128_ddpm().name}/last.ckpt')
    return conf


def horse128_cosine_ddpm_thinner_deep():
    conf = horse128_ddpm()
    conf.postfix = ''
    conf.total_samples = 96_000_000
    conf = thinner_encthin(conf)
    conf.beta_scheduler = 'cosine'
    conf.net_ch_mult = (1, 1, 2, 3, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    return conf


def bedroom128_autoenc():
    conf = ffhq128_autoenc()
    conf.postfix = ''
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    return conf


def bedroom128_ddpm():
    conf = ffhq128_ddpm()
    conf.postfix = ''
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    return conf


def horse128_ddpm():
    conf = ffhq128_ddpm()
    conf.postfix = '_fixseed'
    conf.data_name = 'horse256'
    conf.total_samples = 82_000_000
    return conf


def horse128_ddpm_thick():
    conf = horse128_ddpm()
    conf.net_ch = 192
    # saves 20% memory
    conf.net_beatgans_resnet_use_checkpoint_gnscalesilu = True
    return conf


def pretrain_ffhq64_autoenc48M():
    conf = ffhq64_autoenc_12M()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='real48M',
        path=f'logs/{ffhq64_autoenc_48M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{ffhq64_autoenc_48M().name}.pkl'
    return conf


def pretrain_ffhq128_autoenc48M():
    conf = ffhq128_autoenc()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='48M',
        path=f'logs/{ffhq128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{ffhq128_autoenc().name}.pkl'
    return conf


def pretrain_ffhq128_autoenc72M(flip=False):
    conf = ffhq128_autoenc()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='72Mflip' if flip else '72M',
        path=f'logs/{ffhq128_autoenc_72M().name}/last.ckpt',
    )
    if flip:
        conf.latent_infer_path = f'latent_infer_flip/{ffhq128_autoenc_72M().name}.pkl'
    else:
        conf.latent_infer_path = f'latent_infer/{ffhq128_autoenc_72M().name}.pkl'
    return conf


def pretrain_ffhq128_autoenc130M():
    conf = ffhq128_autoenc()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'logs/{ffhq128_autoenc_200M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{ffhq128_autoenc_200M().name}.pkl'
    return conf


def pretrain_ffhq256_autoenc():
    conf = ffhq256_autoenc()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'logs/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{ffhq256_autoenc().name}.pkl'
    return conf


def pretrain_horse128(flip=False):
    conf = horse128_autoenc()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='82Mflip' if flip else '82M',
        path=f'logs/{horse128_autoenc().name}/last.ckpt',
    )
    if flip:
        conf.latent_infer_path = f'latent_infer_flip/{horse128_autoenc().name}.pkl'
    else:
        conf.latent_infer_path = f'latent_infer/{horse128_autoenc().name}.pkl'
    return conf


def pretrain_horse128_130M():
    conf = horse128_autoenc()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'logs/{horse128_autoenc_200M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{horse128_autoenc_200M().name}.pkl'
    return conf


def pretrain_horse128_thick():
    conf = horse128_autoenc_thick()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='82M',
        path=f'logs/{horse128_autoenc_thick().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{horse128_autoenc_thick().name}.pkl'
    return conf


def pretrain_bedroom128_120M():
    conf = bedroom128_autoenc()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='120M',
        path=f'logs/{bedroom128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{bedroom128_autoenc().name}.pkl'
    return conf


def pretrain_celeba64_72M():
    conf = celeba64_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'logs/{celeba64_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{celeba64_autoenc().name}.pkl'
    return conf


def pretrain_celeba64d2c_72M():
    conf = celeba64d2c_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'logs/{celeba64d2c_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'latent_infer/{celeba64d2c_autoenc().name}.pkl'
    return conf


def latent_diffusion_config(conf: TrainConfig):
    conf.batch_size = 128
    conf.base_dir = 'log-latent'
    conf.postfix = '_fixstats'
    conf.train_mode = TrainMode.latent_diffusion
    conf.latent_gen_type = GenerativeType.ddim
    conf.latent_loss_type = LossType.mse
    conf.latent_model_mean_type = ModelMeanType.eps
    conf.latent_model_var_type = ModelVarType.fixed_large
    conf.latent_model_mse_weight_type = None
    conf.latent_xstart_weight_type = None
    conf.latent_rescale_timesteps = False
    conf.latent_clip_sample = False
    conf.latent_T_eval = 20
    conf.latent_znormalize = True
    conf.latent_detach = True
    conf.latent_unit_normalize = False
    conf.total_samples = 96_000_000
    conf.sample_every_samples = 400_000
    conf.eval_every_samples = 20_000_000
    conf.eval_ema_every_samples = 20_000_000
    conf.save_every_samples = 2_000_000
    return conf


def latent_diffusion128_config(conf: TrainConfig):
    conf = latent_diffusion_config(conf)
    conf.batch_size_eval = 32
    return conf


def latent_diffusion256_config(conf: TrainConfig):
    conf = latent_diffusion_config(conf)
    conf.batch_size_eval = 8
    conf.sample_every_samples = 1_600_000
    conf.eval_every_samples = 400_000_000
    conf.eval_ema_every_samples = 400_000_000
    return conf


def latent_2048_batch_size(conf: TrainConfig):
    conf.batch_size = 2048
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.save_every_samples = 20_000_000
    conf.total_samples = 1_501_000_000
    return conf


def latent_1024_batch_size(conf: TrainConfig):
    conf.batch_size = 1024
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 2_000_000
    conf.save_every_samples = 10_000_000
    conf.total_samples = 1_001_000_000
    return conf


def latent_512_batch_size(conf: TrainConfig):
    conf.batch_size = 512
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 5_000_000
    conf.total_samples = 501_000_000
    return conf


def latent_256_batch_size(conf: TrainConfig):
    conf.batch_size = 256
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 2_000_000
    conf.total_samples = 301_000_000
    return conf


def latent_128_batch_size(conf: TrainConfig):
    conf.batch_size = 128
    conf.eval_ema_every_samples = 50_000_000
    conf.eval_every_samples = 50_000_000
    conf.sample_every_samples = 500_000
    conf.save_every_samples = 2_000_000
    conf.total_samples = 201_000_000
    return conf


def adamw_weight_decay(conf: TrainConfig):
    conf.optimizer = OptimizerType.adamw
    conf.weight_decay = 0.01
    return conf


def latent_mlp_2048(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_num_hid_channels = 2048
    return conf


def latent_mlp_4096(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.skip
    conf.net_latent_layers = 4
    conf.net_latent_skip_layers = (1, 2, 3)
    conf.net_latent_activation = Activation.silu
    conf.net_latent_num_hid_channels = 4096
    conf.net_latent_use_norm = False
    conf.net_latent_condition_bias = 1
    conf.net_latent_condition_2x = False
    return conf


def latent_mlp_8192(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_num_hid_channels = 8192
    return conf


def latent_mlp_4096_10layers(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_layers = 10
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_mlp_2048_10layers(conf: TrainConfig):
    conf = latent_mlp_4096_10layers(conf)
    conf.net_latent_num_hid_channels = 2048
    return conf


def latent_mlp_1024_10layers(conf: TrainConfig):
    conf = latent_mlp_4096_10layers(conf)
    conf.net_latent_num_hid_channels = 1024
    return conf


def latent_mlp_4096_13layers(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_layers = 13
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_mlp_4096_16layers(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_layers = 16
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_mlp_2048_16layers(conf: TrainConfig):
    conf = latent_mlp_4096_16layers(conf)
    conf.net_latent_num_hid_channels = 2048
    return conf


def latent_mlp_1024_16layers(conf: TrainConfig):
    conf = latent_mlp_4096_16layers(conf)
    conf.net_latent_num_hid_channels = 1024
    return conf


def latent_mlp_4096_7layers(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_layers = 7
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_mlp_4096_5layers(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_layers = 5
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_mlp_4096_norm_13layers(conf: TrainConfig):
    conf = latent_mlp_4096_13layers(conf)
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_4096_norm_10layers(conf: TrainConfig):
    conf = latent_mlp_4096_10layers(conf)
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_3072_norm_10layers(conf: TrainConfig):
    conf = latent_mlp_4096(conf)
    conf.net_latent_num_hid_channels = 3072
    conf.net_latent_layers = 10
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_4096_norm_5layers(conf: TrainConfig):
    conf = latent_mlp_4096_5layers(conf)
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_2048_norm_5layers(conf: TrainConfig):
    conf = latent_mlp_2048_10layers(conf)
    conf.net_latent_layers = 5
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_2048_norm_10layers(conf: TrainConfig):
    conf = latent_mlp_2048_10layers(conf)
    conf.net_latent_use_norm = True
    return conf


def latent_mlpprenorm_10layers(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.prenormskip
    conf.net_latent_layers = 10
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_mlp_2048_norm_13layers(conf: TrainConfig):
    conf = latent_mlp_2048_10layers(conf)
    conf.net_latent_layers = 13
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_2048_norm_15layers(conf: TrainConfig):
    conf = latent_mlp_2048_10layers(conf)
    conf.net_latent_layers = 15
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_2048_norm_16layers(conf: TrainConfig):
    conf = latent_mlp_2048_10layers(conf)
    conf.net_latent_layers = 16
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_2048_norm_20layers(conf: TrainConfig):
    conf = latent_mlp_2048_10layers(conf)
    conf.net_latent_layers = 20
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_2048_norm_30layers(conf: TrainConfig):
    conf = latent_mlp_2048_10layers(conf)
    conf.net_latent_layers = 30
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_1024_norm_40layers(conf: TrainConfig):
    conf = latent_mlp_1024_10layers(conf)
    conf.net_latent_layers = 40
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_1024_norm_20layers(conf: TrainConfig):
    conf = latent_mlp_1024_10layers(conf)
    conf.net_latent_layers = 20
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_1024_norm_10layers(conf: TrainConfig):
    conf = latent_mlp_1024_10layers(conf)
    conf.net_latent_use_norm = True
    return conf


def latent_mlp_4096_normbefore_10layers(conf: TrainConfig):
    conf = latent_mlp_4096_10layers(conf)
    conf.net_latent_use_norm = True
    conf.net_latent_condition_type = ConditionType.norm_scale_shift
    return conf


def latent_conv8_256_10layers(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_conv
    conf.net_latent_num_hid_channels = 256
    conf.net_latent_project_size = 8
    conf.net_latent_layers = 10
    return conf


def latent_conv4_512_10layers(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_conv
    conf.net_latent_num_hid_channels = 512
    conf.net_latent_project_size = 4
    conf.net_latent_layers = 10
    return conf


def latent_conv4_512_13layers(conf: TrainConfig):
    # 13 layers = 2 + 2 + 3 + 3 + 3 (just like in UNET512 4x[1, 1] + mid attention)
    conf.net_latent_net_type = LatentNetType.projected_conv
    conf.net_latent_num_hid_channels = 512
    conf.net_latent_project_size = 4
    conf.net_latent_layers = 13
    return conf


def latent_unet8_256(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 256
    conf.net_latent_project_size = 8
    conf.net_latent_channel_mult = (1, 2)
    return conf


def latent_unet8_128_124mult(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 128
    conf.net_latent_project_size = 8
    conf.net_latent_channel_mult = (1, 2, 4)
    return conf


def latent_unet4_256(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 256
    conf.net_latent_project_size = 4
    conf.net_latent_channel_mult = (1, 2)
    return conf


def latent_unet4_512(conf: TrainConfig):
    # unstable due to the attention
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 512
    conf.net_latent_project_size = 4
    conf.net_latent_channel_mult = (1, 2)
    return conf


def latent_unet4_512_nomidatt(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 512
    conf.net_latent_project_size = 4
    conf.net_latent_channel_mult = (1, 2)
    conf.net_latent_use_mid_attn = False
    return conf


def latent_unet2_1024(conf: TrainConfig):
    # unstable due to the attention
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 1024
    conf.net_latent_project_size = 2
    conf.net_latent_channel_mult = (1, )
    return conf


def latent_unet2_1024_nomidatt(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 1024
    conf.net_latent_project_size = 2
    conf.net_latent_channel_mult = (1, )
    conf.net_latent_use_mid_attn = False
    return conf


def latent_unet4_512_11mult(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.projected_unet
    conf.net_latent_num_hid_channels = 512
    conf.net_latent_project_size = 4
    conf.net_latent_channel_mult = (1, 1)
    return conf


def ffhq64_autoenc48M_latent_unet4_256():
    conf = pretrain_ffhq64_autoenc48M()
    conf = latent_diffusion_config(conf)
    conf = latent_unet4_256(conf)
    return conf


def ffhq64_autoenc_interpolate_weight50():
    conf = ffhq64_autoenc_48M()
    conf.postfix = ''
    conf.train_mode = TrainMode.diffusion_interpolate_deterministic_weight
    conf.train_interpolate_prob = 0.5
    conf.pretrain = None
    conf.continue_from = None
    return conf


def ffhq64_autoenc_interpolate_weight100():
    conf = ffhq64_autoenc_48M()
    conf.postfix = ''
    conf.train_mode = TrainMode.diffusion_interpolate_deterministic_weight
    conf.train_interpolate_prob = 0.5
    conf.pretrain = None
    conf.continue_from = None
    return conf


def ffhq64_autoenc_interpolate_weight50_latent():
    conf = ffhq64_autoenc_interpolate_weight50()
    conf = latent_diffusion_config(conf)
    conf = latent_mlp_4096(conf)
    conf.pretrain = PretrainConfig(
        name='intermw-p0.5-48M',
        path=f'logs/{ffhq64_autoenc_interpolate_weight50().name}/last.ckpt',
    )
    return conf


def ffhq64_autoenc_interpolate_weight100_latent():
    conf = ffhq64_autoenc_interpolate_weight50()
    conf = latent_diffusion_config(conf)
    conf = latent_mlp_4096(conf)
    conf.pretrain = PretrainConfig(
        name='intermw-p0.5-48M',
        path=f'logs/{ffhq64_autoenc_interpolate_weight100().name}/last.ckpt',
    )
    return conf