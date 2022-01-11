from run_templates import *


def ffhq64_autoenc_latent():
    conf = pretrain_ffhq64_autoenc48M()
    conf = latent_diffusion_config(conf)
    conf = latent_512_batch_size(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 301_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    return conf


def ffhq128_autoenc_latent():
    conf = pretrain_ffhq128_autoenc130M()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.postfix = '_100M'
    conf.total_samples = 101_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    return conf


def ffhq256_autoenc_latent():
    conf = pretrain_ffhq256_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.postfix = '_100M'
    conf.total_samples = 101_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    return conf


def horse128_autoenc_latent():
    conf = pretrain_horse128_130M()
    conf = latent_diffusion128_config(conf)
    conf = latent_2048_batch_size(conf)
    conf = latent_mlp_2048_norm_20layers(conf)
    conf.total_samples = 2_001_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    return conf


def bedroom128_autoenc_latent():
    conf = pretrain_bedroom128_120M()
    conf = latent_diffusion128_config(conf)
    conf = latent_2048_batch_size(conf)
    conf = latent_mlp_2048_norm_20layers(conf)
    conf.total_samples = 2_001_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    return conf


def celeba64_autoenc_latent():
    conf = pretrain_celeba64_72M()
    conf = latent_diffusion_config(conf)
    conf = latent_512_batch_size(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = adamw_weight_decay(conf)
    conf.postfix = '_300M'
    conf.total_samples = 301_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    return conf


def celeba64d2c_autoenc_latent():
    conf = pretrain_celeba64d2c_72M()
    conf = latent_diffusion_config(conf)
    conf = latent_512_batch_size(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = adamw_weight_decay(conf)
    # just for the name
    conf.continue_from = PretrainConfig('200M',
                                        f'log-latent/{conf.name}/last.ckpt')
    conf.postfix = '_300M'
    conf.total_samples = 301_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    return conf
