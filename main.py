from run_cls_templates import cls_ffhq128_autoenc, cls_ffhq256_autoenc
from run_latent_templates import bedroom128_autoenc_latent, celeba64d2c_autoenc_latent, ffhq128_autoenc_latent, ffhq256_autoenc_latent, horse128_autoenc_latent
from run_templates import *

if __name__ == '__main__':
    gpus = [3]

    # conf = horse128_cosine_autoenc_thinner_deep_morech()
    # conf.net_beatgans_resnet_use_inlayers_cond = True
    # conf.net_beatgans_resnet_use_checkpoint_gnscalesilu = True
    # conf.postfix = '_tmp'
    # conf.batch_size = 16
    # conf.sample_size = 16
    # conf = ffhq64_autoenc_heldout()
    # conf.postfix = '_tmp'
    # conf.batch_size = 32
    # conf.postfix = '_tmp'
    # conf.style_ch = 256
    # conf.net_beatgans_resnet_cond_channels = 256

    # conf = ffhq64_autoenc(
    #     postfix='_tmp',
    #     net_beatgans_resnet_cond_channels=512-128,
    #     net_beatgans_three_cond=True,
    # )

    # conf = ffhq128_autoenc_72M()
    # conf = ffhq128_autoenc_130M()
    # conf = ffhq128_ddpm_130M()
    # conf = ffhq128_ddpm_72M()
    # conf = ffhq256_autoenc()
    # conf = bedroom128_autoenc()
    # conf = bedroom128_ddpm()
    # conf = horse128_autoenc_130M()
    # conf = horse128_ddpm_130M()
    # conf = celeba64d2c_autoenc()
    # conf = celeba64d2c_ddpm()
    # conf = ffhq128_autoenc_latent()
    # conf = celeba64d2c_autoenc_latent()
    # conf = horse128_autoenc_latent()
    # conf = bedroom128_autoenc_latent()
    # conf = ffhq256_autoenc_latent()
    # conf = cls_ffhq128_all()
    # conf = cls_ffhq256_all()
    
    from shutil import copy

    src = f'logs-cls/{conf.name}/last.ckpt' 
    tgt = f'checkpoints/ffhq256_autoenc_cls/last.ckpt'
    if not os.path.exists(os.path.dirname(tgt)):
        os.makedirs(os.path.dirname(tgt))
    print('copying ..')
    print(src, tgt)
    copy(src, tgt)

    # conf.batch_size = 8
    # conf = pretrain_ffhq64_autoenc48M()
    # conf = latent_diffusion_config(conf)
    # # conf = latent_unet4_512(conf)
    # # conf = latent_unet4_512_11mult(conf)
    # conf = latent_unet8_256(conf)
    # # conf = latent_unet4_256(conf)
    # # conf = latent_mlp_4096(conf)
    # conf.postfix = '_fixstats'
    # # conf.eval_programs = ['infer']

    # conf = pretrain_ffhq128_autoenc72M()
    # conf = ffhq128_autoenc_72M()
    # conf = latent_diffusion_config(conf)
    # conf = latent_mlp_4096_norm_10layers(conf)
    # conf = latent_mlp_4096_norm_13layers(conf)
    # conf = latent_mlp_4096_10layers(conf)
    # conf = latent_mlp_4096_13layers(conf)
    # conf = latent_conv8_256_10layers(conf)
    # conf = latent_conv4_512_10layers(conf)
    # conf = latent_conv4_512_13layers(conf)
    # # conf = latent_unet2_1024(conf)
    # # conf = latent_unet2_1024_nomidatt(conf)
    # # conf = latent_unet4_512(conf)
    # # conf = latent_unet4_512_nomidatt(conf)
    # # conf = latent_unet4_512_11mult(conf)
    # conf = latent_unet8_256(conf)
    # # conf = latent_unet4_256(conf)
    # # conf = latent_mlp_4096(conf)
    # # conf = latent_mlp_8192(conf)
    # conf.postfix = '_fixstats'
    # conf.eval_every_samples = 10_000_000
    # conf.eval_ema_every_samples = 10_000_000
    # conf.batch_size_eval = 32
    # conf.eval_programs = ['infer']
    # conf.eval_programs = ['fid(20,20)']

    # train(conf, gpus=gpus)
    # train(conf, gpus=gpus, mode='eval')
