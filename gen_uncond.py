from run_templates import *
from torchvision.utils import save_image

if __name__ == '__main__':
    device = 'cuda:0'
    T = 20
    N = 1

    conf = ffhq128_ddpm()
    state = torch.load(f'logs/{conf.name}/last.ckpt', map_location='cpu')
    print('main step:', state['global_step'])
    model = LitModel(conf)
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    diff_conf = conf._make_diffusion_conf(T=T)
    sampler = diff_conf.make_sampler()

    noise = torch.randn(N, 3, conf.img_size, conf.img_size, device=device)
    # (n, 3, h, w)
    pred_img = sampler.sample(model.ema_model, noise=noise)
    pred_img = (pred_img + 1) / 2

    save_image(pred_img, 'generated/0.png') 

    # put_original_images(64, None, None, 'generated/celeba_All')
    # put_original_images(64, 'Male', False, 'generated/celeba_Male')
    # put_original_images(64, 'Male', True, 'generated/celeba_Female')

    # for data_seed in [0]:
    #     for shots in [10, 20, 50, 100]:
    #         for allneg in [False, True]:
    #             for is_negative in [False, True]:
    #                 cls_name = 'Male'
    #                 if allneg:
    #                     cls_conf = cls_celeba64_fewshot_allneg(
    #                         cls_name, shots, data_seed)
    #                 else:
    #                     cls_conf = cls_celeba64_fewshot(
    #                         cls_name, shots, data_seed)

    #                 name = f'generated/celeba_cond_{cls_name}'
    #                 if is_negative:
    #                     name += '-neg'
    #                 if cls_conf.manipulate_mode.is_fewshot():
    #                     name += f'-shot{cls_conf.manipulate_shots}'
    #                     if cls_conf.manipulate_mode == ManipulateMode.celeba_fewshot_allneg:
    #                         name += '-allneg'
    #                 name += f'_{data_seed}'

    #                 gen = GenCond(
    #                     cls_name,
    #                     is_negative=is_negative,
    #                     threshold=0.5,
    #                     conf=celeba64_autoenc_latent(),
    #                     cls_conf=cls_conf,
    #                     T_img=20,
    #                     T_latent=20,
    #                     device='cuda:1',
    #                 )
    #                 gen.cond_sample(
    #                     50_000,
    #                     name,
    #                     batch_size=40,
    #                     batch_size_latent=2048,
    #                 )
