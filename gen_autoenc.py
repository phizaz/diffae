import torch
from run_templates import *
from tqdm.autonotebook import tqdm

# conf = ffhq128_autoenc_200M()
conf = ffhq256_autoenc()
conf.device = 'cuda:0'
print(conf.name)


def load(name):
    state = torch.load(f'logs/{name}/last.ckpt', map_location='cpu')
    print('step:', state['global_step'])
    model = LitModel(conf)
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.to(conf.device)
    model.ema_model.eval()
    return model


model = load(conf.name)


def render_only(noise, cond, T):
    diff_conf = model.conf._make_diffusion_conf(T=T)
    sampler = diff_conf.make_sampler()

    model.eval()
    with torch.no_grad():
        return sampler.ddim_sample_loop(model.ema_model,
                                        shape=noise.shape,
                                        noise=noise,
                                        model_kwargs={'cond': cond})


def inverse_only(img, inv_T):
    diff_conf = model.conf._make_diffusion_conf(T=inv_T)
    inv_sampler = diff_conf.make_sampler()

    model.eval()
    with torch.no_grad():
        x = img.to(conf.device)
        cond = model.ema_model.encoder(x)
        for t in range(inv_T):
            out = inv_sampler.ddim_reverse_sample(
                model.ema_model,
                x,
                torch.tensor([t] * len(img)).to(conf.device),
                model_kwargs={'cond': cond})
            x = out['sample']

    return x, cond


# data = conf.make_dataset(do_augment=False, as_tensor=True, do_normalize=True)
# data = ImageDataset('/home/konpat/datasets/CelebAMask-HQ/CelebA-HQ-img',
#                     model.conf.img_size,
#                     sort_names=True,
#                     do_augment=False,
#                     do_transform=True,
#                     do_normalize=True)
data = CelebHQLMDB(data_paths['celebahq'][0], image_size=conf.img_size, do_augment=False, do_transform=True, do_normalize=True)

conf.batch_size = 2
loader = conf.make_loader(data, shuffle=False)

from torchvision.utils import save_image, make_grid

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

save_dir = 'generated/autoenc_celebahq_128'
makedirs(f'{save_dir}/real')
makedirs(f'{save_dir}/inv')
makedirs(f'{save_dir}/enc')

N = 10
T_gen = 20
T_inv = 50

with tqdm(total=N) as progress:
    total = 0
    for batch in loader:
        img = batch['img'].to(conf.device)
        cond = model.ema_model.encoder(img)
        noise = torch.randn_like(img)
        x = render_only(noise, cond, T_gen)
        noise2, cond = inverse_only(img, T_inv)
        x2 = render_only(noise2, cond, T_gen)

        img = (img + 1) / 2
        x = (x + 1) / 2
        x2 = (x2 + 1) / 2

        for i in range(len(img)):
            # tmp = make_grid(
            #     [
            #         # original
            #         ((img[i] + 1) / 2).to(conf.device),
            #         # reconstruction
            #         (x[i] + 1) / 2,
            #         # inverted
            #         (x2[i] + 1) / 2,
            #     ],
            #     nrow=3,
            # )
            # save_image(tmp, f'{save_dir}/{total + i}.png')
            save_image(img[i], f'{save_dir}/real/{total+i:05d}.png')
            save_image(x[i], f'{save_dir}/enc/{total+i:05d}.png')
            save_image(x2[i], f'{save_dir}/inv/{total+i:05d}.png')

        total += len(img)
        progress.update(len(img))

        if total >= N:
            break
