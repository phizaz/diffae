from config import *

from torch.cuda import amp


def render_uncondition(conf: TrainConfig,
                       model: BeatGANsAutoencModel,
                       x_T,
                       sampler: Sampler,
                       latent_sampler: Sampler,
                       conds_mean=None,
                       conds_std=None,
                       clip_latent_noise: bool = False):
    device = x_T.device
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.can_sample()
        return sampler.sample(model=model, noise=x_T)
    elif conf.train_mode.is_latent_diffusion():
        model: BeatGANsAutoencModel
        if conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = torch.randn(len(x_T), conf.style_ch, device=device)
        else:
            raise NotImplementedError()

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        cond = latent_sampler.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=conf.latent_clip_sample,
        )

        if conf.latent_znormalize:
            cond = cond * conds_std.to(device) + conds_mean.to(device)

        # the diffusion on the model
        return sampler.sample(model=model, noise=x_T, cond=cond)
    else:
        raise NotImplementedError()


def render_condition(
    conf: TrainConfig,
    model: BeatGANsAutoencModel,
    x_T,
    sampler: Sampler,
    x_start=None,
    cond=None,
):
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        # returns {'cond', 'cond2'}
        if cond is None:
            cond = model.encode(x_start)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs={'cond': cond})
    else:
        raise NotImplementedError()
