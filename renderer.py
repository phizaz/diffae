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
        if conf.model_type.has_noise_to_cond():
            # special case
            model: BeatGANsAutoencModel
            cond = torch.randn(len(x_T), conf.style_ch, device=device)
            cond = model.noise_to_cond(cond)
        else:
            cond = None
        return sampler.sample(model=model, noise=x_T, cond=cond)
    elif conf.train_mode.is_latent_diffusion():
        model: BeatGANsAutoencModel
        if conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = torch.randn(len(x_T), conf.style_ch, device=device)
        elif conf.train_mode == TrainMode.latent_2d_diffusion:
            # (n, c, 4, 4)
            latent_noise = torch.randn(len(x_T),
                                       conf.style_ch,
                                       4,
                                       4,
                                       device=device)
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

        if conf.train_mode == TrainMode.latent_2d_diffusion:
            # apply the pooling and linear of the encoder
            # (n, c)
            cond = model.encoder.forward_flatten(cond)

        # the diffusion on the model
        return sampler.sample(model=model, noise=x_T, cond=cond)
    elif conf.train_mode in [
            TrainMode.parallel_latent_diffusion_pred,
            TrainMode.parallel_latent_diffusion_pred_tt
    ]:
        cond_noise = torch.randn(len(x_T), conf.style_ch, device=x_T.device)
        model_kwargs = []
        for each in latent_sampler.ddim_sample_loop_progressive(
                model=model.latent_net,
                shape=cond_noise.shape,
                noise=cond_noise,
                clip_denoised=False,
        ):
            # (T-1, ..., 0)
            # t is also supplied to the unet (optional, as long as t_cond = t)
            _kwargs = {'cond': each['pred_xstart'], 't_cond': each['t']}
            model_kwargs.append(_kwargs)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs=model_kwargs)
    elif conf.train_mode == TrainMode.parallel_latent_diffusion_noisy:
        cond_noise = torch.randn(len(x_T), conf.style_ch, device=x_T.device)
        model_kwargs = []
        for each in latent_sampler.ddim_sample_loop_progressive(
                model=model.latent_net,
                shape=cond_noise.shape,
                noise=cond_noise,
                clip_denoised=False,
        ):
            # (T-1, ..., 0)
            # cond_t is supplied to the unet
            _kwargs = {'cond': each['sample']}
            model_kwargs.append(_kwargs)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs=model_kwargs)
    else:
        raise NotImplementedError()


def render_condition(
    conf: TrainConfig,
    model: BeatGANsAutoencModel,
    x_T,
    x_start,
    cond,
    sampler: Sampler,
    latent_sampler: Sampler,
):
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        # returns {'cond', 'cond2'}
        conds = model.encode(x_start)
        return sampler.sample(model=model, noise=x_T, model_kwargs=conds)
    elif conf.train_mode in [
            TrainMode.parallel_latent_diffusion_pred,
            TrainMode.parallel_latent_diffusion_pred_tt
    ]:
        # inverse the condition for each t
        with amp.autocast(conf.fp16):
            cond = model.encoder(x_start)
        out = latent_sampler.ddim_reverse_sample_loop(model=model.latent_net,
                                                      x=cond,
                                                      clip_denoised=False)
        # [1, ..., T]
        cond_t = out['sample_t']
        # [0, ..., T-1]
        T = out['T']
        model_kwargs = []
        for i in range(len(cond_t)):
            # predict the latent xstart
            # (1, ..., T) => (0, ..., T-1)
            out = latent_sampler.ddim_sample(model=model.latent_net,
                                             x=cond_t[i],
                                             t=T[i],
                                             clip_denoised=False)
            # use the predicted cond to supply to the Unet
            # supply t_cond to the unet
            _kwargs = {'cond': out['pred_xstart'], 't_cond': T[i]}
            model_kwargs.append(_kwargs)
        # (T-1, ..., 0)
        model_kwargs = model_kwargs[::-1]
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs=model_kwargs)
    elif conf.train_mode == TrainMode.parallel_latent_diffusion_noisy:
        # inverse the condition for each t
        with amp.autocast(conf.fp16):
            cond = model.encoder(x_start)
        out = latent_sampler.ddim_reverse_sample_loop(model=model.latent_net,
                                                      x=cond,
                                                      clip_denoised=False)
        # [1, ..., T]
        cond_t = out['sample_t']
        model_kwargs = []
        for i in range(len(cond_t)):
            _kwargs = {'cond': cond_t[i]}
            model_kwargs.append(_kwargs)
        # (T, ..., 1)
        model_kwargs = model_kwargs[::-1]
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs=model_kwargs)
    else:
        raise NotImplementedError()


def render_condition_no_latent_diffusion(
    conf: TrainConfig,
    model: BeatGANsAutoencModel,
    x_T,
    x_start,
    cond,
    sampler: Sampler,
    latent_sampler: Sampler,
):
    with amp.autocast(conf.fp16):
        cond = model.encoder(x_start)
    # it means t_cond is of very high quality (t=0)
    t_cond = torch.tensor([0] * len(cond)).to(cond.device)
    return sampler.sample(model=model,
                          noise=x_T,
                          model_kwargs={
                              'cond': cond,
                              't_cond': t_cond,
                          })
