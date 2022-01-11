import os
import shutil

import torch
import torchvision
from pytorch_fid import fid_score
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange

from renderer import *
from config import *
from diffusion import Sampler
from dist_utils import *
import lpips
from ssim import ssim


def make_subset_loader(conf: TrainConfig,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool,
                       parallel: bool,
                       drop_last=True):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context('fork'),
    )


def evaluate_lpips(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    use_inverted_noise: bool = False,
):
    """
    compare the generated images from autoencoder on validation dataset

    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    val_loader = make_subset_loader(conf,
                                    dataset=val_data,
                                    batch_size=conf.batch_size_eval,
                                    shuffle=False,
                                    parallel=True)

    model.eval()
    with torch.no_grad():
        scores = {
            'lpips': [],
            'mse': [],
            'ssim': [],
            'psnr': [],
        }
        for batch in tqdm(val_loader, desc='lpips'):
            imgs = batch['img'].to(device)

            if use_inverted_noise:
                # inverse the noise
                # with condition from the encoder
                model_kwargs = {}
                if conf.model_type.has_autoenc():
                    with torch.no_grad():
                        model_kwargs = model.encode(imgs)
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model,
                    x=imgs,
                    clip_denoised=True,
                    model_kwargs=model_kwargs)
                x_T = x_T['sample']
            else:
                x_T = torch.randn((len(imgs), 3, conf.img_size, conf.img_size),
                                  device=device)

            if conf.train_mode.is_parallel_latent_diffusion():
                pred_imgs = render_condition_no_latent_diffusion(
                    conf,
                    model,
                    x_T,
                    imgs,
                    cond=None,
                    sampler=sampler,
                    latent_sampler=latent_sampler)
            else:
                if conf.model_type == ModelType.ddpm:
                    # the case where you want to calculate the inversion capability of the DDIM model
                    assert use_inverted_noise
                    pred_imgs = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                    )
                else:
                    pred_imgs = render_condition(conf=conf,
                                                 model=model,
                                                 x_T=x_T,
                                                 x_start=imgs,
                                                 cond=None,
                                                 sampler=sampler,
                                                 latent_sampler=latent_sampler)
                # # returns {'cond', 'cond2'}
                # conds = model.encode(imgs)
                # pred_imgs = sampler.sample(model=model,
                #                            noise=x_T,
                #                            model_kwargs=conds)
            # (n, 1, 1, 1) => (n, )
            scores['lpips'].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

            # need to normalize into [0, 1]
            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            # (n, )
            scores['ssim'].append(
                ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores['mse'].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
            # (n, )
            scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))
        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    # support multi-gpu
    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ]
        for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    # {'lpips', 'mse', 'ssim'}
    return scores


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def evaluate_fid(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    conds_mean=None,
    conds_std=None,
    normalizer: RunningNormalizer = None,
    remove_cache: bool = True,
    clip_latent_noise: bool = False,
):
    assert conf.fid_cache is not None
    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(conf,
                                        dataset=val_data,
                                        batch_size=conf.batch_size_eval,
                                        shuffle=False,
                                        parallel=False)

        # put the val images to a directory
        cache_dir = f'{conf.fid_cache}_{conf.eval_num_images}'
        if (os.path.exists(cache_dir)
                and len(os.listdir(cache_dir)) < conf.eval_num_images):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)
        os.makedirs(conf.generate_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        if conf.model_type.can_sample():
            eval_num_images = chunk_size(conf.eval_num_images, rank,
                                         world_size)
            desc = "generating images"
            for i in trange(0, eval_num_images, batch_size, desc=desc):
                batch_size = min(batch_size, eval_num_images - i)
                x_T = torch.randn(
                    (batch_size, 3, conf.img_size, conf.img_size),
                    device=device)
                batch_images = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conds_mean=conds_mean,
                    conds_std=conds_std,
                    normalizer=normalizer).cpu()
                # if conf.model_type.has_noise_to_cond():
                #     # special case
                #     model: BeatGANsAutoencModel
                #     cond = torch.randn(len(x_T), conf.style_ch, device=device)
                #     cond = model.noise_to_cond(cond)
                # else:
                #     cond = None

                # if conf.train_mode.is_parallel_latent_diffusion():
                #     batch_images = render_uncondition(
                #         conf,
                #         model,
                #         x_T,
                #         sampler=sampler,
                #         latent_sampler=latent_sampler).cpu()
                # else:
                #     batch_images = sampler.sample(model=model,
                #                                   noise=x_T,
                #                                   cond=cond).cpu()
                batch_images = (batch_images + 1) / 2
                # keep the generated images
                for j in range(len(batch_images)):
                    img_name = filename(i + j)
                    torchvision.utils.save_image(
                        batch_images[j],
                        os.path.join(conf.generate_dir, f'{img_name}.png'))
        elif conf.model_type == ModelType.autoencoder:
            if conf.train_mode.is_latent_diffusion():
                # evaluate autoencoder + latent diffusion (doesn't give the images)
                model: BeatGANsAutoencModel
                eval_num_images = chunk_size(conf.eval_num_images, rank,
                                             world_size)
                desc = "generating images"
                for i in trange(0, eval_num_images, batch_size, desc=desc):
                    batch_size = min(batch_size, eval_num_images - i)
                    x_T = torch.randn(
                        (batch_size, 3, conf.img_size, conf.img_size),
                        device=device)
                    batch_images = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                        conds_mean=conds_mean,
                        conds_std=conds_std,
                        normalizer=normalizer,
                        clip_latent_noise=clip_latent_noise,
                    ).cpu()
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
            else:
                # evaulate autoencoder (given the images)
                # to make the FID fair, autoencoder must not see the validation dataset
                # also shuffle to make it closer to unconditional generation
                train_loader = make_subset_loader(conf,
                                                  dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  parallel=True)

                i = 0
                for batch in tqdm(train_loader, desc='generating images'):
                    imgs = batch['img'].to(device)
                    x_T = torch.randn(
                        (len(imgs), 3, conf.img_size, conf.img_size),
                        device=device)
                    batch_images = render_condition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        x_start=imgs,
                        cond=None,
                        sampler=sampler,
                        latent_sampler=latent_sampler).cpu()
                    # model: BeatGANsAutoencModel
                    # # returns {'cond', 'cond2'}
                    # conds = model.encode(imgs)
                    # batch_images = sampler.sample(model=model,
                    #                               noise=x_T,
                    #                               model_kwargs=conds).cpu()
                    # denormalize the images
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
                    i += len(imgs)
        else:
            raise NotImplementedError()
    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths(
            [cache_dir, conf.generate_dir],
            batch_size,
            device=device,
            dims=2048)

        # remove the cache
        if remove_cache and os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)

    barrier()

    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0., device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f'fid ({get_rank()}):', fid)

    return fid


def evaluate_interpolate_fid(sampler: Sampler, model: Model, conf: TrainConfig,
                             device, train_data: Dataset, val_data: Dataset):
    assert conf.fid_cache is not None

    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(conf,
                                        dataset=val_data,
                                        batch_size=conf.batch_size_eval,
                                        shuffle=False,
                                        parallel=False)

        # put the val images to a directory
        cache_dir = f'{conf.fid_cache}_{conf.eval_num_images}'
        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)
        os.makedirs(conf.generate_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        assert conf.model_type == ModelType.autoencoder
        # evaulate autoencoder (given the images)
        # to make the FID fair, autoencoder must not see the validation dataset
        # also shuffle to make it closer to unconditional generation
        train_loader = make_subset_loader(conf,
                                          dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          parallel=True)

        images = []
        i = 0
        for batch in tqdm(train_loader, desc='generating images'):
            imgs = batch['img'].to(device)
            x_T = torch.randn((len(imgs), 3, conf.img_size, conf.img_size),
                              device=device)

            # linear interpolations within batch
            model: BeatGANsAutoencModel
            # (n, c)
            cond = model.encoder.forward(imgs)
            perm = torch.randperm(len(cond))
            # linear interpolate halfway
            cond = (cond + cond[perm]) / 2

            # generate with interpolated cond
            batch_images = sampler.sample(model=model, noise=x_T,
                                          cond=cond).cpu()

            # denormalize the images
            batch_images = (batch_images + 1) / 2
            images.append(batch_images)
            # keep the generated images
            for j in range(len(batch_images)):
                img_name = filename(i + j)
                torchvision.utils.save_image(
                    batch_images[j],
                    os.path.join(conf.generate_dir, f'{img_name}.png'))
            i += len(imgs)
        images = torch.cat(images, dim=0).numpy()
    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths(
            [cache_dir, conf.generate_dir],
            batch_size,
            device=device,
            dims=2048)

        # remove the cache
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)

    barrier()

    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0., device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f'interpolate fid ({get_rank()}):', fid)

    return fid


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc='copy images'):
        imgs = batch['img']
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j],
                                         os.path.join(path, f'{i+j}.png'))
        i += len(imgs)
