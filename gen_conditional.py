import torch
from run_templates import *
from run_cls_templates import *
from run_latent_templates import *
from experiment import LitModel
from experiment_classifier import ClsModel
from tqdm.autonotebook import tqdm
from torchvision.utils import save_image
from shutil import copyfile


class GenCond:
    def __init__(self, cls_name, is_negative, threshold, conf: TrainConfig,
                 cls_conf: TrainConfig, T_img: int, T_latent: int,
                 device: str) -> None:
        print('conf:', conf.name)
        self.cls_name = cls_name
        self.is_negative = is_negative
        self.threshold = threshold

        self.conf = conf
        if cls_conf is not None:
            self.cls_conf = cls_conf
        self.device = device

        self.model: LitModel = self.load().to(device)

        if cls_conf is not None:
            self.cls_model: ClsModel = self.load_cls().to(device)
            if self.cls_conf.manipulate_mode.is_single_class():
                pass
            else:
                self.cls_data = self.cls_model.load_dataset()
                self.cls_id = self.cls_data.cls_to_id[self.cls_name]
                print('class id:', self.cls_id, self.cls_name)

        # image sampler
        diff_conf = conf._make_diffusion_conf(T=T_img)
        self.sampler = diff_conf.make_sampler()
        # latent sampler
        diff_conf = conf._make_latent_diffusion_conf(T=T_latent)
        self.latent_sampler = diff_conf.make_sampler()

    def uncond_sample(
        self,
        n: int,
        save_dir: str,
        batch_size: int,
        batch_size_latent: int,
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with torch.no_grad():
            total = 0
            Cond = []
            with tqdm(total=n, desc='generating latents') as progress:
                while total < n:
                    # (n, c)
                    latent_noise = torch.randn(batch_size_latent,
                                               self.conf.style_ch,
                                               device=self.device)
                    cond = self.latent_sampler.sample(
                        self.model.ema_model.latent_net,
                        noise=latent_noise,
                        clip_denoised=False,
                    )
                    cond = self.model.denormalize(cond)
                    cond = cond.cpu()
                    Cond.append(cond)
                    total += len(cond)
                    progress.update(len(cond))
            # (n, 512)
            Cond = torch.cat(Cond)[:n]
            assert len(Cond) == n

            # generate images
            dataset = TensorDataset(Cond)
            loader = DataLoader(dataset, batch_size=batch_size)
            j = 0
            for cond, in tqdm(loader, desc='generating images'):
                cond = cond.to(self.device)
                noise = torch.randn(len(cond),
                                    3,
                                    self.conf.img_size,
                                    self.conf.img_size,
                                    device=self.device)
                # (n, 3, h, w)
                pred_img = self.sampler.sample(self.model.ema_model,
                                               noise=noise,
                                               cond=cond)
                pred_img = (pred_img + 1) / 2
                # save
                for i in range(len(pred_img)):
                    path = f'{save_dir}/{j + i}.png'
                    save_image(pred_img[i], path)
                j += len(pred_img)
            return pred_img

    def cond_sample(
        self,
        n: int,
        save_dir: str,
        batch_size: int,
        batch_size_latent: int,
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with torch.no_grad():
            total = 0
            Cond = []
            with tqdm(total=n, desc='generating latents') as progress:
                while total < n:
                    # (n, c)
                    latent_noise = torch.randn(batch_size_latent,
                                               self.conf.style_ch,
                                               device=self.device)
                    cond = self.latent_sampler.sample(
                        self.model.ema_model.latent_net,
                        noise=latent_noise,
                        clip_denoised=False,
                    )
                    # (n, c)
                    pred_cls = torch.sigmoid(
                        self.cls_model.classifier.forward(cond))
                    # (n, )
                    if self.cls_conf.manipulate_mode.is_single_class():
                        pred_cls = pred_cls[:, 0]
                    else:
                        pred_cls = pred_cls[:, self.cls_id]
                    if self.is_negative:
                        pred_cls = 1 - pred_cls

                    # print('pred_cls:', pred_cls)

                    accept = pred_cls > self.threshold
                    cond = cond[accept]
                    cond = self.model.denormalize(cond)
                    cond = cond.cpu()
                    Cond.append(cond)
                    total += len(cond)
                    progress.update(len(cond))
            # (n, 512)
            Cond = torch.cat(Cond)[:n]
            assert len(Cond) == n

            # generate images
            dataset = TensorDataset(Cond)
            loader = DataLoader(dataset, batch_size=batch_size)
            j = 0
            for cond, in tqdm(loader, desc='generating images'):
                cond = cond.to(self.device)
                noise = torch.randn(len(cond),
                                    3,
                                    self.conf.img_size,
                                    self.conf.img_size,
                                    device=self.device)
                # (n, 3, h, w)
                pred_img = self.sampler.sample(self.model.ema_model,
                                               noise=noise,
                                               cond=cond)
                pred_img = (pred_img + 1) / 2
                # save
                for i in range(len(pred_img)):
                    path = f'{save_dir}/{j + i}.png'
                    save_image(pred_img[i], path)
                j += len(pred_img)
            return pred_img

    def cond_sample_rejection(
        self,
        n: int,
        save_dir: str,
        batch_size: int,
        batch_size_latent: int,
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with torch.no_grad():
            total = 0
            Cond = []
            with tqdm(total=n, desc='generating latents') as progress:
                while total < n:
                    # (n, c)
                    latent_noise = torch.randn(batch_size_latent,
                                               self.conf.style_ch,
                                               device=self.device)
                    cond = self.latent_sampler.sample(
                        self.model.ema_model.latent_net,
                        noise=latent_noise,
                        clip_denoised=False,
                    )
                    # rejection sampling requires a good "confidence score"
                    # we use a well-trained and stable EMA model
                    # (n, c)
                    pred_cls = torch.sigmoid(
                        self.cls_model.ema_classifier.forward(cond))
                    # (n, )
                    if self.cls_conf.manipulate_mode.is_single_class():
                        pred_cls = pred_cls[:, 0]
                    else:
                        pred_cls = pred_cls[:, self.cls_id]
                    if self.is_negative:
                        pred_cls = 1 - pred_cls

                    # print('pred_cls:', pred_cls)

                    # rejection sampling
                    accept = pred_cls > self.threshold
                    u = torch.rand_like(pred_cls)
                    accept = accept & (u < pred_cls)

                    cond = cond[accept]
                    cond = self.model.denormalize(cond)
                    cond = cond.cpu()
                    Cond.append(cond)
                    total += len(cond)
                    progress.update(len(cond))
            # (n, 512)
            Cond = torch.cat(Cond)[:n]
            assert len(Cond) == n

            # generate images
            dataset = TensorDataset(Cond)
            loader = DataLoader(dataset, batch_size=batch_size)
            j = 0
            for cond, in tqdm(loader, desc='generating images'):
                cond = cond.to(self.device)
                noise = torch.randn(len(cond),
                                    3,
                                    self.conf.img_size,
                                    self.conf.img_size,
                                    device=self.device)
                # (n, 3, h, w)
                pred_img = self.sampler.sample(self.model.ema_model,
                                               noise=noise,
                                               cond=cond)
                pred_img = (pred_img + 1) / 2
                # save
                for i in range(len(pred_img)):
                    path = f'{save_dir}/{j + i}.png'
                    save_image(pred_img[i], path)
                j += len(pred_img)
            return pred_img

    def load(self):
        state = torch.load(f'log-latent/{self.conf.name}/last.ckpt',
                           map_location='cpu')
        print('main step:', state['global_step'])
        model = LitModel(self.conf)
        model.load_state_dict(state['state_dict'], strict=False)
        model.ema_model.eval()
        return model

    def load_cls(self):
        state = torch.load(f'logs-cls/{self.cls_conf.name}/last.ckpt',
                           map_location='cpu')
        print('latent step:', state['global_step'])
        model = ClsModel(self.cls_conf)
        model.load_state_dict(state['state_dict'], strict=False)
        return model


def put_original_images(img_size,
                        cls_name,
                        is_negative,
                        save_dir,
                        n=50_000,
                        d2c: bool = False,
                        uncrop: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if d2c:
        data = CelebD2CAttrDataset(
            data_paths['celeba'][0],
            img_size,
            data_paths['celeba_anno'],
            do_augment=False,
            only_cls_name=cls_name,
            only_cls_value=-1 if is_negative else 1,
        )
    elif uncrop:
        data = CelebAttrDataset(
            data_paths['celeba'][0],
            img_size,
            data_paths['celeba_anno'],
            ext='jpg',
            do_augment=False,
            only_cls_name=cls_name,
            only_cls_value=-1 if is_negative else 1,
        )
    else:
        data = CelebAttrDataset(
            data_paths['celeba_aligned'][0],
            img_size,
            data_paths['celeba_anno'],
            do_augment=False,
            only_cls_name=cls_name,
            only_cls_value=-1 if is_negative else 1,
        )
    print('data len:', len(data))
    if n > 0:
        if len(data) > n:
            data = SubsetDataset(data, n)
    loader = DataLoader(data, num_workers=4, batch_size=10)
    j = 0
    for batch in tqdm(loader):
        imgs = batch['img']
        imgs = (imgs + 1) / 2
        for i in range(len(imgs)):
            save_image(imgs[i], f'{save_dir}/{j + i}.png')
        j += len(imgs)


def calculate_fid(dir1, dir2, save_path, device, batch_size=128):
    fid = fid_score.calculate_fid_given_paths([dir1, dir2],
                                              batch_size,
                                              device=device,
                                              dims=2048)
    out = {'fid': fid}
    print(out)
    tgt = save_path
    dirname = os.path.dirname(tgt)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(tgt, 'a') as f:
        f.write(json.dumps(out) + "\n")
    return fid


def celeb64_our_gen_cond(
    cls_name,
    data_seed,
    shots,
    allneg,
    is_negative,
    device,
    skip: bool = True,
    d2c: bool = False,
    rejection: bool = False,
    n=50_000,
):
    if d2c:
        if allneg:
            if rejection:
                # rejection requires a longer trained classifier
                cls_conf = cls_celeba64d2c_fewshot_allneg_long(
                    cls_name, shots, data_seed)
            else:
                cls_conf = cls_celeba64d2c_fewshot_allneg(
                    cls_name, shots, data_seed)
        else:
            if rejection:
                cls_conf = cls_celeba64d2c_fewshot_long(
                    cls_name, shots, data_seed)
            else:
                cls_conf = cls_celeba64d2c_fewshot(cls_name, shots, data_seed)
    else:
        if allneg:
            cls_conf = cls_celeba64_fewshot_allneg(cls_name, shots, data_seed)
        else:
            cls_conf = cls_celeba64_fewshot(cls_name, shots, data_seed)

    if d2c:
        # name = f'celeba_d2c_cond_{cls_name}'
        name = f'celeba_d2c_v2_cond_{cls_name}'
    else:
        name = f'celeba_cond_{cls_name}'
    if is_negative:
        name += '-neg'
    if rejection:
        name += '-reject'
    if cls_conf.manipulate_mode.is_fewshot():
        name += f'-shot{cls_conf.manipulate_shots}'
        if cls_conf.manipulate_mode.is_fewshot_allneg():
            name += '-allneg'
    name += f'_{data_seed}'
    save_dir = f'generated/{name}'
    print('save dir:', save_dir)

    if skip and os.path.exists(save_dir):
        if len(os.listdir(save_dir)) == n:
            print('exists skipping ...')
            return name, save_dir
        else:
            pass

    if d2c:
        conf = celeba64d2c_autoenc_latent()
    else:
        conf = celeba64_autoenc_latent()

    gen = GenCond(
        cls_name,
        is_negative=is_negative,
        threshold=0.5,
        conf=conf,
        cls_conf=cls_conf,
        T_img=100,
        T_latent=100,
        device=device,
    )
    if rejection:
        gen.cond_sample_rejection(
            n,
            save_dir,
            batch_size=40,
            batch_size_latent=2048,
        )
    else:
        gen.cond_sample(
            n,
            save_dir,
            batch_size=40,
            batch_size_latent=2048,
        )
    return name, save_dir


def copy_first_n_images(source, target, n):
    files = sorted(os.listdir(source))
    for f in tqdm(files[:n], desc='copying'):
        copyfile(f'{source}/{f}', f'{target}/{f}')


if __name__ == '__main__':
    pass
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
