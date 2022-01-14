from model.unet import ScaleAt
from model.latentnet import *
from model.noisenet import NoiseNetConfig, NoiseNetType
from model.unet_autoenc import CondAt, LatentGenerativeModelConfig, MergerType, TimeMode, VectorizerType
from diffusion.resample import UniformSampler
from diffusion.diffusion import space_timesteps
from typing import Tuple

from torch.utils.data import DataLoader

from config_base import BaseConfig
from dataset import *
from diffusion import *
from diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from model import *
from model.model_our import *
from choices import *
from multiprocessing import get_context
import os
from dataset_util import *
from torch.utils.data.distributed import DistributedSampler

data_paths = {
    'ffhq': (os.path.expanduser('~/datasets/ffhq_256/train'),
             os.path.expanduser('~/datasets/ffhq_256/valid')),
    # no split
    'ffhqfull': (os.path.expanduser('~/datasets/ffhq_256'), None),
    'ffhqlmdb256': (os.path.expanduser('~/datasets/ffhq/ffhq256.lmdb'), None),
    'ffhqlmdbsplit256':
    (os.path.expanduser('~/datasets/ffhq/ffhq256.lmdb'), None),
    'celeba': (os.path.expanduser('~/datasets/celeba_small'), None),
    'celeba_aligned':
    (os.path.expanduser('~/datasets/celeba_small_aligned'), None),
    'celebahq':
    (os.path.expanduser('~/datasets/CelebAMask-HQ/celebahq.lmdb'), None),
    'horse': (os.path.expanduser('~/datasets/lsun/horse'), None),
    'horse256': (os.path.expanduser('~/datasets/lsun/horse256.lmdb'), None),
    'bedroom256':
    (os.path.expanduser('~/datasets/lsun/bedroom256.lmdb'), None),
    'celebalmdb': (os.path.expanduser('~/datasets/celeba.lmdb'), None),
    'celebaalignlmdb': (os.path.expanduser('~/datasets/celeba_aligned.lmdb'),
                        None),
    ####################
    'celeba_anno':
    os.path.expanduser('data/celeba_anno/list_attr_celeba.txt'),
    'celebahq_anno':
    os.path.expanduser('data/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
    'celeba_relight':
    os.path.expanduser('data/celeba_hq_light/celeba_light.txt'),
}


@dataclass
class PretrainConfig(BaseConfig):
    name: str
    path: str


@dataclass
class TrainConfig(BaseConfig):
    # random seed
    seed: int = 0
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: ManipulateMode = ManipulateMode.celeba_all
    manipulate_cls: str = None
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddpm
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_model_mse_weight_type: MSEWeightType = MSEWeightType.var
    beatgans_xstart_weight_type: XStartWeightType = XStartWeightType.uniform
    beatgans_rescale_timesteps: bool = False
    latent_infer_path: str = None
    latent_znormalize: bool = False
    latent_running_znormalize: bool = False
    latent_gen_type: GenerativeType = GenerativeType.ddpm
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_model_mse_weight_type: MSEWeightType = MSEWeightType.var
    latent_xstart_weight_type: XStartWeightType = XStartWeightType.uniform
    latent_rescale_timesteps: bool = False
    latent_T_eval: int = 1_000
    latent_clip_sample: bool = False
    latent_beta_scheduler: str = 'linear'
    beta_scheduler: str = 'linear'
    data_name: str = 'ffhq'
    data_val_name: str = None
    def_beta_1: float = 1e-4
    def_beta_T: float = 0.02
    def_mean_type: str = 'epsilon'
    def_var_type: str = 'fixedlarge'
    device: str = 'cuda:0'
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1
    img_size: int = 64
    kl_coef: float = None
    chamfer_coef: float = 1
    chamfer_type: ChamferType = ChamferType.chamfer
    lr: float = 0.0002
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the the number of style channels
    net_beatgans_embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = 'depthconv'
    net_enc_pool_tail_layer: int = None
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_style_layer: int = 8
    net_beatgans_style_lr_mul: float = 0.1
    net_beatgans_style_time_mode: TimeMode = None
    net_beatgans_time_style_layer: int = 2
    net_beatgans_resnet_condition_scale_bias: float = 1
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_time_emb_2xwidth: bool = True
    net_beatgans_resnet_cond_emb_2xwidth: bool = True
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_beatgans_use_mid_attn: bool = True
    mmd_alphas: Tuple[float] = (0.5, )
    mmd_coef: float = 0.1
    latent_detach: bool = True
    latent_unit_normalize: bool = False
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_k: int = None
    net_enc_name: EncoderName = EncoderName.v1
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_tail_depth: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_enc_vectorizer_type: VectorizerType = None
    net_enc_tanh: bool = False
    net_autoenc_cond_at: CondAt = CondAt.all
    net_autoenc_time_at: CondAt = CondAt.all
    net_autoenc_has_init: bool = False
    net_autoenc_merger_type: MergerType = MergerType.conv1
    net_autoenc_stochastic: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_attn_resolutions: Tuple[int] = tuple()
    net_latent_blocks: int = None
    net_latent_channel_mult: Tuple[int] = (1, 2, 4)
    net_latent_cond_both: bool = True
    net_latent_condition_2x: bool = False
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type: LatentNetType = LatentNetType.none
    net_latent_num_hid_channels: int = 1024
    net_latent_num_res_blocks: int = 2
    net_latent_num_time_layers: int = 2
    net_latent_pooling: str = 'linear'
    net_latent_project_size: int = 4
    net_latent_residual: bool = False
    net_latent_skip_layers: Tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_time_layer_init: bool = False
    net_latent_unpool: str = 'conv'
    net_latent_use_mid_attn: bool = True
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_noise_type: NoiseNetType = None
    net_noise_num_hid_channels: int = 1024
    net_noise_num_layers: int = 3
    net_noise_activation: Activation = Activation.silu
    net_noise_use_norm: bool = False
    net_noise_dropout: float = 0
    net_noise_last_act: Activation = Activation.none
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    sample_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 5000
    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: Tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'logs'
    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.expanduser('~/cache')
    work_cache_dir: str = os.path.expanduser('~/mycache')

    # data_cache_dir: str = os.path.expanduser('/scratch/konpat')
    # work_cache_dir: str = os.path.expanduser('/scratch/konpat')

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

    @property
    def name(self):
        self.make_model_conf()
        names = []
        tmp = f'{self.data_name}{self.img_size}-bs{self.batch_size}'
        if self.accum_batches > 1:
            tmp += f'accum{self.accum_batches}'
        if self.optimizer != OptimizerType.adam:
            tmp += f'-{self.optimizer.value}lr{self.lr}'
        else:
            tmp += f'-lr{self.lr}'
        if self.weight_decay > 0:
            tmp += f'wd{self.weight_decay}'
        if self.grad_clip != 1:
            if self.grad_clip < 0:
                tmp += '-noclip'
            else:
                tmp += f'-clip{self.grad_clip}'
        if self.warmup != 5000:
            tmp += f'-warmup{self.warmup}'

        if self.train_mode.is_manipulate():
            tmp += f'_mani{self.manipulate_mode.value}'
            if self.manipulate_mode.is_single_class():
                tmp += f'-{self.manipulate_cls}'
            if self.manipulate_mode.is_fewshot():
                tmp += f'-{self.manipulate_shots}shots'
            if self.manipulate_znormalize:
                tmp += '-znorm'
            if self.manipulate_mode.is_fewshot():
                tmp += f'-seed{self.manipulate_seed}'

        if self.train_mode.is_diffusion():
            tmp += f'_ddpm-T{self.T}-Tgen{self.T_eval}'
            if self.diffusion_type == 'default':
                tmp += '-default'
            elif self.diffusion_type == 'beatgans':
                tmp += f'-beatgans-gen{self.beatgans_gen_type.value}'
                if self.beta_scheduler != 'linear':
                    tmp += f'-beta{self.beta_scheduler}'
                if self.beatgans_model_mean_type != ModelMeanType.eps:
                    tmp += f'-pred{self.beatgans_model_mean_type.value}'
                if self.beatgans_loss_type != LossType.mse:
                    tmp += f'-loss{self.beatgans_loss_type.value}'
                    if self.beatgans_loss_type == LossType.mse_var_weighted:
                        tmp += f'{self.beatgans_model_mse_weight_type.value}'
                else:
                    if self.beatgans_model_mean_type == ModelMeanType.start_x:
                        tmp += f'-weight{self.beatgans_xstart_weight_type.value}'
                if self.beatgans_model_var_type != ModelVarType.fixed_large:
                    tmp += f'-var{self.beatgans_model_var_type.value}'
                if self.model_type == ModelType.mmdddpm:
                    tmp += f'-mmd{self.mmd_coef}alphas(' + ','.join(
                        str(x) for x in self.mmd_alphas) + ')'

        if self.train_mode.is_interpolate():
            tmp += f'_{self.train_mode.value}-p{self.train_interpolate_prob}'
            if self.train_interpolate_img:
                tmp += '-img'

        if self.train_mode.use_latent_net():
            # latent diffusion configs
            tmp += f'_latentddpm-Tgen{self.latent_T_eval}'
            if self.latent_beta_scheduler != 'linear':
                tmp += f'-beta{self.latent_beta_scheduler}'
            tmp += f'-gen{self.latent_gen_type.value}'
            if self.latent_model_mean_type != ModelMeanType.eps:
                tmp += f'-pred{self.latent_model_mean_type.value}'
            if self.latent_loss_type != LossType.mse:
                tmp += f'-loss{self.latent_loss_type.value}'
                if self.latent_loss_type == LossType.mse_var_weighted:
                    tmp += f'{self.latent_model_mse_weight_type.value}'
            else:
                if self.latent_model_mean_type == ModelMeanType.start_x:
                    tmp += f'-weight{self.latent_xstart_weight_type.value}'
            if self.latent_model_var_type != ModelVarType.fixed_large:
                tmp += f'-var{self.latent_model_var_type.value}'

        if self.train_mode.is_latent_diffusion():
            if self.latent_znormalize:
                tmp += '-znorm'
                if self.latent_running_znormalize:
                    tmp += 'run'
            if self.latent_clip_sample:
                tmp += '-clip'
            if self.train_mode == TrainMode.double_diffusion and self.latent_detach:
                tmp += '-detach'
            if self.latent_unit_normalize:
                tmp += '-unit'

        if self.ema_decay != 0.9999 and not self.train_mode.is_manipulate():
            tmp += f'-ema{self.ema_decay}'

        if self.fp16:
            tmp += '_fp16'

        if self.train_mode.is_manipulate():
            pass
        elif self.train_mode.is_diffusion():
            pass
        elif self.train_mode.is_latent_diffusion():
            pass
        elif self.train_mode == TrainMode.autoenc:
            tmp += '_autoenc'
        elif self.train_mode == TrainMode.latent_mmd:
            tmp += '_latentmmd'
            tmp += f'-mmd{self.mmd_coef}alphas(' + ','.join(
                str(x) for x in self.mmd_alphas) + ')'
        elif self.train_mode == TrainMode.generative_latent:
            tmp += '_genlatent'
            if self.chamfer_coef != 1:
                tmp += f'-coef{self.chamfer_coef}'
            if self.chamfer_type != ChamferType.chamfer:
                tmp += f'-{self.chamfer_type.value}'
        elif self.train_mode == TrainMode.parallel_latent_diffusion_pred:
            tmp += '_latentdiffpred'
            if self.train_cond0_prob > 0:
                tmp += f'-p{self.train_cond0_prob}'
        elif self.train_mode == TrainMode.parallel_latent_diffusion_pred_tt:
            tmp += '_latentdiffpredtt'
            if self.train_cond0_prob > 0:
                tmp += f'-p{self.train_cond0_prob}'
        elif self.train_mode == TrainMode.parallel_latent_diffusion_noisy:
            tmp += '_latentdiffnoisy'
        else:
            raise NotImplementedError()

        if self.train_mode.is_parallel_latent_diffusion():
            if not self.train_pred_xstart_detach:
                tmp += '-notdetach'

        if self.pretrain is not None:
            tmp += f'_pt{self.pretrain.name}'

        if self.continue_from is not None:
            tmp += f'_contd{self.continue_from.name}'

        names.append(tmp)
        names.append(self.model_conf.name)
        return '/'.join(names) + self.postfix

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def fid_cache(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/eval_images/{self.data_name}_size{self.img_size}_{self.eval_num_images}'

    @property
    def data_path(self):
        # may use the cache dir
        path = data_paths[self.data_name][0]
        if self.use_cache_dataset and path is not None:
            path = use_cached_dataset_path(
                path, f'{self.data_cache_dir}/{self.data_name}')
        return path

    @property
    def data_val_path(self):
        path = data_paths[self.data_name][1]
        if self.use_cache_dataset and path is not None:
            path = use_cached_dataset_path(
                path, f'{self.data_cache_dir}/{self.data_name}_val')
        return path

    @property
    def logdir(self):
        return f'{self.base_dir}/{self.name}'

    @property
    def generate_dir(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/gen_images/{self.name}'

    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == 'default':
            assert T == self.T
            assert self.beta_scheduler == 'linear'
            return DiffusionDefaultConfig(beta_1=self.def_beta_1,
                                          beta_T=self.def_beta_T,
                                          T=self.T,
                                          img_size=self.img_size,
                                          mean_type=self.def_mean_type,
                                          var_type=self.def_var_type,
                                          model_type=self.model_type,
                                          kl_coef=self.kl_coef,
                                          fp16=self.fp16)
        elif self.diffusion_type == 'beatgans':
            # can use T < self.T for evaluation
            # follows the guided-diffusion repo conventions
            # t's are evenly spaced
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError()

            return SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, self.T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                model_mse_weight_type=self.beatgans_model_mse_weight_type,
                xstart_weight_type=self.beatgans_xstart_weight_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=self.T,
                                              section_counts=section_counts),
                fp16=self.fp16,
                mmd_alphas=self.mmd_alphas,
                mmd_coef=self.mmd_coef,
            )
        else:
            raise NotImplementedError()

    def _make_latent_diffusion_conf(self, T=None):
        # can use T < self.T for evaluation
        # follows the guided-diffusion repo conventions
        # t's are evenly spaced
        if self.latent_gen_type == GenerativeType.ddpm:
            section_counts = [T]
        elif self.latent_gen_type == GenerativeType.ddim:
            section_counts = f'ddim{T}'
        else:
            raise NotImplementedError()

        return SpacedDiffusionBeatGansConfig(
            train_pred_xstart_detach=self.train_pred_xstart_detach,
            gen_type=self.latent_gen_type,
            # latent's model is always ddpm
            model_type=ModelType.ddpm,
            # latent shares the beta scheduler and full T
            betas=get_named_beta_schedule(self.latent_beta_scheduler, self.T),
            model_mean_type=self.latent_model_mean_type,
            model_var_type=self.latent_model_var_type,
            model_mse_weight_type=self.latent_model_mse_weight_type,
            xstart_weight_type=self.latent_xstart_weight_type,
            loss_type=self.latent_loss_type,
            rescale_timesteps=self.latent_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T,
                                          section_counts=section_counts),
            fp16=self.fp16,
        )

    @property
    def model_out_channels(self):
        if self.diffusion_type == 'beatgans':
            if self.beatgans_model_var_type in [
                    ModelVarType.learned, ModelVarType.learned_range
            ]:
                return 6
            else:
                return 3
        else:
            return 3

    def make_T_sampler(self):
        if self.T_sampler == 'uniform':
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_latent_diffusion_conf(self):
        return self._make_latent_diffusion_conf(T=self.T)

    def make_latent_eval_diffusion_conf(self):
        # latent can have different eval T
        return self._make_latent_diffusion_conf(T=self.latent_T_eval)

    def make_dataset(self, path=None, **kwargs):
        if self.data_name == 'ffhqlmdb256':
            return FFHQlmdb(path=path or self.data_path,
                            image_size=self.img_size,
                            **kwargs)
        elif self.data_name == 'ffhqlmdbsplit256':
            return FFHQlmdb(path=path or self.data_path,
                            image_size=self.img_size,
                            split='train',
                            **kwargs)
        elif self.data_name == 'horse':
            return LSUNHorse(path=path or self.data_path,
                             image_size=self.img_size,
                             **kwargs)
        elif self.data_name == 'horse256':
            return Horse_lmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              **kwargs)
        elif self.data_name == 'bedroom256':
            return Horse_lmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              **kwargs)
        elif self.data_name == 'celebalmdb':
            return CelebAlmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              original_resolution=None,
                              crop_d2c=True,
                              **kwargs)
        elif self.data_name == 'celebaalignlmdb':
            return CelebAlmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              **kwargs)

        elif self.data_name == 'celebahq':
            return CelebHQLMDB(path=path or self.data_path,
                               image_size=self.img_size,
                               **kwargs)
        else:
            return ImageDataset(folder=path or self.data_path,
                                image_size=self.img_size,
                                **kwargs)

    def make_test_dataset(self, **kwargs):
        if self.data_val_name == 'ffhqlmdbsplit256':
            print('test on ffhq split')
            return FFHQlmdb(path=data_paths['ffhqlmdbsplit256'][0],
                            original_resolution=256,
                            image_size=self.img_size,
                            split='test',
                            **kwargs)
        elif self.data_val_name == 'celebhq':
            print('test on celebhq')
            return CelebHQLMDB(path=data_paths['celebahq'][0],
                               original_resolution=256,
                               image_size=self.img_size,
                               **kwargs)
        else:
            return None

    def make_loader(self,
                    dataset,
                    shuffle: bool,
                    num_worker: bool = None,
                    drop_last: bool = True,
                    batch_size: int = None,
                    parallel: bool = False):
        if parallel and distributed.is_initialized():
            # drop last to make sure that there is no added special indexes
            sampler = DistributedSampler(dataset,
                                         shuffle=shuffle,
                                         drop_last=True)
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            # with sampler, use the sample instead of this option
            shuffle=False if sampler else shuffle,
            num_workers=num_worker or self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            multiprocessing_context=get_context('fork'),
        )

    def make_model_conf(self):
        if self.net_enc_name is None:
            style_enc_conf = None
        elif self.net_enc_name == EncoderName.v1:
            style_enc_conf = StyleEncoderConfig(
                img_size=self.img_size,
                style_ch=self.style_ch,
                ch=self.net_ch,
                ch_mult=self.net_ch_mult,
                attn=self.net_enc_attn,
                num_res_blocks=self.net_enc_num_res_blocks,
                dropout=self.dropout)
        elif self.net_enc_name == EncoderName.v2:
            style_enc_conf = StyleEncoder2Config(
                img_size=self.img_size,
                style_ch=self.style_ch,
                ch=self.net_ch,
                ch_mult=self.net_ch_mult,
                attn=self.net_enc_attn,
                num_res_blocks=self.net_enc_num_res_blocks,
                dropout=self.dropout,
                tail_depth=self.net_enc_tail_depth,
                pooling=self.net_enc_pooling,
                k=self.net_enc_k,
            )
        else:
            raise NotImplementedError()

        if self.model_name == ModelName.default_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = UNetConfig(
                img_size=self.img_size,
                T=self.T,
                ch=self.net_ch,
                ch_mult=self.net_ch_mult,
                attn=self.net_attn,
                num_res_blocks=self.net_num_res_blocks,
                dropout=self.dropout,
            )
        elif self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_fp16=False,
                use_new_attention_order=False,
                resnet_condition_scale_bias=self.
                net_beatgans_resnet_condition_scale_bias,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_time_emb_2xwidth=self.
                net_beatgans_resnet_time_emb_2xwidth,
                resnet_cond_emb_2xwidth=self.
                net_beatgans_resnet_cond_emb_2xwidth,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
                resnet_scale_at=self.net_beatgans_resnet_scale_at,
                use_mid_attn=self.net_beatgans_use_mid_attn,
            )
        elif self.model_name in [
                ModelName.beatgans_autoenc,
                ModelName.beatgans_vaeddpm,
                ModelName.beatgans_mmddpm,
                ModelName.beatgans_gen_latent,
        ]:
            cls = BeatGANsAutoencConfig
            # supports both autoenc and vaeddpm
            if self.model_name == ModelName.beatgans_autoenc:
                self.model_type = ModelType.autoencoder
            elif self.model_name == ModelName.beatgans_vaeddpm:
                self.model_type = ModelType.vaeddpm
            elif self.model_name == ModelName.beatgans_mmddpm:
                self.model_type = ModelType.mmdddpm
            elif self.model_name == ModelName.beatgans_gen_latent:
                self.model_type = ModelType.genlatent
                cls = LatentGenerativeModelConfig
            else:
                raise NotImplementedError()

            if self.net_latent_net_type == LatentNetType.none:
                latent_net_conf = None
            elif self.net_latent_net_type == LatentNetType.skip:
                latent_net_conf = MLPSkipNetConfig(
                    num_channels=self.style_ch,
                    skip_layers=self.net_latent_skip_layers,
                    num_hid_channels=self.net_latent_num_hid_channels,
                    num_layers=self.net_latent_layers,
                    num_time_emb_channels=self.net_latent_time_emb_channels,
                    activation=self.net_latent_activation,
                    use_norm=self.net_latent_use_norm,
                    condition_2x=self.net_latent_condition_2x,
                    condition_bias=self.net_latent_condition_bias,
                    dropout=self.net_latent_dropout,
                    last_act=self.net_latent_net_last_act,
                    num_time_layers=self.net_latent_num_time_layers,
                    time_layer_init=self.net_latent_time_layer_init,
                    residual=self.net_latent_residual,
                    time_last_act=self.net_latent_time_last_act,
                )
            else:
                raise NotImplementedError()

            if self.net_noise_type is not None:
                noise_net_conf = NoiseNetConfig(
                    type=self.net_noise_type,
                    num_channels=self.style_ch,
                    num_hid_channels=self.net_noise_num_hid_channels,
                    num_layers=self.net_noise_num_layers,
                    activation=self.net_noise_activation,
                    use_norm=self.net_noise_use_norm,
                    dropout=self.net_noise_dropout,
                    last_act=self.net_noise_last_act,
                )
            else:
                noise_net_conf = None

            self.model_conf = cls(
                is_stochastic=(self.model_type == ModelType.vaeddpm
                               or self.net_autoenc_stochastic),
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_pool_tail_layer=self.net_enc_pool_tail_layer,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                style_layer=self.net_beatgans_style_layer,
                style_lr_mul=self.net_beatgans_style_lr_mul,
                style_time_mode=self.net_beatgans_style_time_mode,
                time_style_layer=self.net_beatgans_time_style_layer,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_fp16=False,
                use_new_attention_order=False,
                enc_tanh=self.net_enc_tanh,
                resnet_condition_scale_bias=self.
                net_beatgans_resnet_condition_scale_bias,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_time_emb_2xwidth=self.
                net_beatgans_resnet_time_emb_2xwidth,
                resnet_cond_emb_2xwidth=self.
                net_beatgans_resnet_cond_emb_2xwidth,
                vectorizer_type=self.net_enc_vectorizer_type,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
                cond_at=self.net_autoenc_cond_at,
                time_at=self.net_autoenc_time_at,
                has_init=self.net_autoenc_has_init,
                merger_type=self.net_autoenc_merger_type,
                latent_net_conf=latent_net_conf,
                noise_net_conf=noise_net_conf,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
                use_mid_attn=self.net_beatgans_use_mid_attn,
            )
        elif self.model_name == ModelName.default_autoenc:
            self.model_type = ModelType.autoencoder
            self.model_conf = StyleUNetConfig(
                img_size=self.img_size,
                T=self.T,
                ch=self.net_ch,
                ch_mult=self.net_ch_mult,
                attn=self.net_attn,
                num_res_blocks=self.net_num_res_blocks,
                dropout=self.dropout,
                style_ch=self.style_ch,
                mid_attn=self.autoenc_mid_attn,
                style_enc_conf=style_enc_conf,
            )
        elif self.model_name == ModelName.default_vaeddpm:
            self.model_type = ModelType.vaeddpm
            self.model_conf = VAEStyleUNetConfig(
                img_size=self.img_size,
                T=self.T,
                ch=self.net_ch,
                ch_mult=self.net_ch_mult,
                attn=self.net_attn,
                num_res_blocks=self.net_num_res_blocks,
                dropout=self.dropout,
                style_ch=self.style_ch,
            )
        elif self.model_name == ModelName.beatgans_encoder:
            self.model_type = ModelType.encoder
            self.model_conf = BeatGANsEncoderConfig(
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                out_hid_channels=None,
                out_channels=self.net_enc_num_cls,
                num_res_blocks=self.net_enc_num_res_blocks,
                attention_resolutions=self.net_enc_attn,
                dropout=self.dropout,
                channel_mult=self.net_enc_channel_mult,
                num_head_channels=1,
                use_time_condition=self.net_enc_use_time,
                conv_resample=True,
                dims=2,
                resblock_updown=True,
                pool=self.net_enc_pool,
                use_checkpoint=self.net_enc_grad_checkpoint,
            )
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf
