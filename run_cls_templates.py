from run_templates import *


def cls_celeba64_all():
    conf = celeba64_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celeba_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'latent_infer/{celeba64_autoenc().name}.pkl'
    conf.postfix = ''
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 600_000
    conf.pretrain = PretrainConfig(
        '72M',
        f'logs/{celeba64_autoenc().name}/last.ckpt',
    )
    return conf


def cls_celeba64_fewshot(cls_name, shots, seed):
    conf = celeba64_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celeba_fewshot
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'latent_infer/{celeba64_autoenc().name}.pkl'
    conf.batch_size = 32
    conf.total_samples = 100_000
    conf.lr = 1e-3
    conf.pretrain = PretrainConfig(
        '72M',
        f'logs/{celeba64_autoenc().name}/last.ckpt',
    )
    conf.manipulate_cls = cls_name
    conf.manipulate_shots = shots
    conf.manipulate_seed = seed
    conf.total_samples = shots * 1000
    return conf


def cls_celeba64_fewshot_allneg(cls_name, shots, seed):
    conf = cls_celeba64_fewshot(cls_name, shots, seed)
    conf.manipulate_mode = ManipulateMode.celeba_fewshot_allneg
    conf.total_samples = 100_000
    return conf


def cls_celeba64d2c_fewshot(cls_name, shots, seed):
    # total_sample = 10_000 like D2C paper
    conf = cls_celeba64_fewshot(cls_name, shots, seed)
    conf.manipulate_mode = ManipulateMode.d2c_fewshot
    conf.latent_infer_path = f'latent_infer/{celeba64d2c_autoenc().name}.pkl'
    conf.pretrain = PretrainConfig(
        '72M',
        f'logs/{celeba64d2c_autoenc().name}/last.ckpt',
    )
    return conf


def cls_celeba64d2c_fewshot_long(cls_name, shots, seed):
    conf = cls_celeba64d2c_fewshot(cls_name, shots, seed)
    conf.postfix = '_long'
    conf.total_samples = 600_000
    return conf


def cls_celeba64d2c_fewshot_allneg(cls_name, shots, seed):
    conf = cls_celeba64d2c_fewshot(cls_name, shots, seed)
    conf.manipulate_mode = ManipulateMode.d2c_fewshot_allneg
    conf.total_samples = 100_000

    return conf


def cls_celeba64d2c_fewshot_allneg_long(cls_name, shots, seed, is_negative=False):
    if is_negative:  cls_name +=  '_neg'   #TODO: I added this 
    conf = cls_celeba64d2c_fewshot_allneg(cls_name, shots, seed)
    conf.is_negative = is_negative    #TODO: I added this  
    conf.postfix = '_long'
    conf.total_samples = 600_000

    return conf

# ness
def cls_celeba64d2c_fewshot_allneg_long_negative(cls_name, shots, seed):
    conf = cls_celeba64d2c_fewshot_allneg(cls_name, shots, seed)
    conf.postfix = '_long'
    conf.total_samples = 600_000
    return conf

def cls_ffhq128_all():
    conf = ffhq128_autoenc_200M()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'latent_infer/{ffhq128_autoenc_200M().name}.pkl'
    conf.postfix = ''
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'logs/{ffhq128_autoenc_200M().name}/last.ckpt',
    )
    conf.continue_from = None
    return conf


def cls_ffhq256_all():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    # conf.latent_infer_path = f'latent_infer/{ffhq256_autoenc().name}.pkl'  # we train on Celeb dataset, not FFHQ
    conf.postfix = ''
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'logs/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.continue_from = None
    return conf
