from templates import *
from templates_latent import *

if __name__ == '__main__':
    gpus = [0, 1, 2, 3]
    conf = ffhq128_ddpm_130M()
    train(conf, gpus=gpus)

    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid10']
    train(conf, gpus=gpus, mode='eval')