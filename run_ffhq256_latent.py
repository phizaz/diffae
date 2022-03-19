from templates import *
from templates_latent import *

if __name__ == '__main__':
    # do run the run_ffhq256 before using the file to train the latent DPM

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf = ffhq256_autoenc()
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = ffhq256_autoenc_latent()
    train(conf, gpus=gpus)
