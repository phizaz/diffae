from templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    # need to first train the diffae autoencoding model & infer the latents
    # this requires only a single GPU.
    gpus = [0]
    conf = ffhq128_autoenc_cls()
    train_cls(conf, gpus=gpus)

    # after this you can do the manipulation!
