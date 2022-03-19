from templates import *
from templates_latent import *

if __name__ == '__main__':
    # 256 requires 8x v100s, in our case, on two nodes.
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    gpus = [0, 1, 2, 3]
    nodes = 2
    conf = ffhq256_autoenc()
    train(conf, gpus=gpus, nodes=nodes)