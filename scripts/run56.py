from run_templates import *

if __name__ == '__main__':
    gpus = [0, 1, 2, 3]
    nodes = 2

    # baseline autoenc 256
    conf = ffhq128_autoenc()
    conf.postfix = ''
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 8
    conf.scale_up_gpus(len(gpus), nodes)

    conf.eval_every_samples = 200_000_000
    conf.eval_ema_every_samples = 200_000_000

    # always run on the shared home
    conf.use_cache_dataset = False
    conf.data_cache_dir = os.path.expanduser('~/cache')
    conf.work_cache_dir = os.path.expanduser('~/mycache')

    train(conf, gpus=gpus, nodes=nodes)
