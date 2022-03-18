# Official implementation of Diffusion Autoencoders

A CVPR 2022 paper:

> Preechakul, Konpat, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. 2021. “Diffusion Autoencoders: Toward a Meaningful and Decodable Representation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2111.15640.

## Usage

Note: Since we expect a lot of changes on the codebase, please fork the repo before using.

### Quick start

A jupyter notebook.

For unconditional generation: `sample.ipynb`

For manipulation: `manipulate.ipynb`

### Checkpoints

We provide checkpoints for the following models:

1. DDIM: **FFHQ128** ([72M](https://drive.google.com/drive/folders/1-J8FPNZOQxSqpfTpwRXawLi2KKGL1qlK?usp=sharing), [130M](https://drive.google.com/drive/folders/17T5YJXpYdgE6cWltN8gZFxRsJzpVxnLh?usp=sharing)), [**Bedroom128**](https://drive.google.com/drive/folders/19s-lAiK7fGD5Meo5obNV5o0L3MfqU0Sk?usp=sharing), [**Horse128**](https://drive.google.com/drive/folders/1PiC5JWLcd8mZW9cghDCR0V4Hx0QCXOor?usp=sharing)
2. DiffAE (autoencoding only): [**FFHQ256**](https://drive.google.com/drive/folders/1hTP9QbYXwv_Nl5sgcZNH0yKprJx7ivC5?usp=sharing), **FFHQ128** ([72M](https://drive.google.com/drive/folders/15QHmZP1G5jEMh80R1Nbtdb4ZKb6VvfII?usp=sharing), [130M](https://drive.google.com/drive/folders/1UlwLwgv16cEqxTn7g-V2ykIyopmY_fVz?usp=sharing)), [**Bedroom128**](https://drive.google.com/drive/folders/1okhCb1RezlWmDbdEAGWMHMkUBRRXmey0?usp=sharing), [**Horse128**](https://drive.google.com/drive/folders/1Ujmv3ajeiJLOT6lF2zrQb4FimfDkMhcP?usp=sharing)
3. DiffAE (with latent DPM, can sample): [**FFHQ256**](https://drive.google.com/drive/folders/1MonJKYwVLzvCFYuVhp-l9mChq5V2XI6w?usp=sharing), [**FFHQ128**](https://drive.google.com/drive/folders/1E3Ew1p9h42h7UA1DJNK7jnb2ERybg9ji?usp=sharing), [**Bedroom128**](https://drive.google.com/drive/folders/1okhCb1RezlWmDbdEAGWMHMkUBRRXmey0?usp=sharing), [**Horse128**](https://drive.google.com/drive/folders/1Ujmv3ajeiJLOT6lF2zrQb4FimfDkMhcP?usp=sharing)
4. DiffAE's classifiers (for manipulation): [**FFHQ256's latent on CelebAHQ**](https://drive.google.com/drive/folders/1QGkTfvNhgi_TbbV8GbX1Emrp0lStsqLj?usp=sharing), [**FFHQ128's latent on CelebAHQ**](https://drive.google.com/drive/folders/1E3Ew1p9h42h7UA1DJNK7jnb2ERybg9ji?usp=sharing)

Checkpoints ought to be put into a separate directory `checkpoints`. 
Download the checkpoints and put them into `checkpoints` directory. It should look like this:

```
checkpoints/
- bedroom128_autoenc
    - last.ckpt # diffae checkpoint
    - latent.ckpt # predicted z_sem on the dataset
- bedroom128_autoenc_latent
    - last.ckpt # diffae + latent DPM checkpoint
- bedroom128_ddpm
- ...
```


### LMDB Datasets

We do not own any of the following datasets. We provide the LMDB ready-to-use dataset for the sake of convenience.

- [FFHQ](https://drive.google.com/drive/folders/1ww7itaSo53NDMa0q-wn-3HWZ3HHqK1IK?usp=sharing)
- [CelebAHQ](https://drive.google.com/drive/folders/1SX3JuVHjYA8sA28EGxr_IoHJ63s4Btbl?usp=sharing) 
- [LSUN Bedroom](https://drive.google.com/drive/folders/1O_3aT3LtY1YDE2pOQCp6MFpCk7Pcpkhb?usp=sharing)
- [LSUN Horse](https://drive.google.com/drive/folders/1ooHW7VivZUs4i5CarPaWxakCwfeqAK8l?usp=sharing)

The directory tree should be:

```
datasets/
- bedroom256.lmdb
- celebahq256.lmdb
- ffhq256.lmdb
- horse256.lmdb
```

You can also download from the original sources, and use our provided codes to package them as LMDB files.
Original sources for each dataset is as follows:

- FFHQ (https://github.com/NVlabs/ffhq-dataset)
- CelebAHQ (https://github.com/switchablenorms/CelebAMask-HQ)
- LSUN (https://github.com/fyu/lsun)

The conversion codes are provided as:

```
data_resize_bedroom.py
data_resize_celebhq.py
data_resize_ffhq.py
data_resize_horse.py
```

Google drive: https://drive.google.com/drive/folders/1abNP4QKGbNnymjn8607BF0cwxX2L23jh?usp=sharing


## Training

Soon ...