import argparse
import multiprocessing
import os
import shutil
from functools import partial
from io import BytesIO
from multiprocessing import Process, Queue
from os.path import exists, join
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import LSUNClass
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm


def resize_and_convert(img, size, resample, quality=100):
    if size is not None:
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)

    buffer = BytesIO()
    img.save(buffer, format="webp", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(idx, img, sizes, resample):
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)
    return idx, out


class ConvertDataset(Dataset):
    def __init__(self, data, size) -> None:
        self.data = data
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        bytes = resize_and_convert(img, self.size, Image.LANCZOS, quality=100)
        return bytes


class ImageFolder(Dataset):
    def __init__(self, folder, ext='jpg'):
        super().__init__()
        paths = sorted([p for p in Path(f'{folder}').glob(f'*.{ext}')])
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.paths[index])
        img = Image.open(path)
        return img


if __name__ == "__main__":
    from tqdm import tqdm

    out_path = 'datasets/celeba.lmdb'
    in_path = 'datasets/celeba'
    ext = 'jpg'
    size = None

    dataset = ImageFolder(in_path, ext)
    print('len:', len(dataset))
    dataset = ConvertDataset(dataset, size)
    loader = DataLoader(dataset,
                        batch_size=50,
                        num_workers=12,
                        collate_fn=lambda x: x,
                        shuffle=False)

    target = os.path.expanduser(out_path)
    if os.path.exists(target):
        shutil.rmtree(target)

    with lmdb.open(target, map_size=1024**4, readahead=False) as env:
        with tqdm(total=len(dataset)) as progress:
            i = 0
            for batch in loader:
                with env.begin(write=True) as txn:
                    for img in batch:
                        key = f"{size}-{str(i).zfill(7)}".encode("utf-8")
                        # print(key)
                        txn.put(key, img)
                        i += 1
                        progress.update()
                # if i == 1000:
                #     break
                # if total == len(imgset):
                #     break

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(i).encode("utf-8"))
