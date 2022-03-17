import argparse
import multiprocessing
import os
from os.path import join, exists
from functools import partial
from io import BytesIO
import shutil

import lmdb
from PIL import Image
from torchvision.datasets import LSUNClass
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm

from multiprocessing import Process, Queue


def resize_and_convert(img, size, resample, quality=100):
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


from torch.utils.data import Dataset, DataLoader


class ConvertDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, _ = self.data[index]
        bytes = resize_and_convert(img, 256, Image.LANCZOS, quality=90)
        return bytes


if __name__ == "__main__":
    """
    converting lsun' original lmdb to our lmdb, which is somehow more performant.
    """
    from tqdm import tqdm

    # path to the original lsun's lmdb
    src_path = 'datasets/bedroom_train_lmdb'
    out_path = 'datasets/bedroom256.lmdb'

    dataset = LSUNClass(root=os.path.expanduser(src_path))
    dataset = ConvertDataset(dataset)
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
                        key = f"{256}-{str(i).zfill(7)}".encode("utf-8")
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
