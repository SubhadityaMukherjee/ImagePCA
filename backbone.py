
import matplotlib.pyplot as plt

import random
import numpy as np
import pandas as pd
import concurrent
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import manifold
import sklearn
import tqdm
import glob
from PIL import Image
from typing import *
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import os
from pathlib import Path

def ifnone(a, b):
    """
    Return if None
    """
    return b if a is None else a


def listify(o):
    """
    Convert to list
    """
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


_default_cpus = max(16, num_cpus())
defaults = SimpleNamespace(
    cpus=_default_cpus, cmap="viridis", return_fig=False, silent=False
)


def parallel(func, arr: Collection, max_workers: int = None, leave=False):  # %t
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers < 2:
        results = [
            func(o, i)
            for i, o in tqdm.tqdm(enumerate(arr), total=len(arr), leave=leave)
        ]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func, o, i) for i, o in enumerate(arr)]
            results = []
            for f in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(arr), leave=leave
            ):
                results.append(f.result())
    if any([o is not None for o in results]):
        return results



def get_label(fname, *args, **kwargs):
    return Path(fname).parent.name


def run_clustering(mnist_tr, files,nclust):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters = nclust).fit_predict(mnist_tr)
    df = pd.DataFrame.from_dict({x:km[i] for i,x in enumerate(files)}.items())
    df.to_csv("output_cluster.csv")
    return df

def plot_unsupervised(mnist_tr, features, image_size):
    tx, ty = mnist_tr[:, 0], mnist_tr[:, 1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 100
    print("CREATING FIGURE")
    full_image = Image.new('RGBA', (width, height), color=(1, 1, 1))
    for img, x, y in zip(features, tx, ty):
        # tile = Image.open(img)
        # print(img)
        tile = Image.fromarray(img.reshape(image_size[0], image_size[1], 3))
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize(
            (int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x),
                         int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    plt.figure(figsize=(226, 212))
    print(full_image.size)
    full_image.convert("RGB").save("output.pdf")
