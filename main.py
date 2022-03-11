# %%
import matplotlib.pyplot as plt
from backbone import *
from functools import partial
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
# %%

# Partly from https://nextjournal.com/ml4a/image-t-sne

def get_images(file, *args, **kwargs):
    return np.array(Image.open(file).convert("RGB").resize((image_size, image_size), Image.ANTIALIAS))

def preprocess(im, *args, **kwargs):
    out_im = im[30:250, 100:230] #example
    return out_im
# %%
main_dir = "/media/hdd/Datasets/Fish_Dataset/Fish_Dataset/" #folder path
k = 100 # subset image count
type_im = "png" #image time
image_size = 128 #px
iters = 2000 # tsne iterations
nclust = 12 # No of types you have. Approximately works too

# %%
all_files = glob.glob(str(Path(main_dir)/f"**/*.{type_im}"), recursive=True)
random.shuffle(all_files)
if k == None:
    files = all_files
else:
    files = all_files[:k]

features = parallel(partial(get_images, image_size = image_size), files)

#features = parallel(preprocess, features) #uncomment these two if you need it
#image_size = features[1].shape #this too

pca = sklearn.decomposition.PCA(n_components=2)
tsne = manifold.TSNE(n_components=2,n_iter=iters, n_jobs = -1)
if k != None:
    features = features[:k]
features = np.array(features).reshape(k, -1)
print("FITTING")
mnist_tr = pca.fit_transform(tsne.fit_transform(features))

# clustering
if nclust != None:
    df = run_clustering(mnist_tr, files, nclust)
print(image_size)
# Plot the output of the unsupervised TSNE+ PCA
plot_unsupervised(mnist_tr, features,image_size)
print("DONE")

# %%
