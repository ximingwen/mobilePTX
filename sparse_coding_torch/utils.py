import numpy as np
import torchvision as tv
import torch
import tensorflow as tf
from tqdm import tqdm
from torchvision.datasets.video_utils import VideoClips
from typing import Sequence, Iterator
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

def get_sample_weights(train_idx, dataset):
    dataset = list(dataset)

    num_positive = len([clip[0] for clip in dataset if clip[0] == 'Positives'])
    negative_weight = num_positive / len(dataset)
    positive_weight = 1.0 - negative_weight
    
    weights = []
    for idx in train_idx:
        label = dataset[idx][0]
        if label == 'Positives':
            weights.append(positive_weight)
        elif label == 'Negatives':
            weights.append(negative_weight)
        else:
            raise Exception('Sampler encountered invalid label')
    
    return weights

class SubsetWeightedRandomSampler(torch.utils.data.Sampler[int]):
    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], indicies: Sequence[int],
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.indicies = indicies
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, len(self.indicies), self.replacement, generator=self.generator)
        for i in rand_tensor:
            yield self.indicies[i]

    def __len__(self) -> int:
        return len(self.indicies)

class MinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __init__(self, min_val=0, max_val=254):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)

class VideoGrayScaler(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.grayscale = tv.transforms.Grayscale(num_output_channels=1)
        
    def forward(self, video):
        # shape = channels, time, width, height
        video = self.grayscale(video.swapaxes(-4, -3).swapaxes(-2, -1))
        video = video.swapaxes(-4, -3).swapaxes(-2, -1)
        # print(video.shape)
        return video

def plot_video(video):

    fig = plt.gcf()
    ax = plt.gca()

    DPI = fig.get_dpi()
    fig.set_size_inches(video.shape[2]/float(DPI), video.shape[3]/float(DPI))

    ax.set_title("Video")

    T = video.shape[1]
    im = ax.imshow(video[0, 0, :, :],
                     cmap=cm.Greys_r)

    def update(i):
        t = i % T
        im.set_data(video[0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/20)

def plot_original_vs_recon(original, reconstruction, idx=0):

    # create two subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.set_title("Original")
    ax2.set_title("Reconstruction")

    T = original.shape[2]
    im1 = ax1.imshow(original[idx, 0, 0, :, :],
                     cmap=cm.Greys_r)
    im2 = ax2.imshow(reconstruction[idx, 0, 0, :, :],
                     cmap=cm.Greys_r)

    def update(i):
        t = i % T
        im1.set_data(original[idx, 0, t, :, :])
        im2.set_data(reconstruction[idx, 0, t, :, :])

    return FuncAnimation(plt.gcf(), update, interval=1000/30)


def plot_filters(filters):
    filters = filters.astype('float32')
    num_filters = filters.shape[4]
    ncol = 3
    # ncol = int(np.sqrt(num_filters))
    # nrow = int(np.sqrt(num_filters))
    T = filters.shape[0]

    if num_filters // ncol == num_filters / ncol:
        nrow = num_filters // ncol
    else:
        nrow = num_filters // ncol + 1

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True,
                             figsize=(ncol*2, nrow*2))

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[0, :, :, 0, i],
                                        cmap=cm.Greys_r)

    def update(i):
        t = i % T
        for i in range(num_filters):
            r = i // ncol
            c = i % ncol
            ims[(r, c)].set_data(filters[t, :, :, 0, i])

    return FuncAnimation(plt.gcf(), update, interval=1000/20)

def plot_filters_image(filters):
    filters = filters.astype('float32')
    num_filters = filters.shape[4]
    ncol = 3
    T = filters.shape[0]

    if num_filters // ncol == num_filters / ncol:
        nrow = num_filters // ncol
    else:
        nrow = num_filters // ncol + 1

    fig, axes = plt.subplots(ncols=ncol, nrows=nrow,
                             constrained_layout=True,
                             figsize=(ncol*2, nrow*2))

    ims = {}
    for i in range(num_filters):
        r = i // ncol
        c = i % ncol
        ims[(r, c)] = axes[r, c].imshow(filters[0, :, :, 0, i],
                                        cmap=cm.Greys_r)

    return plt.gcf()