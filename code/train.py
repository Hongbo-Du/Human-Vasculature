
import os
import glob
import random

import torch
import torch.utils.data

import PIL
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook

from node_unet import ConvODEUNet, ConvResUNet, ODEBlock, Unet
from data_loader import ImageDataLoader
from train_utils import plot_losses

from IPython.display import clear_output

torch.manual_seed(0)

val_set_idx = torch.LongTensor(10).random_(0, 85)
train_set_idx = torch.arange(0, 85)

overlapping = (train_set_idx[..., None] == val_set_idx).any(-1)
train_set_idx = torch.masked_select(train_set_idx, ~overlapping)

trainset = ImageDataLoader((352, 512), dataset_repeat=1, images=train_set_idx)
valset = ImageDataLoader((352, 512), dataset_repeat=1, images=val_set_idx, validation=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=10)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=10)


