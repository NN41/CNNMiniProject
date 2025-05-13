# %% Set up

import numpy as np
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device\n")

# %% Load MNIST

training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor()
)

fig, axes = plt.subplots(3,3,figsize=(8,8))
axes = axes.flatten()
for i in range(len(axes)):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(str(label))

# %%
