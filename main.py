# %% Set up

import numpy as np
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device\n")

# %% Load MNIST

transform_to_apply = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.13066047430038452,), std=(0.30810782313346863,))
])

training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform_to_apply
)

test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform_to_apply
)

fig, axes = plt.subplots(3,3,figsize=(8,8))
axes = axes.flatten()
for i in range(len(axes)):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(str(label))

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)


# %%

# dataloader_for_normalization = DataLoader(training_data, batch_size=len(training_data))
# data_for_normalization = next(iter(dataloader_for_normalization))[0]
# mean_for_normalization = torch.mean(data_for_normalization).item() # = 0.13066047430038452
# std_for_normalization = torch.std(data_for_normalization).item() # = 0.30810782313346863
# print(f'The mean and std of the pixel values over the training data is mean {mean_for_normalization} and std {std_for_normalization}')

training_data[0]