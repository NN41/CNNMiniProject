# %% Set up

import numpy as np
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F

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

next(iter(train_dataloader))[0].shape # must be of shape (N, C, H, W)

# %%

# dataloader_for_normalization = DataLoader(training_data, batch_size=len(training_data))
# data_for_normalization = next(iter(dataloader_for_normalization))[0]
# mean_for_normalization = torch.mean(data_for_normalization).item() # = 0.13066047430038452
# std_for_normalization = torch.std(data_for_normalization).item() # = 0.30810782313346863
# print(f'The mean and std of the pixel values over the training data is mean {mean_for_normalization} and std {std_for_normalization}')

# %% define model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=4*14*14, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x # note that x are logits


# %% train model

model = CNN().to(device)
# model.named_parameters()

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0)

n_batches = len(train_dataloader)
running_loss = 0
running_n_batches = 0
EPOCHS = 5

model.train()
for epoch in range(EPOCHS):
    for batch, (X, y) in enumerate(train_dataloader):

        X = X.to(device)
        y = y.to(device)

        # perform forward pass and compute loss
        logits = model(X)
        loss = criterion(logits, y)

        # perform back-prop and update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute and print statistics
        running_loss += loss
        running_n_batches += 1
        if batch % 300 == 0:
            print(f"epoch {epoch+1:3d} / {EPOCHS} | batch {batch+1:5d} / {n_batches} | loss {loss.item():.5f} | running loss {running_loss.item() / running_n_batches:.5f}")
            running_loss = 0
            running_n_batches = 0
print("Finished Training!")

# %%

running_loss = 0
accuracy = 0
n_correct = 0
n_examples = 0

model.eval()
with torch.no_grad():
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        batch_loss = criterion(logits, y)
        running_loss += batch_loss.item() * len(y)
        n_examples += len(y)
        n_correct += (logits.argmax(dim=1) == y).sum().item()
        # if batch % 100 == 0:
        #     print(f"batch {batch+1:5d} / {len(test_dataloader)}")

accuracy = n_correct / n_examples
loss = running_loss / n_examples

print(f"Test error: loss = {loss:.5f}, accuracy = {accuracy*100:.2f}%\n")