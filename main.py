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
    axes[i].imshow(img.squeeze(0), cmap="gray")
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
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=4*7*7, out_features=10) # in_features depends on channels of last conv layer and the strides used in pooling

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
        x = self.fc(x)
        return x # note that x are logits

model = CNN().to(device)
XX, _ = next(iter(train_dataloader))
model(XX.to(device)).shape
model

# %%

def test(dataloader, model, criterion):
    running_loss = 0
    accuracy = 0
    n_correct = 0
    n_examples = 0

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            batch_loss = criterion(logits, y)
            running_loss += batch_loss.item() * len(y)
            n_examples += len(y)
            n_correct += (logits.argmax(dim=1) == y).sum().item()
            # if batch % 100 == 0:
            #     print(f"batch {batch+1:5d} / {len(dataloader)}")

    accuracy = n_correct / n_examples
    loss = running_loss / n_examples
    print(f"Test error: loss = {loss:.5f}, accuracy = {accuracy*100:.2f}%\n")
    model.train()

# %% train model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_batches = len(train_dataloader)
running_loss = 0
running_n_batches = 0
EPOCHS = 25


test(train_dataloader, model, criterion)
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
            print(f"epoch {epoch+1:3d} / {EPOCHS} | batch {batch+1:5d} / {n_batches} | avg loss between updates {running_loss.item() / running_n_batches:.5f}")
            running_loss = 0
            running_n_batches = 0

    test(train_dataloader, model, criterion)
print("Finished Training!")

# %%

def visualize_kernel_weights(layer, layer_name):
    weights_all = layer.weight.data.clone().to("cpu") # of shape (out_channels, in_channels, k_H, k_W)

    num_kernels = weights_all.shape[0]
    num_in_channels = weights_all.shape[1]

    weights_to_plot = []

    for idx_kernel in range(num_kernels):
        for idx_channel in range(num_in_channels):
            kernel = weights_all[idx_kernel]
            weights = kernel[idx_channel]
            name_to_display = f"ker {idx_kernel}: chan {idx_channel}"
            weights_to_display = (weights - weights.min()) / (weights.max() - weights.min())
            weights_to_plot.append([name_to_display, weights_to_display])

    cols = num_kernels
    rows = num_in_channels
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for i in range(cols * rows):
        name, weights = weights_to_plot[i]
        axes[i].imshow(weights, cmap="gray")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(name)
    fig.suptitle(f"{layer_name} kernels")
    plt.show()

visualize_kernel_weights(model.conv1, "Conv1")
# visualize_kernel_weights(model.conv2, "Conv2")

# %% visualize feature maps (activations)

# --- set up hooks for activation maps ---
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# --- register hooks ---
handles = []
handles.append(model.relu1.register_forward_hook(get_activation('relu1_out')))
handles.append(model.relu2.register_forward_hook(get_activation('relu2_out')))

# --- get sample data for activation maps ---
sample_idx = 9
batch = next(iter(test_dataloader))
sample_images, sample_labels = batch
sample_images = sample_images[sample_idx:sample_idx+3]
sample_labels =  sample_labels[sample_idx:sample_idx+3]

rows = 4
cols = 1
fig, axes = plt.subplots(1,3, figsize=(6,6))
axes = axes.flatten()
for i in range(len(sample_images)):
    img = sample_images[i].squeeze(0)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(sample_labels[i].item())
    axes[i].set_xticks([])
    axes[i].set_yticks([])

# --- perform forward pass using sample data ---
with torch.no_grad():
    model.eval()
    model(sample_images.to(device))

# --- define function for visualizing activation maps ---
def visualize_activations(activations_dict, layer_name):
    all_activations = activations_dict[layer_name] # of shape (batch_size, num_channels, img_H, img_W)
    num_kernels = all_activations.shape[1]
    num_samples = all_activations.shape[0]
    cols = num_samples
    rows = num_kernels
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for idx_sample in range(num_samples):
        for idx_kernel in range(num_kernels):
            activation_map = all_activations[idx_sample, idx_kernel].cpu()
            ax = axes[idx_kernel, idx_sample]
            ax.imshow(activation_map, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"sample {idx_sample} | ker {idx_kernel}")
    fig.suptitle(f'Activations of kernels (rows) \nper sample (cols) for layer {layer_name}\n')

# --- plot activation maps ---
visualize_activations(activations, 'relu1_out')
visualize_activations(activations, 'relu2_out')

# --- remove hooks ---
for handle in handles:
    handle.remove()
activations = {}

# %% feature visualization by maximizing activations

model.eval()
verbose = False
# target_layers = [model.relu1, model.relu2]

# target_layer_idx = 0
# target_layer = target_layers[target_layer_idx]
idx_target_channel = 1
target_layer = model.conv2

def find_activation_maximizer(target_layer, idx_target_channel):

    height = width = 28
    num_samples = 1
    num_channels = 1
    opt_img = torch.rand(num_samples, num_channels, height, width, device=device, requires_grad=True)
    with torch.no_grad():
        opt_img.data = (opt_img.data - opt_img.mean()) / opt_img.std()
    plt.imshow(opt_img.squeeze().cpu().detach().numpy(), cmap='gray')

    img_optimizer = torch.optim.Adam(
        [opt_img], 
        lr=0.005,
        weight_decay=1e-3
    )

    layer_name = target_layer.__class__.__name__
    activation_store = {}
    def hook(module, input, output):
        activation_store[layer_name] = output
    hook_handle = target_layer.register_forward_hook(hook)


    iterations = 5000

    for i in range(iterations):

        model(opt_img)
        layer_activation = activation_store[layer_name]
        channel_activation = layer_activation[0, idx_target_channel]

        objective = channel_activation.mean()
        loss = -objective
        
        img_optimizer.zero_grad()
        loss.backward()
        img_optimizer.step()

        with torch.no_grad():
            opt_img.data = torch.clamp(opt_img.data, -5, 5)
            # opt_img.data = nn.Tanh()(opt_img.data)
            
        if i % 100 == 0:
            with torch.no_grad():
                noise = torch.rand(num_samples, num_channels, height, width, device=device) * 0.05
                opt_img.data = opt_img.data + noise.data

        if i % 500 == 0 and verbose:
            print(f"iteration {i+1} / {iterations} | loss = {loss.item()}")
            plt.imshow(opt_img.squeeze().cpu().detach().numpy(), cmap='gray')
            plt.show()


    hook_handle.remove()
    activation_store.clear()

    return opt_img

# opt_img = find_activation_maximizer(target_layer, idx_target_channel)
# plt.imshow(opt_img.squeeze().cpu().detach().numpy(), cmap='gray')

# %%

layers = {'conv1': model.conv1, 'conv2': model.conv2}
optimal_images = []
label_names = []

for name, target_layer in layers.items():
    num_channels = target_layer.weight.shape[0]
    for idx_target_channel in range(num_channels):
        label_name = f"{name}: chan {idx_target_channel}"
        label_names.append(label_name)
        print(f"optimizing image for {label_name}")
        opt_img = find_activation_maximizer(target_layer, idx_target_channel)
        optimal_images.append(opt_img)

rows = 2
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()
for i, img in enumerate(optimal_images):
    axes[i].imshow(img.detach().cpu().squeeze(), cmap='gray')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(label_names[i])