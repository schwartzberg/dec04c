"""Problem 1, step 1: load and inspect MNIST. Run from the repository folder."""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

# Fixed seeds make later splits and model initialisation more repeatable.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Python executable: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"Device: {device}")

# ToTensor converts each greyscale image to a float tensor of shape
# (1, 28, 28) and scales its pixel values from 0-255 to 0-1.
transform = transforms.ToTensor()
DATA_DIR = Path("data")

mnist_train_full = datasets.MNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
)

# Inspect a training example; keep the test set for final evaluation.
image, label = mnist_train_full[0]
print(f"Training pool: {len(mnist_train_full):,} images")
print(f"Test set: {len(mnist_test):,} images")
print(f"One image: {tuple(image.shape)}")
print(f"Tensor type: {image.dtype}")
print(f"Pixel range in this image: {image.min().item():.1f} to {image.max().item():.1f}")
print(f"Label: {label}")

# Show one training image for each digit.
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit, ax in enumerate(axes.flat):
    index = (mnist_train_full.targets == digit).nonzero(as_tuple=True)[0][0].item()
    image, label = mnist_train_full[index]
    ax.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Label: {label}")
    ax.axis("off")
fig.suptitle("MNIST: one training example per digit")
plt.tight_layout()
plt.show()
