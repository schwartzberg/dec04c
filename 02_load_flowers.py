"""Problem 2, step 1: load and inspect the flower photos. Run from the repository folder."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

# Separate names keep the flower data distinct from Problem 1's MNIST data.
FLOWER_DATA_DIR = Path("data")
FLOWER_IMAGE_SIZE = 64
flower_root = FLOWER_DATA_DIR / "flower_photos"

# Reuse the downloaded photos on later runs.
if not flower_root.exists():
    download_and_extract_archive(
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        download_root=str(FLOWER_DATA_DIR),
        md5="6f87fb78e9cc9ab41eff2015b380011d",
    )

# Resize to a common size and convert to tensors. No data augmentation.
flower_transform = transforms.Compose([
    transforms.Resize((FLOWER_IMAGE_SIZE, FLOWER_IMAGE_SIZE)),
    transforms.ToTensor(),
])
flower_catalog = datasets.ImageFolder(flower_root, transform=flower_transform)
flower_class_names = flower_catalog.classes
print(f"Images: {len(flower_catalog):,}")
print(f"Folder-to-label mapping: {flower_catalog.class_to_idx}")
flower_image, flower_label = flower_catalog[0]
print(f"One image: {tuple(flower_image.shape)}")
print(f"Label: {flower_label} ({flower_class_names[flower_label]})")

# Show one example per class to check the images and folder labels.
fig, axes = plt.subplots(1, len(flower_class_names), figsize=(15, 3))
for label, ax in enumerate(axes):
    index = flower_catalog.targets.index(label)
    image, target = flower_catalog[index]
    # RGB plotting needs (height, width, channels), rather than (channels, height, width).
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(flower_class_names[target])
    ax.axis("off")
fig.suptitle("Flower Photos: one example per class")
plt.tight_layout()
plt.show()
