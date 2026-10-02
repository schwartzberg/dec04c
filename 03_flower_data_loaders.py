"""Problem 2: split the flower data and prepare batches.

Run from the repository folder after 02_load_flowers.py has downloaded the photos.
"""
from pathlib import Path
import torch
from torchvision import datasets, transforms

FLOWER_IMAGE_SIZE = 64
flower_catalog = datasets.ImageFolder(
    Path("data") / "flower_photos",
    transform=transforms.Compose([
        transforms.Resize((FLOWER_IMAGE_SIZE, FLOWER_IMAGE_SIZE)),
        transforms.ToTensor(),
    ]),
)
flower_class_names = flower_catalog.classes

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

FLOWER_SEED = 42
flower_indices = np.arange(len(flower_catalog))
flower_targets = np.asarray(flower_catalog.targets)

# Reserve 30% for validation and testing; use 70% for training.
flower_train_indices, flower_remaining_indices = train_test_split(
    flower_indices,
    test_size=0.30,
    random_state=FLOWER_SEED,
    stratify=flower_targets,
)

# Divide the reserved images equally between validation and testing.
flower_val_indices, flower_test_indices = train_test_split(
    flower_remaining_indices,
    test_size=0.50,
    random_state=FLOWER_SEED,
    stratify=flower_targets[flower_remaining_indices],
)

# Each Subset retrieves only its assigned images from the original catalogue.
flower_train_dataset = Subset(flower_catalog, flower_train_indices.tolist())
flower_val_dataset = Subset(flower_catalog, flower_val_indices.tolist())
flower_test_dataset = Subset(flower_catalog, flower_test_indices.tolist())

# Check that every image belongs to exactly one split.
flower_split_sets = [set(indices.tolist()) for indices in
                     (flower_train_indices, flower_val_indices, flower_test_indices)]
assert flower_split_sets[0].isdisjoint(flower_split_sets[1])
assert flower_split_sets[0].isdisjoint(flower_split_sets[2])
assert flower_split_sets[1].isdisjoint(flower_split_sets[2])
assert set.union(*flower_split_sets) == set(flower_indices.tolist())

for name, indices in [("Training", flower_train_indices),
                      ("Validation", flower_val_indices),
                      ("Test", flower_test_indices)]:
    counts = np.bincount(flower_targets[indices], minlength=len(flower_class_names))
    print(f"{name}: {len(indices):,} images")
    print(dict(zip(flower_class_names, counts.tolist())))

FLOWER_BATCH_SIZE = 64

# Shuffle training examples into a new order on each pass through the data.
flower_train_loader = DataLoader(
    flower_train_dataset,
    batch_size=FLOWER_BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(FLOWER_SEED),
    num_workers=0,
)
flower_val_loader = DataLoader(
    flower_val_dataset, batch_size=FLOWER_BATCH_SIZE,
    shuffle=False, num_workers=0,
)
flower_test_loader = DataLoader(
    flower_test_dataset, batch_size=FLOWER_BATCH_SIZE,
    shuffle=False, num_workers=0,
)

# Use a separate loader for this preview so the training shuffle is untouched.
flower_preview_loader = DataLoader(
    flower_train_dataset, batch_size=FLOWER_BATCH_SIZE,
    shuffle=False, num_workers=0,
)
flower_batch_images, flower_batch_labels = next(iter(flower_preview_loader))
print(f"Image batch: {tuple(flower_batch_images.shape)}")
print(f"Label batch: {tuple(flower_batch_labels.shape)}")
print(f"Label type: {flower_batch_labels.dtype}")
print(f"Training batches per epoch: {len(flower_train_loader)}")
assert flower_batch_images.shape == (FLOWER_BATCH_SIZE, 3, FLOWER_IMAGE_SIZE, FLOWER_IMAGE_SIZE)
assert flower_batch_labels.dtype == torch.int64
