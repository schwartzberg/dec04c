"""The five-block flower CNN required by Problem 2. No training runs on import."""

import torch
from torch import nn

class DeeperFlowerCNN(nn.Module):
    """Five convolutional blocks and three linear layers for 64 x 64 RGB images."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: three colour channels -> 16 feature maps.
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 64 x 64 -> 32 x 32
            # Block 2.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 32 x 32 -> 16 x 16
            # Block 3: the last block in the original model.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 16 x 16 -> 8 x 8
            # NEW block 4.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 8 x 8 -> 4 x 4
            # NEW block 5.
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 4 x 4 -> 2 x 2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # 256 * 2 * 2 = 1,024 values
            nn.Linear(256 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),                     # NEW hidden linear layer
            nn.ReLU(),                            # Required activation after it
            nn.Linear(32, num_classes),           # Five raw class scores
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)
