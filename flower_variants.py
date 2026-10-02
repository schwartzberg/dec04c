"""Reconstruct the flower architectures for training and inference."""
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


# Each configuration changes one setting relative to the first deeper model.
FLOWER_DEFAULT_CONFIG = {
    "learning_rate": 0.001, "activation": "relu",
    "first_kernel": 3, "last_pooling": "max",
}
FLOWER_EXPERIMENTS = [
    ("lower_learning_rate", {**FLOWER_DEFAULT_CONFIG, "learning_rate": 0.0005}),
    ("leaky_relu", {**FLOWER_DEFAULT_CONFIG, "activation": "leaky_relu"}),
    ("first_kernel_5", {**FLOWER_DEFAULT_CONFIG, "first_kernel": 5}),
    ("last_average_pooling", {**FLOWER_DEFAULT_CONFIG, "last_pooling": "average"}),
]

def build_flower_variant(config, num_classes):
    """Rebuild an experiment's exact architecture, including for later inference."""
    model = DeeperFlowerCNN(num_classes)
    if config["first_kernel"] == 5:
        # Padding of two keeps the first convolution's height and width unchanged.
        model.features[0] = nn.Conv2d(3, 16, kernel_size=5, padding=2)
    if config["activation"] == "leaky_relu":
        # Keep ReLU in the classifier, including after the required extra linear layer.
        for index in (1, 4, 7, 10, 13):
            model.features[index] = nn.LeakyReLU(negative_slope=0.01)
    if config["last_pooling"] == "average":
        for index in (11, 14):
            model.features[index] = nn.AvgPool2d(2)
    return model
