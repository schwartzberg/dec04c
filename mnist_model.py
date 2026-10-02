"""Small CNN for torchvision MNIST and personal handwriting inference."""
from torch import nn

class MNISTCNN(nn.Module):
    """Adapt the course's two-block CNN to one-channel 28 x 28 images."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 28 x 28 -> 14 x 14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 14 x 14 -> 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 16 * 7 * 7 = 784 values
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10),                    # Raw scores for digits 0-9
        )

    def forward(self, images):
        return self.classifier(self.features(images))
