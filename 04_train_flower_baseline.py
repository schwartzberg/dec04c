"""Train the course flower baseline. Run from the repository folder.

First run 02_load_flowers.py to download the photos. This script runs the
existing split/batch setup, then trains and saves the baseline.
"""
import runpy
from pathlib import Path
import matplotlib.pyplot as plt

# Reuse the same dataset and split code as the notebook's previous step.
globals().update(runpy.run_path(str(Path(__file__).with_name("03_flower_data_loaders.py"))))
flower_root = Path("data") / "flower_photos"

import copy
import json
import random
import time
from torch import nn

# Match the course baseline: 64 x 64 images, Adam, 0.001 learning rate, 15 epochs.
FLOWER_EPOCHS = 15
FLOWER_LEARNING_RATE = 0.001
flower_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
random.seed(FLOWER_SEED)
np.random.seed(FLOWER_SEED)
torch.manual_seed(FLOWER_SEED)

class BaselineFlowerCNN(nn.Module):
    """The course model: three convolutional blocks and two linear layers."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (3, 64, 64) -> (16, 32, 32).
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: (16, 32, 32) -> (32, 16, 16).
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: (32, 16, 16) -> (64, 8, 8).
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (FLOWER_IMAGE_SIZE // 8) ** 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)

flower_baseline = BaselineFlowerCNN(len(flower_class_names)).to(flower_device)
flower_criterion = nn.CrossEntropyLoss()
flower_optimizer = torch.optim.Adam(
    flower_baseline.parameters(), lr=FLOWER_LEARNING_RATE
)

print(f"Device: {flower_device}")
print(flower_baseline)
print(f"Trainable parameters: {sum(p.numel() for p in flower_baseline.parameters()):,}")
with torch.no_grad():
    flower_example_scores = flower_baseline(flower_batch_images.to(flower_device))
print(f"Output shape: {tuple(flower_example_scores.shape)}")
assert flower_example_scores.shape == (len(flower_batch_images), len(flower_class_names))

def evaluate_flower_model(model, loader):
    """Measure average loss and accuracy without updating model weights."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(flower_device)
            labels = labels.to(flower_device)
            scores = model(images)
            loss = flower_criterion(scores, labels)
            # Weight each batch by its size; the last batch can be smaller.
            loss_sum += loss.item() * len(labels)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return loss_sum / count, correct / count

# Recreate the loader so earlier notebook activity does not change its shuffle.
flower_train_loader = DataLoader(
    flower_train_dataset, batch_size=FLOWER_BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(FLOWER_SEED), num_workers=0,
)
flower_baseline_history = {
    "train_loss": [], "val_loss": [],
    "train_accuracy": [], "val_accuracy": [],
}
flower_best_val_loss = float("inf")
flower_best_state = None
flower_best_epoch = 0
flower_start_time = time.perf_counter()

for epoch in range(1, FLOWER_EPOCHS + 1):
    flower_baseline.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in flower_train_loader:
        images = images.to(flower_device)
        labels = labels.to(flower_device)
        flower_optimizer.zero_grad()                 # Clear previous gradients.
        scores = flower_baseline(images)             # Calculate class scores.
        loss = flower_criterion(scores, labels)       # Measure prediction error.
        loss.backward()                              # Calculate gradients.
        flower_optimizer.step()                      # Update model weights.
        loss_sum += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        count += len(labels)

    train_loss = loss_sum / count
    train_accuracy = correct / count
    val_loss, val_accuracy = evaluate_flower_model(flower_baseline, flower_val_loader)
    flower_baseline_history["train_loss"].append(train_loss)
    flower_baseline_history["train_accuracy"].append(train_accuracy)
    flower_baseline_history["val_loss"].append(val_loss)
    flower_baseline_history["val_accuracy"].append(val_accuracy)

    if val_loss < flower_best_val_loss:
        flower_best_val_loss = val_loss
        flower_best_epoch = epoch
        # Copy the weights so later training does not change this saved state.
        flower_best_state = copy.deepcopy(flower_baseline.state_dict())

    print(f"Epoch {epoch:02d}/{FLOWER_EPOCHS} | "
          f"train loss {train_loss:.3f}, accuracy {train_accuracy:.1%} | "
          f"validation loss {val_loss:.3f}, accuracy {val_accuracy:.1%}", flush=True)

flower_training_seconds = time.perf_counter() - flower_start_time
flower_baseline.load_state_dict(flower_best_state)
flower_baseline.eval()
flower_selected_val_loss, flower_selected_val_accuracy = evaluate_flower_model(
    flower_baseline, flower_val_loader
)
assert abs(flower_selected_val_loss - flower_best_val_loss) < 1e-6
print(f"Restored epoch {flower_best_epoch}, chosen by lowest validation loss.")
print(f"Selected validation accuracy: {flower_selected_val_accuracy:.2%}")
print(f"Training time: {flower_training_seconds / 60:.1f} minutes")

flower_epochs = np.arange(1, FLOWER_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["loss", "accuracy"]):
    ax.plot(flower_epochs, flower_baseline_history["train_" + metric], label="Training")
    ax.plot(flower_epochs, flower_baseline_history["val_" + metric], label="Validation")
    ax.axvline(flower_best_epoch, color="gray", linestyle="--", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title="Baseline " + metric)
    ax.legend()
    ax.grid(alpha=0.25)
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()

flower_output_dir = Path("outputs") / "flowers"
flower_output_dir.mkdir(parents=True, exist_ok=True)
flower_baseline_path = flower_output_dir / "baseline_cnn.pth"
torch.save({
    "state_dict": {k: v.detach().cpu() for k, v in flower_baseline.state_dict().items()},
    "architecture": "BaselineFlowerCNN",
    "class_names": flower_class_names,
    "image_size": FLOWER_IMAGE_SIZE,
    "seed": FLOWER_SEED,
    "best_epoch": flower_best_epoch,
}, flower_baseline_path)

# Check that saving and reloading preserves the model's scores.
flower_checkpoint = torch.load(flower_baseline_path, map_location=flower_device, weights_only=True)
flower_reloaded_baseline = BaselineFlowerCNN(len(flower_class_names)).to(flower_device)
flower_reloaded_baseline.load_state_dict(flower_checkpoint["state_dict"])
flower_reloaded_baseline.eval()
with torch.no_grad():
    flower_check_batch = flower_batch_images.to(flower_device)
    torch.testing.assert_close(
        flower_baseline(flower_check_batch), flower_reloaded_baseline(flower_check_batch)
    )

flower_baseline_metrics = {
    "experiment": "course_baseline",
    "architecture": "BaselineFlowerCNN",
    "epochs": FLOWER_EPOCHS,
    "learning_rate": FLOWER_LEARNING_RATE,
    "batch_size": FLOWER_BATCH_SIZE,
    "image_size": FLOWER_IMAGE_SIZE,
    "seed": FLOWER_SEED,
    "device": str(flower_device),
    "pytorch_version": str(torch.__version__),
    "best_epoch": flower_best_epoch,
    "selected_validation_loss": flower_selected_val_loss,
    "selected_validation_accuracy": flower_selected_val_accuracy,
    "training_seconds": flower_training_seconds,
    "history": flower_baseline_history,
    "test_evaluated": False,
}
(flower_output_dir / "baseline_metrics.json").write_text(
    json.dumps(flower_baseline_metrics, indent=2), encoding="utf-8"
)
# Keep the exact split and class mapping for later experiments and visualisations.
flower_split_manifest = {
    "seed": FLOWER_SEED,
    "class_to_idx": flower_catalog.class_to_idx,
    "splits": {
        name: [Path(flower_catalog.samples[int(i)][0]).relative_to(flower_root).as_posix()
               for i in indices]
        for name, indices in [("train", flower_train_indices),
                              ("validation", flower_val_indices),
                              ("test", flower_test_indices)]
    },
}
(flower_output_dir / "split_manifest.json").write_text(
    json.dumps(flower_split_manifest, indent=2), encoding="utf-8"
)
print(f"Saved baseline weights to {flower_baseline_path}; reload verified.")
print("Saved validation results and the exact data split. Test evaluation remains pending.")
