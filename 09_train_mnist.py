"""Train and evaluate MNIST. Run from the repository folder."""
from pathlib import Path
import random, numpy as np, matplotlib.pyplot as plt, torch
from torchvision import datasets, transforms
from mnist_model import MNISTCNN
mnist_train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
from torch import nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

MNIST_SEED = 42
MNIST_BATCH_SIZE = 128
# Reserve validation images from the official training pool, not the test set.
mnist_indices = np.arange(len(mnist_train_full))
mnist_train_indices, mnist_val_indices = train_test_split(
    mnist_indices, test_size=5000, random_state=MNIST_SEED,
    stratify=mnist_train_full.targets.numpy(),
)
mnist_train_dataset = Subset(mnist_train_full, mnist_train_indices.tolist())
mnist_val_dataset = Subset(mnist_train_full, mnist_val_indices.tolist())
assert set(mnist_train_indices).isdisjoint(set(mnist_val_indices))
assert len(mnist_train_dataset) + len(mnist_val_dataset) == 60000
mnist_val_loader = DataLoader(mnist_val_dataset, batch_size=256, shuffle=False, num_workers=0)
mnist_test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=0)
print(f"Training: {len(mnist_train_dataset):,}; validation: {len(mnist_val_dataset):,}; test: {len(mnist_test):,}")

import copy
import json
import time

# This cell starts a fresh training run, even if it is executed again.
MNIST_EPOCHS = 5
MNIST_LEARNING_RATE = 0.001
torch.set_num_threads(4)
random.seed(MNIST_SEED)
np.random.seed(MNIST_SEED)
torch.manual_seed(MNIST_SEED)
mnist_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mnist_model = MNISTCNN().to(mnist_device)
mnist_criterion = nn.CrossEntropyLoss()
mnist_optimizer = torch.optim.Adam(mnist_model.parameters(), lr=MNIST_LEARNING_RATE)
mnist_train_loader = DataLoader(
    mnist_train_dataset, batch_size=MNIST_BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(MNIST_SEED), num_workers=0,
)

def evaluate_mnist(model, loader):
    """Measure loss and accuracy without changing weights."""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(mnist_device), labels.to(mnist_device)
            scores = model(images)
            loss_sum += mnist_criterion(scores, labels).item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            count += len(labels)
    return loss_sum / count, correct / count

mnist_history = {key: [] for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}
mnist_best_loss, mnist_best_epoch, mnist_best_state = float("inf"), 0, None
mnist_started = time.perf_counter()
for epoch in range(1, MNIST_EPOCHS + 1):
    mnist_model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in mnist_train_loader:
        images, labels = images.to(mnist_device), labels.to(mnist_device)
        mnist_optimizer.zero_grad()                 # Clear previous gradients.
        scores = mnist_model(images)                # Calculate class scores.
        loss = mnist_criterion(scores, labels)      # Compare with correct digits.
        loss.backward()                            # Calculate gradients.
        mnist_optimizer.step()                     # Update weights.
        loss_sum += loss.item() * len(labels)
        correct += (scores.argmax(1) == labels).sum().item()
        count += len(labels)
    val_loss, val_accuracy = evaluate_mnist(mnist_model, mnist_val_loader)
    for key, value in [("train_loss", loss_sum / count), ("train_accuracy", correct / count),
                       ("val_loss", val_loss), ("val_accuracy", val_accuracy)]:
        mnist_history[key].append(value)
    if val_loss < mnist_best_loss:
        mnist_best_loss, mnist_best_epoch = val_loss, epoch
        mnist_best_state = copy.deepcopy(mnist_model.state_dict())
    print(f"Epoch {epoch}/{MNIST_EPOCHS} | train accuracy {correct/count:.2%} | "
          f"validation accuracy {val_accuracy:.2%}, loss {val_loss:.4f}", flush=True)
mnist_training_seconds = time.perf_counter() - mnist_started
mnist_model.load_state_dict(mnist_best_state)
mnist_model.eval()
print(f"Restored epoch {mnist_best_epoch}, selected by validation loss.")

from sklearn.metrics import ConfusionMatrixDisplay

# Evaluate once after selecting weights by validation loss.
mnist_true, mnist_pred = [], []
mnist_test_loss_sum = 0.0
with torch.no_grad():
    for images, labels in mnist_test_loader:
        scores = mnist_model(images.to(mnist_device))
        mnist_test_loss_sum += mnist_criterion(scores, labels.to(mnist_device)).item() * len(labels)
        mnist_true.extend(labels.tolist())
        mnist_pred.extend(scores.argmax(1).cpu().tolist())
mnist_test_accuracy = float(np.mean(np.asarray(mnist_true) == np.asarray(mnist_pred)))
mnist_test_loss = mnist_test_loss_sum / len(mnist_true)
print(f"Test images: {len(mnist_true):,}")
print(f"Test accuracy: {mnist_test_accuracy:.2%}; test loss: {mnist_test_loss:.4f}")

mnist_output_dir = Path("outputs") / "mnist"
mnist_output_dir.mkdir(parents=True, exist_ok=True)
mnist_model_path = mnist_output_dir / "mnist_cnn.pth"
torch.save({
    "state_dict": {k: v.detach().cpu() for k, v in mnist_model.state_dict().items()},
    "architecture": "MNISTCNN", "image_size": 28, "class_names": list(range(10)),
    "seed": MNIST_SEED, "best_epoch": mnist_best_epoch,
    "preprocessing": "ToTensor: bright digit on dark background; values in [0, 1]",
}, mnist_model_path)
# Check that the saved model produces the same scores as the current model.
mnist_checkpoint = torch.load(mnist_model_path, map_location=mnist_device, weights_only=True)
mnist_reloaded = MNISTCNN().to(mnist_device)
mnist_reloaded.load_state_dict(mnist_checkpoint["state_dict"])
mnist_reloaded.eval()
with torch.no_grad():
    sample = mnist_train_full[0][0].unsqueeze(0).to(mnist_device)
    torch.testing.assert_close(mnist_model(sample), mnist_reloaded(sample))
mnist_metrics = {
    "seed": MNIST_SEED, "epochs": MNIST_EPOCHS, "batch_size": MNIST_BATCH_SIZE,
    "learning_rate": MNIST_LEARNING_RATE, "best_epoch": mnist_best_epoch,
    "selected_validation_loss": mnist_best_loss,
    "selected_validation_accuracy": mnist_history["val_accuracy"][mnist_best_epoch-1],
    "test_images": len(mnist_true), "test_accuracy": mnist_test_accuracy,
    "test_loss": mnist_test_loss, "training_seconds": mnist_training_seconds,
    "history": mnist_history, "device": str(mnist_device), "pytorch_version": str(torch.__version__),
}
(mnist_output_dir / "metrics.json").write_text(json.dumps(mnist_metrics, indent=2), encoding="utf-8")
print("Saved MNIST weights and metrics; reloaded scores match.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ["loss", "accuracy"]):
    ax.plot(range(1, MNIST_EPOCHS+1), mnist_history["train_"+metric], marker="o", label="Training")
    ax.plot(range(1, MNIST_EPOCHS+1), mnist_history["val_"+metric], marker="o", label="Validation")
    ax.axvline(mnist_best_epoch, color="gray", linestyle="--", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title="MNIST " + metric)
    ax.legend()
axes[1].set_ylim(0, 1)
plt.tight_layout()
fig.savefig(mnist_output_dir / "learning_curves.png", dpi=120)
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(mnist_true, mnist_pred, labels=list(range(10)),
                                       cmap="Blues", colorbar=False, ax=ax)
ax.set_title("MNIST: final test predictions")
plt.tight_layout()
fig.savefig(mnist_output_dir / "test_confusion_matrix.png", dpi=120)
plt.show()
