"""Run the first deeper-model experiment after the saved baseline exists.
Run from the repository folder. No test evaluation is performed.
"""
from pathlib import Path
import runpy
import matplotlib.pyplot as plt

globals().update(runpy.run_path("03_flower_data_loaders.py"))
globals().update(runpy.run_path("05_deeper_flower_model.py"))
flower_root = Path("data") / "flower_photos"

# Reset the experiment so rerunning this cell starts from fresh weights.
import copy
import json
import random
import time

FLOWER_EPOCHS = 15
FLOWER_LEARNING_RATE = 0.001
torch.set_num_threads(4)
random.seed(FLOWER_SEED)
np.random.seed(FLOWER_SEED)
torch.manual_seed(FLOWER_SEED)
flower_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flower_deeper = DeeperFlowerCNN(len(flower_class_names)).to(flower_device)
flower_criterion = nn.CrossEntropyLoss()
flower_deeper_optimizer = torch.optim.Adam(
    flower_deeper.parameters(), lr=FLOWER_LEARNING_RATE
)

# Check that the saved baseline used these same experiment settings and split.
flower_output_dir = Path("outputs") / "flowers"
flower_reference_metrics = json.loads((flower_output_dir / "baseline_metrics.json").read_text())
for key, value in [("epochs", FLOWER_EPOCHS), ("learning_rate", FLOWER_LEARNING_RATE),
                   ("batch_size", FLOWER_BATCH_SIZE), ("image_size", FLOWER_IMAGE_SIZE),
                   ("seed", FLOWER_SEED)]:
    assert flower_reference_metrics[key] == value, f"Baseline setting differs: {key}"
flower_saved_split = json.loads((flower_output_dir / "split_manifest.json").read_text())
assert flower_saved_split["class_to_idx"] == flower_catalog.class_to_idx
for name, indices in [("train", flower_train_indices),
                      ("validation", flower_val_indices), ("test", flower_test_indices)]:
    paths = [Path(flower_catalog.samples[int(i)][0]).relative_to(flower_root).as_posix()
             for i in indices]
    assert paths == flower_saved_split["splits"][name], f"Different split: {name}"
print("Baseline settings and exact image split verified. Starting a fresh deeper model.")

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
flower_deeper_history = {
    "train_loss": [], "val_loss": [],
    "train_accuracy": [], "val_accuracy": [],
}
flower_deeper_best_val_loss = float("inf")
flower_deeper_best_state = None
flower_deeper_best_epoch = 0
flower_deeper_start_time = time.perf_counter()

for epoch in range(1, FLOWER_EPOCHS + 1):
    flower_deeper.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in flower_train_loader:
        images = images.to(flower_device)
        labels = labels.to(flower_device)
        flower_deeper_optimizer.zero_grad()                 # Clear previous gradients.
        scores = flower_deeper(images)             # Calculate class scores.
        loss = flower_criterion(scores, labels)       # Measure prediction error.
        loss.backward()                              # Calculate gradients.
        flower_deeper_optimizer.step()                      # Update model weights.
        loss_sum += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        count += len(labels)

    train_loss = loss_sum / count
    train_accuracy = correct / count
    val_loss, val_accuracy = evaluate_flower_model(flower_deeper, flower_val_loader)
    flower_deeper_history["train_loss"].append(train_loss)
    flower_deeper_history["train_accuracy"].append(train_accuracy)
    flower_deeper_history["val_loss"].append(val_loss)
    flower_deeper_history["val_accuracy"].append(val_accuracy)

    if val_loss < flower_deeper_best_val_loss:
        flower_deeper_best_val_loss = val_loss
        flower_deeper_best_epoch = epoch
        # Copy the weights so later training does not change this saved state.
        flower_deeper_best_state = copy.deepcopy(flower_deeper.state_dict())

    print(f"Epoch {epoch:02d}/{FLOWER_EPOCHS} | "
          f"train loss {train_loss:.3f}, accuracy {train_accuracy:.1%} | "
          f"validation loss {val_loss:.3f}, accuracy {val_accuracy:.1%}", flush=True)

flower_deeper_training_seconds = time.perf_counter() - flower_deeper_start_time
flower_deeper.load_state_dict(flower_deeper_best_state)
flower_deeper.eval()
flower_deeper_selected_val_loss, flower_deeper_selected_val_accuracy = evaluate_flower_model(
    flower_deeper, flower_val_loader
)
assert abs(flower_deeper_selected_val_loss - flower_deeper_best_val_loss) < 1e-6
print(f"Restored epoch {flower_deeper_best_epoch}, chosen by lowest validation loss.")
print(f"Selected validation accuracy: {flower_deeper_selected_val_accuracy:.2%}")
print(f"Training time: {flower_deeper_training_seconds / 60:.1f} minutes")

# Save the selected deeper checkpoint separately from the baseline.
flower_deeper_path = flower_output_dir / "deeper_cnn.pth"
torch.save({
    "state_dict": {k: v.detach().cpu() for k, v in flower_deeper.state_dict().items()},
    "architecture": "DeeperFlowerCNN",
    "class_names": flower_class_names,
    "image_size": FLOWER_IMAGE_SIZE,
    "seed": FLOWER_SEED,
    "best_epoch": flower_deeper_best_epoch,
}, flower_deeper_path)
flower_deeper_checkpoint = torch.load(flower_deeper_path, map_location=flower_device, weights_only=True)
flower_reloaded_deeper = DeeperFlowerCNN(len(flower_class_names)).to(flower_device)
flower_reloaded_deeper.load_state_dict(flower_deeper_checkpoint["state_dict"])
flower_reloaded_deeper.eval()
with torch.no_grad():
    check_images = flower_batch_images.to(flower_device)
    torch.testing.assert_close(flower_deeper(check_images), flower_reloaded_deeper(check_images))

flower_deeper_metrics = {
    "experiment": "deeper_same_settings",
    "architecture": "DeeperFlowerCNN",
    "epochs": FLOWER_EPOCHS,
    "learning_rate": FLOWER_LEARNING_RATE,
    "batch_size": FLOWER_BATCH_SIZE,
    "image_size": FLOWER_IMAGE_SIZE,
    "seed": FLOWER_SEED,
    "device": str(flower_device),
    "pytorch_version": str(torch.__version__),
    "best_epoch": flower_deeper_best_epoch,
    "selected_validation_loss": flower_deeper_selected_val_loss,
    "selected_validation_accuracy": flower_deeper_selected_val_accuracy,
    "training_seconds": flower_deeper_training_seconds,
    "history": flower_deeper_history,
    "test_evaluated": False,
}
(flower_output_dir / "deeper_metrics.json").write_text(
    json.dumps(flower_deeper_metrics, indent=2), encoding="utf-8"
)
print("Deeper weights saved and reload verified.")
print(f"{'Model':<12} {'Epoch':>6} {'Val loss':>10} {'Val accuracy':>14}")
for name, metrics in [("Baseline", flower_reference_metrics), ("Deeper", flower_deeper_metrics)]:
    print(f"{name:<12} {metrics['best_epoch']:>6} "
          f"{metrics['selected_validation_loss']:>10.4f} "
          f"{metrics['selected_validation_accuracy']:>14.2%}")
flower_accuracy_change = (flower_deeper_selected_val_accuracy
                          - flower_reference_metrics["selected_validation_accuracy"]) * 100
print(f"Validation accuracy change: {flower_accuracy_change:+.2f} percentage points")

# Show each run's training and validation curves, marking its selected epoch.
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for row, (name, metrics) in enumerate([("Baseline", flower_reference_metrics),
                                      ("Deeper", flower_deeper_metrics)]):
    epochs = np.arange(1, metrics["epochs"] + 1)
    for col, metric in enumerate(["loss", "accuracy"]):
        ax = axes[row, col]
        ax.plot(epochs, metrics["history"]["train_" + metric], label="Training")
        ax.plot(epochs, metrics["history"]["val_" + metric], label="Validation")
        ax.axvline(metrics["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), title=name + " " + metric)
        ax.legend()
        ax.grid(alpha=0.25)
    axes[row, 1].set_ylim(0, 1)
plt.tight_layout()
fig.savefig(flower_output_dir / "baseline_vs_deeper.png", dpi=120)
plt.show()
print("Test evaluation remains pending while we compare experiments.")
