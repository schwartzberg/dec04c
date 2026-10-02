"""Run the four planned trials and select a model without using test results.
Run from the repository folder after saving the baseline and deeper reference.
"""
from pathlib import Path
import runpy, copy, json, random, time
import matplotlib.pyplot as plt
from flower_variants import DeeperFlowerCNN, build_flower_variant, FLOWER_DEFAULT_CONFIG, FLOWER_EXPERIMENTS
from torch import nn

globals().update(runpy.run_path("03_flower_data_loaders.py"))
flower_root = Path("data") / "flower_photos"

def measure_flower_model(model, loader, criterion, device):
    """Return per-image mean loss and accuracy with no parameter updates."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            total_loss += criterion(scores, labels).item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def run_flower_experiment(name, config):
    """Train a fresh variant using only training and validation data."""
    random.seed(FLOWER_SEED)
    np.random.seed(FLOWER_SEED)
    torch.manual_seed(FLOWER_SEED)
    model = build_flower_variant(config, len(flower_class_names)).to(flower_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loader = DataLoader(
        flower_train_dataset, batch_size=FLOWER_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(FLOWER_SEED), num_workers=0,
    )
    history = {key: [] for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}
    best_loss, best_epoch, best_state = float("inf"), 0, None
    started = time.perf_counter()
    for epoch in range(1, FLOWER_EPOCHS + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(flower_device), labels.to(flower_device)
            optimizer.zero_grad()
            scores = model(images)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            count += len(labels)
        val_loss, val_accuracy = measure_flower_model(model, flower_val_loader, criterion, flower_device)
        for key, value in [("train_loss", total_loss / count), ("train_accuracy", correct / count),
                           ("val_loss", val_loss), ("val_accuracy", val_accuracy)]:
            history[key].append(value)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        # Keep notebook output short while retaining every epoch in the saved history.
        if epoch == 1 or epoch % 5 == 0:
            print(f"{name}: epoch {epoch:02d}/{FLOWER_EPOCHS} | "
                  f"train {correct / count:.1%} | val {val_accuracy:.1%} | "
                  f"val loss {val_loss:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    selected_loss, selected_accuracy = measure_flower_model(model, flower_val_loader, criterion, flower_device)
    assert abs(selected_loss - best_loss) < 1e-6
    checkpoint_path = flower_output_dir / (name + ".pth")
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": "DeeperFlowerCNN_variant", "config": config,
        "class_names": flower_class_names, "image_size": FLOWER_IMAGE_SIZE,
        "seed": FLOWER_SEED, "best_epoch": best_epoch,
    }, checkpoint_path)
    # Verify that the configuration rebuilds the saved model correctly.
    checkpoint = torch.load(checkpoint_path, map_location=flower_device, weights_only=True)
    reloaded = build_flower_variant(checkpoint["config"], len(checkpoint["class_names"])).to(flower_device)
    reloaded.load_state_dict(checkpoint["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        images = flower_batch_images.to(flower_device)
        torch.testing.assert_close(model(images), reloaded(images))
    result = {
        "experiment": name, "config": config, "epochs": FLOWER_EPOCHS,
        "batch_size": FLOWER_BATCH_SIZE, "seed": FLOWER_SEED,
        "image_size": FLOWER_IMAGE_SIZE, "device": str(flower_device),
        "pytorch_version": str(torch.__version__), "best_epoch": best_epoch,
        "selected_validation_loss": selected_loss,
        "selected_validation_accuracy": selected_accuracy,
        "history": history, "training_seconds": elapsed,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": checkpoint_path.name, "test_evaluated": False,
    }
    (flower_output_dir / (name + "_metrics.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{name}: retained epoch {best_epoch}, validation {selected_accuracy:.2%}; reload verified.", flush=True)
    return result


# Fix the protocol before starting the four experiments.
FLOWER_EPOCHS = 15
FLOWER_BATCH_SIZE = 64
FLOWER_SEED = 42
torch.set_num_threads(4)
flower_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flower_output_dir = Path("outputs") / "flowers"
flower_saved_split = json.loads((flower_output_dir / "split_manifest.json").read_text())
assert flower_saved_split["class_to_idx"] == flower_catalog.class_to_idx
for name, indices in [("train", flower_train_indices), ("validation", flower_val_indices),
                      ("test", flower_test_indices)]:
    paths = [Path(flower_catalog.samples[int(i)][0]).relative_to(flower_root).as_posix() for i in indices]
    assert paths == flower_saved_split["splits"][name]

flower_trial_results = []
for name, config in FLOWER_EXPERIMENTS:
    flower_trial_results.append(run_flower_experiment(name, config))


# Include the already-trained deeper reference; do not retrain it here.
flower_deep_reference = json.loads((flower_output_dir / "deeper_metrics.json").read_text())
flower_deep_reference.update({"config": FLOWER_DEFAULT_CONFIG, "checkpoint": "deeper_cnn.pth"})
flower_candidates = [flower_deep_reference] + flower_trial_results
flower_baseline_reference = json.loads((flower_output_dir / "baseline_metrics.json").read_text())
print(f"{'Experiment':<24} {'Epoch':>5} {'Val loss':>10} {'Val accuracy':>13}")
for result in [flower_baseline_reference] + flower_candidates:
    print(f"{result['experiment']:<24} {result['best_epoch']:>5} "
          f"{result['selected_validation_loss']:>10.4f} {result['selected_validation_accuracy']:>13.2%}")

# Select only among the five-block models, as required by the assignment.
# Use the same lowest-validation-loss rule for epochs and for experiment selection.
flower_winner = min(flower_candidates, key=lambda row: row["selected_validation_loss"])
flower_winner_checkpoint = torch.load(
    flower_output_dir / flower_winner["checkpoint"], map_location="cpu", weights_only=True
)
flower_winner_checkpoint["config"] = flower_winner["config"]
flower_winner_checkpoint["experiment"] = flower_winner["experiment"]
flower_winner_checkpoint["selection_rule"] = "lowest validation loss; no test scores used"
flower_selected_path = flower_output_dir / "selected_flower_cnn.pth"
torch.save(flower_winner_checkpoint, flower_selected_path)
flower_selection = {
    "experiment": flower_winner["experiment"], "config": flower_winner["config"],
    "best_epoch": flower_winner["best_epoch"],
    "selected_validation_loss": flower_winner["selected_validation_loss"],
    "selected_validation_accuracy": flower_winner["selected_validation_accuracy"],
    "selection_rule": "lowest validation loss among the five-block models",
    "test_used_for_selection": False,
}
(flower_output_dir / "model_selection.json").write_text(json.dumps(flower_selection, indent=2), encoding="utf-8")
(flower_output_dir / "experiment_comparison.json").write_text(
    json.dumps([flower_baseline_reference] + flower_candidates, indent=2), encoding="utf-8"
)
print(f"Selected: {flower_winner['experiment']} at epoch {flower_winner['best_epoch']}.")

# Same axis limits make comparisons between experiments easier.
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
for ax, result in zip(axes.flat, [flower_baseline_reference] + flower_candidates):
    epochs = np.arange(1, result["epochs"] + 1)
    ax.plot(epochs, result["history"]["train_loss"], label="Training")
    ax.plot(epochs, result["history"]["val_loss"], label="Validation")
    ax.axvline(result["best_epoch"], linestyle="--", color="gray", label="Selected epoch")
    ax.set(title=result["experiment"], xlabel="Epoch", ylabel="Loss")
    ax.grid(alpha=0.25)
axes[0, 0].legend()
plt.tight_layout()
fig.savefig(flower_output_dir / "experiment_learning_curves.png", dpi=120)
plt.show()

