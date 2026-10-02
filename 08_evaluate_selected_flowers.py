"""Evaluate the fixed, validation-selected model. Run after 07_flower_experiments.py.
Do not use the reported test results to tune further experiments.
"""
from pathlib import Path
import runpy,json
import matplotlib.pyplot as plt
from torch import nn
from flower_variants import build_flower_variant

globals().update(runpy.run_path("03_flower_data_loaders.py"))
flower_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)

# The experiment choice is already fixed in model_selection.json.
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, ConfusionMatrixDisplay

flower_output_dir = Path("outputs") / "flowers"
flower_final_selection = json.loads((flower_output_dir / "model_selection.json").read_text())
flower_final_checkpoint = torch.load(
    flower_output_dir / "selected_flower_cnn.pth", map_location=flower_device, weights_only=True
)
assert flower_final_checkpoint["experiment"] == flower_final_selection["experiment"]
assert flower_final_checkpoint["class_names"] == flower_class_names
assert flower_final_checkpoint["image_size"] == FLOWER_IMAGE_SIZE
flower_final_model = build_flower_variant(
    flower_final_checkpoint["config"], len(flower_class_names)
).to(flower_device)
flower_final_model.load_state_dict(flower_final_checkpoint["state_dict"])
flower_final_model.eval()
assert sum(isinstance(layer, nn.Conv2d) for layer in flower_final_model.modules()) == 5
assert sum(isinstance(layer, nn.Linear) for layer in flower_final_model.modules()) == 3
assert isinstance(flower_final_model.classifier[4], nn.ReLU)

# One final test evaluation; these results do not change model selection.
flower_true_labels, flower_predicted_labels = [], []
flower_test_loss_sum = 0.0
flower_final_criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for images, labels in flower_test_loader:
        scores = flower_final_model(images.to(flower_device))
        assert torch.isfinite(scores).all()
        flower_test_loss_sum += flower_final_criterion(scores, labels.to(flower_device)).item() * len(labels)
        flower_true_labels.extend(labels.tolist())
        flower_predicted_labels.extend(scores.argmax(1).cpu().tolist())
assert len(flower_true_labels) == len(flower_test_dataset)
flower_final_test_loss = flower_test_loss_sum / len(flower_true_labels)
flower_final_test_accuracy = accuracy_score(flower_true_labels, flower_predicted_labels)
flower_final_balanced_accuracy = balanced_accuracy_score(flower_true_labels, flower_predicted_labels)
flower_report = classification_report(
    flower_true_labels, flower_predicted_labels,
    labels=list(range(len(flower_class_names))), target_names=flower_class_names,
    output_dict=True, zero_division=0,
)
print(f"Selected experiment: {flower_final_selection['experiment']}")
print(f"Selected epoch: {flower_final_selection['best_epoch']}")
print(f"Test images: {len(flower_true_labels)}")
print(f"Test loss: {flower_final_test_loss:.4f}")
print(f"Test accuracy: {flower_final_test_accuracy:.2%}")
print(f"Balanced test accuracy: {flower_final_balanced_accuracy:.2%}")
print(classification_report(
    flower_true_labels, flower_predicted_labels,
    labels=list(range(len(flower_class_names))), target_names=flower_class_names,
    zero_division=0,
))
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    flower_true_labels, flower_predicted_labels,
    labels=list(range(len(flower_class_names))), display_labels=flower_class_names,
    cmap="Blues", colorbar=False, ax=ax,
)
ax.set_title("Selected flower model: final test predictions")
plt.tight_layout()
fig.savefig(flower_output_dir / "final_test_confusion_matrix.png", dpi=120)
plt.show()
flower_final_test_metrics = {
    "selection": flower_final_selection, "test_images": len(flower_true_labels),
    "test_loss": flower_final_test_loss, "test_accuracy": flower_final_test_accuracy,
    "balanced_test_accuracy": flower_final_balanced_accuracy, "classification_report": flower_report,
    "true_labels": flower_true_labels, "predicted_labels": flower_predicted_labels,
    "checkpoint": "selected_flower_cnn.pth", "test_used_for_selection": False,
}
(flower_output_dir / "final_test_metrics.json").write_text(
    json.dumps(flower_final_test_metrics, indent=2), encoding="utf-8"
)
print("Final test results saved. Keep this model fixed for Problem 3.")
