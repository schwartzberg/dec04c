"""Problem 3: inspect the saved flower model, with no training."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

# Reuse the architecture definitions above, or import them in a fresh kernel.
if "build_flower_variant" not in globals():
    from flower_variants import build_flower_variant

p3_checkpoint_path = Path("outputs/flowers/selected_flower_cnn.pth")
p3_checkpoint = torch.load(p3_checkpoint_path, map_location="cpu", weights_only=True)
p3_classes = p3_checkpoint["class_names"]
p3_model = build_flower_variant(p3_checkpoint["config"], len(p3_classes))
p3_model.load_state_dict(p3_checkpoint["state_dict"], strict=True)
p3_model.eval()
p3_model.requires_grad_(False)  # Grad-CAM will track the input instead of parameter gradients.
p3_output = Path("outputs") / "problem3"
p3_output.mkdir(parents=True, exist_ok=True)
p3_transform = transforms.Compose([
    transforms.Resize((p3_checkpoint["image_size"], p3_checkpoint["image_size"])),
    transforms.ToTensor(),
])
assert sum(isinstance(layer, nn.Conv2d) for layer in p3_model.modules()) == 5
assert sum(isinstance(layer, nn.Linear) for layer in p3_model.modules()) == 3
assert isinstance(p3_model.classifier[4], nn.ReLU)
print("Loaded experiment:", p3_checkpoint["experiment"], "| epoch:", p3_checkpoint["best_epoch"])
print("Configuration:", p3_checkpoint["config"])
for index, layer in enumerate(p3_model.features):
    if isinstance(layer, nn.Conv2d):
        print(f"features[{index}]: weights {tuple(layer.weight.shape)}")

# Pick the first filename in each chosen class from the saved validation split.
# Selection does not depend on predictions or how attractive a heatmap looks.
p3_manifest = json.loads(Path("outputs/flowers/split_manifest.json").read_text())
p3_photos = {}
for name in ["daisy", "roses", "tulips"]:
    relative_path = sorted(s for s in p3_manifest["splits"]["validation"] if Path(s).parts[0] == name)[0]
    path = Path("data/flower_photos") / relative_path
    with Image.open(path) as picture:
        tensor = p3_transform(picture.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        scores = p3_model(tensor)
        probabilities = scores.softmax(1)[0]
    predicted = int(probabilities.argmax())
    p3_photos[name] = {"path": relative_path, "input": tensor, "true_index": p3_classes.index(name),
                       "predicted_index": predicted, "score": float(probabilities[predicted])}
    print(f"{name}: {relative_path} | predicted {p3_classes[predicted]} ({float(probabilities[predicted]):.1%})")
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, (name, item) in zip(axes, p3_photos.items()):
    ax.imshow(item["input"][0].permute(1, 2, 0))
    ax.set_title(f"True: {name}\nPredicted: {p3_classes[item['predicted_index']]}")
    ax.axis("off")
plt.tight_layout();fig.savefig(p3_output / "selected_inputs.png", dpi=120);plt.show()

# Read six first-layer filters. Each has three 5 x 5 channel slices.
p3_filters = p3_model.features[0].weight.detach()
p3_limit = max(float(p3_filters.abs().max()), 1e-8)
fig, axes = plt.subplots(6, 3, figsize=(6, 9), layout="constrained")
for f in range(6):
    for channel in range(3):
        ax = axes[f, channel]
        picture = ax.imshow(p3_filters[f, channel], cmap="RdBu_r", vmin=-p3_limit, vmax=p3_limit,
                            interpolation="nearest")
        ax.set_xticks([]);ax.set_yticks([])
        if f == 0:ax.set_title(["Red input", "Green input", "Blue input"][channel])
        if channel == 0:ax.set_ylabel(f"Filter {f}")
fig.colorbar(picture, ax=axes, shrink=0.6, label="Learned weight")
fig.suptitle("Six learned filters: three channel slices per filter")
fig.savefig(p3_output / "first_layer_filters.png", dpi=120);plt.show()

def trace_flower_features(input_tensor):
    """Return feature maps after each ReLU, before pooling, and verify the forward pass."""
    maps = {}
    current = input_tensor
    block = 0
    with torch.no_grad():
        for layer in p3_model.features:
            if isinstance(layer, nn.Conv2d):
                block += 1
            current = layer(current)
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                maps[block] = current.clone()
        scores = p3_model.classifier(current)
        torch.testing.assert_close(scores, p3_model(input_tensor))
    return maps

p3_daisy_maps = trace_flower_features(p3_photos["daisy"]["input"])
fig, axes = plt.subplots(5, 8, figsize=(15, 9), layout="constrained")
for row, (block, maps) in enumerate(p3_daisy_maps.items()):
    print(f"Daisy, block {block}: {tuple(maps.shape)} after ReLU, before pooling")
    shown = maps[0, :8]
    upper = max(float(shown.max()), 1e-8)
    for channel, ax in enumerate(axes[row]):
        ax.imshow(shown[channel], cmap="viridis", vmin=0, vmax=upper, interpolation="nearest")
        ax.set_xticks([]);ax.set_yticks([])
        if row == 0:ax.set_title(f"Channel {channel}")
        if channel == 0:ax.set_ylabel(f"Block {block}\n{maps.shape[-1]} x {maps.shape[-1]}")
fig.suptitle("Daisy: first eight feature maps at each block (scale shared within each row)")
fig.savefig(p3_output / "daisy_all_blocks.png", dpi=120);plt.show()

# Compare the same channel indices for roses and tulips with a shared scale.
p3_last_maps = {name: trace_flower_features(p3_photos[name]["input"])[5][0, :8]
                for name in ["roses", "tulips"]}
upper = max(max(float(maps.max()) for maps in p3_last_maps.values()), 1e-8)
fig, axes = plt.subplots(2, 8, figsize=(15, 4), layout="constrained")
for row, (name, maps) in enumerate(p3_last_maps.items()):
    for channel, ax in enumerate(axes[row]):
        picture = ax.imshow(maps[channel], cmap="viridis", vmin=0, vmax=upper, interpolation="nearest")
        ax.set_xticks([]);ax.set_yticks([])
        if row == 0:ax.set_title(f"Channel {channel}")
        if channel == 0:ax.set_ylabel(name)
fig.colorbar(picture, ax=axes, shrink=0.7, label="Activation after last ReLU")
fig.suptitle("Last block: first eight of 256 feature maps, each 4 x 4")
fig.savefig(p3_output / "roses_tulips_last_block.png", dpi=120);plt.show()

import torch.nn.functional as F

def flower_gradcam(input_tensor, target_index):
    """Explain a named class using the final ReLU maps, without changing weights."""
    # Evaluation mode does not disable gradients. Track a fresh input for this calculation.
    tracked_input = input_tensor.detach().clone().requires_grad_(True)
    maps = p3_model.features[:14](tracked_input)  # Includes last convolution [12] and ReLU [13].
    pooled = p3_model.features[14:](maps)         # Last pooling layer [14].
    scores = p3_model.classifier(pooled)
    torch.testing.assert_close(scores.detach(), p3_model(input_tensor).detach())
    gradients = torch.autograd.grad(scores[0, target_index], maps)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    coarse = (weights * maps).sum(dim=1, keepdim=True).relu().detach()
    assert coarse.shape == (1, 1, 4, 4)
    enlarged = F.interpolate(coarse, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    peak = float(enlarged.max())
    heatmap = enlarged / max(peak, 1e-8)
    assert torch.isfinite(heatmap).all()
    return heatmap.numpy(), coarse[0, 0].numpy(), peak

# Explain each true flower class explicitly, even if the predicted class differs.
p3_cam_results = {}
fig, axes = plt.subplots(3, 3, figsize=(10, 10), layout="constrained")
for row, (name, item) in enumerate(p3_photos.items()):
    heatmap, coarse, peak = flower_gradcam(item["input"], item["true_index"])
    rgb = item["input"][0].permute(1, 2, 0).numpy()
    axes[row, 0].imshow(rgb)
    axes[row, 0].set_title(f"True: {name}\nPredicted: {p3_classes[item['predicted_index']]}")
    picture = axes[row, 1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    axes[row, 1].set_title(f"Grad-CAM for {name}")
    axes[row, 2].imshow(rgb)
    axes[row, 2].imshow(heatmap, cmap="inferno", vmin=0, vmax=1, alpha=0.45)
    axes[row, 2].set_title("Overlay (4 x 4 map enlarged)")
    for ax in axes[row]:ax.axis("off")
    fig.colorbar(picture, ax=axes[row, 1], shrink=0.7)
    np.save(p3_output / (name + "_gradcam.npy"), heatmap)
    p3_cam_results[name] = {
        "image": item["path"], "true_class": name,
        "predicted_class": p3_classes[item["predicted_index"]], "model_score": item["score"],
        "explained_class": name, "coarse_shape": list(coarse.shape),
        "positive_response": peak > 0,
    }
    print(f"{name}: explaining {name}; positive Grad-CAM response: {peak > 0}")
fig.savefig(p3_output / "gradcam_overlays.png", dpi=140);plt.show()
# Confirm that inspection and gradient calculations have not changed any model parameters.
for name, value in p3_model.state_dict().items():
    torch.testing.assert_close(value, p3_checkpoint["state_dict"][name], rtol=0, atol=0)
(p3_output / "visualisation_results.json").write_text(json.dumps(p3_cam_results, indent=2), encoding="utf-8")
print("All model weights and biases are unchanged. No training was performed.")
