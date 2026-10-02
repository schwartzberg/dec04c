"""Preprocess personal digit images and load the saved MNIST model for inference."""
# This section can run on its own: it loads the trained checkpoint from disk.
from pathlib import Path
import base64
import io
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

# MNISTCNN is defined in Section 1.5. Import it if this is a fresh kernel.
if "MNISTCNN" not in globals():
    from mnist_model import MNISTCNN

HANDWRITING_DIR = Path("handwriting")
HANDWRITING_DIR.mkdir(exist_ok=True)
handwriting_checkpoint = torch.load("outputs/mnist/mnist_cnn.pth", map_location="cpu", weights_only=True)
handwriting_model = MNISTCNN()
handwriting_model.load_state_dict(handwriting_checkpoint["state_dict"])
handwriting_model.eval()


def prepare_handwritten_image(source):
    """Convert a black-on-white drawing to a centred, bright 28 x 28 MNIST input."""
    # Support a file path or an image passed directly by the drawing widget.
    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        with Image.open(source) as opened:
            image = opened.copy()
    # Composite transparent PNG pixels on white before converting to greyscale.
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    grey = background.convert("L")
    ink = ImageOps.invert(grey)  # MNIST uses bright strokes on a dark background.
    bounds = ink.point(lambda value: 255 if value > 30 else 0).getbbox()
    if bounds is None:
        raise ValueError("The canvas is blank. Draw one digit before saving.")
    ink = ink.crop(bounds)
    # Keep the original proportions; fit the digit inside a 20 x 20 area.
    scale = 20 / max(ink.size)
    new_size = (max(1, round(ink.width * scale)), max(1, round(ink.height * scale)))
    ink = ink.resize(new_size, Image.Resampling.LANCZOS)
    prepared = Image.new("L", (28, 28), 0)
    prepared.paste(ink, ((28 - ink.width) // 2, (28 - ink.height) // 2))
    tensor = torch.from_numpy(np.asarray(prepared, dtype=np.float32).copy() / 255.0).unsqueeze(0)
    return prepared, tensor


def predict_handwriting(source):
    """Run inference only: no loss.backward(), optimiser or weight updates."""
    prepared, tensor = prepare_handwritten_image(source)
    with torch.no_grad():
        probabilities = handwriting_model(tensor.unsqueeze(0)).softmax(dim=1)[0]
    prediction = int(probabilities.argmax())
    return prepared, prediction, float(probabilities[prediction])

print("Saved MNIST model loaded. Ready for your drawings; no training will run here.")