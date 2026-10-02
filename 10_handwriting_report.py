"""Run inference on five Paint PNGs and save the static report."""
from handwriting_helpers import *
# The filename supplies the known answer; the model sees only the image pixels.
# If you choose different digits, change this mapping to match your five files.
HANDWRITING_FILES = {
    "digit_0.png": 0,
    "digit_2.png": 2,
    "digit_5.png": 5,
    "digit_7.png": 7,
    "digit_8.png": 8,
}
missing = [name for name in HANDWRITING_FILES if not (HANDWRITING_DIR / name).is_file()]
if missing:
    print("Save your five Paint PNGs in:", HANDWRITING_DIR.resolve())
    print("Still missing:", ", ".join(missing))
    print("Then run this cell again. No handwriting predictions have been made yet.")
else:
    # Preserve the originals and display the exact inputs used for inference.
    results = []
    fig, axes = plt.subplots(len(HANDWRITING_FILES), 2, figsize=(6, 2.3 * len(HANDWRITING_FILES)), squeeze=False)
    for row, (filename, true_digit) in enumerate(HANDWRITING_FILES.items()):
        path = HANDWRITING_DIR / filename
        prepared, predicted, score = predict_handwriting(path)
        with Image.open(path) as original:
            axes[row, 0].imshow(original.convert("RGB"))
        axes[row, 0].set_title(f"My drawing: {true_digit}")
        axes[row, 1].imshow(prepared, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"28 x 28 input; predicted {predicted}\nModel score: {score:.1%}",
                             color="green" if predicted == true_digit else "red")
        for ax in axes[row]:
            ax.axis("off")
        results.append({"image": filename, "true_digit": true_digit,
                        "predicted_digit": predicted, "model_score": score,
                        "correct": predicted == true_digit,
                        "source": "user drawing in Paint",
                        "checkpoint": "outputs/mnist/mnist_cnn.pth"})
    plt.tight_layout()
    results_dir = Path("outputs") / "mnist"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "my_handwriting_predictions.png", dpi=120)
    plt.show()
    correct = sum(item["correct"] for item in results)
    print(f"Correct on my five drawings: {correct}/{len(results)}.")
    print("This is a small inference demonstration, not a reliable accuracy estimate.")
    print("The drawings were not used for training. Keep incorrect predictions too.")
    (results_dir / "my_handwriting_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
