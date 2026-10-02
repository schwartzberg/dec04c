"""Regenerate the static handwriting report from saved personal drawings."""
from handwriting_helpers import *
# Run AFTER saving three drawings. This creates ordinary outputs that remain in
# the notebook and HTML export even when the interactive widget is unavailable.
handwriting_records = []
for record_path in sorted(HANDWRITING_DIR.glob("digit_*.json")):
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("source") == "user drawing in notebook":
        handwriting_records.append(record)
if not handwriting_records:
    print("No drawings saved yet. Use the box above, then run this cell again.")
else:
    # Show at most six examples to avoid long output; the original files remain saved.
    examples = handwriting_records[:6]
    fig, axes = plt.subplots(len(examples), 2, figsize=(6, 2.5 * len(examples)), squeeze=False)
    refreshed_results = []
    for row, record in enumerate(examples):
        path = HANDWRITING_DIR / record["image"]
        prepared, predicted, score = predict_handwriting(path)
        with Image.open(path) as original:
            axes[row, 0].imshow(original.copy())
        axes[row, 0].set_title(f"My drawing: {record['true_digit']}")
        axes[row, 1].imshow(prepared, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"28 x 28 input; predicted {predicted}\nModel score: {score:.1%}",
                             color="green" if predicted == record["true_digit"] else "red")
        for ax in axes[row]:
            ax.axis("off")
        refreshed_results.append({**record, "predicted_digit": predicted, "model_score": score,
                                  "correct": predicted == record["true_digit"]})
    plt.tight_layout()
    results_dir = Path("outputs") / "mnist"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "my_handwriting_predictions.png", dpi=120)
    plt.show()
    correct = sum(record["correct"] for record in refreshed_results)
    print(f"Correct on these personal examples: {correct}/{len(examples)}. This is a small demonstration, not a reliable accuracy estimate.")
    print("Predictions used saved weights only; the drawings were not used for training.")
    if len(handwriting_records) > len(examples):
        print(f"Showing the first {len(examples)} of {len(handwriting_records)} saved examples.")
    (results_dir / "my_handwriting_results.json").write_text(json.dumps(refreshed_results, indent=2), encoding="utf-8")