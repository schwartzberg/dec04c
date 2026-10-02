# CSCI E-89: Homework 4

Paul Schwartzberg

## Status

- Problem 1: MNIST training and evaluation complete (98.26% test accuracy). Sections 1.8-1.9 load five Paint PNGs and create an inference report; the personal drawings and results remain to be added.
- Problem 2: complete. The notebook contains the baseline, deeper model, four controlled trials, selection by validation loss, final test evaluation and conclusions.
- Problem 3: not yet implemented. The selected flower checkpoint is ready for it.

Use `e89-Schwartzberg-Paul-HW04.ipynb` with the `cscie89` kernel. The HTML file is a readable snapshot of the current notebook; the whole assignment is not yet complete.

## Problem 2 results

The selected five-block model uses a 5 x 5 first kernel, ReLU and max pooling, with three linear layers. It was selected at epoch 14 by validation loss (0.8457), with 67.27% validation accuracy. Final test accuracy is 69.15% on 551 images; test loss is 0.8353. No augmentation was used. The notebook discusses overfitting and the limits of one-seed comparisons.

## Run

Run notebook cells in order. Training cells take several minutes on the CPU. Setup cells initialise fresh weights, so rerun each experiment's setup before repeating its training. Checkpoints on disk are separate from models in notebook memory.

Script alternatives, run from this folder:

1. `01_load_mnist.py`: load and inspect MNIST.
2. `02_load_flowers.py`: download and inspect Flower Photos (about 220 MB).
3. `03_flower_data_loaders.py`: create the fixed, stratified split and data loaders.
4. `04_train_flower_baseline.py`: train and save the original model.
5. `05_deeper_flower_model.py`: define the five-block model; importing it does not train it.
6. `06_train_deeper_flowers.py`: run the deeper model under the baseline settings.
7. `07_flower_experiments.py`: run four additional trials and select by validation loss.
8. `08_evaluate_selected_flowers.py`: evaluate the fixed selected checkpoint on test images. Do not tune from these results.

`flower_variants.py` reconstructs each variant from its saved configuration. The notebook also includes those definitions, so it contains the solution code in one place.

## Saved results

`outputs/flowers/selected_flower_cnn.pth` is the checkpoint for Problem 3. It includes weights, configuration, class names and image size. `model_selection.json` records the selection rule. `split_manifest.json` records exact image assignments. Individual checkpoints, histories, comparison plots and final test metrics are also saved under `outputs/flowers/`.

Downloaded datasets are kept in `data/`, which Git ignores. Required packages include PyTorch, torchvision, NumPy, matplotlib, scikit-learn, Pillow and Jupyter.

## Personal handwriting from Paint

Save five black-on-white digit PNGs in `handwriting/`: `digit_0.png`, `digit_2.png`, `digit_5.png`, `digit_7.png` and `digit_8.png`. Use one digit per image, a roughly 280 x 280 canvas, thick strokes and a clear margin. Then run Sections 1.8 and 1.9 and save the notebook with Ctrl+S. The code handles resizing and colour inversion. Keep incorrect predictions.

The original PNGs remain unchanged. The static plot and inference report are saved under `outputs/mnist/`. The notebook no longer uses a drawing widget or requires anywidget.

`09_train_mnist.py` reproduces training. `mnist_model.py` defines the CNN; `handwriting_helpers.py` provides preprocessing and inference. `10_handwriting_report.py` regenerates the Paint-image report. The checkpoint is `outputs/mnist/mnist_cnn.pth`.
