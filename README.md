# CSCI E-89: Homework 4

Work through the assignment in `e89-Schwartzberg-Paul-HW04.ipynb`, using the `cscie89` Jupyter kernel.

## Progress

Problem 1, step 1: load MNIST through torchvision and inspect one image per digit.
The model, training, evaluation and personal handwriting examples will follow.

The starting point is the course notebook `1_pytorch_basic_cnn_digits.ipynb`.

## Run

Open the notebook and run its cells in order. The first run downloads MNIST into `data/`, which Git ignores.
`01_load_mnist.py` contains the same code as this first step. Run it from this folder with `python 01_load_mnist.py`.

The assignment calls for one notebook covering all three problems and an HTML export. We will extend this notebook as we proceed.

## Problem 2

The notebook now loads and previews the Flower Photos dataset. `02_load_flowers.py` contains the same code for this step. The first run downloads about 220 MB. The baseline uses the course model with three convolutional blocks and two linear layers. The deeper-model experiments remain to be done.

The next cells create a fixed, stratified 70%/15%/15% split and batches of 64 images. `03_flower_data_loaders.py` contains this step as a script. Run it after downloading the photos.

The new baseline cells (2.5-2.7) train for 15 epochs with Adam and a learning rate of 0.001. They restore the weights with the lowest validation loss, plot learning curves and save weights, metrics and the exact split under `outputs/flowers/`. The test set is not evaluated during this step.

`04_train_flower_baseline.py` runs the same baseline outside Jupyter and reuses `03_flower_data_loaders.py`. Run it from this folder after downloading the photos. To repeat training in the notebook, rerun Section 2.5 first to reset the weights and optimiser, then run Sections 2.6 and 2.7.

Sections 2.8-2.10 explain the baseline curves and define the required five-block, three-linear-layer model. `05_deeper_flower_model.py` provides the same model class. Shape and layer-count checks have passed; training this deeper model is the next step.
