# PyTorch ResNet Image Classifier

This project implements a multiclass image classification pipeline in PyTorch. The classifier uses a ResNet-18 style architecture trained from scratch to classify images into six categories: bus, car, light, sign, truck, and vegetation.

The project includes dataset preparation utilities, a custom PyTorch Dataset class, Albumentations-based image preprocessing and augmentation, training and validation loops, model checkpointing, learning-curve visualization, and an inference script that exports predictions to CSV.

This repository focuses on the implementation of the classification pipeline, including the model architecture, dataset handling, training workflow, and inference workflow. The full dataset and trained model checkpoint are not included because of file-size.

## Classes

The model predicts one of the following classes:

| Class ID | Class |
|---|---|
| 0 | bus |
| 1 | car |
| 2 | light |
| 3 | sign |
| 4 | truck |
| 5 | vegetation |

## Model

The model is a ResNet-18 style convolutional neural network implemented in PyTorch and trained from scratch. It is built from residual `BasicBlock` modules and adapted for six output classes.

## Attribution

The custom ResNet-18 implementation was adapted from the DebuggerCafe tutorial [Implementing ResNet18 in PyTorch from Scratch](https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/#download-code) and modified for this six-class image classification task.

## Project structure

```text
.
├── README.md
├── final_learning_curves.png
└── classification/
    ├── dataset.py
    ├── inference.py
    ├── network.py
    ├── training.py
    └── requirements.txt
```

## Dataset format

The code expects the dataset to be organized into class-specific folders:

```text
dataset/
├── bus/
├── car/
├── light/
├── sign/
├── truck/
└── vegetation/
```

Each folder should contain images belonging to that class.

During training, the script splits the provided dataset into training and validation subsets using a 90/10 split.

The training script also contains optional logic for creating separate train and test folders from an original class-structured dataset. This is controlled by the `TRAIN_TEST_FLAG` variable in `training.py`.

## Requirements

Install the required dependencies from the repository root:

```bash
pip install -r classification/requirements.txt
```

Note: PyTorch installation may vary depending on whether CPU or GPU/CUDA support is needed. Model architecture visualization with `torchview` may also require Graphviz to be installed on the system.

## Usage

### Training

To train the model from the repository root, run:

```bash
python classification/training.py path/to/train_dataset
```

The training script:

- loads images from class-specific folders,
- applies preprocessing and augmentation,
- splits the data into training and validation subsets,
- trains the custom ResNet-18 model from scratch,
- saves the best model checkpoint based on validation loss,
- saves learning curves.

Generated training outputs:

```text
model.pt
learning_curves.png
model_architecture.png
```

The trained model checkpoint is not included in this repository.

### Inference

After training, the best checkpoint is saved as `model.pt`. It can then be used for inference:

```bash
python classification/inference.py path/to/test_dataset model.pt
```

To run inference on only the first `N` samples:

```bash
python classification/inference.py path/to/test_dataset model.pt 100
```

Predictions are saved to:

```text
output_predictions/predictions.csv
```

The output CSV has the following format:

```text
filename,class_id
```

## Training results

The training and validation losses from one training run are shown below:

![Learning curves](final_learning_curves.png)

## Features

- Custom ResNet-18 style model implemented in PyTorch and trained from scratch
- Custom `torch.utils.data.Dataset` class
- Optional train/test dataset preparation utility
- Albumentations preprocessing and augmentation
- 90/10 train-validation split
- Cross-entropy loss
- Optional class-weighted loss for imbalanced data
- Adam optimizer
- `ReduceLROnPlateau` learning-rate scheduler
- Model checkpointing based on validation loss
- Learning-curve plotting
- Inference script with CSV prediction output

## Generated files

Running the training and inference scripts may create the following outputs:

```text
model.pt
learning_curves.png
model_architecture.png
output_predictions/
```

These files are generated during training or inference and are not required to be committed to the repository.
