# CIFAR-Vision

A notebook-based project for building and evaluating a convolutional neural network on the CIFAR-10 image dataset.

## Contents

- `CIFAR10_CNN.ipynb` — Jupyter notebook containing the CIFAR-10 exploration and CNN workflow.
- `data/` — CIFAR-10 binary files, including five training batches, a test batch, and the class-label metadata.

## Dataset

CIFAR-10 contains 60,000 color images with 10 classes:

- 50,000 training images
- 10,000 test images
- 32 × 32 pixels per image
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck

## Setup

Install Python 3, Jupyter, and the dependencies used by the notebook:

```bash
python -m pip install jupyter tensorflow numpy pandas matplotlib seaborn scikit-learn
```

Run Jupyter from the project directory so the notebook can access `data/`:

```bash
python -m jupyter notebook
```

Then open `CIFAR10_CNN.ipynb`.

## Loading the local dataset

Run the following in a notebook cell when the current working directory is the project root:

```python
import pickle
from pathlib import Path
import numpy as np

data_dir = Path("data")

def load_batch(filename):
    with open(data_dir / filename, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"])[:, None]
    return images, labels

train_batches = [load_batch(f"data_batch_{i}") for i in range(1, 6)]
x_train = np.concatenate([images for images, _ in train_batches])
y_train = np.concatenate([labels for _, labels in train_batches])
x_test, y_test = load_batch("test_batch")

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

The expected shapes are `(50000, 32, 32, 3)` and `(10000, 32, 32, 3)` for the image arrays.

To load CIFAR-10 through TensorFlow instead, use:

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

This alternative downloads the dataset if it is not already cached by TensorFlow.
