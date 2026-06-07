# vect-micrograd-plus

Stress test of vect_mircograd [vect_micrograd](https://github.com/pvilanova/vect_micrograd).

This is an educational project.

## Demo results

These are the recorded results saved in the executed notebooks inspected for this README.

| Demo | Dataset / split | Model | Training setup | Best checkpoint | Test result | Saved output status |
|---|---:|---|---|---:|---:|---|
| `fer2013_demo_ema.ipynb` | 28,709 train images, 2,868 validation images, 7,178 test images | `2304 -> 512 -> 256 -> 7` ReLU MLP | Adam, mixed natural/balanced mini-batches, light augmentation, EMA checkpoint evaluation | step 3450, validation accuracy 47.98%, validation loss 1.5180 | **47.44%** test accuracy, 3,405 / 7,178 correct | Executed output present |
| `cifar10_demo_ema.ipynb` | 50,000 train images, 5,000 validation images, 10,000 test images | `3072 -> 768 -> 384 -> 10` ReLU MLP | Adam, per-pixel standardization, horizontal flip + random crop, EMA checkpoint evaluation | step 3999, validation accuracy 59.26%, validation loss 1.1990 | **58.20%** test accuracy, 5,820 / 10,000 correct | Executed output present |

## FER2013 example

The FER2013 notebook uses the Kaggle image-folder layout:

```text
train/<class_name>/*.jpg or *.png
test/<class_name>/*.jpg or *.png
```

The recorded run loaded:

```text
Train : (28709, 2304)
Test  : (7178, 2304)
Classes: angry, disgust, fear, happy, neutral, sad, surprise
Pixel range: [0.00, 1.00]
```

The data is split into train-fit and validation sets before normalization:

```text
Train-fit raw : (25841, 2304)
Validation    : (2868, 2304)
Test          : (7178, 2304)
Norm check    : mean=0.000, std=1.000
```

The model printed by the notebook:

```text
MLP of [Layer(2304, 512), Layer(512, 256), Layer(256, 7)]
Scalar parameters : 1,313,287
Value nodes       : 6  (vs ~1,313,287 in scalar micrograd)
```

Training configuration:

```python
STEPS      = 4000
BATCH_SIZE = 256
L2_ALPHA   = 1e-3
EVAL_EVERY = 50
EMA_START  = 500
EMA_DECAY  = 0.995
optimizer  = Adam(model.parameters(), lr=5e-4, total_steps=STEPS, weight_decay=0)
```

Mini-batches are sampled with `balanced_frac=0.50`. Augmentation uses horizontal flips with `p_flip=0.5`, no translation by default (`max_shift=0`), and small Gaussian pixel noise with `noise_std=0.01`.

Recorded best checkpoint:

```text
Best EMA checkpoint at step 3450, val acc 47.98%, val loss 1.5180
```

Recorded final evaluation:

```text
Restored checkpoint from step 3450
Test accuracy : 47.44%  (3,405 / 7,178 correct)
Chance baseline : 14.29%
Human accuracy  : ~65 %
```

The notebook also records:

```text
3773 misclassifications out of 7178 test images
```

This is a reasonable result for a raw-pixel MLP on FER2013. The run strongly benefits from EMA checkpoint selection because the training mini-batch accuracy rises much higher than validation accuracy, indicating overfitting.

## CIFAR-10 example

The CIFAR-10 notebook is the cleaner bridge toward future NICE work. It keeps labels for now because it is still a discriminative MLP stress test, but the data shape and preprocessing are close to what a later NICE notebook would need.

The recorded run loaded the standard CIFAR-10 Python archive:

```text
Train : (50000, 3072)  |  Test : (10000, 3072)
Classes (10): ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Pixel range: [0.00, 1.00]
```

The data is balanced across all ten classes:

```text
Train counts: {'airplane': 5000, 'automobile': 5000, 'bird': 5000, 'cat': 5000, 'deer': 5000, 'dog': 5000, 'frog': 5000, 'horse': 5000, 'ship': 5000, 'truck': 5000}
Test counts : {'airplane': 1000, 'automobile': 1000, 'bird': 1000, 'cat': 1000, 'deer': 1000, 'dog': 1000, 'frog': 1000, 'horse': 1000, 'ship': 1000, 'truck': 1000}
```

The data is split into train-fit and validation sets before normalization:

```text
Train-fit raw : (45000, 3072)
Validation    : (5000, 3072)
Test          : (10000, 3072)
```

Recorded preprocessing:

```text
Preprocessing: standardize
Norm check on sample: mean=0.000, std=0.999
```

The model printed by the notebook:

```text
MLP of [Layer(3072, 768), Layer(768, 384), Layer(384, 10)]
Scalar parameters : 2,659,210
Value nodes       : 6  (vs ~2,659,210 scalar nodes in scalar micrograd)
```

Recorded training setup:

```text
STEPS=4,000, BATCH_SIZE=256, profile=medium, preprocessing=standardize
```

The training cell records:

```text
Training time: 16.0 min
Best EMA checkpoint at step 3999, val acc 59.26%, val loss 1.1990
```

Recorded final evaluation:

```text
Restored checkpoint from step 3999
Test loss     : 1.2051
Test accuracy : 58.20%  (5,820 / 10,000 correct)
Chance baseline : 10.00%
```

The notebook also records:

```text
4180 misclassifications out of 10000 test images
```

For a raw-pixel MLP without convolutions, 58.20% on CIFAR-10 is a strong sanity check for the vectorized engine, batching, normalization, optimizer, and checkpointing. A CNN should score much higher, but this notebook is meant to stress-test vectorized micrograd rather than compete with modern CIFAR-10 models.

The notebook expects either an extracted CIFAR-10 Python archive at:

```text
data/cifar-10-batches-py/
```

or the archive file:

```text
data/cifar-10-python.tar.gz
```

If neither is present and `DOWNLOAD=True`, the notebook attempts to download the archive.

## Re-running the demos

### FER2013

1. Download FER2013 from Kaggle or another source that provides the folder layout.
2. Put `train/` and `test/` in the notebook working directory, or edit `DATA_ROOT`.
3. Run all cells in `fer2013_demo_ema.ipynb`.

### CIFAR-10

1. Put the CIFAR-10 Python archive or extracted folder under `data/`.
2. Open `cifar10_demo_ema.ipynb`.
3. Leave `PREPROCESSING='standardize'` for a fast educational run.
4. Try `PREPROCESSING='zca'` only after the standard run is working.

For a quick smoke test, set:

```python
FAST_DEV_RUN = True
TRAIN_SUBSET = 10_000
```

For a more meaningful run, use the full train split and leave `FAST_DEV_RUN=False`.

## License

MIT.