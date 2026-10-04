# CIFAR-100 CNN

A PyTorch baseline convolutional neural network for CIFAR-100 image
classification. The project includes deterministic data splits, training-only
RGB normalization, validation-based checkpointing, evaluation, and metric
artifacts.

## Setup

Install the project and development tools with:

```powershell
uv sync
```

CIFAR-100 is loaded through torchvision and downloaded into the repository's
data directory when it is not already available.

## Train

Run the baseline training script:

```powershell
uv run python scripts/train_baseline.py
```

The script selects CUDA, MPS, or CPU automatically. Use `--help` to see the
available options. For example, a short run without TensorBoard is:

```powershell
uv run python scripts/train_baseline.py --epochs 1 --batch-size 512 --no-use-tensorboard
```

Each run writes its checkpoint, training curves, classification report, and
confusion matrix under `artifacts/baseline_cnn/<training_name>_<run_id>/`.

### Architecture variants

Two additional training scripts use less frequent max pooling: after every
second convolution and after the final convolution.

| Script | Classifier | Artifact directory |
|---|---|---|
| `scripts/train_reduced_pooling.py` | Flatten, hidden linear layer, ReLU, output linear layer | `artifacts/reduced_pooling/` |
| `scripts/train_gap_classifier.py` | Global average pooling and a single output linear layer | `artifacts/gap_classifier/` |

Both scripts accept `--conv-channels`, `--training-name`, and the optimizer,
TensorBoard, and early-stopping options from the baseline workflow. Only the
flattened classifier accepts `--fc-hidden`; the GAP classifier's input width
is the number of channels in the final convolution.

```powershell
uv run python scripts/train_reduced_pooling.py --training-name reduced-pooling --conv-channels 32 64 128 --fc-hidden 256
uv run python scripts/train_gap_classifier.py --training-name gap --conv-channels 32 64 128 --early-stop --patience 10
```

Early stopping is disabled by default. Enable it with `--early-stop`; patience
counts consecutive epochs without an improvement in validation loss. Evaluation
uses the checkpoint with the lowest validation loss, whether or not early
stopping is enabled.

### GAP weight initialization

The GAP model enables custom initialization by default (`--initialize-weights`):

- Convolution weights: Kaiming normal, using `fan_out` and ReLU scaling.
- Linear classifier weights: Xavier uniform.
- Biases: zero.

Use `--no-initialize-weights` to retain PyTorch's default layer initialization
for a comparison run. This flag disables the custom initialization pass; the
layers still receive PyTorch's normal initial values.

```powershell
uv run python scripts/train_gap_classifier.py --training-name gap-default-init --no-initialize-weights
```

The initialization choice is saved in `training_args.json` and the checkpoint's
constructor arguments. Loading a checkpoint reconstructs the model and then
restores its saved weights, replacing the initial values.

### GAP learning rate scheduling

The GAP training script accepts `--scheduler none|step|cosine|plateau`.
The default, `none`, keeps the optimizer learning rate constant.

| Scheduler | Behavior | Options used |
|---|---|---|
| `step` | Multiply the rate every specified number of epochs | `--lr-step-size` (5), `--lr-decay-factor` (0.5) |
| `cosine` | Cosine annealing with fixed `T_max=100` | `--min-lr` (0.000001) |
| `plateau` | Reduce the rate when validation loss stops improving, with patience 3 | `--lr-decay-factor` (0.5), `--min-lr` (0.000001) |

```powershell
uv run python scripts/train_gap_classifier.py --training-name gap-step --scheduler step --lr-step-size 10 --lr-decay-factor 0.5
uv run python scripts/train_gap_classifier.py --training-name gap-plateau --scheduler plateau --min-lr 0.000001
```

Scheduler updates happen at the end of an epoch; the logged rate is the rate
used during that epoch. An epoch that triggers early stopping exits before
the scheduler update. Scheduler settings are saved in `training_args.json`.
Step scheduling does not use `--min-lr`. Changing `--epochs` does not change
the cosine period; after 100 scheduler steps, its rate can rise again.
The `--patience` option controls early stopping, while plateau scheduler
patience remains fixed at 3.

### Optional GAP BatchNorm

The model constructor accepts `use_batchnorm=True` to insert `BatchNorm2d`
between every convolution and ReLU:

```python
from cifar100_cnn.model.gap_classifier import ReducedPoolingGAPCNN

model = ReducedPoolingGAPCNN(
    in_dims=(3, 32, 32),
    conv_channels=[32, 64, 128],
    use_batchnorm=True,
)
```

BatchNorm is disabled by default. Enable it in the GAP training script with
`--use-batchnorm`, or explicitly disable it with `--no-use-batchnorm`:

```powershell
uv run python scripts/train_gap_classifier.py --training-name gap-batchnorm --use-batchnorm
```

With BatchNorm enabled, convolution biases are disabled because BatchNorm
provides a learned offset. The output linear layer retains its bias. The CLI
saves the choice in `training_args.json` and checkpoint constructor arguments.

When training through the Python API, include `use_batchnorm=True` in the
`model_init_args` passed to `train` so checkpoint loading reconstructs the
same architecture. Custom initialization sets BatchNorm scales to one and
biases to zero.

### GAP training augmentation

Use `--use-augmentation` to enable random 32x32 crops with four-pixel zero
padding and horizontal flips with probability 0.5. Augmentation is disabled
by default and can be explicitly disabled with `--no-use-augmentation`.

```powershell
uv run python scripts/train_gap_classifier.py --training-name gap-augmented --use-augmentation
```

On CUDA, the loader converts training images to float32 in [0, 1], and the
training loop applies augmentation followed by normalization after moving the
batch to the GPU. `PerSampleBatchTransform` applies the transform separately
to each image and stacks the results, so crop offsets and flip decisions are
sampled independently within a batch. This uses per-image calls on the GPU
rather than one vectorized augmentation call. On CPU or MPS, augmentation and
normalization run per image in the dataset transform before batching.

Validation and test images are always normalized without random augmentation.
RGB means and standard deviations are computed from the unaugmented training
subset and exposed as `CIFARDataset.means` and `CIFARDataset.stds`.
The augmentation flag is recorded in `training_args.json`.

When using `get_cifar_dataset(augment=True)` directly on a CUDA-capable machine,
pass `get_gpu_train_transform(dataset.means, dataset.stds)` to the trainer's
`transform` argument. The returned training batches still need augmentation
and normalization. The GAP script performs this wiring automatically.

## Inference app

Launch the Streamlit dashboard to select a saved model and run inference on
an image from the CIFAR-100 test set:

```powershell
uv run streamlit run app.py
```

The dashboard groups checkpoints by model type, lists the available training
runs, and shows the predicted class, confidence, and top predictions.

## Project layout

- `src/cifar100_cnn/args`: Training configuration dataclasses.
- `src/cifar100_cnn/data`: CIFAR-100 loading, splits, normalization, and previews.
- `src/cifar100_cnn/model`: Baseline architecture, training, evaluation, and checkpoints.
- `src/cifar100_cnn/utils`: Diagnostics, timing, console formatting, and plots.
- `scripts`: Executable training workflows.
- `app.py`: Streamlit model-selection and inference dashboard.
- `tests`: Unit tests for the model and data helpers.

Public package exports are listed in each subpackage's `__init__.py`.

## Checks

Run the test suite and lint checks with:

```powershell
uv run pytest
uv run ruff check src scripts tests
uv run mypy src tests scripts app.py
```
