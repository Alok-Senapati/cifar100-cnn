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
