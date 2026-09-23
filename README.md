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
confusion matrix under `artifacts/baseline_cnn/<run_id>/`.

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
