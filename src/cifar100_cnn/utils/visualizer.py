"""Plot metrics and inspect model outputs with Matplotlib.

Metric plots and confusion matrices support general classification tasks.
Image-based diagnostic helpers assume 28-by-28 grayscale digit images. Use
the CIFAR dataset preview function for RGB CIFAR-100 samples.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _to_floats(values: Iterable) -> list[float]:
    """Convert a sequence of numeric values to a list of floats.

    Args:
        values: Iterable of numeric values to coerce to Python floats.

    Returns:
        A list of float values preserving the iterable order.
    """
    return [float(value) for value in values]


def visualize_loss(
    train_losses: Iterable,
    val_losses: Iterable,
    save_path: Path | None = None,
) -> None:
    """Plot training and validation loss over time.

    Args:
        train_losses: Per-epoch training-loss values.
        val_losses: Per-epoch validation-loss values.
        save_path: Optional output path for the PNG file.
    """
    train = _to_floats(train_losses)
    val = _to_floats(val_losses)
    epochs = list(range(1, len(train) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", label="Train Loss", color="tab:red")
    ax.plot(epochs, val, marker="x", linestyle="--", label="Val Loss", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def visualize_accuracy(
    train_acc: Iterable,
    val_acc: Iterable,
    save_path: Path | None = None,
) -> None:
    """Plot training and validation accuracy over time.

    Args:
        train_acc: Per-epoch training accuracy values.
        val_acc: Per-epoch validation accuracy values.
        save_path: Optional output path for the PNG file.
    """
    train = _to_floats(train_acc)
    val = _to_floats(val_acc)
    epochs = list(range(1, len(train) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", label="Train Acc", color="tab:blue")
    ax.plot(epochs, val, marker="x", linestyle="--", label="Val Acc", color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def visualize_lr(
    lrs: Iterable,
    save_path: Path | None = None,
) -> None:
    """Plot learning rate decay over epochs.

    Args:
        lrs: Per-epoch learning rate values.
        save_path: Optional output path for the PNG file.
    """
    lr_values = _to_floats(lrs)
    epochs = list(range(1, len(lr_values) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, lr_values, marker="o", label="Learning Rate", color="tab:purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def visualize_confusion_matrix(
    y_true: np.ndarray | Iterable,
    y_pred: np.ndarray | Iterable,
    classes: list[int] | list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot the multi-class confusion matrix heatmap.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        classes: Ordered list of unique class labels.
        save_path: Optional output path for the PNG file.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_misclassified(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None,
    max_samples: int = 10,
    save_path: Path | None = None,
) -> None:
    """Plot misclassified 28-by-28 grayscale images.

    When probabilities are supplied, examples with the highest probability
    assigned to an incorrect class are shown first.

    Args:
        images: Flattened grayscale images with shape (N, 784).
        y_true: Ground-truth labels with shape (N,).
        y_pred: Predicted labels with shape (N,).
        y_probs: Optional class probabilities with shape (N, num_classes).
        max_samples: Maximum number of mistakes to display.
        save_path: Optional path for the saved PNG image.
    """
    mistake_indices = np.where(y_pred != y_true)[0]
    if len(mistake_indices) == 0:
        return

    if y_probs is not None:
        confidences = [y_probs[i, y_pred[i]] for i in mistake_indices]
        sorted_order = np.argsort(confidences)[::-1]
        selected_indices = mistake_indices[sorted_order[:max_samples]]
    else:
        selected_indices = mistake_indices[:max_samples]

    n_samples = len(selected_indices)
    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for idx, sample_idx in enumerate(selected_indices):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img = images[sample_idx].reshape(28, 28)
        true_lbl = y_true[sample_idx]
        pred_lbl = y_pred[sample_idx]

        title = f"True: {true_lbl} | Pred: {pred_lbl}"
        if y_probs is not None:
            conf = y_probs[sample_idx, pred_lbl] * 100.0
            title += f"\nConf: {conf:.1f}%"

        ax.imshow(img, cmap="gray")
        ax.set_title(title, color="darkred", fontsize=10)
        ax.axis("off")

    for idx in range(n_samples, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_feature_maps(
    feature_maps: dict[str, torch.Tensor],
    raw_image: np.ndarray | torch.Tensor | None = None,
    max_channels_per_layer: int = 16,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot convolutional feature maps for one grayscale digit image.

    Args:
        feature_maps: Mapping from layer names to activations shaped as
            (1, channels, height, width) or (channels, height, width).
        raw_image: Optional 28-by-28 or flattened 784-value input image.
        max_channels_per_layer: Maximum number of channels shown per layer.
        save_path: Optional path for the saved PNG image.

    Returns:
        The Matplotlib figure containing the input and activation maps.
    """
    n_layers = len(feature_maps)
    if n_layers == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No Feature Maps Available", ha="center", va="center")
        ax.axis("off")
        return fig

    # Prepare the optional raw input image.
    has_raw = raw_image is not None

    # Limit each activation-map row to at most eight channels.
    cols = min(8, max_channels_per_layer)
    section_rows: list[int] = []

    if has_raw:
        section_rows.append(1)

    for layer_tensor in feature_maps.values():
        n_ch = min(
            max_channels_per_layer,
            layer_tensor.shape[1] if layer_tensor.ndim == 4 else layer_tensor.shape[0],
        )
        rows_for_layer = (n_ch + cols - 1) // cols
        section_rows.append(rows_for_layer)

    total_grid_rows = sum(section_rows)
    fig = plt.figure(figsize=(2.0 * cols, 2.2 * total_grid_rows))
    grid_spec = fig.add_gridspec(total_grid_rows, cols)

    current_row = 0

    # Place the raw input image above the activation maps.
    if has_raw:
        ax_raw = fig.add_subplot(grid_spec[0, :2])
        if isinstance(raw_image, torch.Tensor):
            arr_raw = raw_image.detach().cpu().numpy()
        else:
            arr_raw = np.asarray(raw_image)
        if arr_raw.ndim == 1:
            arr_raw = arr_raw.reshape((28, 28))
        elif arr_raw.ndim == 3 and arr_raw.shape[0] == 1:
            arr_raw = arr_raw[0]

        ax_raw.imshow(arr_raw, cmap="gray")
        ax_raw.set_title("Input Image (28x28)", fontsize=11, fontweight="bold")
        ax_raw.axis("off")

        # Hide unused cells in the input-image row.
        for c in range(2, cols):
            ax_blank = fig.add_subplot(grid_spec[0, c])
            ax_blank.axis("off")

        current_row += 1

    # Render the activation maps for each layer.
    for layer_name, tensor in feature_maps.items():
        if tensor.ndim == 4:
            activations = tensor[0].detach().cpu().numpy()
        else:
            activations = tensor.detach().cpu().numpy()

        num_channels = min(max_channels_per_layer, activations.shape[0])
        h, w = activations.shape[1], activations.shape[2]

        for ch in range(num_channels):
            r = current_row + (ch // cols)
            c = ch % cols
            ax = fig.add_subplot(grid_spec[r, c])
            feature_slice = activations[ch]

            # Use color to show the relative activation intensity.
            ax.imshow(feature_slice, cmap="viridis")
            ax.set_title(f"{layer_name} ch{ch + 1}\n({h}x{w})", fontsize=8)
            ax.axis("off")

        # Hide unused cells in the rows allocated to this layer.
        allocated_rows = (num_channels + cols - 1) // cols
        for leftover in range(num_channels, allocated_rows * cols):
            r = current_row + (leftover // cols)
            c = leftover % cols
            ax = fig.add_subplot(grid_spec[r, c])
            ax.axis("off")

        current_row += allocated_rows

    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def visualize_augmentations(
    sample_image: np.ndarray | torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    num_variations: int = 8,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a grayscale digit beside independently augmented variations.

    Args:
        sample_image: Original image as a flattened 784-value array or a
            28-by-28 array.
        transform: Optional transform applied to each generated variation.
        num_variations: Number of augmented samples to draw.
        save_path: Optional path for the saved PNG image.

    Returns:
        The Matplotlib figure containing the input and generated variations.
    """
    if num_variations <= 0:
        raise ValueError("num_variations must be positive.")

    if isinstance(sample_image, torch.Tensor):
        base_arr = sample_image.detach().cpu().numpy()
    else:
        base_arr = np.asarray(sample_image)

    if base_arr.ndim == 1:
        base_arr = base_arr.reshape(28, 28)

    total_cols = num_variations + 1
    fig, axes = plt.subplots(1, total_cols, figsize=(2.2 * total_cols, 2.8))

    # Show the original image in the first panel.
    axes[0].imshow(base_arr, cmap="gray")
    axes[0].set_title("Original\n(Input)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    tensor_input = torch.tensor(base_arr, dtype=torch.float32).reshape(1, 28, 28)

    # Generate one independently transformed image per remaining panel.
    for i in range(1, total_cols):
        if transform is not None:
            aug_tensor = transform(tensor_input)
            aug_arr = aug_tensor.squeeze().numpy()
        else:
            aug_arr = base_arr

        axes[i].imshow(aug_arr, cmap="gray")
        axes[i].set_title(f"Augmented\n#{i}", fontsize=9)
        axes[i].axis("off")

    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig
