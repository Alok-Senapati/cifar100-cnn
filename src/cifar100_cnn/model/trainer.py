"""Training, checkpointing, and evaluation helpers for classification models.

The trainer records metrics and checkpoints the model with the lowest
validation loss. Evaluation returns predictions, probabilities, input images,
and a classification report.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cifar100_cnn.model.loader import load_model
from cifar100_cnn.utils.diagnose import compute_gradient_norms
from cifar100_cnn.utils.printer import section_printer
from cifar100_cnn.utils.timer import Timer
from cifar100_cnn.utils.visualizer import visualize_accuracy, visualize_loss, visualize_lr


def get_optimizer(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> Optimizer:
    """Create the requested optimizer instance for the supplied model.

    Args:
        model: The PyTorch model whose parameters will be optimized.
        optimizer: One of ``adam``, ``adamw``, or ``sgd``.
        lr: Learning rate used by the optimizer.
        weight_decay: L2 penalty coefficient.
        momentum: Momentum coefficient for SGD.

    Returns:
        The matching PyTorch optimizer.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )

    valid = ["adam", "adamw", "sgd"]
    raise ValueError(f"Invalid optimizer. Please select from {valid}.")


@section_printer("Model Training")
def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Literal["cpu", "cuda", "mps"] | str | torch.device,
    model_init_args: dict[str, Any],
    lr_scheduler: LRScheduler | None = None,
    output_path: Path | None = None,
    patience: int = 10,
    use_tensorboard: bool = True,
    writer: SummaryWriter | None = None,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[nn.Module, nn.Module]:
    """Train a model and return its final and best-checkpoint versions.

    Args:
        model: Network to optimize. It is moved to the selected device.
        criterion: Loss function applied to model logits and targets.
        optimizer: Optimizer configured with the model parameters.
        epochs: Maximum number of training epochs.
        train_loader: Batches used to update model parameters.
        val_loader: Batches used to measure validation loss and accuracy.
        device: Device used for model execution and batch tensors.
        model_init_args: Constructor keyword arguments stored in the checkpoint.
        lr_scheduler: Optional scheduler stepped once per epoch. Plateau
            schedulers receive the validation loss.
        output_path: Directory for the best checkpoint and metric plots.
        patience: Consecutive validation epochs without improvement allowed
            before early stopping.
        use_tensorboard: Whether to create a writer when none is supplied.
        writer: Optional caller-owned writer for graph and metric logging.
        transform: Optional transform applied to training batches only.

    Returns:
        The final in-memory model and a separate evaluation-mode model restored
        from the checkpoint with the lowest validation loss.

    Raises:
        ValueError: If output_path is not supplied, training limits are not
            positive, or either data loader is empty.
    """

    if output_path is None:
        raise ValueError("output_path is required to save training artifacts.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if patience <= 0:
        raise ValueError("patience must be positive.")
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("train_loader and val_loader must both contain at least one batch.")

    output_path.mkdir(parents=True, exist_ok=True)
    model.to(device)
    checkpoint_path: Path = output_path / "best_model.pt"
    train_accuracies: list[float] = []
    train_losses: list[float] = []
    val_accuracies: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    degrade_counter = 0
    learning_rates: list[float] = []

    created_writer = False
    if writer is None and use_tensorboard:
        # Only close writers created here; callers retain ownership of supplied writers.
        writer = SummaryWriter(log_dir=str(output_path / "tensorboard"))
        created_writer = True

    if writer is not None:
        sample_input = next(iter(train_loader))[0][:1].to(device)
        writer.add_graph(model, sample_input)

    for epoch in range(1, epochs + 1):
        with Timer() as elapsed_timer:
            running_training_loss = 0.0
            running_training_correct_predictions = 0.0
            training_seen = 0

            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                xb: torch.Tensor = cast(torch.Tensor, data[0]).to(device=device, non_blocking=True)
                yb: torch.Tensor = cast(torch.Tensor, data[1]).to(device=device, non_blocking=True)

                if transform is not None:
                    xb = transform(xb)

                logits = cast(torch.Tensor, model(xb))
                predictions: torch.Tensor = logits.argmax(dim=1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                training_seen += xb.size(0)
                running_training_loss += loss.item() * xb.size(0)
                running_training_correct_predictions += (predictions == yb).sum().item()

            train_loss = running_training_loss / training_seen
            train_accuracy = running_training_correct_predictions / training_seen
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            running_val_loss = 0.0
            running_val_correct_predictions = 0.0
            val_seen = 0

            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    xb = cast(torch.Tensor, data[0]).to(device=device, non_blocking=True)
                    yb = cast(torch.Tensor, data[1]).to(device=device, non_blocking=True)
                    val_logits = cast(torch.Tensor, model(xb))
                    predictions = val_logits.argmax(dim=1)
                    loss = criterion(val_logits, yb)
                    val_seen += xb.size(0)
                    running_val_loss += loss.item() * xb.size(0)
                    running_val_correct_predictions += (predictions == yb).sum().item()

            val_accuracy = running_val_correct_predictions / val_seen
            val_loss = running_val_loss / val_seen
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                degrade_counter = 0
                best_val_loss = val_loss
                checkpoint = {
                    "model_meta": {
                        "class_name": model.__class__.__name__,
                        "module_path": model.__class__.__module__,
                        "init_args": model_init_args,
                        "arch_str": str(model),
                    },
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "val_acc": val_accuracy,
                }
                # Keep the best epoch weights because validation loss may worsen later.
                torch.save(checkpoint, checkpoint_path)
            else:
                degrade_counter += 1

            if degrade_counter >= patience:
                print(f"Early stopping at epoch: {epoch}...")
                break

        current_lr = optimizer.param_groups[0]["lr"]

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("accuracy/train", train_accuracy, epoch)
            writer.add_scalar("accuracy/val", val_accuracy, epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

            for name, param in model.named_parameters():
                writer.add_histogram(f"parameters/{name}", param, epoch)

            gradient_norms = compute_gradient_norms(model)
            for tag, value in gradient_norms.items():
                writer.add_scalar(tag, value, epoch)

        learning_rates.append(current_lr)
        if lr_scheduler is not None:
            # Plateau schedulers consume validation loss; other schedulers advance per epoch.
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

        print(
            f"Epoch: {epoch:02d}/{epochs}, "
            f"Time: {elapsed_timer.elapsed:.2f}s, LR: {current_lr:.6f} "
            f"| Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} "
            f"| Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f} "
        )

    if created_writer and writer is not None:
        writer.close()

    visualize_accuracy(
        train_accuracies, val_accuracies, save_path=output_path / "accuracy_plot.png"
    )
    visualize_loss(train_losses, val_losses, save_path=output_path / "loss_plot.png")
    visualize_lr(learning_rates, save_path=output_path / "lr_plot.png")

    return model, load_model(checkpoint_path, device, eval_mode=True)


@torch.no_grad()
@section_printer("Evaluating on Test Set")
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "cpu",
    classes: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate a model and compute classification metrics for a data loader.

    The model is moved to the requested device and evaluated without gradient
    tracking. Input images retain the shape provided by the data loader.

    Args:
        model: Classification model to evaluate.
        data_loader: Batches of input tensors and integer target labels.
        device: Device used for model execution. Defaults to CPU.
        classes: Optional ordered numeric label IDs included in the report.

    Returns:
        A dictionary containing concatenated images, true labels, predicted
        labels, class probabilities, and the classification report in dictionary
        and text forms.

    Raises:
        ValueError: If the data loader is empty.
    """
    if len(data_loader) == 0:
        raise ValueError("Cannot evaluate on an empty DataLoader.")

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()

    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch_features, batch_targets in data_loader:
        batch_features = batch_features.to(target_device)

        logits = cast(torch.Tensor, model(batch_features))
        batch_probs = torch.softmax(logits, dim=1)
        batch_preds = torch.argmax(batch_probs, dim=1)

        all_inputs.append(batch_features.detach().cpu().numpy())
        all_targets.append(batch_targets.detach().cpu().numpy())
        all_predictions.append(batch_preds.detach().cpu().numpy())
        all_probabilities.append(batch_probs.detach().cpu().numpy())

    images = np.concatenate(all_inputs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    y_probs = np.concatenate(all_probabilities, axis=0)

    report_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=classes,
        output_dict=False,
        zero_division=0,
    )

    return {
        "images": images,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "report_dict": report_dict,
        "report_str": report_str,
    }
