"""Training loop, checkpointing, and metric logging for classification models.

The train function keeps the model that was trained in memory while
also returning a freshly loaded copy of the checkpoint with the best
validation loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cifar100_cnn.model.loader import load_model
from cifar100_cnn.utils.diagnose import compute_gradient_norms
from cifar100_cnn.utils.printer import section_printer
from cifar100_cnn.utils.timer import Timer
from cifar100_cnn.utils.visualizer import visualize_accuracy, visualize_loss, visualize_lr


@section_printer("Model Training")
def train(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Literal['cpu', 'cuda', 'mps'] | str | torch.device,
        model_init_args: dict,
        lr_scheduler: LRScheduler | None = None,
        output_path: Path | None = None,
        patience: int = 10,
        use_tensorboard: bool = True,
        writer: SummaryWriter | None = None,
        transform: nn.Module | None = None
) -> tuple[nn.Module, nn.Module]:
    """Train a model, save its best checkpoint, and generate metric plots.

    The model is updated in place and returned first. The second return value
    is a separate evaluation-mode model restored from the checkpoint with the
    best validation loss.

    output_path must be provided because training always writes a best
    checkpoint and metric plots there. model_init_args must contain the
    keyword arguments needed to reconstruct the model from checkpoint metadata.
    """

    checkpoint_path: Path = output_path / "best_model.pt"
    train_accuracies: list[float] = []
    train_losses: list[float] = []
    val_accuracies: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float('inf')
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

                logits: torch.Tensor = cast(torch.Tensor, model(xb))
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
                    logits: torch.Tensor = model(xb)
                    predictions = logits.argmax(dim=1)
                    loss = criterion(logits, yb)
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
