"""Unit tests for training, evaluation, and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.trainer import evaluate, train


def _loader() -> DataLoader:
    """Build a small deterministic binary classification loader."""
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(features, targets), batch_size=2)


def test_evaluate_returns_predictions_probabilities_and_report() -> None:
    """Evaluation returns aligned arrays and a complete classification report."""
    model = nn.Linear(2, 2)

    results = evaluate(model, _loader(), classes=[0, 1])

    assert results["images"].shape == (4, 2)
    assert results["y_true"].shape == (4,)
    assert results["y_pred"].shape == (4,)
    assert results["y_probs"].shape == (4, 2)
    assert set(results["report_dict"]) >= {"0", "1", "accuracy"}


def test_train_creates_checkpoint_and_loads_best_model(tmp_path: Path) -> None:
    """Training writes artifacts and reconstructs the best checkpoint."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _, best_model = train(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        epochs=1,
        train_loader=_loader(),
        val_loader=_loader(),
        device="cpu",
        model_init_args={"in_features": 2, "out_features": 2},
        output_path=tmp_path / "run",
        use_tensorboard=False,
    )

    assert (tmp_path / "run" / "best_model.pt").exists()
    assert (tmp_path / "run" / "accuracy_plot.png").exists()
    assert isinstance(best_model, nn.Linear)
    assert not best_model.training


@pytest.mark.parametrize("epochs, patience", [(0, 1), (1, 0)])
def test_train_rejects_non_positive_control_values(
    tmp_path: Path, epochs: int, patience: int
) -> None:
    """Training limits must be positive to guarantee a checkpoint is produced."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="positive"):
        train(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            epochs=epochs,
            train_loader=_loader(),
            val_loader=_loader(),
            device="cpu",
            model_init_args={"in_features": 2, "out_features": 2},
            output_path=tmp_path,
            patience=patience,
            use_tensorboard=False,
        )


def test_load_model_rejects_malformed_metadata(tmp_path: Path) -> None:
    """Checkpoint metadata must be a mapping before model reconstruction."""
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({"model_meta": [], "model_state": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="model metadata"):
        load_model(checkpoint_path)


@pytest.mark.parametrize("early_stopping, expected_epochs", [(True, 2), (False, 4)])
def test_early_stopping_controls_training_duration(
    tmp_path: Path, early_stopping: bool, expected_epochs: int
) -> None:
    """A flat validation loss stops training only when early stopping is enabled."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loader = _loader()

    with patch.object(optimizer, "step", wraps=optimizer.step) as step:
        train(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            epochs=4,
            train_loader=loader,
            val_loader=loader,
            device="cpu",
            model_init_args={"in_features": 2, "out_features": 2},
            output_path=tmp_path,
            early_stopping=early_stopping,
            patience=1,
            use_tensorboard=False,
        )

    assert step.call_count == expected_epochs * len(loader)
