"""Unit tests for the BaselineCNN model architecture."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cifar100_cnn.model.baseline import BaselineCNN


def test_baseline_cnn_initialization_defaults() -> None:
    """Model initializes with correct layer structures and default class count."""
    model = BaselineCNN(
        in_dims=(3, 32, 32),
        conv_channels=[16, 32],
        fc_hidden=64,
        n_classes=100,
    )

    assert model.in_dims == (3, 32, 32)
    assert model.conv_channels == (16, 32)
    assert model.fc_hidden == 64
    assert model.n_classes == 100

    assert isinstance(model.conv_layers, nn.ModuleDict)
    assert list(model.conv_layers.keys()) == ["conv_0", "conv_1"]
    assert isinstance(model.classifier_layer, nn.Sequential)


def test_baseline_cnn_accepts_tuples_and_lists() -> None:
    """Constructor accepts both lists and tuples for dimensions and channels."""
    model = BaselineCNN(
        in_dims=[3, 32, 32],
        conv_channels=(16, 32, 64),
        fc_hidden=128,
        n_classes=10,
    )

    assert model.in_dims == (3, 32, 32)
    assert model.conv_channels == (16, 32, 64)
    assert model.n_classes == 10


def test_baseline_cnn_forward_call_and_output_shape() -> None:
    """Model callable computes forward pass producing expected output tensor shape."""
    model = BaselineCNN(
        in_dims=(3, 32, 32),
        conv_channels=[16, 32],
        fc_hidden=64,
        n_classes=100,
    )
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)

    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 100)
    assert output.dtype == torch.float32


def test_baseline_cnn_forward_method_directly() -> None:
    """Explicit forward method call produces identical output to __call__."""
    model = BaselineCNN(
        in_dims=(3, 32, 32),
        conv_channels=[16, 32],
        fc_hidden=64,
        n_classes=100,
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        call_output = model(x)
        forward_output = model.forward(x)

    assert torch.equal(call_output, forward_output)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_baseline_cnn_handles_variable_batch_sizes(batch_size: int) -> None:
    """Model processes arbitrary batch sizes correctly."""
    model = BaselineCNN(
        in_dims=(3, 32, 32),
        conv_channels=[16, 32],
        fc_hidden=64,
        n_classes=100,
    )
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    assert output.shape == (batch_size, 100)


def test_baseline_cnn_backward_pass_computes_gradients() -> None:
    """Loss backpropagation populates gradients for all learnable parameters."""
    model = BaselineCNN(
        in_dims=(3, 32, 32),
        conv_channels=[16, 32],
        fc_hidden=64,
        n_classes=100,
    )
    x = torch.randn(4, 3, 32, 32)
    target = torch.randint(0, 100, (4,))
    criterion = nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN."


def test_baseline_cnn_training_step_reduces_loss() -> None:
    """Single optimization step updates weights and decreases or changes loss."""
    torch.manual_seed(42)
    model = BaselineCNN(
        in_dims=(3, 32, 32),
        conv_channels=[8, 16],
        fc_hidden=32,
        n_classes=10,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32)
    target = torch.randint(0, 10, (8,))

    # Record the loss before the optimizer updates the model.
    initial_loss = criterion(model(x), target).item()

    # Apply several optimization steps to the same batch.
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss


@pytest.mark.parametrize(
    "invalid_in_dims",
    [
        "invalid",
        (3, 32),
        (3, 32, 32, 1),
        (0, 32, 32),
        (3, -32, 32),
        (3, 32, 0),
        (3.0, 32, 32),
        (True, 32, 32),
    ],
)
def test_baseline_cnn_rejects_invalid_in_dims(invalid_in_dims: object) -> None:
    """Constructor validates in_dims elements and shape."""
    with pytest.raises((ValueError, TypeError)):
        BaselineCNN(
            in_dims=invalid_in_dims,  # type: ignore[arg-type]
            conv_channels=[16, 32],
            fc_hidden=64,
        )


@pytest.mark.parametrize(
    "invalid_conv_channels",
    [
        "invalid",
        [],
        (),
        [0],
        [-16, 32],
        [16, "32"],
        [16.5, 32],
        [True, 32],
    ],
)
def test_baseline_cnn_rejects_invalid_conv_channels(invalid_conv_channels: object) -> None:
    """Constructor validates conv_channels sequence and values."""
    with pytest.raises((ValueError, TypeError)):
        BaselineCNN(
            in_dims=(3, 32, 32),
            conv_channels=invalid_conv_channels,  # type: ignore[arg-type]
            fc_hidden=64,
        )


@pytest.mark.parametrize("invalid_fc_hidden", [0, -64, 64.5, "64", False])
def test_baseline_cnn_rejects_invalid_fc_hidden(invalid_fc_hidden: object) -> None:
    """Constructor validates fc_hidden positive integer constraint."""
    with pytest.raises((ValueError, TypeError)):
        BaselineCNN(
            in_dims=(3, 32, 32),
            conv_channels=[16, 32],
            fc_hidden=invalid_fc_hidden,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_n_classes", [0, -100, 100.0, "100", True])
def test_baseline_cnn_rejects_invalid_n_classes(invalid_n_classes: object) -> None:
    """Constructor validates n_classes positive integer constraint."""
    with pytest.raises((ValueError, TypeError)):
        BaselineCNN(
            in_dims=(3, 32, 32),
            conv_channels=[16, 32],
            fc_hidden=64,
            n_classes=invalid_n_classes,  # type: ignore[arg-type]
        )


def test_baseline_cnn_detects_spatial_dimension_collapse() -> None:
    """Constructor raises ValueError when conv layers cause spatial collapse to zero."""
    # Pooling reduces 4x4 to 2x2 and then 1x1, so a third block cannot pool.
    with pytest.raises(ValueError, match="too small for 2x2 max-pooling"):
        BaselineCNN(
            in_dims=(3, 4, 4),
            conv_channels=[16, 32, 64],
            fc_hidden=64,
        )


def test_baseline_cnn_forward_rejects_non_tensor() -> None:
    """Forward pass requires a torch.Tensor input."""
    model = BaselineCNN(in_dims=(3, 32, 32), conv_channels=[16], fc_hidden=32)
    with pytest.raises(TypeError, match="Expected input to be a torch.Tensor"):
        model([[1.0]])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (3, 32, 32),  # 3D
        (1, 1, 3, 32, 32),  # 5D
        (2,),  # 1D
    ],
)
def test_baseline_cnn_forward_rejects_non_4d_tensor(invalid_shape: tuple[int, ...]) -> None:
    """Forward pass rejects inputs that are not 4D tensors."""
    model = BaselineCNN(in_dims=(3, 32, 32), conv_channels=[16], fc_hidden=32)
    x = torch.randn(*invalid_shape)
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model(x)


@pytest.mark.parametrize(
    "mismatched_shape",
    [
        (2, 1, 32, 32),  # Wrong channels (1 instead of 3)
        (2, 3, 16, 16),  # Wrong spatial dims (16x16 instead of 32x32)
        (2, 3, 64, 64),  # Wrong spatial dims (64x64 instead of 32x32)
    ],
)
def test_baseline_cnn_forward_rejects_dimension_mismatch(
    mismatched_shape: tuple[int, ...],
) -> None:
    """Forward pass rejects tensors whose dimensions do not match in_dims."""
    model = BaselineCNN(in_dims=(3, 32, 32), conv_channels=[16], fc_hidden=32)
    x = torch.randn(*mismatched_shape)
    with pytest.raises(ValueError, match="Expected input tensor with shape"):
        model(x)
