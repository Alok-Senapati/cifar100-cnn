"""Unit tests for the from-scratch CIFAR residual network."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.resnet import ResidualBlock, SmallResNet


def test_identity_residual_block_preserves_shape() -> None:
    """A stride-1 same-channel block keeps NCHW dimensions unchanged."""
    block = ResidualBlock(32, 32)
    x = torch.randn(4, 32, 16, 16)

    y = block(x)

    assert y.shape == x.shape
    assert isinstance(block.shortcut, nn.Identity)


def test_projection_residual_block_changes_channels_and_resolution() -> None:
    """Stage-transition blocks downsample and use a learned projection shortcut."""
    block = ResidualBlock(32, 64, stride=2)
    x = torch.randn(4, 32, 16, 16)

    y = block(x)

    assert y.shape == (4, 64, 8, 8)
    assert isinstance(block.shortcut, nn.Sequential)


def test_small_resnet_returns_cifar100_logits() -> None:
    """The default model maps CIFAR-shaped input to 100 class logits."""
    model = SmallResNet(in_dims=(3, 32, 32), n_classes=100)
    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 100)


def test_small_resnet_downsamples_only_at_stage_transitions() -> None:
    """The first block in stages 2-4 performs stride-2 downsampling."""
    model = SmallResNet(in_dims=(3, 32, 32), n_classes=100)

    expected_strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
    for stage, expected_stride in zip(model.stages, expected_strides, strict=True):
        assert isinstance(stage, nn.Sequential)
        first_block = stage[0]
        assert isinstance(first_block, ResidualBlock)
        assert first_block.conv1.stride == expected_stride


@pytest.mark.parametrize(
    "in_dims, n_classes, stage_channels, blocks_per_stage",
    [
        ((3, 32), 100, (32, 64), 2),
        ((3, 0, 32), 100, (32, 64), 2),
        ((3, 32, 32), 0, (32, 64), 2),
        ((3, 32, 32), 100, (), 2),
        ((3, 32, 32), 100, (32, 0), 2),
        ((3, 32, 32), 100, (32, 64), 0),
    ],
)
def test_small_resnet_rejects_invalid_configuration(
    in_dims, n_classes, stage_channels, blocks_per_stage
) -> None:
    """Invalid input geometry and architecture dimensions fail early."""
    with pytest.raises(ValueError):
        SmallResNet(in_dims, n_classes, stage_channels, blocks_per_stage)


def test_small_resnet_rejects_wrong_input_dimensions() -> None:
    """Forward pass checks the configured channel and spatial dimensions."""
    model = SmallResNet(in_dims=(3, 32, 32), stage_channels=(8, 16))
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(2, 3, 16, 16))


def test_small_resnet_checkpoint_round_trip(tmp_path: Path) -> None:
    """ResNet metadata and tensors reload through the restricted checkpoint loader."""
    model = SmallResNet(
        in_dims=(3, 8, 8),
        n_classes=4,
        stage_channels=(4, 8),
        blocks_per_stage=1,
    ).eval()
    inputs = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        expected = model(inputs)

    checkpoint_path = tmp_path / "resnet.pt"
    torch.save(
        {
            "model_meta": {
                "module_path": type(model).__module__,
                "class_name": type(model).__name__,
                "init_args": {
                    "in_dims": (3, 8, 8),
                    "n_classes": 4,
                    "stage_channels": [4, 8],
                    "blocks_per_stage": 1,
                },
            },
            "model_state": model.state_dict(),
            "optimizer_state": {},
        },
        checkpoint_path,
    )

    restored = load_model(checkpoint_path)
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), expected)
