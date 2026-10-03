"""Unit tests for the from-scratch CIFAR residual network."""

from __future__ import annotations

import torch
import torch.nn as nn

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

    assert model.stages[0][0].conv1.stride == (1, 1)
    assert model.stages[1][0].conv1.stride == (2, 2)
    assert model.stages[2][0].conv1.stride == (2, 2)
    assert model.stages[3][0].conv1.stride == (2, 2)
