"""Tests for the plain staged residual-ablation model."""

from __future__ import annotations

import torch

from cifar100_cnn.model.plain_staged import PlainStageCNN


def test_plain_staged_matches_residual_stage_output_shape() -> None:
    """The control network returns CIFAR-100 logits with the shared stage layout."""
    model = PlainStageCNN(in_dims=(3, 32, 32), n_classes=100)
    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 100)


def test_plain_staged_uses_stride_two_at_stage_transitions() -> None:
    """Stages 2-4 downsample exactly like the residual model."""
    model = PlainStageCNN(in_dims=(3, 32, 32), n_classes=100)

    assert model.stages[0][0].layers[0].stride == (1, 1)
    assert model.stages[1][0].layers[0].stride == (2, 2)
    assert model.stages[2][0].layers[0].stride == (2, 2)
    assert model.stages[3][0].layers[0].stride == (2, 2)
