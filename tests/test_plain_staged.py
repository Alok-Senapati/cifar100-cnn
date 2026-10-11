"""Tests for the plain-staged control model and checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.plain_staged import PlainBlock, PlainStageCNN


def test_plain_staged_returns_logits_and_matches_transition_strides() -> None:
    """The control network preserves CIFAR logits and residual-stage downsampling."""
    model = PlainStageCNN(in_dims=(3, 32, 32), n_classes=100)
    images = torch.randn(2, 3, 32, 32)

    logits = model(images)

    assert logits.shape == (2, 100)
    strides = [
        cast(torch.nn.Conv2d, stage.get_submodule("0.layers.0")).stride for stage in model.stages
    ]
    assert strides == [
        (1, 1),
        (2, 2),
        (2, 2),
        (2, 2),
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"in_dims": (3, 32)}, ValueError, "in_dims"),
        ({"in_dims": (3, 32, 32), "n_classes": 0}, ValueError, "n_classes"),
        ({"in_dims": (3, 32, 32), "stage_channels": []}, ValueError, "stage_channels"),
        ({"in_dims": (3, 32, 32), "stage_channels": (8, True)}, TypeError, "integers"),
        ({"in_dims": (3, 32, 32), "blocks_per_stage": 0}, ValueError, "blocks_per_stage"),
    ],
)
def test_plain_staged_rejects_invalid_configuration(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    """Reject invalid constructor values with a clear error before building layers."""
    with pytest.raises(error, match=message):
        cast(Any, PlainStageCNN)(**kwargs)


def test_plain_staged_checks_input_dimensions() -> None:
    """Reject non-tensor and incorrectly shaped image batches consistently."""
    model = PlainStageCNN(in_dims=(3, 8, 8), n_classes=4, stage_channels=(4,))

    with pytest.raises(TypeError, match="torch.Tensor"):
        model("not a tensor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.zeros(2, 1, 8, 8))


def test_plain_staged_checkpoint_can_be_reloaded(tmp_path: Path) -> None:
    """Preserve constructor metadata needed by the generic checkpoint loader."""
    init_args: dict[str, Any] = {
        "in_dims": (3, 8, 8),
        "n_classes": 4,
        "stage_channels": (4, 8),
        "blocks_per_stage": 1,
        "initialize_weights": False,
    }
    model = PlainStageCNN(**init_args).eval()
    checkpoint_path = tmp_path / "plain-staged.pt"
    torch.save(
        {
            "model_meta": {
                "class_name": type(model).__name__,
                "module_path": type(model).__module__,
                "init_args": init_args,
            },
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    restored = load_model(checkpoint_path)
    images = torch.randn(2, 3, 8, 8)

    torch.testing.assert_close(restored(images), model(images))


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "stride", "error"),
    [(0, 4, 1, ValueError), (3, 4, 3, ValueError), (3, True, 1, TypeError)],
)
def test_plain_block_validates_channels_and_stride(
    in_channels: int, out_channels: int, stride: int, error: type[Exception]
) -> None:
    """Validate block dimensions and stride at its public constructor boundary."""
    with pytest.raises(error):
        PlainBlock(in_channels, out_channels, stride)
