"""Check pooling, gradients, and checkpoint compatibility for CNN variants."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cifar100_cnn.model.gap_classifier import ReducedPoolingGAPCNN
from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.reduced_pooling import BaselineReducedPoolingCNN


@pytest.mark.parametrize("model_class", [BaselineReducedPoolingCNN, ReducedPoolingGAPCNN])
@pytest.mark.parametrize("channels", [[4], [4, 8], [4, 8, 16], [4, 8, 16, 16]])
def test_variant_forward_backward_and_checkpoint(model_class, channels, tmp_path: Path) -> None:
    """Odd/even depths preserve the intended resolution and reload exact predictions."""
    kwargs = {"in_dims": (3, 16, 16), "conv_channels": channels, "n_classes": 5}
    if model_class is BaselineReducedPoolingCNN:
        kwargs["fc_hidden"] = 8
    model = model_class(**kwargs)
    inputs = torch.randn(2, 3, 16, 16)
    features = inputs
    for block in model.conv_layers.values():
        features = block(features)
    expected_size = 16 // (2 ** ((len(channels) + 1) // 2))
    assert features.shape == (2, channels[-1], expected_size, expected_size)

    logits = model(inputs)
    assert logits.shape == (2, 5)
    nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_meta": {
                "module_path": model_class.__module__,
                "class_name": model_class.__name__,
                "init_args": kwargs,
            },
            "model_state": model.state_dict(),
        },
        path,
    )
    restored = load_model(path)
    torch.testing.assert_close(restored(inputs), logits)


def test_gap_head_does_not_grow_with_spatial_resolution() -> None:
    """GAP keeps parameter count fixed when the input image dimensions increase."""
    models = [ReducedPoolingGAPCNN((3, size, size), [4, 8], 5) for size in (16, 32)]
    assert sum(p.numel() for p in models[0].parameters()) == sum(
        p.numel() for p in models[1].parameters()
    )
    for model in models:
        assert model(torch.randn(2, *model.in_dims)).shape == (2, 5)


@pytest.mark.parametrize("model_class", [BaselineReducedPoolingCNN, ReducedPoolingGAPCNN])
def test_variant_rejects_pooling_collapse(model_class) -> None:
    kwargs = {"fc_hidden": 8} if model_class is BaselineReducedPoolingCNN else {}
    with pytest.raises(ValueError, match="too small for 2x2 max-pooling"):
        model_class((3, 2, 2), [4, 8, 16], **kwargs)
