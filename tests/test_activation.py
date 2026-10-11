"""Tests for bounded intermediate-activation capture."""

import pytest
import torch

from cifar100_cnn.model.resnet import SmallResNet
from cifar100_cnn.utils.activation import capture_layer_activations


def test_capture_layer_activations_includes_residual_outputs_and_removes_hooks() -> None:
    """Capture leaf operations and residual sums without retaining hooks."""
    model = SmallResNet(
        in_dims=(3, 8, 8),
        n_classes=3,
        stage_channels=(4, 8),
        blocks_per_stage=1,
        initialize_weights=False,
    ).eval()
    image = torch.randn(3, 8, 8)

    activations, logits = capture_layer_activations(model, image, max_channels=2)

    names = [activation.name for activation in activations]
    assert "stages.1.0.shortcut.0" in names
    assert "stages.1.0" in names
    assert names[-1] == "classifier"
    assert logits.shape == (1, 3)
    residual_output = activations[names.index("stages.1.0")]
    assert residual_output.output_shape == (1, 8, 4, 4)
    assert residual_output.values.shape == (1, 2, 4, 4)
    assert all(not module._forward_hooks for module in model.modules())


def test_capture_layer_activations_rejects_invalid_image_and_channel_limit() -> None:
    """Reject malformed inputs before registering hooks or running inference."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))

    with pytest.raises(ValueError, match="CHW"):
        capture_layer_activations(model, torch.zeros(1, 3, 2, 2))

    with pytest.raises(ValueError, match="positive integer"):
        capture_layer_activations(model, torch.zeros(3, 2, 2), max_channels=0)
