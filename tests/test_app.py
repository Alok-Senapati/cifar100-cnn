"""Unit tests for Streamlit app helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app import architecture_sections, discover_checkpoints, predict, tensor_to_image
from cifar100_cnn.model.baseline import BaselineCNN
from cifar100_cnn.model.resnet import SmallResNet


def test_discover_checkpoints_groups_model_types_and_runs(tmp_path: Path) -> None:
    """Checkpoint discovery returns model types and run directory names."""
    baseline_run = tmp_path / "baseline_cnn" / "run-001"
    other_run = tmp_path / "resnet" / "run-002"
    baseline_run.mkdir(parents=True)
    other_run.mkdir(parents=True)
    (baseline_run / "best_model.pt").touch()
    (other_run / "best_model.pt").touch()

    assert discover_checkpoints(tmp_path) == {
        "baseline_cnn": {"run-001": baseline_run / "best_model.pt"},
        "resnet": {"run-002": other_run / "best_model.pt"},
    }


def test_discover_checkpoints_supports_legacy_direct_runs(tmp_path: Path) -> None:
    """Older direct run directories are available under the legacy model type."""
    run_dir = tmp_path / "123456"
    run_dir.mkdir()
    checkpoint = run_dir / "best_model.pt"
    checkpoint.touch()

    assert discover_checkpoints(tmp_path) == {"legacy": {"123456": checkpoint}}


def test_tensor_to_image_converts_chw_rgb_tensor() -> None:
    """Display conversion changes channel order and preserves valid pixel range."""
    image = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])

    converted = tensor_to_image(image)

    assert converted.shape == (1, 1, 3)
    assert np.array_equal(converted[0, 0], np.array([0.0, 0.5, 1.0]))


def test_tensor_to_image_rejects_non_rgb_input() -> None:
    """The app only supports the RGB CIFAR-100 input format."""
    with pytest.raises(ValueError, match="RGB"):
        tensor_to_image(torch.zeros(1, 28, 28))


def test_predict_returns_class_confidence_and_probabilities() -> None:
    """Inference returns a valid class index, confidence, and probability vector."""
    model = torch.nn.Linear(3, 2)
    image = torch.ones(3)

    class_index, confidence, probabilities = predict(model, image)

    assert class_index in {0, 1}
    assert 0.0 <= confidence <= 1.0
    assert probabilities.shape == (2,)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_architecture_sections_follow_the_model_family_structure() -> None:
    """The explainer flow maps to actual CNN blocks and ResNet stages."""
    cnn = BaselineCNN((3, 32, 32), [4, 8], fc_hidden=16, n_classes=5)
    resnet = SmallResNet(
        (3, 32, 32),
        n_classes=5,
        stage_channels=(4, 8),
        blocks_per_stage=1,
        initialize_weights=False,
    )

    cnn_sections = architecture_sections(cnn)
    resnet_sections = architecture_sections(resnet)

    assert [section[1] for section in cnn_sections] == [
        "conv_layers.conv_0",
        "conv_layers.conv_1",
        "classifier_layer",
    ]
    assert [section[1] for section in resnet_sections] == [
        "stem",
        "stages.0",
        "stages.1",
        "global_pool",
        "classifier",
    ]
