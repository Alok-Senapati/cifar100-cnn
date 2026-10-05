"""Tests for CIFAR-100 augmentation helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from cifar100_cnn.data.transforms import PerSampleBatchTransform


class AddSampleCounter(nn.Module):
    """Deterministic stand-in proving that a batch is mapped sample by sample."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return a distinct constant offset on every transform invocation."""
        self.calls += 1
        return image + float(self.calls)


def test_per_sample_batch_transform_maps_each_image_independently() -> None:
    """A batched input invokes the wrapped image transform once per sample."""
    inner = AddSampleCounter()
    transform = PerSampleBatchTransform(inner)
    batch = torch.zeros(3, 3, 4, 4)

    output = transform(batch)

    assert inner.calls == 3
    assert torch.all(output[0] == 1)
    assert torch.all(output[1] == 2)
    assert torch.all(output[2] == 3)


def test_per_sample_batch_transform_supports_single_image() -> None:
    """Single CHW tensors pass directly through the wrapped image transform."""
    inner = AddSampleCounter()
    transform = PerSampleBatchTransform(inner)
    image = torch.zeros(3, 4, 4)

    output = transform(image)

    assert inner.calls == 1
    assert torch.all(output == 1)
