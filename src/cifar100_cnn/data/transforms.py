"""Device-side augmentation and normalization for CIFAR-100 training batches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2_transforms


class PerSampleBatchTransform(nn.Module):
    """Apply an image transform independently to every sample in a batch.

    Torchvision v2 transforms accept batched tensors, but a single batched call
    may share one sampled random decision across the batch. CIFAR augmentation
    should sample crop offsets and flip decisions independently per image, so
    this wrapper maps the transform over the batch after it has moved to the
    training device.
    """

    def __init__(self, transform: nn.Module) -> None:
        """Register the image transform applied to each CHW tensor."""
        super().__init__()
        self.transform = transform

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Transform a CHW image or stack transformed images from an NCHW batch.

        The wrapped transform determines each output image's shape. Batch
        inputs must be nonempty and produce equally shaped output images so
        they can be stacked. Other input ranks raise ValueError.
        """
        if images.ndim == 3:
            return self.transform(images)
        if images.ndim != 4:
            raise ValueError(
                "Expected image tensor with shape (C, H, W) or (N, C, H, W), "
                f"got {tuple(images.shape)}."
            )
        # Separate calls let random transforms draw new parameters for each image.
        return torch.stack([self.transform(image) for image in images], dim=0)


def get_gpu_train_transform(
    means: list[float],
    stds: list[float],
) -> nn.Module:
    """Build per-image training augmentation for device-side execution.

    Args:
        means: Three RGB means computed from the unaugmented training subset.
        stds: Three positive RGB standard deviations from the training subset.

    Returns:
        A module that applies a random 32x32 crop with four-pixel zero padding,
        horizontal flip with probability 0.5, and channel normalization.
        Random crop and flip decisions are sampled independently for each image
        in a batch. Inputs must already be floating-point images scaled to
        [0, 1], with shape (C, H, W) or (N, C, H, W).

    Notes:
        Validation and test images continue to use the deterministic evaluation
        transform from the data loader. The returned module does not move input
        tensors to CUDA or scale them; the training loop must move the batch first.
        Images are processed one at a time on that device and then stacked,
        rather than using a single vectorized augmentation call for the batch.
    """

    image_transform = v2_transforms.Compose(
        [
            v2_transforms.RandomCrop(
                size=(32, 32),
                padding=4,
            ),
            v2_transforms.RandomHorizontalFlip(p=0.5),
            v2_transforms.Normalize(
                mean=means,
                std=stds,
            ),
        ]
    )

    return PerSampleBatchTransform(image_transform)
