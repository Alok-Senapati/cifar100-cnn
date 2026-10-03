"""Device-side augmentation and normalization for CIFAR-100 training batches."""

from __future__ import annotations

import torchvision.transforms.v2 as v2_transforms


def get_gpu_train_transform(
    means: list[float],
    stds: list[float],
) -> v2_transforms.Compose:
    """Build the training transform used after a batch is moved to its device.

    Args:
        means: Three RGB means computed from the unaugmented training subset.
        stds: Three positive RGB standard deviations from the training subset.

    Returns:
        A random 32x32 crop with four-pixel zero padding, a horizontal flip
        with probability 0.5, and channel normalization. Inputs must already
        be floating-point images scaled to [0, 1], with shape (C, H, W) or
        (N, C, H, W). This helper does not move inputs to CUDA or scale them.

    Notes:
        A batched call shares the sampled crop and flip across its images.
        Validation and test images use the loader's deterministic evaluation
        transform instead.
    """

    return v2_transforms.Compose(
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
