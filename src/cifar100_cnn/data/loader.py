"""Load CIFAR-100 data with deterministic splits and training-only normalization.

The loader creates independent training and evaluation transforms, reusable data
loaders, and class metadata. Normalization statistics are computed from the
training subset to avoid incorporating validation or test data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload

import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

RANDOM_SEED = 42
BASE_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
Transform = Callable[[Any], Tensor]
IMAGE_TO_TENSOR = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)


@dataclass
class CIFARDataset:
    """Data loaders and metadata required by model training.

    Attributes:
        train_loader: Shuffled batches from the training split.
        val_loader: Ordered batches from the validation split.
        test_loader: Ordered batches from the official test split.
        classes: Class names ordered by numeric label.
        classmap: Mapping from numeric label to class name.
        img_size: Image dimensions in channel-height-width order.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: list[str]
    classmap: dict[int, str]
    img_size: tuple[int, int, int]


def get_train_val_split_indices(
    dataset: Dataset,
    train_split_ratio: float = 0.9,
) -> tuple[Sequence[int], Sequence[int]]:
    """Return deterministic, non-overlapping training and validation indices.

    Args:
        dataset: Dataset to split.
        train_split_ratio: Fraction of examples assigned to the training split.

    Returns:
        Two index sequences for the training and validation splits.

    Raises:
        TypeError: If the dataset does not report its length.
        ValueError: If the ratio cannot create two non-empty splits.
    """
    if not isinstance(dataset, Sized):
        raise TypeError("dataset must report its length.")
    if not 0.0 < train_split_ratio < 1.0:
        raise ValueError("train_split_ratio must be greater than 0 and less than 1.")

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_split_ratio)
    val_size = dataset_size - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            "dataset and train_split_ratio must create non-empty train and validation splits."
        )

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = random_split(dataset, (train_size, val_size), generator=generator)
    return train_set.indices, val_set.indices


@overload
def load_datasets(
    train_transform: Transform | None = None,
    eval_transform: Transform | None = None,
    train_indices: Sequence[int] | None = None,
    val_indices: Sequence[int] | None = None,
    train_only: Literal[False] = False,
) -> tuple[Subset, Subset, datasets.CIFAR100]: ...


@overload
def load_datasets(
    train_transform: Transform | None = None,
    eval_transform: Transform | None = None,
    train_indices: Sequence[int] | None = None,
    val_indices: Sequence[int] | None = None,
    train_only: Literal[True] = ...,
) -> tuple[datasets.CIFAR100]: ...


def load_datasets(
    train_transform: Transform | None = None,
    eval_transform: Transform | None = None,
    train_indices: Sequence[int] | None = None,
    val_indices: Sequence[int] | None = None,
    train_only: bool = False,
) -> tuple[datasets.CIFAR100] | tuple[Subset, Subset, datasets.CIFAR100]:
    """Load CIFAR-100 and construct the requested dataset splits.

    Training and validation use separate dataset instances so they can apply
    different transforms while selecting examples from the same source split.

    Args:
        train_transform: Transform applied to training examples.
        eval_transform: Transform applied to validation and test examples.
        train_indices: Training indices in the original training dataset.
        val_indices: Validation indices in the original training dataset.
        train_only: If true, return only the unpartitioned training dataset.

    Returns:
        A one-element tuple containing the training dataset when train_only is
        true; otherwise, a tuple containing the training subset, validation
        subset, and official test dataset.

    Raises:
        ValueError: If split indices are omitted when all datasets are requested.
    """
    if not train_only and (train_indices is None or val_indices is None):
        raise ValueError("train_indices and val_indices are required when train_only is False.")
    if train_transform is None:
        train_transform = IMAGE_TO_TENSOR
    if eval_transform is None:
        eval_transform = IMAGE_TO_TENSOR

    train_dataset = datasets.CIFAR100(
        root=str(BASE_DATA_DIR),
        train=True,
        download=True,
        transform=train_transform,
    )
    if train_only:
        return (train_dataset,)

    assert train_indices is not None
    assert val_indices is not None

    val_dataset = datasets.CIFAR100(
        root=str(BASE_DATA_DIR),
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=str(BASE_DATA_DIR),
        train=False,
        download=True,
        transform=eval_transform,
    )
    return (
        Subset(train_dataset, train_indices),
        Subset(val_dataset, val_indices),
        test_dataset,
    )


def compute_mean_and_std(dataset: Dataset, indices: Sequence[int]) -> tuple[Tensor, Tensor]:
    """Calculate per-channel population statistics for selected RGB images.

    Dataset items must yield image tensors in channel-height-width format.
    Accumulation uses float64 to limit precision loss across the selected pixels.

    Args:
        dataset: Dataset yielding an image and target for each item.
        indices: Dataset indices whose pixels contribute to the statistics.

    Returns:
        Three-element tensors containing the RGB means and population
        standard deviations.

    Raises:
        ValueError: If no indices are supplied or images are not three-channel batches.
    """
    if len(indices) == 0:
        raise ValueError("indices must contain at least one dataset index.")

    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False, num_workers=0)
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    for images, _ in loader:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("dataset images must have shape (batch, 3, height, width).")

        images = images.to(dtype=torch.float64)
        batch_pixels = images.shape[0] * images.shape[2] * images.shape[3]
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += images.square().sum(dim=(0, 2, 3))
        num_pixels += batch_pixels

    means = channel_sum / num_pixels
    variances = channel_squared_sum / num_pixels - means.square()
    return means.float(), variances.clamp_min(0).sqrt().float()


def create_transformers(
    means: Tensor, stds: Tensor
) -> tuple[transforms.Compose, transforms.Compose]:
    """Create matching image conversion and RGB normalization transforms.

    Args:
        means: Per-channel means in RGB order.
        stds: Per-channel standard deviations in RGB order.

    Returns:
        Equivalent training and evaluation transforms.

    Raises:
        ValueError: If either statistic does not contain three values or a
            standard deviation is non-positive.
    """
    if means.numel() != 3 or stds.numel() != 3:
        raise ValueError("means and stds must each contain one value for every RGB channel.")
    if torch.any(stds <= 0):
        raise ValueError("stds must be strictly positive.")

    normalize = transforms.Normalize(mean=means.tolist(), std=stds.tolist())
    transform = transforms.Compose([*IMAGE_TO_TENSOR.transforms, normalize])
    return transform, transform


def get_cifar_dataset(
    train_batchsize: int,
    eval_batchsize: int,
    num_workers: int = 2,
) -> CIFARDataset:
    """Build reproducible CIFAR-100 loaders and class metadata.

    Normalization statistics come from the training subset only, so validation
    and test examples do not influence preprocessing.

    Args:
        train_batchsize: Number of examples in each training batch.
        eval_batchsize: Number of examples in validation and test batches.
        num_workers: Number of worker processes used by the data loaders.

    Returns:
        A dataset bundle containing the loaders, class names, label mapping,
        and image dimensions.

    Raises:
        ValueError: If a batch size is non-positive or num_workers is negative.
    """
    if train_batchsize <= 0 or eval_batchsize <= 0:
        raise ValueError("train_batchsize and eval_batchsize must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    train_dataset = cast(tuple[datasets.CIFAR100], load_datasets(train_only=True))[0]
    train_indices, val_indices = get_train_val_split_indices(train_dataset)
    means, stds = compute_mean_and_std(train_dataset, train_indices)
    train_transform, eval_transform = create_transformers(means, stds)
    train_subset, val_subset, test_dataset = cast(
        tuple[Subset, Subset, datasets.CIFAR100],
        load_datasets(
            train_transform=train_transform,
            eval_transform=eval_transform,
            train_indices=train_indices,
            val_indices=val_indices,
        ),
    )

    # Store names by numeric label because model outputs use label indices.
    classmap = {index: name for name, index in train_dataset.class_to_idx.items()}
    image, _ = train_dataset[0]
    if not isinstance(image, Tensor) or image.ndim != 3:
        raise ValueError("the training transform must return an image tensor with shape (C, H, W).")
    img_size = cast(tuple[int, int, int], tuple(image.shape))
    pin_memory = torch.cuda.is_available()

    return CIFARDataset(
        train_loader=DataLoader(
            train_subset,
            batch_size=train_batchsize,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        val_loader=DataLoader(
            val_subset,
            batch_size=eval_batchsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=eval_batchsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        classes=[v for k, v in sorted(classmap.items())],
        classmap=classmap,
        img_size=img_size,
    )


def visualize_cifar_dataset(
    data_dir: str | Path = BASE_DATA_DIR,
    output_path: str | Path | None = None,
    show: bool = True,
    download: bool = True,
    columns: int = 10,
) -> Figure:
    """Plot one representative test image for every CIFAR-100 fine class.

    Args:
        data_dir: Directory containing, or receiving, the CIFAR-100 data files.
        output_path: Optional destination for the generated figure.
        show: Whether to open the figure in an interactive Matplotlib window.
        download: Whether torchvision may download missing dataset files.
        columns: Number of image columns in the output grid.

    Returns:
        The generated Matplotlib figure. The caller can close it with ``plt.close``.

    Raises:
        ValueError: If ``columns`` is not positive or an expected class has no image.
    """
    if columns <= 0:
        raise ValueError("columns must be positive.")

    dataset = datasets.CIFAR100(
        root=str(data_dir),
        train=False,
        download=download,
        transform=None,
    )
    classes = list(dataset.classes)
    if not classes:
        raise ValueError("CIFAR-100 dataset does not define any classes.")

    # Keep the first matching index for each class while scanning targets once.
    first_indices: dict[int, int] = {}
    for dataset_index, target in enumerate(dataset.targets):
        first_indices.setdefault(target, dataset_index)
        if len(first_indices) == len(classes):
            break

    missing_classes = [
        class_name for index, class_name in enumerate(classes) if index not in first_indices
    ]
    if missing_classes:
        raise ValueError(f"CIFAR-100 dataset is missing images for: {', '.join(missing_classes)}.")

    rows = (len(classes) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.8, rows * 2.1), squeeze=False)
    figure.suptitle("CIFAR-100: Representative Test Images", fontsize=16, fontweight="bold")

    for class_index, class_name in enumerate(classes):
        axis = axes.flat[class_index]
        image, _ = dataset[first_indices[class_index]]
        if isinstance(image, Tensor) and image.ndim == 3:
            image = image.detach().cpu().permute(1, 2, 0).numpy()
        axis.imshow(image)
        axis.set_title(class_name.replace("_", " "), fontsize=8)
        axis.axis("off")

    # Hide unused cells when the class count does not fill the final row.
    for axis in axes.flat[len(classes) :]:
        axis.axis("off")

    figure.tight_layout()
    if output_path is not None:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    return figure


if __name__ == "__main__":
    visualize_cifar_dataset()
