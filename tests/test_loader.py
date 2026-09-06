"""Unit tests for CIFAR-100 data loading helpers."""

from __future__ import annotations

import math
from collections.abc import Sized
from pathlib import Path

import matplotlib
import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, TensorDataset

matplotlib.use("Agg")

from cifar100_cnn.data import loader


class FakeCIFAR100(Dataset):
    """Small in-memory stand-in for torchvision's CIFAR-100 dataset."""

    classes = ["apple", "baby"]
    class_to_idx = {"apple": 0, "baby": 1}

    def __init__(self, root: str, train: bool, download: bool, transform=None) -> None:
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        size = 10 if train else 4
        self.images = torch.arange(size * 3 * 2 * 2, dtype=torch.float32).reshape(size, 3, 2, 2)
        self.targets = [index % len(self.classes) for index in range(size)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, index % len(self.classes)


def test_split_indices_are_reproducible_and_non_overlapping() -> None:
    """Splits should be deterministic and cover every dataset element exactly once."""
    dataset = TensorDataset(torch.zeros(10, 3, 1, 1), torch.arange(10))

    first_train, first_val = loader.get_train_val_split_indices(dataset, 0.8)
    second_train, second_val = loader.get_train_val_split_indices(dataset, 0.8)

    assert first_train == second_train
    assert first_val == second_val
    assert len(first_train) == 8
    assert len(first_val) == 2
    assert set(first_train).isdisjoint(first_val)
    assert set(first_train) | set(first_val) == set(range(len(dataset)))


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_split_indices_reject_invalid_ratios(ratio: float) -> None:
    """A split ratio must reserve data for both train and validation sets."""
    dataset = TensorDataset(torch.zeros(10, 3, 1, 1), torch.arange(10))

    with pytest.raises(ValueError, match="train_split_ratio"):
        loader.get_train_val_split_indices(dataset, ratio)


def test_load_datasets_requires_indices_before_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing split indices should fail without instantiating a downloadable dataset."""
    monkeypatch.setattr(
        loader.datasets,
        "CIFAR100",
        lambda **_: pytest.fail("CIFAR100 should not be created without split indices."),
    )

    with pytest.raises(ValueError, match="train_indices"):
        loader.load_datasets()


def test_load_datasets_applies_independent_train_and_eval_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train and validation subsets use separate dataset instances and transforms."""
    monkeypatch.setattr(loader.datasets, "CIFAR100", FakeCIFAR100)

    def train_transform(image: Tensor) -> Tensor:
        return image + 1

    def eval_transform(image: Tensor) -> Tensor:
        return image * 2

    train_subset, val_subset, test_dataset = loader.load_datasets(
        train_transform=train_transform,
        eval_transform=eval_transform,
        train_indices=[0, 2],
        val_indices=[1],
    )

    assert isinstance(train_subset, Subset)
    assert isinstance(val_subset, Subset)
    assert isinstance(train_subset.dataset, FakeCIFAR100)
    assert isinstance(val_subset.dataset, FakeCIFAR100)
    assert train_subset.indices == [0, 2]
    assert val_subset.indices == [1]
    assert train_subset.dataset.transform is train_transform
    assert val_subset.dataset.transform is eval_transform
    assert test_dataset.transform is eval_transform


def test_compute_mean_and_std_returns_channel_population_statistics() -> None:
    """Per-channel means and population standard deviations use every image pixel."""
    images = torch.tensor(
        [
            [[[0.0, 2.0]], [[1.0, 3.0]], [[2.0, 4.0]]],
            [[[4.0, 6.0]], [[5.0, 7.0]], [[6.0, 8.0]]],
        ]
    )
    dataset = TensorDataset(images, torch.tensor([0, 1]))

    means, stds = loader.compute_mean_and_std(dataset, [0, 1])

    assert torch.allclose(means, torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(stds, torch.full((3,), math.sqrt(5.0)))


def test_compute_mean_and_std_rejects_empty_indices() -> None:
    """Statistics are undefined when no training examples are selected."""
    dataset = TensorDataset(torch.zeros(1, 3, 1, 1), torch.zeros(1, dtype=torch.long))

    with pytest.raises(ValueError, match="indices"):
        loader.compute_mean_and_std(dataset, [])


def test_create_transformers_rejects_invalid_statistics() -> None:
    """RGB normalization requires three positive standard deviations."""
    with pytest.raises(ValueError, match="RGB"):
        loader.create_transformers(torch.ones(2), torch.ones(2))
    with pytest.raises(ValueError, match="strictly positive"):
        loader.create_transformers(torch.ones(3), torch.tensor([1.0, 0.0, 1.0]))


def test_get_cifar_dataset_builds_loaders_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public factory builds expected loader sizes and index-to-class metadata."""
    raw_train_dataset = FakeCIFAR100(root="data", train=True, download=False)

    def fake_load_datasets(**kwargs):
        if kwargs.get("train_only"):
            return (raw_train_dataset,)
        return (
            Subset(raw_train_dataset, kwargs["train_indices"]),
            Subset(raw_train_dataset, kwargs["val_indices"]),
            FakeCIFAR100(root="data", train=False, download=False),
        )

    monkeypatch.setattr(loader, "load_datasets", fake_load_datasets)
    monkeypatch.setattr(loader, "compute_mean_and_std", lambda *_: (torch.zeros(3), torch.ones(3)))
    monkeypatch.setattr(
        loader, "create_transformers", lambda *_: (lambda image: image, lambda image: image)
    )

    cifar_dataset = loader.get_cifar_dataset(train_batchsize=4, eval_batchsize=3, num_workers=0)

    assert isinstance(cifar_dataset.train_loader.dataset, Sized)
    assert len(cifar_dataset.train_loader.dataset) == 9
    assert isinstance(cifar_dataset.val_loader.dataset, Sized)
    assert len(cifar_dataset.val_loader.dataset) == 1
    assert isinstance(cifar_dataset.test_loader.dataset, Sized)
    assert len(cifar_dataset.test_loader.dataset) == 4
    assert cifar_dataset.classes == ["apple", "baby"]
    assert cifar_dataset.classmap == {0: "apple", 1: "baby"}
    assert cifar_dataset.img_size == (3, 2, 2)


def test_visualize_cifar_dataset_saves_one_image_for_each_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Visualization selects each class's first matching dataset image and saves the figure."""
    monkeypatch.setattr(loader.datasets, "CIFAR100", FakeCIFAR100)
    show_calls: list[None] = []

    def fake_show() -> None:
        show_calls.append(None)

    monkeypatch.setattr(loader.plt, "show", fake_show)
    output_path = tmp_path / "figures" / "cifar100.png"

    figure = loader.visualize_cifar_dataset(
        data_dir=tmp_path,
        output_path=output_path,
        show=True,
        download=False,
        columns=1,
    )

    assert output_path.exists()
    assert len(figure.axes) == len(FakeCIFAR100.classes)
    assert [axis.get_title() for axis in figure.axes] == ["apple", "baby"]
    assert show_calls == [None]
    loader.plt.close(figure)


def test_visualize_cifar_dataset_rejects_non_positive_column_count() -> None:
    """The grid must contain at least one column."""
    with pytest.raises(ValueError, match="columns"):
        loader.visualize_cifar_dataset(columns=0, show=False)
