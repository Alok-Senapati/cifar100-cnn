"""CIFAR-100 dataset construction and preprocessing.

The package exports the dataset bundle and helpers for splitting, loading,
normalizing, and previewing CIFAR-100 data.
"""

from cifar100_cnn.data.loader import (
    CIFARDataset,
    compute_mean_and_std,
    create_transformers,
    get_cifar_dataset,
    get_train_val_split_indices,
    load_datasets,
    visualize_cifar_dataset,
)

__all__ = [
    "CIFARDataset",
    "compute_mean_and_std",
    "create_transformers",
    "get_cifar_dataset",
    "get_train_val_split_indices",
    "load_datasets",
    "visualize_cifar_dataset",
]
