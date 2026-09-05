"""
CIFAR-100 Dataset Loader and Visualizer
---------------------------------------
Loads CIFAR-100, displays key dataset metrics, and generates visualizations:
1. Random sample grid with fine and coarse (superclass) labels.
2. Superclass hierarchy showcase (each superclass with its 5 fine classes).
3. Pixel intensity & RGB channel distribution analysis (with normalization stats).
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms


def load_cifar100(
    data_dir: str | Path = "data",
    download: bool = False,
) -> tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100, dict]:
    """Loads the CIFAR-100 train and test sets, and extracts metadata."""
    data_path = Path(data_dir)

    print(f"[*] Loading CIFAR-100 dataset from: {data_path.resolve()}")
    train_dataset = torchvision.datasets.CIFAR100(
        root=str(data_path),
        train=True,
        download=download,
        transform=transforms.ToTensor(),
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(data_path),
        train=False,
        download=download,
        transform=transforms.ToTensor(),
    )

    # Load metadata (fine & coarse class names)
    meta_path = data_path / "cifar-100-python" / "meta"
    train_batch_path = data_path / "cifar-100-python" / "train"

    coarse_names: list[str] = []
    fine_to_coarse: dict[int, int] = {}
    coarse_to_fines: dict[int, list[int]] = defaultdict(list)

    if meta_path.exists() and train_batch_path.exists():
        with open(meta_path, "rb") as f:
            meta_dict = pickle.load(f, encoding="latin1")
        coarse_names = meta_dict.get("coarse_label_names", [])

        with open(train_batch_path, "rb") as f:
            train_batch = pickle.load(f, encoding="latin1")

        fine_labels = train_batch.get("fine_labels", [])
        coarse_labels = train_batch.get("coarse_labels", [])

        for f_idx, c_idx in zip(fine_labels, coarse_labels):
            if f_idx not in fine_to_coarse:
                fine_to_coarse[f_idx] = c_idx
                coarse_to_fines[c_idx].append(f_idx)

        # Sort fine classes for each coarse class
        for c_idx in coarse_to_fines:
            coarse_to_fines[c_idx].sort()

    metadata = {
        "classes": train_dataset.classes,
        "coarse_classes": coarse_names,
        "fine_to_coarse": fine_to_coarse,
        "coarse_to_fines": coarse_to_fines,
    }

    return train_dataset, test_dataset, metadata


def print_dataset_summary(
    train_set: torchvision.datasets.CIFAR100,
    test_set: torchvision.datasets.CIFAR100,
    metadata: dict,
) -> None:
    """Prints detailed overview of dataset structure and statistics."""
    data_arr = train_set.data  # shape: (50000, 32, 32, 3), uint8

    mean_rgb = data_arr.mean(axis=(0, 1, 2)) / 255.0
    std_rgb = data_arr.std(axis=(0, 1, 2)) / 255.0

    print("\n" + "=" * 60)
    print("           CIFAR-100 DATASET SUMMARY")
    print("=" * 60)
    print(f" Training Samples:      {len(train_set):,}")
    print(f" Testing Samples:       {len(test_set):,}")
    print(f" Total Samples:         {len(train_set) + len(test_set):,}")
    print(f" Image Dimensions:      32 x 32 (RGB 3 channels)")
    print(f" Number of Fine Classes:   {len(metadata['classes'])}")
    print(f" Number of Superclasses:   {len(metadata['coarse_classes'])}")
    print(f" Classes per Superclass:   5")
    print(f" Images per Fine Class:    500 (train) + 100 (test) = 600")
    print("-" * 60)
    print(f" Raw Data Shape (Train):   {data_arr.shape} ({data_arr.dtype})")
    print(f" Pixel Value Range:        [{data_arr.min()}, {data_arr.max()}]")
    print(f" RGB Means:                [{mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f}]")
    print(f" RGB Stds:                 [{std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f}]")
    print("=" * 60 + "\n")


def visualize_random_samples(
    dataset: torchvision.datasets.CIFAR100,
    metadata: dict,
    rows: int = 5,
    cols: int = 5,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Plots a grid of randomly sampled images with class & superclass names."""
    num_samples = rows * cols
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.7))
    fig.suptitle(
        f"CIFAR-100: Random Samples ({num_samples} images)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    classes = metadata["classes"]
    coarse_classes = metadata.get("coarse_classes", [])
    fine_to_coarse = metadata.get("fine_to_coarse", {})

    for idx, ax in enumerate(axes.flat):
        img_idx = sample_indices[idx]
        img_np = dataset.data[img_idx]
        target = dataset.targets[img_idx]

        fine_name = classes[target].replace("_", " ")
        coarse_idx = fine_to_coarse.get(target, None)
        coarse_name = (
            coarse_classes[coarse_idx].replace("_", " ")
            if coarse_idx is not None and coarse_idx < len(coarse_classes)
            else "N/A"
        )

        ax.imshow(img_np)
        ax.set_title(
            f"#{target}: {fine_name}\n({coarse_name})",
            fontsize=8.5,
            fontweight="semibold",
            pad=4,
        )
        ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[+] Saved random sample grid to: {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)


def visualize_superclass_hierarchy(
    dataset: torchvision.datasets.CIFAR100,
    metadata: dict,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Visualizes all 20 superclasses, showing 1 example for each of the 5 constituent fine classes."""
    classes = metadata["classes"]
    coarse_classes = metadata.get("coarse_classes", [])
    coarse_to_fines = metadata.get("coarse_to_fines", {})

    if not coarse_classes or not coarse_to_fines:
        print("[-] Skipping superclass hierarchy visualization: metadata not available.")
        return

    # Find one representative image for each fine class
    fine_to_sample: dict[int, np.ndarray] = {}
    for i, target in enumerate(dataset.targets):
        if target not in fine_to_sample:
            fine_to_sample[target] = dataset.data[i]
        if len(fine_to_sample) == 100:
            break

    # 20 rows (superclasses), 5 columns (classes per superclass)
    num_superclasses = len(coarse_classes)
    fig, axes = plt.subplots(
        num_superclasses,
        5,
        figsize=(12, num_superclasses * 1.6),
        squeeze=False,
    )
    fig.suptitle(
        "CIFAR-100: Superclass Hierarchy (20 Superclasses x 5 Fine Classes)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for r, c_name in enumerate(coarse_classes):
        fines = coarse_to_fines.get(r, [])
        for c, f_idx in enumerate(fines):
            ax = axes[r, c]
            if f_idx in fine_to_sample:
                ax.imshow(fine_to_sample[f_idx])
            fine_name = classes[f_idx].replace("_", " ")
            ax.set_title(fine_name, fontsize=8, pad=3)
            ax.axis("off")

            # Label superclass on the leftmost column
            if c == 0:
                clean_super = c_name.replace("_", " ").title()
                ax.text(
                    -0.3,
                    0.5,
                    clean_super,
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    va="center",
                    ha="right",
                )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[+] Saved superclass hierarchy grid to: {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)


def visualize_channel_distributions(
    dataset: torchvision.datasets.CIFAR100,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Analyzes and plots RGB pixel intensity distributions and normalization parameters."""
    data_arr = dataset.data  # shape: (50000, 32, 32, 3), uint8

    r_vals = data_arr[:, :, :, 0].flatten()
    g_vals = data_arr[:, :, :, 1].flatten()
    b_vals = data_arr[:, :, :, 2].flatten()

    mean_rgb = [r_vals.mean() / 255.0, g_vals.mean() / 255.0, b_vals.mean() / 255.0]
    std_rgb = [r_vals.std() / 255.0, g_vals.std() / 255.0, b_vals.std() / 255.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    channels = [
        ("Red Channel", r_vals, "#e74c3c", mean_rgb[0], std_rgb[0]),
        ("Green Channel", g_vals, "#2ecc71", mean_rgb[1], std_rgb[1]),
        ("Blue Channel", b_vals, "#3498db", mean_rgb[2], std_rgb[2]),
    ]

    for ax, (name, vals, color, mean, std) in zip(axes, channels):
        # Sample for fast histogram rendering
        sample = np.random.choice(vals, size=200000, replace=False)
        ax.hist(sample, bins=64, color=color, alpha=0.75, edgecolor="black", density=True)
        ax.axvline(
            mean * 255.0,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=f"Mean: {mean:.4f} ({mean*255.0:.1f})",
        )
        ax.set_title(f"{name}\nMean={mean:.4f}, Std={std:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Pixel Value (0-255)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("Probability Density", fontsize=10)
    fig.suptitle(
        "CIFAR-100: Color Channel Pixel Intensity Distribution (Train Set)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[+] Saved channel distribution plot to: {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and visualize CIFAR-100 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to dataset root directory containing cifar-100-python",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory where output visualization figures will be saved",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively using matplotlib GUI window",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CIFAR-100
    train_set, test_set, metadata = load_cifar100(data_dir=data_dir, download=False)

    # 2. Print Summary Statistics
    print_dataset_summary(train_set, test_set, metadata)

    # 3. Generate Visualizations
    print("[*] Generating visualizations...")
    visualize_random_samples(
        train_set,
        metadata,
        rows=5,
        cols=5,
        output_path=output_dir / "cifar100_samples.png",
        show=args.show,
    )

    visualize_superclass_hierarchy(
        train_set,
        metadata,
        output_path=output_dir / "cifar100_superclasses.png",
        show=args.show,
    )

    visualize_channel_distributions(
        train_set,
        output_path=output_dir / "cifar100_channel_distributions.png",
        show=args.show,
    )

    print(f"\n[v] All visualizations successfully generated and saved to '{output_dir.resolve()}'!\n")


if __name__ == "__main__":
    main()
