"""Reusable error-analysis helpers for CIFAR-100 classification runs."""

from __future__ import annotations

from collections import Counter

import numpy as np


def top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 10,
) -> list[tuple[int, int, int]]:
    """Return the most frequent true-class -> predicted-class mistakes."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")

    mistakes = Counter(
        (int(true), int(pred))
        for true, pred in zip(y_true, y_pred, strict=True)
        if int(true) != int(pred)
    )
    return [(true, pred, count) for (true, pred), count in mistakes.most_common(top_k)]


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[int, float]:
    """Compute recall-style accuracy independently for each target class."""
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")

    result: dict[int, float] = {}
    for class_id in range(n_classes):
        mask = y_true == class_id
        total = int(mask.sum())
        result[class_id] = float((y_pred[mask] == class_id).mean()) if total else 0.0
    return result
