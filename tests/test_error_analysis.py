"""Tests for classification error-analysis helpers."""

from __future__ import annotations

import numpy as np

from cifar100_cnn.utils.error_analysis import per_class_accuracy, top_confusions


def test_top_confusions_orders_mistakes_by_frequency() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([1, 1, 0, 0, 1, 1])

    result = top_confusions(y_true, y_pred, top_k=2)

    assert result[0] == (0, 1, 2)
    assert result[1] in {(1, 0, 1), (2, 1, 1)}


def test_per_class_accuracy_computes_classwise_recall() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])

    result = per_class_accuracy(y_true, y_pred, n_classes=3)

    assert result == {0: 0.5, 1: 1.0, 2: 0.5}
