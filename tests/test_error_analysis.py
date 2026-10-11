"""Tests for classification error-analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cifar100_cnn.utils.error_analysis import per_class_recall, top_confusions


def test_top_confusions_orders_mistakes_by_frequency() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([1, 1, 0, 0, 1, 1])

    result = top_confusions(y_true, y_pred, top_k=2)

    assert result[0] == (0, 1, 2)
    assert result[1] in {(1, 0, 1), (2, 1, 1)}


def test_per_class_recall_computes_target_class_recall() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])

    result = per_class_recall(y_true, y_pred, n_classes=3)

    assert result == {0: 0.5, 1: 1.0, 2: 0.5}


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_confusions_rejects_non_positive_top_k(top_k: int) -> None:
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="positive"):
        top_confusions(y, y, top_k=top_k)
