"""Tests for the imbalance-aware classification/segmentation metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.training.metrics import (  # noqa: E402
    ClassificationMetrics,
    classification_metrics,
)


def test_macro_f1_exposes_majority_class_bias():
    # Always predict class 0: accuracy is high but the rare classes are missed.
    predictions = torch.tensor([0, 0, 0, 0, 0, 0])
    targets = torch.tensor([0, 0, 0, 0, 1, 2])
    metrics = classification_metrics(predictions, targets)

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 0.0 + 0.0) / 3)
    assert metrics["weighted_f1"] == pytest.approx((0.8 * 4) / 6)
    assert metrics["macro_precision"] == pytest.approx((2 / 3) / 3)


def test_per_class_and_confusion_matrix_are_reported():
    metrics = classification_metrics(
        torch.tensor([0, 1, 1, 2]),
        torch.tensor([0, 1, 2, 2]),
    )
    classes = {entry["class_index"]: entry for entry in metrics["per_class"]}
    assert classes[0]["support"] == 1
    assert classes[0]["f1"] == pytest.approx(1.0)
    assert classes[1]["support"] == 1
    assert classes[1]["precision"] == pytest.approx(0.5)
    assert classes[1]["recall"] == pytest.approx(1.0)
    assert classes[2]["support"] == 2
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]


def test_accumulator_matches_single_batch():
    first = ClassificationMetrics()
    first.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    second = ClassificationMetrics()
    second.update(torch.tensor([2, 2]), torch.tensor([2, 1]))

    merged = ClassificationMetrics()
    merged.merge(first)
    merged.merge(second)
    expected = classification_metrics(
        torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 2, 1])
    )
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        assert merged.compute()[key] == pytest.approx(expected[key])


def test_empty_metrics_are_zero():
    metrics = ClassificationMetrics().compute()
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["balanced_accuracy"] == 0.0
    assert metrics["per_class"] == []


def test_ignore_index_is_dropped():
    metrics = classification_metrics(
        torch.tensor([0, 1, 1]),
        torch.tensor([0, -100, 1]),
    )
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ClassificationMetrics().update(torch.tensor([0, 1]), torch.tensor([0]))


def test_pixel_level_metrics_report_iou():
    # 2x2 segmentation, class 0 background, class 1 foreground.
    predictions = torch.tensor([[0, 0], [1, 1]])
    targets = torch.tensor([[0, 1], [1, 1]])
    metrics = classification_metrics(predictions, targets)
    # class 0: tp=1, union=2 -> iou=.5; class 1: tp=2, union=3 -> iou=2/3
    assert metrics["macro_iou"] == pytest.approx((0.5 + 2 / 3) / 2)
