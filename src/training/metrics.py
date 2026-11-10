"""Torch-only classification and segmentation metrics.

The held-out test split originally reported only sample-weighted accuracy,
which hides class imbalance: a model that always predicts the majority class
can score high accuracy while being useless for the minority classes. This
module adds the metrics that stay informative on small, imbalanced test sets:

* accuracy (micro, the existing metric),
* balanced accuracy (mean per-class recall),
* macro precision / recall / F1 (every present class weighs the same),
* weighted precision / recall / F1 (weighted by class support),
* per-class precision / recall / F1 / support,
* the confusion matrix.

Only present classes (``support > 0``) enter the macro/balanced averages, so a
class that happens to be absent from the held-out split does not zero the
score. Everything is implemented with ``torch`` so the toolkit keeps its
dependency set unchanged (no scikit-learn).

Example:

    >>> import torch
    >>> metrics = ClassificationMetrics()
    >>> metrics.update(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 0]))
    >>> round(metrics.compute()["macro_f1"], 4)
    0.6667
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch


def _as_flat_long(values: Any) -> torch.Tensor:
    """Return ``values`` as a flat ``int64`` CPU tensor.

    Args:
        values: Tensor, numpy array, or nested sequence of integer labels.

    Returns:
        A one-dimensional ``torch.long`` tensor on the CPU.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to("cpu").reshape(-1)
        return tensor.to(torch.long)
    return torch.as_tensor(values, dtype=torch.long, device="cpu").reshape(-1)


class ClassificationMetrics:
    """Accumulate sample- or pixel-level classification metrics.

    The accumulator starts empty and grows its confusion matrix on demand, so
    callers do not need to know ``num_classes`` up front. Labels must be
    non-negative integers; negative targets (``-100``/ignore indices) are
    dropped together with their prediction.

    Args:
        num_classes: Optional initial number of classes. The matrix grows if a
            later batch contains a larger label.

    Attributes:
        num_classes: Current number of classes represented by the matrix.
        num_samples: Number of valid (prediction, target) pairs added.
    """

    def __init__(self, num_classes: Optional[int] = None) -> None:
        initial = max(0, int(num_classes or 0))
        self._confusion = torch.zeros((initial, initial), dtype=torch.long)
        self.num_classes = initial
        self.num_samples = 0

    def _grow(self, size: int) -> None:
        """Grow the confusion matrix to at least ``size`` classes."""
        if size <= self.num_classes:
            return
        grown = torch.zeros((size, size), dtype=torch.long)
        if self.num_classes:
            grown[: self.num_classes, : self.num_classes] = self._confusion
        self._confusion = grown
        self.num_classes = size

    def update(
        self,
        predictions: Iterable | torch.Tensor,
        targets: Iterable | torch.Tensor,
    ) -> None:
        """Add one batch of predictions and ground-truth labels.

        Args:
            predictions: Flat or shaped integer predictions.
            targets: Flat or shaped integer targets with the same number of
                elements as ``predictions``.

        Raises:
            ValueError: If the two tensors do not have the same number of
                elements.
        """
        preds = _as_flat_long(predictions)
        truth = _as_flat_long(targets)
        if preds.numel() != truth.numel():
            raise ValueError(
                "prediction and target tensors must have the same number of "
                f"elements, got {preds.numel()} and {truth.numel()}"
            )
        if preds.numel() == 0:
            return

        valid = (truth >= 0) & (preds >= 0)
        if not bool(valid.any()):
            return
        preds = preds[valid]
        truth = truth[valid]

        largest = int(max(int(preds.max()), int(truth.max())))
        self._grow(largest + 1)

        flat = truth * self.num_classes + preds
        counts = torch.bincount(
            flat, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        self._confusion += counts
        self.num_samples += int(valid.sum().item())

    def merge(self, other: "ClassificationMetrics") -> None:
        """Merge another accumulator into this one.

        Args:
            other: Accumulator with the same class indexing.
        """
        self._grow(other.num_classes)
        self._confusion[: other.num_classes, : other.num_classes] += other._confusion
        self.num_samples += other.num_samples

    def compute(self) -> Dict[str, Any]:
        """Return every metric computed from the accumulated confusion matrix.

        Returns:
            Dict with ``accuracy``, ``balanced_accuracy``, ``macro_precision``,
            ``macro_recall``, ``macro_f1``, ``macro_iou``,
            ``weighted_precision``, ``weighted_recall``, ``weighted_f1``,
            ``per_class`` (list of dicts) and ``confusion_matrix`` (nested
            lists). All scalars are floats in ``[0, 1]``; empty accumulators
            return zeros.
        """
        confusion = self._confusion
        total = int(confusion.sum().item())
        if total == 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "macro_iou": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1": 0.0,
                "per_class": [],
                "confusion_matrix": confusion.tolist(),
            }

        true_positive = confusion.diag().float()
        support = confusion.sum(dim=1).float()
        predicted = confusion.sum(dim=0).float()

        precision = torch.where(
            predicted > 0, true_positive / predicted.clamp(min=1), torch.zeros_like(true_positive)
        )
        recall = torch.where(
            support > 0, true_positive / support.clamp(min=1), torch.zeros_like(true_positive)
        )
        denominator = precision + recall
        f1 = torch.where(
            denominator > 0,
            2 * precision * recall / denominator.clamp(min=1e-12),
            torch.zeros_like(denominator),
        )
        union = support + predicted - true_positive
        iou = torch.where(
            union > 0, true_positive / union.clamp(min=1), torch.zeros_like(union)
        )

        present = support > 0
        num_present = int(present.sum().item())
        if num_present:
            macro_precision = float(precision[present].mean().item())
            macro_recall = float(recall[present].mean().item())
            macro_f1 = float(f1[present].mean().item())
            macro_iou = float(iou[present].mean().item())
            balanced_accuracy = macro_recall
            weighted_precision = float(
                (precision * support).sum().item() / total
            )
            weighted_recall = float((recall * support).sum().item() / total)
            weighted_f1 = float((f1 * support).sum().item() / total)
        else:  # pragma: no cover - unreachable when total > 0
            macro_precision = macro_recall = macro_f1 = macro_iou = 0.0
            balanced_accuracy = 0.0
            weighted_precision = weighted_recall = weighted_f1 = 0.0

        per_class: List[Dict[str, Any]] = []
        for index in range(self.num_classes):
            per_class.append(
                {
                    "class_index": index,
                    "support": int(support[index].item()),
                    "predicted": int(predicted[index].item()),
                    "precision": float(precision[index].item()),
                    "recall": float(recall[index].item()),
                    "f1": float(f1[index].item()),
                    "iou": float(iou[index].item()),
                }
            )

        return {
            "accuracy": float(true_positive.sum().item() / total),
            "balanced_accuracy": balanced_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "macro_iou": macro_iou,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
        }


def classification_metrics(
    predictions: Iterable | torch.Tensor,
    targets: Iterable | torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute :class:`ClassificationMetrics` for one pair of tensors.

    Args:
        predictions: Integer class predictions.
        targets: Integer class targets.
        num_classes: Optional initial class count.

    Returns:
        The dict returned by :meth:`ClassificationMetrics.compute`.
    """
    accumulator = ClassificationMetrics(num_classes=num_classes)
    accumulator.update(predictions, targets)
    return accumulator.compute()
