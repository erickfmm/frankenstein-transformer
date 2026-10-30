"""Held-out test split and end-of-training evaluation.

Implements the ``training.test_percentage`` contract: a deterministic
percentage of the task dataset is held out before training, the trainer sees
only the complementary part, and the final weights are evaluated on the held
out part after the last epoch. The metrics are written next to the run's CSV
log as ``test_metrics.json`` so downstream tooling (the ``full_tests`` harness,
CI, or an embedding host) can report them without re-running inference.

The split is deterministic for a fixed seed and dataset length: the same
configuration always reserves the same examples, independently of the
DataLoader shuffle order. ``test_percentage=0`` disables the feature and
returns the original loader untouched.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

#: File written by :func:`evaluate_and_write` next to the run's CSV log.
TEST_METRICS_FILENAME = "test_metrics.json"

#: Seed for the deterministic hold-out permutation.
DEFAULT_TEST_SEED = 42

#: Tasks whose trainer "accuracy" slot carries an error rate instead.
_ERROR_RATE_TASKS = {"asr"}


def resolve_test_percentage(training_runtime: Dict[str, Any]) -> float:
    """Return the clamped ``training.test_percentage`` value.

    Args:
        training_runtime: The validated ``training`` block from the YAML.

    Returns:
        A percentage in ``[0, 100]``; invalid or missing values yield ``0.0``.
    """
    raw = training_runtime.get("test_percentage", 0)
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        logging.warning(
            "Ignoring non-numeric training.test_percentage=%r", raw
        )
        return 0.0
    return max(0.0, min(100.0, value))


@dataclasses.dataclass(frozen=True)
class TestSplit:
    """A deterministic train/test partition of one DataLoader.

    Attributes:
        train_loader: Loader over the training partition.
        test_loader: Loader over the held-out partition, or ``None`` when the
            split is disabled.
        num_train: Number of training examples.
        num_test: Number of held-out examples (``0`` when disabled).
        test_percentage: Effective percentage that was requested.
    """

    train_loader: Any
    test_loader: Any
    num_train: int
    num_test: int
    test_percentage: float


def maybe_split_dataloader(
    dataloader: Any,
    training_runtime: Dict[str, Any],
    *,
    seed: int = DEFAULT_TEST_SEED,
) -> TestSplit:
    """Split ``dataloader`` according to ``training.test_percentage``.

    The original DataLoader settings (batch size, collate function, worker
    count, pin-memory and ``drop_last``) are preserved for the training
    partition; the held-out loader never drops its last partial batch and is
    not shuffled.

    Args:
        dataloader: The task DataLoader built by the caller.
        training_runtime: The validated ``training`` block from the YAML.
        seed: Seed for the deterministic permutation.

    Returns:
        A :class:`TestSplit`. When the percentage is zero, the dataset is not
        map-style, or it holds fewer than two examples, ``test_loader`` is
        ``None`` and ``train_loader`` is the original ``dataloader``.
    """
    from torch.utils.data import DataLoader, Subset

    percentage = resolve_test_percentage(training_runtime)
    if percentage <= 0.0:
        dataset = getattr(dataloader, "dataset", None)
        size = (
            len(dataset)
            if dataset is not None and hasattr(dataset, "__len__")
            else 0
        )
        return TestSplit(dataloader, None, size, 0, 0.0)

    dataset = getattr(dataloader, "dataset", None)
    batch_size = getattr(dataloader, "batch_size", None)
    if (
        not isinstance(dataloader, DataLoader)
        or dataset is None
        or not hasattr(dataset, "__len__")
        or not hasattr(dataset, "__getitem__")
        or batch_size is None
    ):
        logging.warning(
            "training.test_percentage=%s ignored: the task DataLoader is not a "
            "map-style DataLoader with a fixed batch size",
            percentage,
        )
        size = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else 0
        return TestSplit(dataloader, None, size, 0, 0.0)

    total = len(dataset)
    if total < 2:
        logging.warning(
            "training.test_percentage=%s ignored: the dataset has %d example(s) "
            "(need at least 2)",
            percentage,
            total,
        )
        return TestSplit(dataloader, None, total, 0, 0.0)

    num_test = int(round(total * percentage / 100.0))
    num_test = max(1, min(num_test, total - 1))
    num_train = total - num_test

    import torch

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(total, generator=generator).tolist()
    test_indices = permutation[:num_test]
    train_indices = permutation[num_test:]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    def _rebuild(subset: Any, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(getattr(dataloader, "num_workers", 0) or 0),
            pin_memory=bool(getattr(dataloader, "pin_memory", False)),
            drop_last=drop_last,
            collate_fn=getattr(dataloader, "collate_fn", None),
        )

    logging.info(
        "Held-out test split: %d train / %d test examples (%.2f%% held out)",
        num_train,
        num_test,
        percentage,
    )
    return TestSplit(
        train_loader=_rebuild(train_subset, shuffle=True, drop_last=bool(dataloader.drop_last)),
        test_loader=_rebuild(test_subset, shuffle=False, drop_last=False),
        num_train=num_train,
        num_test=num_test,
        test_percentage=percentage,
    )


def build_test_metrics(
    trainer: Any,
    split: TestSplit,
    task: str,
) -> Optional[Dict[str, Any]]:
    """Evaluate the trained model on the held-out partition.

    Args:
        trainer: The :class:`~src.training.trainer.TitanTrainer` used for
            training (its model is evaluated in ``eval()`` mode, no gradients).
        split: The :class:`TestSplit` produced by :func:`maybe_split_dataloader`.
        task: Training task identifier (``mlm``, ``asr``, ...).

    Returns:
        The metrics payload, or ``None`` when the split is disabled.
    """
    if split.test_loader is None:
        return None
    metrics = trainer.evaluate(split.test_loader)
    metric_name = "cer" if str(task).strip().lower() in _ERROR_RATE_TASKS else "accuracy"
    payload: Dict[str, Any] = {
        "task": str(task),
        "metric": metric_name,
        "test_percentage": float(split.test_percentage),
        "num_train_samples": int(split.num_train),
        "num_test_samples": int(split.num_test),
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "num_batches": int(metrics["num_batches"]),
    }
    return payload


def write_test_metrics(payload: Dict[str, Any], out_dir: str | Path = ".") -> Path:
    """Write ``payload`` as ``test_metrics.json`` inside ``out_dir``.

    Args:
        payload: Metrics returned by :func:`build_test_metrics`.
        out_dir: Directory that receives the file (the run's working dir).

    Returns:
        The path of the written file.
    """
    path = Path(out_dir) / TEST_METRICS_FILENAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logging.info(
        "Held-out test metrics: loss=%.4f %s=%.4f (%d examples) -> %s",
        payload["test_loss"],
        payload["metric"],
        payload["test_accuracy"],
        payload["num_test_samples"],
        path,
    )
    return path


def evaluate_and_write(
    trainer: Any,
    split: TestSplit,
    task: str,
    out_dir: str | Path = ".",
) -> Optional[Dict[str, Any]]:
    """Evaluate on the held-out split and persist ``test_metrics.json``.

    Args:
        trainer: The trainer whose model is evaluated.
        split: The :class:`TestSplit` produced by :func:`maybe_split_dataloader`.
        task: Training task identifier.
        out_dir: Directory that receives ``test_metrics.json``.

    Returns:
        The written payload, or ``None`` when the split is disabled.
    """
    payload = build_test_metrics(trainer, split, task)
    if payload is not None:
        write_test_metrics(payload, out_dir)
    return payload
