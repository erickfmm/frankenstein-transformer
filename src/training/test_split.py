"""Held-out test split and end-of-training evaluation.

Implements the ``training.test_percentage`` contract: a deterministic
percentage of the task dataset is held out before training, the trainer sees
only the complementary part, and the final weights are evaluated on the held
out part after the last epoch. The metrics are written next to the run's CSV
log as ``test_metrics.json`` so downstream tooling (the ``full_tests`` harness,
CI, or an embedding host) can report them without re-running inference.

Classification tasks (``classification``, ``text_classification`` and
``audio_classification``) are split **stratified by default**
(``training.test_stratify``): every class keeps approximately the same share in
the train and test partitions, so a rare class is not randomly swallowed by
either side. The split stays deterministic for a fixed ``training.seed`` and
dataset length. ``test_percentage=0`` disables the feature and returns the
original loader untouched.

When a split is created, its indices are also persisted as ``test_split.json``
next to ``test_metrics.json`` so the deployment/inference evaluation can score
the exact held-out examples instead of the training rows.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from ..utils.reproducibility import make_generator, seed_worker
except ImportError:  # pragma: no cover - direct ``training.*`` imports
    from utils.reproducibility import make_generator, seed_worker

#: File written by :func:`evaluate_and_write` next to the run's CSV log.
TEST_METRICS_FILENAME = "test_metrics.json"

#: Manifest with the exact held-out indices (consumed by ``full_tests``).
TEST_SPLIT_FILENAME = "test_split.json"

#: Seed for the deterministic hold-out permutation.
DEFAULT_TEST_SEED = 42

#: Tasks whose trainer "accuracy" slot carries an error rate instead.
_ERROR_RATE_TASKS = {"asr"}

#: Tasks with one integer class label per example (stratifiable).
_CLASSIFICATION_TASKS = {
    "classification",
    "text_classification",
    "audio_classification",
}

#: Maximum number of dataset items probed for labels when no fast path exists.
_MAX_LABEL_PROBE = 50_000


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


def resolve_test_stratify(training_runtime: Dict[str, Any]) -> bool:
    """Return ``training.test_stratify`` (default ``True``).

    Args:
        training_runtime: The validated ``training`` block from the YAML.

    Returns:
        ``True`` unless the value is an explicit negative boolean/string.
    """
    raw = training_runtime.get("test_stratify", True)
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


@dataclasses.dataclass(frozen=True)
class TestSplit:
    """A deterministic, optionally stratified train/test partition.

    Attributes:
        train_loader: Loader over the training partition.
        test_loader: Loader over the held-out partition, or ``None`` when the
            split is disabled.
        num_train: Number of training examples.
        num_test: Number of held-out examples (``0`` when disabled).
        test_percentage: Effective percentage that was requested.
        stratified: Whether the split used class labels for stratification.
        num_classes: Number of distinct labels found (``0`` when unknown).
        class_counts_train: Per-class training counts keyed by ``str(label)``.
        class_counts_test: Per-class held-out counts keyed by ``str(label)``.
        train_indices: Dataset indices assigned to the training partition.
        test_indices: Dataset indices assigned to the held-out partition.
    """

    train_loader: Any
    test_loader: Any
    num_train: int
    num_test: int
    test_percentage: float
    stratified: bool = False
    num_classes: int = 0
    class_counts_train: Dict[str, int] = dataclasses.field(default_factory=dict)
    class_counts_test: Dict[str, int] = dataclasses.field(default_factory=dict)
    train_indices: Tuple[int, ...] = ()
    test_indices: Tuple[int, ...] = ()


def _coerce_stratum(value: Any) -> Any:
    """Return ``value`` as a hashable class label, or ``None`` if unusable.

    Args:
        value: Raw label from a dataset item or label column.

    Returns:
        ``int``/``str`` labels are kept; one-element numeric tensors and
        NumPy scalars are unwrapped; multi-element tensors return ``None``.
    """
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001 - exotic label containers
            return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        return value
    return None


def _label_from_item(item: Any, label_key: Optional[str] = None) -> Any:
    """Extract a class label from one dataset item.

    Args:
        item: Dataset item (dict, tuple/list, or bare label).
        label_key: Preferred dict key (from the dataset's label column).

    Returns:
        A hashable label or ``None`` when the item carries no usable label.
    """
    if isinstance(item, dict):
        keys: List[str] = []
        if label_key is not None:
            keys.append(label_key)
        keys.extend(
            ["labels", "label", "target", "targets", "class", "class_index", "y"]
        )
        for key in keys:
            if key in item:
                value = _coerce_stratum(item[key])
                if value is not None:
                    return value
        return None
    if isinstance(item, (tuple, list)):
        for candidate in reversed(item):
            value = _coerce_stratum(candidate)
            if value is not None:
                return value
        return None
    return _coerce_stratum(item)


def _fast_dataset_labels(dataset: Any) -> Optional[List[Any]]:
    """Read labels from well-known dataset attributes without per-item I/O.

    Args:
        dataset: The task dataset (possibly a project ``ImageDataset``,
            ``AudioDataset`` or ``_MapTextDataset``).

    Returns:
        One label per dataset item, or ``None`` when no fast path applies.
    """
    items = getattr(dataset, "items", None)
    if isinstance(items, list) and items:
        label_column = getattr(dataset, "label_column", None)
        if label_column:
            values: List[Any] = []
            for item in items:
                if not isinstance(item, dict) or label_column not in item:
                    return None
                value = _coerce_stratum(item[label_column])
                if value is None:
                    return None
                values.append(value)
            return values

    try:
        size = len(dataset)
    except TypeError:
        size = None

    for attribute in ("targets", "labels", "y"):
        values = getattr(dataset, attribute, None)
        if values is None or size is None:
            continue
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, (list, tuple)) or len(values) < size:
            continue
        coerced = [_coerce_stratum(value) for value in values[:size]]
        if all(value is not None for value in coerced):
            return coerced

    source = getattr(dataset, "_source", None)
    label_column = getattr(dataset, "label_column", None) or getattr(
        dataset, "_label_col", None
    )
    if source is not None and label_column and size is not None:
        try:
            values = source[label_column]
        except Exception:  # noqa: BLE001 - not a column-addressable source
            values = None
        if values is not None and len(values) >= size:
            coerced = [_coerce_stratum(value) for value in values[:size]]
            if all(value is not None for value in coerced):
                return coerced
    return None


def _dataset_labels(dataset: Any, task: Optional[str]) -> Optional[List[Any]]:
    """Return one class label per dataset item, or ``None`` when unavailable.

    Only classification tasks are attempted. Fast paths cover the project's
    image/audio/text datasets, torchvision-style ``targets``/``samples``
    attributes and HF ``labels`` columns; anything else is probed item by
    item, bounded by :data:`_MAX_LABEL_PROBE`.

    Args:
        dataset: The underlying (map-style) dataset.
        task: Training task identifier.

    Returns:
        A list of hashable labels aligned with ``dataset`` indices, or ``None``.
    """
    if str(task or "").strip().lower() not in _CLASSIFICATION_TASKS:
        return None
    try:
        from torch.utils.data import Subset
    except Exception:  # pragma: no cover - torch is a hard dependency
        Subset = ()  # type: ignore[assignment]

    if Subset and isinstance(dataset, Subset):
        parent = _dataset_labels(dataset.dataset, task)
        if parent is None:
            return None
        try:
            return [parent[int(index)] for index in dataset.indices]
        except (IndexError, TypeError, ValueError):
            return None

    fast = _fast_dataset_labels(dataset)
    if fast is not None:
        return fast

    try:
        size = len(dataset)
    except TypeError:
        return None
    if size <= 0:
        return None
    if size > _MAX_LABEL_PROBE:
        logging.warning(
            "training.test_stratify ignored: dataset has %d items and exposes "
            "no label fast path (probe cap is %d)",
            size,
            _MAX_LABEL_PROBE,
        )
        return None

    label_key = getattr(dataset, "label_column", None) or getattr(
        dataset, "_label_col", None
    )
    labels: List[Any] = []
    for index in range(size):
        try:
            item = dataset[index]
        except Exception as exc:  # noqa: BLE001 - dataset may be I/O bound
            logging.warning(
                "training.test_stratify ignored: could not read item %d (%s)",
                index,
                exc,
            )
            return None
        value = _label_from_item(item, label_key)
        if value is None:
            return None
        labels.append(value)
    return labels


def _class_order(label: Any) -> Tuple[int, Any]:
    """Deterministic sort key that also works with mixed label types."""
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return (0, float(label))
    return (1, str(label))


def _stratified_split_indices(
    labels: Sequence[Any],
    num_test: int,
    total: int,
    seed: int,
) -> Optional[Tuple[List[int], List[int], Dict[Any, int]]]:
    """Allocate a deterministic stratified train/test partition.

    The allocation is proportional to each class frequency (largest-remainder
    rounding) and caps every class at ``count - 1`` so at least one example
    stays in training. When the requested test size allows it, every
    splittable class (``count >= 2``) receives at least one test example, so a
    rare class is never absent from the held-out evaluation.

    Args:
        labels: One hashable label per dataset index.
        num_test: Requested number of held-out examples.
        total: Dataset length (``len(labels)``).
        seed: Seed for the deterministic within-class permutation.

    Returns:
        ``(test_indices, train_indices, allocation)`` or ``None`` when
        stratification is impossible (fewer than two classes, no splittable
        class, or no test example).
    """
    if total <= 1 or num_test < 1 or num_test > total - 1:
        return None

    groups: Dict[Any, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    classes = sorted(groups, key=_class_order)
    if len(classes) < 2:
        return None
    counts = {label: len(groups[label]) for label in classes}
    splittable = [label for label in classes if counts[label] >= 2]
    if not splittable:
        return None

    import torch

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(total, generator=generator).tolist()
    rank = [0] * total
    for position, index in enumerate(permutation):
        rank[index] = position
    for label in classes:
        groups[label].sort(key=lambda index: rank[index])

    fraction = float(num_test) / float(total)
    caps = {label: max(0, counts[label] - 1) for label in classes}
    ideal = {label: counts[label] * fraction for label in classes}
    allocation = {
        label: min(int(math.floor(ideal[label])), caps[label])
        for label in classes
    }
    remaining = num_test - sum(allocation.values())

    def remainder(label: Any) -> float:
        return ideal[label] - math.floor(ideal[label])

    order = sorted(
        classes,
        key=lambda label: (-remainder(label), -counts[label], _class_order(label)),
    )
    while remaining > 0:
        progressed = False
        for label in order:
            if remaining <= 0:
                break
            if allocation[label] < caps[label]:
                allocation[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    # Guarantee representation for every splittable class when the budget
    # allows it, borrowing from the largest allocation.
    if num_test >= len(splittable):
        for label in splittable:
            if allocation[label] > 0:
                continue
            donors = [
                candidate
                for candidate in splittable
                if allocation[candidate] >= 2
            ]
            if not donors:
                continue
            donor = max(
                donors,
                key=lambda candidate: (
                    allocation[candidate],
                    counts[candidate],
                    _class_order(candidate),
                ),
            )
            allocation[donor] -= 1
            allocation[label] = 1

    if sum(allocation.values()) == 0:
        return None

    test_indices: List[int] = []
    train_indices: List[int] = []
    for label in classes:
        held_out = int(allocation[label])
        members = groups[label]
        test_indices.extend(members[:held_out])
        train_indices.extend(members[held_out:])

    return sorted(test_indices), sorted(train_indices), allocation


def _class_count_payload(
    labels: Optional[Sequence[Any]],
    indices: Sequence[int],
) -> Dict[str, int]:
    """Count labels over ``indices`` and stringify the keys for JSON."""
    if labels is None:
        return {}
    counts = Counter(labels[index] for index in indices)
    return {
        str(label): int(counts[label])
        for label in sorted(counts, key=_class_order)
    }


def maybe_split_dataloader(
    dataloader: Any,
    training_runtime: Dict[str, Any],
    *,
    seed: int = DEFAULT_TEST_SEED,
    task: Optional[str] = None,
) -> TestSplit:
    """Split ``dataloader`` according to ``training.test_percentage``.

    The original DataLoader settings (batch size, collate function, worker
    count, pin-memory and ``drop_last``) are preserved for the training
    partition; the held-out loader never drops its last partial batch and is
    not shuffled.

    Classification tasks are stratified when ``training.test_stratify`` is
    true and class labels can be extracted from the dataset; every other task
    (or a dataset without labels) falls back to the seeded random split.

    Args:
        dataloader: The task DataLoader built by the caller.
        training_runtime: The validated ``training`` block from the YAML.
        seed: Seed for the deterministic permutation.
        task: Training task identifier (``mlm``, ``classification``, ...).
            Required for stratification; ``None`` disables it.

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

    requested_test = int(round(total * percentage / 100.0))
    requested_test = max(1, min(requested_test, total - 1))

    labels = _dataset_labels(dataset, task)
    if labels is not None and len(labels) != total:
        logging.warning(
            "training.test_stratify ignored: extracted %d labels for %d items",
            len(labels),
            total,
        )
        labels = None

    stratified = False
    allocation: Optional[Dict[Any, int]] = None
    if labels is not None and resolve_test_stratify(training_runtime):
        result = _stratified_split_indices(labels, requested_test, total, seed)
        if result is not None:
            test_indices, train_indices, allocation = result
            stratified = True
        else:
            logging.warning(
                "training.test_stratify=%s could not stratify this dataset "
                "(classes=%d, requested_test=%d); using a random split",
                training_runtime.get("test_stratify", True),
                len(set(labels)),
                requested_test,
            )

    if not stratified:
        import torch

        generator = torch.Generator().manual_seed(int(seed))
        permutation = torch.randperm(total, generator=generator).tolist()
        test_indices = sorted(permutation[:requested_test])
        train_indices = sorted(permutation[requested_test:])

    num_test = len(test_indices)
    num_train = total - num_test
    if num_test != requested_test:
        logging.info(
            "Stratified split adjusted the test size from %d to %d to keep at "
            "least one training example per class",
            requested_test,
            num_test,
        )

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
            generator=make_generator(seed) if shuffle else None,
            worker_init_fn=seed_worker,
        )

    class_counts_train = _class_count_payload(labels, train_indices)
    class_counts_test = _class_count_payload(labels, test_indices)
    num_classes = len(set(labels)) if labels is not None else 0

    if labels is not None:
        all_counts = Counter(labels)
        absent = [
            str(label)
            for label, count in sorted(
                all_counts.items(), key=lambda item: _class_order(item[0])
            )
            if count >= 2 and class_counts_test.get(str(label), 0) == 0
        ]
        if absent:
            logging.warning(
                "Held-out split has no test examples for class(es) %s; their "
                "F1/recall are undefined on this split",
                absent,
            )

    logging.info(
        "Held-out test split: %d train / %d test examples (%.2f%% held out, "
        "stratified=%s)",
        num_train,
        num_test,
        percentage,
        stratified,
    )
    if stratified and allocation is not None:
        logging.info(
            "Stratified per-class test counts: %s",
            {str(label): int(allocation[label]) for label in sorted(allocation, key=_class_order)},
        )
    return TestSplit(
        train_loader=_rebuild(train_subset, shuffle=True, drop_last=bool(dataloader.drop_last)),
        test_loader=_rebuild(test_subset, shuffle=False, drop_last=False),
        num_train=num_train,
        num_test=num_test,
        test_percentage=percentage,
        stratified=stratified,
        num_classes=num_classes,
        class_counts_train=class_counts_train,
        class_counts_test=class_counts_test,
        train_indices=tuple(train_indices),
        test_indices=tuple(test_indices),
    )


#: Trainer metric keys copied as ``test_<key>`` when present.
_METRIC_KEYS = (
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_iou",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "perplexity",
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
    task_key = str(task).strip().lower()
    metric_name = "cer" if task_key in _ERROR_RATE_TASKS else "accuracy"
    if task_key in _CLASSIFICATION_TASKS:
        primary_metric = "macro_f1"
    elif task_key in _ERROR_RATE_TASKS:
        primary_metric = "cer"
    else:
        primary_metric = "accuracy"
    payload: Dict[str, Any] = {
        "task": str(task),
        "metric": metric_name,
        "primary_metric": primary_metric,
        "test_percentage": float(split.test_percentage),
        "num_train_samples": int(split.num_train),
        "num_test_samples": int(split.num_test),
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "num_batches": int(metrics["num_batches"]),
        "stratified": bool(split.stratified),
        "num_classes": int(split.num_classes),
        "class_counts_train": split.class_counts_train,
        "class_counts_test": split.class_counts_test,
    }
    for key in _METRIC_KEYS:
        if metrics.get(key) is not None:
            payload[f"test_{key}"] = float(metrics[key])
    for key in ("per_class", "confusion_matrix"):
        if key in metrics:
            payload[key] = metrics[key]
    if metrics.get("num_samples") is not None:
        payload["num_samples"] = int(metrics["num_samples"])
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
    primary_metric = str(payload.get("primary_metric") or "accuracy")
    primary_key = f"test_{primary_metric}"
    if primary_key not in payload:
        primary_key = "test_accuracy"
        primary_metric = "accuracy"
    logging.info(
        "Held-out test metrics: loss=%.4f accuracy=%.4f %s=%.4f "
        "(%d examples, stratified=%s) -> %s",
        payload["test_loss"],
        payload["test_accuracy"],
        primary_metric,
        payload[primary_key],
        payload["num_test_samples"],
        payload.get("stratified", False),
        path,
    )
    return path


def write_test_split_manifest(
    split: TestSplit,
    task: str,
    out_dir: str | Path = ".",
) -> Optional[Path]:
    """Persist the exact held-out indices as ``test_split.json``.

    The deployment/inference evaluation can then score the same examples that
    produced ``test_metrics.json`` instead of the training rows.

    Args:
        split: The split returned by :func:`maybe_split_dataloader`.
        task: Training task identifier.
        out_dir: Directory that receives the file (the run's working dir).

    Returns:
        The path of the written file, or ``None`` when the split is disabled.
    """
    if split.test_loader is None:
        return None
    payload = {
        "task": str(task),
        "test_percentage": float(split.test_percentage),
        "stratified": bool(split.stratified),
        "num_train": int(split.num_train),
        "num_test": int(split.num_test),
        "num_classes": int(split.num_classes),
        "class_counts_train": split.class_counts_train,
        "class_counts_test": split.class_counts_test,
        "train_indices": list(split.train_indices),
        "test_indices": list(split.test_indices),
    }
    path = Path(out_dir) / TEST_SPLIT_FILENAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def evaluate_and_write(
    trainer: Any,
    split: TestSplit,
    task: str,
    out_dir: str | Path = ".",
) -> Optional[Dict[str, Any]]:
    """Evaluate on the held-out split and persist the metrics/manifest.

    Args:
        trainer: The trainer whose model is evaluated.
        split: The :class:`TestSplit` produced by :func:`maybe_split_dataloader`.
        task: Training task identifier.
        out_dir: Directory that receives ``test_metrics.json`` and
            ``test_split.json``.

    Returns:
        The written payload, or ``None`` when the split is disabled.
    """
    payload = build_test_metrics(trainer, split, task)
    if payload is not None:
        write_test_metrics(payload, out_dir)
        write_test_split_manifest(split, task, out_dir)
    return payload
