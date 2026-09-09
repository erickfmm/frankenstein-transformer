"""HuggingFace ``datasets`` loaders for text tasks driven by ``text_dataset``.

Builds dict-batch DataLoaders (``{"input_ids", "attention_mask", "labels"}``)
from a :class:`~src.training.config_loader.LoadedTrainingConfig`'s
``text_dataset`` block for the three NLP tasks:

- ``mlm``: BERT-style masking at collate time (80/10/10), ``-100`` outside
  the mask.
- ``causal_lm``: ``labels == input_ids`` (the trainer shifts internally).
- ``text_classification``: integer class ids in ``labels``.

Data sources: HF hub id (``load_dataset(name, split=...)``, optionally
streaming) or a local ``data_dir`` with ``*.parquet`` / ``*.json(l)`` files.
Padding is ``max_length``; special/pad positions are never supervised.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

PAD_LABEL = -100


@dataclass
class TextDatasetSpec:
    """Parsed ``text_dataset`` block."""

    dataset_name: str = ""
    split: str = "train"
    text_column: str = "text"
    label_column: str = "label"
    data_dir: str = ""
    streaming: bool = False
    use_labels: bool = False
    max_samples: int = 0

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "TextDatasetSpec":
        cfg = cfg or {}
        return cls(
            dataset_name=str(cfg.get("dataset_name", "") or ""),
            split=str(cfg.get("split", "train") or "train"),
            text_column=str(cfg.get("text_column", "text") or "text"),
            label_column=str(cfg.get("label_column", "label") or "label"),
            data_dir=str(cfg.get("data_dir", "") or ""),
            streaming=bool(cfg.get("streaming", False)),
            use_labels=bool(cfg.get("use_labels", False)),
            max_samples=int(cfg.get("max_samples", 0) or 0),
        )

    @property
    def configured(self) -> bool:
        """True when the block names an actual data source."""
        return bool(self.dataset_name or self.data_dir)


def _load_hf_source(cfg: TextDatasetSpec) -> Any:
    """Load the raw HF dataset (local dir first, then hub id)."""
    from datasets import load_dataset

    if cfg.data_dir:
        path = Path(cfg.data_dir)
        parquet_files = sorted(path.glob("*.parquet"))
        json_files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
        if parquet_files or json_files:
            fmt = "parquet" if parquet_files else "json"
            pattern = str(path / ("*.parquet" if parquet_files else "*.json*"))
            kwargs: Dict[str, Any] = {"data_files": pattern}
            if not cfg.streaming:
                kwargs["split"] = cfg.split
            return load_dataset(fmt, **kwargs)
        return load_dataset(str(path), split=cfg.split)

    if cfg.dataset_name:
        return load_dataset(
            cfg.dataset_name, split=cfg.split, streaming=cfg.streaming
        )
    raise ValueError(
        "text_dataset requires 'dataset_name' (HF id) or 'data_dir' "
        "(local parquet/json files)"
    )


def _source_columns(source: Any) -> List[str]:
    """Flatten ``column_names`` (DatasetDict → first split's columns)."""
    cols = getattr(source, "column_names", None)
    if isinstance(cols, dict):
        for v in cols.values():
            if v:
                return list(v)
        return []
    return list(cols or [])


def _resolve_columns(cfg: TextDatasetSpec, source: Any) -> tuple:
    """Validate the configured columns against the loaded source."""
    cols = _source_columns(source)
    if cols and cfg.text_column not in cols:
        raise ValueError(
            f"text_dataset.text_column '{cfg.text_column}' not in source "
            f"columns {cols}"
        )
    if cfg.use_labels and cols and cfg.label_column not in cols:
        raise ValueError(
            f"text_dataset.label_column '{cfg.label_column}' not in source "
            f"columns {cols}"
        )
    return cfg.text_column, cfg.label_column


def _build_label_map(source: Any, label_col: str) -> Dict[str, int]:
    """Build a label->id map from a ClassLabel feature or unique strings."""
    features = getattr(source, "features", None)
    if features is None:
        return {}
    if isinstance(features, dict):
        feat = features.get(label_col)
    else:
        feat = getattr(features, label_col, None)
    if feat is not None and getattr(feat, "names", None):
        return {str(n): i for i, n in enumerate(feat.names)}

    # Fallback: scan unique string values (small map-style sources only).
    if hasattr(source, "__len__") and not hasattr(source, "data"):
        pass
    try:
        if not hasattr(source, "__len__"):
            return {}
        values = set()
        for i in range(len(source)):
            v = source[i].get(label_col)
            if isinstance(v, str):
                values.add(v)
        if values:
            return {n: i for i, n in enumerate(sorted(values))}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("label scan failed: %s", exc)
    return {}


def _resolve_label(value: Any, label_map: Dict[str, int]) -> int:
    """Coerce a raw label value to an integer class id."""
    if isinstance(value, bool):
        raise ValueError(f"Unsupported label {value!r}")
    if isinstance(value, int):
        return value
    if value is None:
        raise ValueError("None label encountered in text_dataset")
    if isinstance(value, str):
        if label_map and value in label_map:
            return label_map[value]
        if value.isdigit():
            return int(value)
        raise ValueError(f"Unknown label {value!r} (not in dataset ClassLabel)")
    raise ValueError(f"Unsupported label type {type(value).__name__}: {value!r}")


def _make_collate(
    tokenizer: Any,
    max_length: int,
    task: str,
    use_labels: bool,
    label_map: Dict[str, int],
    mlm_probability: float,
    text_col: str,
    label_col: str,
):
    """Create the per-batch collate function for the task objective."""
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [str(b[text_col]) for b in batch]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        if task == "mlm":
            labels = input_ids.clone()
            specials = torch.isin(
                input_ids, torch.tensor(special_ids, dtype=torch.long)
            ) | (attention_mask == 0)
            prob = torch.full(input_ids.shape, mlm_probability)
            prob[specials] = 0.0
            masked = torch.bernoulli(prob).bool()
            if masked.any():
                labels[~masked] = PAD_LABEL
                rand = torch.rand(input_ids.shape)
                keep = masked & (rand < 0.1)          # 10% keep original
                rnd_tok = masked & (rand >= 0.1) & (rand < 0.2)  # 10% random
                mask_tok = masked & (rand >= 0.2)     # 80% [MASK]
                if rnd_tok.any():
                    ids_rand = torch.randint(
                        0, len(tokenizer), input_ids.shape
                    )
                    input_ids = torch.where(rnd_tok, ids_rand, input_ids)
                if mask_tok.any() and mask_token_id is not None:
                    input_ids = torch.where(
                        mask_tok,
                        torch.full_like(input_ids, mask_token_id),
                        input_ids,
                    )
                labels[keep] = PAD_LABEL
            else:
                labels[:] = PAD_LABEL
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        if task == "causal_lm":
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

        # text_classification (supervised): integer class ids.
        if not use_labels:
            raise ValueError(
                "task=text_classification requires text_dataset.use_labels=true "
                "and a label_column"
            )
        labels = torch.tensor(
            [_resolve_label(b[label_col], label_map) for b in batch],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


class _MapTextDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Wrap a map-style HF dataset, exposing only the configured columns."""

    def __init__(
        self, source: Any, text_col: str, label_col: str, max_samples: int
    ):
        self._source = source
        self._text_col = text_col
        self._label_col = label_col
        self._max_samples = max_samples

    def __len__(self) -> int:
        total = len(self._source)
        if self._max_samples:
            return min(total, self._max_samples)
        return total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._source[idx]
        return {self._text_col: row[self._text_col],
                self._label_col: row.get(self._label_col)}


class _IterableTextDataset(IterableDataset):  # type: ignore[type-arg]
    """Wrap a streaming HF dataset, exposing only the configured columns."""

    def __init__(
        self, source: Any, text_col: str, label_col: str, max_samples: int
    ):
        self._source = source
        self._text_col = text_col
        self._label_col = label_col
        self._max_samples = max_samples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        count = 0
        for row in self._source:
            if self._max_samples and count >= self._max_samples:
                break
            yield {self._text_col: row[self._text_col],
                   self._label_col: row.get(self._label_col)}
            count += 1


def build_text_dataloader_from_config(
    loaded: Any,
    tokenizer: Any,
    resolved_device: str,
    cli_batch_size: Optional[int],
    task: str,
) -> Optional[DataLoader]:
    """Build a dict-batch DataLoader from the ``text_dataset`` config block.

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` whose
            ``text_dataset_config`` names the source.
        tokenizer: Tokenizer (HF or SPM) for ``text_column``.
        resolved_device: Resolved torch device (only for pin_memory).
        cli_batch_size: Optional CLI ``--batch-size`` override.
        task: One of ``mlm`` / ``causal_lm`` / ``text_classification``.

    Returns:
        A ``DataLoader`` yielding ``{"input_ids", "attention_mask", "labels"}``
        dict batches, or ``None`` when no ``text_dataset`` block is configured
        (the caller falls back to the legacy path).

    Raises:
        ValueError: If the spec is inconsistent with the task (e.g.
            ``text_classification`` without ``use_labels``/``label_column``,
            unknown columns, or a non-positive CLI batch size).
    """
    cfg = TextDatasetSpec.from_config(getattr(loaded, "text_dataset_config", None))
    if not cfg.configured:
        return None

    task = str(task or "mlm").strip().lower()
    if task not in {"mlm", "causal_lm", "text_classification"}:
        raise ValueError(
            f"text_dataset does not support task '{task}' "
            "(expected mlm, causal_lm or text_classification)"
        )
    if task == "text_classification":
        cfg.use_labels = True

    source = _load_hf_source(cfg)
    text_col, label_col = _resolve_columns(cfg, source)

    runtime = getattr(loaded, "training_runtime", None) or {}
    max_length = int(runtime.get("max_length", 512))
    mlm_probability = float(runtime.get("mlm_probability", 0.15))
    batch_size = runtime.get("batch_size") or 1
    if cli_batch_size is not None:
        if cli_batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        batch_size = cli_batch_size

    label_map = (
        _build_label_map(source, label_col)
        if task == "text_classification"
        else {}
    )

    collate_fn = _make_collate(
        tokenizer, max_length, task, cfg.use_labels, label_map,
        mlm_probability, text_col, label_col,
    )

    is_streaming = cfg.streaming and not hasattr(source, "__len__")
    if is_streaming:
        dataset: Any = _IterableTextDataset(
            source, text_col, label_col, cfg.max_samples
        )
    else:
        dataset = _MapTextDataset(source, text_col, label_col, cfg.max_samples)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=not is_streaming,
        num_workers=0,
        pin_memory=resolved_device.startswith("cuda"),
        drop_last=not is_streaming and len(dataset) >= batch_size * 2,
    )
    logger.info(
        "text_dataset: %s (split=%s, text=%s, task=%s, streaming=%s)",
        cfg.dataset_name or cfg.data_dir, cfg.split, text_col, task, is_streaming,
    )
    return dataloader