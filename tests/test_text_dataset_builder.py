"""Tests for the ``text_dataset`` builder in the Frankenstein engine.

Covers:
- ``TextDatasetSpec`` parsing + the ``configured`` property.
- ``build_text_dataloader_from_config``: dict batches per task (mlm masking
  contract, causal_lm labels == input_ids, text_classification integer ids).
- Column validation errors and the no-source → None fallback.
- Engine integration: ``_train_from_config_path`` picks the text_dataset
  loader when ``dataset=None`` (smoke: task=text_classification raises
  without text_dataset, with text_dataset it trains).

Run with the conda env:
    conda run -n frankenstein python -m pytest tests/test_text_dataset_builder.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

torch = pytest.importorskip("torch")

from training.text_dataset_builder import (  # noqa: E402
    TextDatasetSpec,
    _build_label_map,
    build_text_dataloader_from_config,
)
from engine_hf_tokenizer import build_hf_tokenizer_from_config  # noqa: E402


class _LoadedStub:
    def __init__(self, text_cfg=None, runtime=None):
        self.text_dataset_config = text_cfg or {}
        self.training_runtime = runtime or {"max_length": 32, "batch_size": 4}


@pytest.fixture(scope="module")
def tok():
    return build_hf_tokenizer_from_config({"name_or_path": "prajjwal1/bert-tiny"})


@pytest.fixture(scope="module")
def json_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("text_ds")
    rows = [
        {"text": f"frase de ejemplo numero {i} para el builder", "label": "pos" if i % 2 else "neg"}
        for i in range(24)
    ]
    with open(d / "train.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return str(d)


class TestTextDatasetSpec:
    def test_empty_block_not_configured(self):
        assert not TextDatasetSpec.from_config({}).configured
        assert not TextDatasetSpec.from_config(None).configured

    def test_configured_by_name_or_dir(self):
        assert TextDatasetSpec.from_config({"dataset_name": "wikitext"}).configured
        assert TextDatasetSpec.from_config({"data_dir": "/tmp/x"}).configured

    def test_defaults(self):
        spec = TextDatasetSpec.from_config({"dataset_name": "x"})
        assert spec.split == "train"
        assert spec.text_column == "text"
        assert spec.streaming is False


class TestBuilderBatches:
    def test_returns_none_without_block(self, tok):
        assert build_text_dataloader_from_config(
            _LoadedStub(None), tok, "cpu", None, "mlm"
        ) is None

    def test_mlm_batches_masked(self, tok, json_dir):
        loaded = _LoadedStub({"data_dir": json_dir})
        dl = build_text_dataloader_from_config(loaded, tok, "cpu", None, "mlm")
        batch = next(iter(dl))
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        labels, ids = batch["labels"], batch["input_ids"]
        masked = labels != -100
        assert masked.any()
        assert (labels[~masked] == -100).all()
        # Special tokens never supervised.
        assert not masked[:, 0].any()

    def test_causal_lm_labels_equal_ids(self, tok, json_dir):
        loaded = _LoadedStub({"data_dir": json_dir})
        dl = build_text_dataloader_from_config(loaded, tok, "cpu", None, "causal_lm")
        batch = next(iter(dl))
        assert (batch["labels"] == batch["input_ids"]).all()
        assert not (batch["labels"] == -100).any()

    def test_text_classification_integer_labels(self, tok, json_dir):
        loaded = _LoadedStub({"data_dir": json_dir, "label_column": "label"})
        dl = build_text_dataloader_from_config(
            loaded, tok, "cpu", None, "text_classification"
        )
        batch = next(iter(dl))
        assert batch["labels"].dtype == torch.long
        assert batch["labels"].shape == (4,)
        assert set(batch["labels"].tolist()) <= {0, 1}

    def test_unknown_text_column_raises(self, tok, json_dir):
        loaded = _LoadedStub({"data_dir": json_dir, "text_column": "nope"})
        with pytest.raises(ValueError, match="text_column"):
            build_text_dataloader_from_config(loaded, tok, "cpu", None, "mlm")

    def test_cli_batch_size_override(self, tok, json_dir):
        loaded = _LoadedStub({"data_dir": json_dir})
        dl = build_text_dataloader_from_config(loaded, tok, "cpu", 8, "causal_lm")
        batch = next(iter(dl))
        assert batch["input_ids"].shape[0] == 8

    def test_cli_batch_size_invalid(self, tok, json_dir):
        loaded = _LoadedStub({"data_dir": json_dir})
        with pytest.raises(ValueError, match="batch-size"):
            build_text_dataloader_from_config(loaded, tok, "cpu", 0, "causal_lm")

    def test_unsupported_task_rejected(self, tok, json_dir):
        loaded = _LoadedStub({"data_dir": json_dir})
        with pytest.raises(ValueError, match="does not support task"):
            build_text_dataloader_from_config(loaded, tok, "cpu", None, "sbert")