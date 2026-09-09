"""Tests for MLM pretraining integration in DashAI (plugin).

Covers:
- The MLM dict dataloader: masking contract (-100 on unmasked, 15% target
  ratio, special tokens untouched).
- ``MaskedLanguageModelingTask`` contract (metadata, schema, num_labels,
  prepare_for_task validation).
- End-to-end: a tiny MLM pretraining through ``engine.train_from_config``
  streams STEP/EPOCH telemetry into a real DashAI SQLite DB.

Run with the DashAI project's uv env:
    cd /home/erickfmm/programas/dashAI
    uv run --extra cpu pytest \
        /home/erickfmm/programas/transformer-encoder-frankestein/dashai-frankenstein/tests/test_mlm_pretraining.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
_FRANKENSTEIN_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PLUGIN_SRC), str(_FRANKENSTEIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyarrow as pa
import torch

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum  # noqa: E402
from DashAI.back.models.base_model import BaseModel  # noqa: E402
from DashAI.back.types.value_types import Text  # noqa: E402

_TEXT_TYPE = Text(pa.string())

from dashai_frankenstein.adapters.dataset import mlm_dataloader_dict  # noqa: E402
from dashai_frankenstein.adapters.telemetry import (  # noqa: E402
    STEP_FIELDS,
    DashAITelemetryCallback,
)
from dashai_frankenstein.models.base import _inject_engine_overrides  # noqa: E402
from dashai_frankenstein.tasks.mlm_task import MaskedLanguageModelingTask  # noqa: E402


# ---------------------------------------------------------------------------
# Text dataset stub (DashAIDataset-shaped: column_names / types / __getitem__)
# ---------------------------------------------------------------------------

class _TextDataset:
    def __init__(self, texts):
        self._texts = list(texts)
        self.types = {"text": _TEXT_TYPE}

    @property
    def column_names(self):
        return ["text"]

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        if isinstance(idx, str):  # column access: dataset["text"]
            return self._texts
        return {"text": self._texts[idx]}


# ---------------------------------------------------------------------------
# MLM dataloader masking contract
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal HF-tokenizer stand-in for masking logic tests."""

    mask_token_id = 103
    vocab_size = 30522
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, texts, truncation=True, padding="max_length", max_length=32):
        # Deterministic pseudo-encoding: alternate ids, always pad to max_length.
        input_ids = []
        attention = []
        for t in texts:
            n = min(len(t.split()) + 2, max_length)
            ids = [101] + [2000 + (i % 500) for i in range(n - 2)] + [102]
            ids = (ids + [0] * max_length)[:max_length]
            input_ids.append(ids)
            attention.append([1 if v != 0 else 0 for v in ids])
        return {"input_ids": input_ids, "attention_mask": attention}

    def __len__(self):
        return self.vocab_size


class TestMLMDataloader:
    def test_labels_use_minus100_outside_mask(self):
        torch.manual_seed(42)
        ds = _TextDataset(["hola mundo uno dos tres cuatro cinco"] * 8)
        loader = mlm_dataloader_dict(ds, _FakeTokenizer(), "text", batch_size=8,
                                     max_length=32, shuffle=False)
        batch = next(iter(loader))
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        labels, ids = batch["labels"], batch["input_ids"]
        # Unmasked positions are -100; masked positions carry the ORIGINAL
        # id (HF convention) — the input_ids are the masked version.
        masked = labels != -100
        assert masked.any(), "no tokens were masked"
        assert not torch.equal(ids, labels)  # inputs differ from targets
        assert (labels[~masked] == -100).all()
        # Masked positions in input were replaced by [MASK] (~80%) or kept.
        assert (ids[masked] == _FakeTokenizer.mask_token_id).any()

    def test_special_and_padding_not_masked(self):
        torch.manual_seed(42)
        ds = _TextDataset(["hola mundo"] * 8)
        loader = mlm_dataloader_dict(ds, _FakeTokenizer(), "text", batch_size=8,
                                     max_length=16, shuffle=False)
        batch = next(iter(loader))
        labels = batch["labels"]
        assert labels[:, 0].ne(-100).sum() == 0  # CLS never supervised
        for tok in (101, 102, 0):
            assert not (labels == tok).any()

    def test_mask_ratio_approximately_respected(self):
        torch.manual_seed(42)
        ds = _TextDataset([f"texto numero {i} con palabras variadas" for i in range(64)])
        loader = mlm_dataloader_dict(ds, _FakeTokenizer(), "text", batch_size=64,
                                     max_length=32, mlm_probability=0.5, shuffle=False)
        batch = next(iter(loader))
        attended = batch["attention_mask"] == 1
        supervised = (batch["labels"] != -100).sum().float() / attended.sum().float()
        assert 0.3 < supervised.item() < 0.7  # ~50% of attended (real) tokens

    def test_resampling_each_epoch(self):
        torch.manual_seed(7)
        ds = _TextDataset([f"palabras distintas {i} para el sampler" for i in range(32)])
        loader = mlm_dataloader_dict(ds, _FakeTokenizer(), "text", batch_size=16,
                                     max_length=16, mlm_probability=0.5, shuffle=False)
        it = iter(loader)
        b1 = next(it)["labels"]
        b2 = next(it)["labels"]
        assert not (b1 == b2).all()  # stochastic masking re-sampled


# ---------------------------------------------------------------------------
# Task contract
# ---------------------------------------------------------------------------

class TestMLMTask:
    def test_metadata_types(self):
        t = MaskedLanguageModelingTask()
        md = t.get_metadata()
        assert md["inputs_types"] == ["Text"]
        assert md["outputs_types"] == ["Text"]
        assert md["inputs_cardinality"] == 1

    def test_schema_binds_pretrainer(self):
        t = MaskedLanguageModelingTask()
        assert t.schema == {"models": ["FrankensteinPretrainer"], "metrics": []}

    def test_num_labels_none(self):
        t = MaskedLanguageModelingTask()
        assert t.num_labels(_TextDataset(["x"]), "text") is None

    def test_rejects_non_text_column(self):
        from DashAI.back.types.categorical import Categorical

        cat = Categorical(["a", "b"])

        class _CatDS:
            column_names = ["text", "label"]
            types = {"text": _TEXT_TYPE, "label": cat}

            def __getitem__(self, idx):
                return {"text": "t", "label": "a"}

        with pytest.raises(TypeError):
            MaskedLanguageModelingTask().prepare_for_task(_CatDS(), ["label"], ["label"])


# ---------------------------------------------------------------------------
# E2E: real engine MLM pretraining -> DashAI metric DB
# ---------------------------------------------------------------------------

_MLM_CFG = {
    "model": {
        "dims": {
            "vocab_size": 30522,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "num_loops": 1,
            "layer_pattern": ["standard_attn"],
            "mode": "encoder",
            "dropout": 0.0,
        },
        "norm": {"type": "layer_norm"},
        "use_moe": False,
        "use_bitnet": False,
        "ffn_activation": "gelu",
        "ffn_hidden_size": 64,
    },
    "training": {
        "task": "mlm",
        "batch_size": 4,
        "num_epochs": 2,
        "max_length": 16,
        "optimizer": {
            "optimizer_class": "adamw",
            "parameters": {"adamw-lr_other": 0.001},
        },
        "csv_log_path": "e2e_mlm.csv",
        "checkpoint_every_n_steps": 100000,
        "gpu_temp_guard_enabled": False,
    },
    "tokenizer": {"name_or_path": "prajjwal1/bert-tiny"},
}


@pytest.fixture()
def dashai_db(tmp_path):
    from DashAI.back.dependencies.database.models import (
        Base, Dataset, ModelSession, Run,
    )
    from DashAI.back.dependencies.database.sqlite_database import setup_sqlite_db
    from kink import di

    config = {"SQLITE_DB_PATH": str(tmp_path / "dashai_mlm_e2e.db"),
              "LOGGING_LEVEL": "ERROR"}
    engine, session_factory = setup_sqlite_db(config)
    Base.metadata.create_all(engine)

    def _get(name):
        try:
            return di[name]
        except KeyError:
            return None

    prev_sf, prev_eng = _get("session_factory"), _get("engine")
    di["session_factory"] = session_factory
    di["engine"] = engine

    with session_factory() as db:
        ds = Dataset(name="t", file_path=str(tmp_path / "t.csv"))
        db.add(ds)
        db.flush()
        ms = ModelSession(
            dataset_id=ds.id, name="mlm-e2e-session",
            task_name="MaskedLanguageModelingTask",
            input_columns=["text"], output_columns=["text"], splits="{}",
        )
        db.add(ms)
        db.flush()
        run = Run(
            model_session_id=ms.id, model_name="FrankensteinPretrainer",
            parameters={}, optimizer_name="adamw", optimizer_parameters={},
            goal_metric="Accuracy", name="mlm-e2e-run",
        )
        db.add(run)
        db.commit()
        run_id = run.id
    yield session_factory, run_id
    if prev_sf is not None:
        di["session_factory"] = prev_sf
    if prev_eng is not None:
        di["engine"] = prev_eng
    engine.dispose()


class _NativeModel(BaseModel):
    DISPLAY_NAME = "MLME2EModel"

    def train(self, x_train, y_train, x_validation=None, y_validation=None):
        return self

    def save(self, filename):
        pass

    @classmethod
    def load(cls, filename):
        return cls()

    def predict(self, x_pred):
        import numpy as np

        return np.zeros((1, 2))


def test_engine_mlm_pretraining_streams_telemetry(dashai_db, tmp_path, monkeypatch):
    """Tiny real MLM pretraining -> STEP rows + LAST/EPOCH rows in DashAI DB."""
    pytest.importorskip("transformers")
    # Avoid hub revalidation hangs in restricted-network environments: the
    # tokenizer files are small and get cached on first use.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from sqlalchemy import func, select

    from DashAI.back.dependencies.database.models import Metric
    from src.engine import train_from_config

    session_factory, run_id = dashai_db

    native = _NativeModel()
    native.run_id = run_id
    callback = DashAITelemetryCallback(native, split=SplitEnum.TRAIN,
                                       log_every_n_steps=1)

    engine_cfg = _inject_engine_overrides(
        _MLM_CFG, task="mlm", num_epochs=2, batch_size=4,
    )
    monkeypatch.chdir(tmp_path)

    texts = [
        "el gato duerme en la silla junto a la ventana abierta",
        "los pájaros cantan al amanecer sobre los árboles del parque",
        "el tren llega tarde a la estación central de la ciudad",
        "la niña lee un libro de aventuras junto al fuego",
    ] * 16  # 64 rows so the loader has several batches
    dataset = _TextDataset(texts)

    import json
    from dashai_frankenstein.engine import (
        build_model_from_json, resolve_device, resolve_tokenizer,
    )
    from dashai_frankenstein.adapters.dataset import mlm_dataloader_dict

    model, loaded, _ = build_model_from_json(_MLM_CFG)
    device = resolve_device("cpu")
    tokenizer = resolve_tokenizer(loaded)
    loader = mlm_dataloader_dict(
        dataset, tokenizer, "text", batch_size=4, max_length=16,
        mlm_probability=0.15, device=device,
    )

    result = train_from_config(
        engine_cfg, dataset=loader, device=device, supervisor="off",
        metrics_callback=callback,
    )
    assert result is not None
    assert result.model is not None

    with session_factory() as db:
        step_rows = db.execute(
            select(func.count()).select_from(Metric).where(
                Metric.run_id == run_id, Metric.level == LevelEnum.STEP)
        ).scalar()
        assert step_rows > 0
        assert step_rows % len(STEP_FIELDS) == 0

        losses = db.execute(
            select(Metric.value).where(
                Metric.run_id == run_id, Metric.level == LevelEnum.STEP,
                Metric.name == "loss",
            ).order_by(Metric.step)
        ).scalars().all()
        assert len(losses) == step_rows // len(STEP_FIELDS)
        assert all(v > 0 for v in losses)

        # MLM accuracy recorded (masked-token accuracy).
        accs = db.execute(
            select(Metric.value).where(
                Metric.run_id == run_id, Metric.level == LevelEnum.STEP,
                Metric.name == "accuracy",
            )
        ).scalars().all()
        assert accs and all(0.0 <= a <= 1.0 for a in accs)