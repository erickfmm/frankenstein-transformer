"""Tests for the ``use_dashai_dataset`` toggle + causal-LM integration.

Covers:
- ``causal_lm_dataloader_dict``: dict batches with ``labels == input_ids``
  (unshifted; the trainer shifts internally).
- ``CausalLMPretrainingTask`` contract (metadata, schema, num_labels,
  prepare_for_task validation).
- ``use_dashai_dataset`` flag wiring: when true (default) each trainable
  model passes a prebuilt loader via ``dataset=``; when false it passes
  ``dataset=None`` so the engine resolves the corpus from the JSON.
- End-to-end: tiny causal-LM pretraining (decoder) through
  ``engine.train_from_config`` streams STEP telemetry into DashAI's DB.

Run with the DashAI project's uv env:
    cd /home/erickfmm/programas/dashAI
    uv run --extra cpu pytest \
        /home/erickfmm/programas/transformer-encoder-frankestein/dashai-frankenstein/tests/test_dashai_dataset_toggle.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
_FRANKENSTEIN_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PLUGIN_SRC), str(_FRANKENSTEIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyarrow as pa  # noqa: E402
import torch  # noqa: E402

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum  # noqa: E402
from DashAI.back.models.base_model import BaseModel  # noqa: E402
from DashAI.back.types.value_types import Text  # noqa: E402

_TEXT_TYPE = Text(pa.string())

from dashai_frankenstein.adapters.dataset import (  # noqa: E402
    causal_lm_dataloader_dict,
    mlm_dataloader_dict,
)
from dashai_frankenstein.models.base import _inject_engine_overrides  # noqa: E402
from dashai_frankenstein.tasks.causal_lm_task import CausalLMPretrainingTask  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
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
        if isinstance(idx, str):
            return self._texts
        return {"text": self._texts[idx]}


class _FakeTokenizer:
    mask_token_id = 103
    vocab_size = 30522
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, texts, truncation=True, padding="max_length", max_length=32):
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


# ---------------------------------------------------------------------------
# causal_lm_dataloader_dict contract
# ---------------------------------------------------------------------------

class TestCausalLMDataloader:
    def test_labels_equal_input_ids(self):
        ds = _TextDataset(["hola mundo uno dos tres"] * 6)
        loader = causal_lm_dataloader_dict(ds, _FakeTokenizer(), "text",
                                           batch_size=6, max_length=16,
                                           shuffle=False)
        batch = next(iter(loader))
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        # Unshifted targets: labels == input_ids (trainer shifts internally).
        assert (batch["labels"] == batch["input_ids"]).all()
        assert not (batch["labels"] == -100).any()

    def test_shapes_and_mask(self):
        ds = _TextDataset(["a b c", "d e f"] * 3)
        loader = causal_lm_dataloader_dict(ds, _FakeTokenizer(), "text",
                                           batch_size=4, max_length=8,
                                           shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (4, 8)
        assert batch["attention_mask"].shape == (4, 8)
        # Padded positions are never attended.
        assert (batch["attention_mask"][batch["input_ids"] == 0] == 0).all()

    def test_empty_column_raises(self):
        ds = _TextDataset([])
        with pytest.raises(ValueError, match="no rows"):
            causal_lm_dataloader_dict(ds, _FakeTokenizer(), "text", batch_size=2)


# ---------------------------------------------------------------------------
# CausalLMPretrainingTask contract
# ---------------------------------------------------------------------------

class TestCausalLMTask:
    def test_metadata_types(self):
        task = CausalLMPretrainingTask()
        assert task.metadata["inputs_types"] == [Text]
        assert task.metadata["outputs_types"] == [Text]

    def test_schema_binds_causal_model(self):
        task = CausalLMPretrainingTask()
        assert task.schema["models"] == ["FrankensteinCausalLMModel"]
        assert task.schema["metrics"] == []

    def test_num_labels_none(self):
        task = CausalLMPretrainingTask()
        assert task.num_labels(None, "text") is None

    def test_rejects_non_text_column(self):
        from DashAI.back.types.categorical import Categorical

        task = CausalLMPretrainingTask()
        cat = Categorical(["a", "b"])

        class _CatDS:
            column_names = ["text", "label"]
            types = {"text": _TEXT_TYPE, "label": cat}

            def __getitem__(self, idx):
                return {"text": "t", "label": "a"}

        with pytest.raises(TypeError):
            task.prepare_for_task(_CatDS(), ["label"], ["label"])


# ---------------------------------------------------------------------------
# use_dashai_dataset toggle: models honor the flag in their engine call
# ---------------------------------------------------------------------------

class _CapturedEngine:
    """Monkeypatched ``train_from_config`` capturing the dataset kwarg."""

    def __init__(self):
        self.calls = []

    def __call__(self, cfg, *, dataset=None, device="cpu", supervisor="off",
                 metrics_callback=None, **kw):
        self.calls.append({"cfg": cfg, "dataset": dataset})
        return type("R", (), {"model": None})()


class _StubFrankModel:
    """Minimal nn.Module-like stand-in exposing ``.to``."""

    def __init__(self):
        self.emb = None

    def to(self, device):
        return self

    def eval(self):
        return self


class TestUseDashaiDatasetFlag:
    def _instance(self, cls, use_flag, use_tok_flag=True):
        return cls(
            use_dashai_dataset=use_flag,
            use_dashai_dataset_for_tokenizer=use_tok_flag,
            frankenstein_json="{}",
        )

    def test_pretrainer_flag_persisted(self):
        from dashai_frankenstein.models.pretrainer import FrankensteinPretrainer

        m = self._instance(FrankensteinPretrainer, False)
        assert m.use_dashai_dataset is False
        m2 = self._instance(FrankensteinPretrainer, True)
        assert m2.use_dashai_dataset is True

    def test_pretrainer_tokenizer_checkbox_persisted(self):
        from dashai_frankenstein.models.pretrainer import FrankensteinPretrainer

        m = self._instance(FrankensteinPretrainer, True, use_tok_flag=False)
        assert m.use_dashai_dataset_for_tokenizer is False
        m2 = self._instance(FrankensteinPretrainer, True, use_tok_flag=True)
        assert m2.use_dashai_dataset_for_tokenizer is True

    def test_pretrainer_passes_none_when_unchecked(self, monkeypatch):
        from dashai_frankenstein.models import pretrainer as pm
        from dashai_frankenstein.models.pretrainer import FrankensteinPretrainer

        captured = _CapturedEngine()
        monkeypatch.setattr(
            "dashai_frankenstein.engine.train_from_config", captured
        )
        m = self._instance(FrankensteinPretrainer, False)
        m._frank_model = None
        # Bypass heavy model building by stubbing the internals used by train.
        monkeypatch.setattr(pm, "resolve_json", lambda self: "{}")
        monkeypatch.setattr(pm, "validate_training_json", lambda txt: None)

        class _Loaded:
            training_runtime = {"batch_size": 2, "num_epochs": 1,
                                "max_length": 16, "mlm_probability": 0.15}
            tokenizer_config = {"name_or_path": "prajjwal1/bert-tiny"}

        class _Tok:
            def __len__(self):
                return 30522

            def __call__(self, texts, **k):
                n = int(k.get("max_length", 16))
                out = []
                for t in texts:
                    ids = [101, 2000, 102] + [0] * (n - 3)
                    out.append(ids[:n])
                return {"input_ids": out,
                        "attention_mask": [[1, 1, 1] + [0] * (n - 3)] * len(texts)}

        monkeypatch.setattr(pm, "build_model_from_json",
                            lambda *a, **k: (_StubFrankModel(), _Loaded(), None))
        monkeypatch.setattr(pm, "resolve_tokenizer", lambda loaded: _Tok())
        monkeypatch.setattr(pm, "resolve_device", lambda dev: "cpu")

        ds = _TextDataset(["x y z"] * 4)
        m.train(ds, ds)
        assert captured.calls and captured.calls[0]["dataset"] is None

    def test_pretrainer_passes_loader_when_checked(self, monkeypatch):
        from dashai_frankenstein.models import pretrainer as pm
        from dashai_frankenstein.models.pretrainer import FrankensteinPretrainer

        captured = _CapturedEngine()
        monkeypatch.setattr(
            "dashai_frankenstein.engine.train_from_config", captured
        )
        m = self._instance(FrankensteinPretrainer, True)
        monkeypatch.setattr(pm, "resolve_json", lambda self: "{}")
        monkeypatch.setattr(pm, "validate_training_json", lambda txt: None)

        class _Loaded:
            training_runtime = {"batch_size": 2, "num_epochs": 1,
                                "max_length": 16, "mlm_probability": 0.15}
            tokenizer_config = {"name_or_path": "prajjwal1/bert-tiny"}

        class _Tok:
            def __len__(self):
                return 30522

            def __call__(self, texts, **k):
                n = int(k.get("max_length", 16))
                out = []
                for t in texts:
                    ids = [101, 2000, 102] + [0] * (n - 3)
                    out.append(ids[:n])
                return {"input_ids": out,
                        "attention_mask": [[1, 1, 1] + [0] * (n - 3)] * len(texts)}

        monkeypatch.setattr(pm, "build_model_from_json",
                            lambda *a, **k: (_StubFrankModel(), _Loaded(), None))
        monkeypatch.setattr(pm, "resolve_tokenizer", lambda loaded: _Tok())
        monkeypatch.setattr(pm, "resolve_device", lambda dev: "cpu")

        ds = _TextDataset(["x y z"] * 4)
        m.train(ds, ds)
        assert captured.calls and captured.calls[0]["dataset"] is not None

    def test_causal_model_passes_none_when_unchecked(self, monkeypatch):
        from dashai_frankenstein.models import causal_lm_model as cmm

        captured = _CapturedEngine()
        monkeypatch.setattr(
            "dashai_frankenstein.engine.train_from_config", captured
        )
        m = self._instance(
            __import__("dashai_frankenstein.models.causal_lm_model",
                       fromlist=["FrankensteinCausalLMModel"]
                       ).FrankensteinCausalLMModel, False)
        monkeypatch.setattr(cmm, "resolve_json", lambda self: "{}")
        monkeypatch.setattr(cmm, "validate_training_json", lambda txt: None)

        class _Loaded:
            training_runtime = {"batch_size": 2, "num_epochs": 1,
                                "max_length": 16}
            tokenizer_config = {"name_or_path": "prajjwal1/bert-tiny"}

        class _Tok:
            def __len__(self):
                return 30522

        monkeypatch.setattr(cmm, "build_model_from_json",
                            lambda *a, **k: (_StubFrankModel(), _Loaded(), None))
        monkeypatch.setattr(cmm, "resolve_tokenizer", lambda loaded: _Tok())
        monkeypatch.setattr(cmm, "resolve_device", lambda dev: "cpu")

        ds = _TextDataset(["x y z"] * 4)
        m.train(ds, ds)
        assert captured.calls and captured.calls[0]["dataset"] is None

    def test_classifier_passes_none_when_unchecked(self, monkeypatch):
        from dashai_frankenstein.models import base as bmod

        captured = _CapturedEngine()
        monkeypatch.setattr(
            "dashai_frankenstein.engine.train_from_config", captured
        )
        # Drive classification_train directly with a stubbed component.
        comp = type("C", (), {
            "frankenstein_json": "{}",
            "use_dashai_dataset": False,
            "num_labels": None,
            "_frank_model": None,
            "_loaded_config": None,
            "_tokenizer": None,
            "_device": "cpu",
            "_batch_size": 2,
            "_label_column": None,
            "_text_column_fn": staticmethod(lambda ds: "text"),
        })()

        class _Loaded:
            training_runtime = {"batch_size": 2, "num_epochs": 1}
            tokenizer_config = {"name_or_path": "prajjwal1/bert-tiny"}

        class _Tok:
            def __len__(self):
                return 30522

            def __call__(self, texts, **k):
                n = 8
                return {"input_ids": [[101, 2000, 102] + [0] * (n - 3)] * len(texts),
                        "attention_mask": [[1, 1, 1] + [0] * (n - 3)] * len(texts)}

        monkeypatch.setattr(bmod, "resolve_json", lambda self: "{}")
        monkeypatch.setattr(bmod, "validate_training_json", lambda txt: None)
        monkeypatch.setattr(bmod, "build_model_from_json",
                            lambda *a, **k: (_StubFrankModel(), _Loaded(), None))
        monkeypatch.setattr(bmod, "resolve_tokenizer", lambda loaded: _Tok())
        monkeypatch.setattr(bmod, "resolve_device", lambda dev: "cpu")

        ds = _TextDataset(["x y z"] * 4)

        class _YDs:
            """y_train stand-in: one integer column (already encoded)."""

            def __init__(self):
                self._vals = [0, 1, 0, 1]
                self.types = {"label": _TEXT_TYPE}

            @property
            def column_names(self):
                return ["label"]

            def __len__(self):
                return len(self._vals)

            def __getitem__(self, idx):
                if isinstance(idx, str):
                    return self._vals
                return {"label": self._vals[idx]}

            def unique(self, col):
                return [0, 1]

        bmod.classification_train(
            comp, ds, _YDs(), None, None,
            num_labels=2, label_column="label",
            text_column_fn=lambda d: "text",
        )
        assert captured.calls and captured.calls[0]["dataset"] is None


@pytest.fixture()
def dashai_db(tmp_path):
    from DashAI.back.dependencies.database.models import (
        Base, Dataset, ModelSession, Run,
    )
    from DashAI.back.dependencies.database.sqlite_database import setup_sqlite_db
    from kink import di

    config = {"SQLITE_DB_PATH": str(tmp_path / "dashai_clm_e2e.db"),
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
            dataset_id=ds.id, name="clm-e2e-session",
            task_name="CausalLMPretrainingTask",
            input_columns=["text"], output_columns=["text"], splits="{}",
        )
        db.add(ms)
        db.flush()
        run = Run(
            model_session_id=ms.id, model_name="FrankensteinCausalLMModel",
            parameters={}, optimizer_name="adamw", optimizer_parameters={},
            goal_metric="Accuracy", name="clm-e2e-run",
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


# ---------------------------------------------------------------------------
# Engine-level: causal_lm accepts a caller dataset (DashAI path)
# ---------------------------------------------------------------------------

def test_engine_causal_lm_accepts_caller_dataset(dashai_db, tmp_path, monkeypatch):
    """task=causal_lm + caller loader: engine uses it (no RedPajama build)."""
    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)

    from sqlalchemy import func, select

    from DashAI.back.dependencies.database.models import Metric
    from src.engine import train_from_config

    session_factory, run_id = dashai_db

    texts = [
        "el gato duerme en la silla junto a la ventana abierta",
        "los pájaros cantan al amanecer sobre los árboles del parque",
        "el tren llega tarde a la estación central de la ciudad",
        "la niña lee un libro de aventuras junto al fuego",
    ] * 8
    dataset = _TextDataset(texts)

    from dashai_frankenstein.engine import (
        build_model_from_json, resolve_device, resolve_tokenizer,
    )
    from dashai_frankenstein.adapters.dataset import causal_lm_dataloader_dict

    cfg = {
        "model_class": "frankensteindecoder",
        "model": {"dims": {"hidden_size": 64, "num_layers": 2,
                           "num_heads": 4, "layer_pattern": ["standard_attn"]}},
        "tokenizer": {"name_or_path": "prajjwal1/bert-tiny"},
        "training": {"task": "causal_lm",
                     "optimizer": {"optimizer_class": "AdamW",
                                   "parameters": {"lr": 1e-3}}},
    }
    model, loaded, _ = build_model_from_json(cfg, model_class_override="frankensteindecoder")
    device = resolve_device("cpu")
    tokenizer = resolve_tokenizer(loaded)
    loader = causal_lm_dataloader_dict(
        dataset, tokenizer, "text", batch_size=4, max_length=16, device=device,
    )

    class _StubModel(BaseModel):
        DISPLAY_NAME = "CLME2EModel"

        def train(self, x, y, xv=None, yv=None):
            return self

        def save(self, filename):
            pass

        @classmethod
        def load(cls, filename):
            return cls()

        def predict(self, x_pred):
            import numpy as np

            return np.zeros((1, 2))

    stub = _StubModel()
    stub.run_id = run_id
    from dashai_frankenstein.adapters.telemetry import (
        STEP_FIELDS,
        DashAITelemetryCallback,
    )
    callback = DashAITelemetryCallback(stub, split=SplitEnum.TRAIN,
                                       log_every_n_steps=1)

    engine_cfg = _inject_engine_overrides(
        cfg, task="causal_lm", num_epochs=2, batch_size=4,
    )
    result = train_from_config(
        engine_cfg, dataset=loader, device=device, supervisor="off",
        metrics_callback=callback,
    )
    assert result is not None

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
            )
        ).scalars().all()
        assert losses and all(v > 0 for v in losses)


# ---------------------------------------------------------------------------
# tokenizer.source=train_from_dataset (plugin facade)
# ---------------------------------------------------------------------------

class TestTokenizerTrainFromDataset:
    def test_detector_off_for_empty_json(self):
        from dashai_frankenstein.models.pretrainer import (
            _tokenizer_train_from_dataset,
        )

        assert _tokenizer_train_from_dataset("{}") is False
        assert _tokenizer_train_from_dataset("not json") is False

    def test_detector_on_for_train_from_dataset(self):
        from dashai_frankenstein.models.pretrainer import (
            _tokenizer_train_from_dataset,
        )

        cfg = {"tokenizer": {"source": "train_from_dataset"}}
        assert _tokenizer_train_from_dataset(cfg) is True

    def test_schema_field_exists(self):
        from dashai_frankenstein.config import FrankensteinPassthroughSchema

        fields = getattr(FrankensteinPassthroughSchema, "model_fields", None)
        if fields is None:
            fields = FrankensteinPassthroughSchema.__fields__
        assert "use_dashai_dataset_for_tokenizer" in fields

    def test_build_model_and_trained_tokenizer(self, monkeypatch, tmp_path):
        pytest.importorskip("tokenizers")
        pytest.importorskip("transformers")
        monkeypatch.chdir(tmp_path)

        from dashai_frankenstein.engine import build_model_and_trained_tokenizer

        cfg = {
            "model_class": "frankenstein",
            "model": {"dims": {"hidden_size": 32, "num_layers": 1,
                               "num_heads": 4,
                               "layer_pattern": ["standard_attn"]}},
            "tokenizer": {
                "source": "train_from_dataset",
                "training": {
                    "algorithm": "bpe",
                    "vocab_size": 64,
                    "min_frequency": 1,
                    "special_tokens": ["[PAD]", "[UNK]", "[MASK]"],
                },
            },
            "training": {"task": "mlm",
                         "optimizer": {"optimizer_class": "adamw",
                                       "parameters": {"adamw-lr": 1e-3}}},
        }
        dataset = _TextDataset(["el zorro corre rapido", "el perro ladra"] * 20)
        model, loaded, tok = build_model_and_trained_tokenizer(cfg, raw_dataset=dataset)
        assert model is not None
        # The trained vocab was injected into the model config.
        assert loaded.model_config.vocab_size <= 64
        assert tok is None  # tokenizer resolved by the engine flow