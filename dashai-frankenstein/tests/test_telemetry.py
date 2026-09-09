"""Tests for the DashAI telemetry adapter (DashAITelemetryCallback).

Two layers:
- Unit tests with a mocked DashAI model instance (no DB) asserting the
  ``_save_metrics`` / ``calculate_metrics`` contract.
- Integration test against a real in-memory DashAI SQLite database
  (``setup_sqlite_db`` + ``Metric`` model), exercising the native
  ``BaseModel._save_metrics`` path.

Run with the DashAI project's uv env (DashAI importable):
    cd /home/erickfmm/programas/dashAI
    uv run --extra cpu pytest \
        /home/erickfmm/programas/transformer-encoder-frankestein/dashai-frankenstein/tests/test_telemetry.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the plugin sources are importable when running from the DashAI repo.
_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "src"
_FRANKENSTEIN_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PLUGIN_SRC), str(_FRANKENSTEIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashai_frankenstein.adapters.telemetry import (  # noqa: E402
    EPOCH_FIELDS,
    STEP_FIELDS,
    DashAITelemetryCallback,
    _coerce,
)

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests (mocked model, no DB)
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal DashAI model stand-in recording _save_metrics calls."""

    def __init__(self, run_id: int = 1):
        self.run_id = run_id
        self.saved = []

    def _save_metrics(self, split, level, results, log_index=None):
        self.saved.append(
            {"split": split, "level": level, "results": dict(results), "log_index": log_index}
        )

    def calculate_metrics(self, split=SplitEnum.VALIDATION, level=LevelEnum.LAST,
                          x_data=None, y_data=None, log_index=None):
        self.scored.append({"split": split, "level": level, "log_index": log_index})


def _sample_step(global_step: int = 10) -> dict:
    """A TitanTrainer step payload mirroring trainer.py:708-745."""
    return {
        "level": "step",
        "timestamp": "2026-09-09T00:00:00",
        "epoch": 0,
        "step": 9,
        "global_step": global_step,
        "loss": 0.5,
        "accuracy": 0.75,
        "learning_rate": 1e-4,
        "grad_norm": 0.9,
        "scaler_scale": 65536.0,
        "gpu_memory_gb": 2.5,
        "gpu_cached_gb": 3.0,
        "has_nan": False,
        "has_inf": False,
        "has_zero": False,
        "repair_action": "none",
        "gpu_temp_c": 71.5,
        "gpu_power_w": 250.0,
        "gpu_util_pct": 95.0,
        "gpu_mem_used_mib": 2560.0,
        "grad_norm_embeddings": 0.1,
        "grad_norm_attention": 0.2,
        "grad_norm_ffn": 0.3,
        "grad_norm_experts": 0.0,
        "grad_norm_router": 0.0,
        "grad_norm_ode": 0.0,
        "grad_norm_retnet": 0.0,
        "grad_norm_mamba": 0.0,
        "grad_norm_norms": 0.05,
        "grad_norm_head": 0.06,
        "grad_norm_other": 0.07,
        "step_time_ms": 120.5,
        "tokens_per_sec": 4096.0,
        "clip_ratio": 0.9,
        "effective_batch_size": 32,
    }


def _sample_epoch() -> dict:
    """A TitanTrainer epoch payload mirroring trainer.py:1813-1821."""
    return {
        "level": "epoch",
        "epoch": 0,
        "loss": 0.45,
        "accuracy": 0.8,
        "best_loss": 0.45,
        "global_step": 100,
    }


class TestCoerce:
    def test_booleans(self):
        assert _coerce(True) == 1.0
        assert _coerce(False) == 0.0

    def test_numbers(self):
        assert _coerce(0.5) == 0.5
        assert _coerce(3) == 3.0

    def test_non_finite_dropped(self):
        assert _coerce(float("nan")) is None
        assert _coerce(float("inf")) is None

    def test_strings_dropped(self):
        assert _coerce("none") is None
        assert _coerce("") is None


class TestStepTelemetry:
    def test_all_numeric_fields_written_as_step_rows(self):
        model = FakeModel()
        cb = DashAITelemetryCallback(model)
        cb(_sample_step(global_step=10))

        assert len(model.saved) == 1
        call = model.saved[0]
        assert call["split"] == SplitEnum.TRAIN
        assert call["level"] == LevelEnum.STEP
        assert call["log_index"] == 10
        # All 29 scalar STEP fields present; strings/booleans coerced.
        expected = {f: _coerce(_sample_step()[f]) for f in STEP_FIELDS}
        assert call["results"] == expected

    def test_non_numeric_fields_not_persisted(self):
        model = FakeModel()
        cb = DashAITelemetryCallback(model)
        cb(_sample_step())
        names = set(model.saved[0]["results"])
        assert "timestamp" not in names
        assert "repair_action" not in names
        assert "epoch" not in names
        assert "step" not in names
        assert "level" not in names

    def test_throttle_every_n_steps(self):
        model = FakeModel()
        cb = DashAITelemetryCallback(model, log_every_n_steps=50)
        for gs in range(1, 101):
            cb(_sample_step(global_step=gs))
        # Only steps 50 and 100 written.
        assert [c["log_index"] for c in model.saved] == [50, 100]

    def test_every_step_by_default(self):
        model = FakeModel()
        cb = DashAITelemetryCallback(model)
        for gs in range(1, 6):
            cb(_sample_step(global_step=gs))
        assert len(model.saved) == 5


class TestEpochTelemetry:
    def test_epoch_fields_and_scored_metrics(self):
        model = FakeModel()
        model.scored = []
        x, y = object(), object()
        xv, yv = object(), object()
        cb = DashAITelemetryCallback(model, x, y, xv, yv)
        cb(_sample_epoch())

        # 1 EPOCH-level save + 2 scored splits.
        assert len(model.saved) == 1
        call = model.saved[0]
        assert call["split"] == SplitEnum.TRAIN
        assert call["level"] == LevelEnum.EPOCH
        assert call["log_index"] == 1  # 1-based (0-based epoch 0)
        expected = {f: _coerce(_sample_epoch()[f]) for f in EPOCH_FIELDS}
        assert call["results"] == expected

        assert [s["split"] for s in model.scored] == [
            SplitEnum.TRAIN, SplitEnum.VALIDATION,
        ]
        assert all(s["level"] == LevelEnum.EPOCH for s in model.scored)

    def test_epoch_skips_scored_when_no_data(self):
        model = FakeModel()
        model.scored = []
        cb = DashAITelemetryCallback(model)
        cb(_sample_epoch())
        assert model.scored == []

    def test_epoch_not_written_twice(self):
        model = FakeModel()
        model.scored = []
        cb = DashAITelemetryCallback(model)
        cb(_sample_epoch())
        cb(_sample_epoch())
        assert len(model.saved) == 1

    def test_scored_failure_does_not_break_epoch(self):
        class FailingScoreModel(FakeModel):
            def calculate_metrics(self, **kwargs):
                raise RuntimeError("predict exploded")

        model = FailingScoreModel()
        cb = DashAITelemetryCallback(model, object(), object())
        cb(_sample_epoch())  # must not raise
        assert len(model.saved) == 1


class TestRobustness:
    def test_db_failure_does_not_raise(self):
        class BrokenModel(FakeModel):
            def _save_metrics(self, **kwargs):
                raise RuntimeError("db gone")

        cb = DashAITelemetryCallback(BrokenModel())
        cb(_sample_step())  # must not raise
        cb(_sample_epoch())  # must not raise

    def test_empty_payload_is_noop(self):
        model = FakeModel()
        cb = DashAITelemetryCallback(model)
        cb({"level": "step"})
        cb({"level": "epoch", "epoch": 0})
        assert model.saved == []


# ---------------------------------------------------------------------------
# Integration test (real DashAI SQLite DB, native _save_metrics path)
# ---------------------------------------------------------------------------

@pytest.fixture()
def dashai_db(tmp_path):
    """Real DashAI SQLite engine + session_factory + a Run row."""
    from DashAI.back.dependencies.database.models import (
        Base, Dataset, ModelSession, Run,
    )
    from DashAI.back.dependencies.database.sqlite_database import setup_sqlite_db
    from kink import di

    config = {"SQLITE_DB_PATH": str(tmp_path / "dashai_test.db"), "LOGGING_LEVEL": "ERROR"}
    engine, session_factory = setup_sqlite_db(config)
    Base.metadata.create_all(engine)
    # kink's Container is not a Mapping; save/restore manually instead of
    # monkeypatch.setitem.
    def _container_get(name: str):
        try:
            return di[name]
        except KeyError:
            return None

    prev_session_factory = _container_get("session_factory")
    prev_engine = _container_get("engine")
    di["session_factory"] = session_factory
    di["engine"] = engine

    with session_factory() as db:
        ds = Dataset(name="t", file_path=str(tmp_path / "t.csv"))
        db.add(ds)
        db.flush()
        ms = ModelSession(
            dataset_id=ds.id,
            name="telemetry-test-session",
            task_name="TextClassificationTask",
            input_columns=["text"],
            output_columns=["label"],
            splits='{"train": [0, 1], "test": [2]}',
        )
        db.add(ms)
        db.flush()
        run = Run(
            model_session_id=ms.id,
            model_name="FrankensteinMLMModel",
            parameters={},
            optimizer_name="adamw",
            optimizer_parameters={},
            goal_metric="Accuracy",
            name="telemetry-test-run",
        )
        db.add(run)
        db.commit()
        run_id = run.id
    yield session_factory, run_id
    # Restore previous DI values (may be unset on a fresh container).
    if prev_session_factory is not None:
        di["session_factory"] = prev_session_factory
    if prev_engine is not None:
        di["engine"] = prev_engine
    engine.dispose()


class TestNativeDashAIPersistence:
    def test_step_and_epoch_rows_in_metric_table(self, dashai_db):
        from sqlalchemy import func, select

        from DashAI.back.dependencies.database.models import Metric
        from DashAI.back.models.base_model import BaseModel

        session_factory, run_id = dashai_db

        # A real DashAI BaseModel subclass so the native ``_save_metrics``
        # (DB-writing, @final) path is exercised — not a mock.
        class _NativeModel(BaseModel):
            DISPLAY_NAME = "TelemetryTestModel"

            def train(self, x_train, y_train, x_validation=None, y_validation=None):
                return self

            def save(self, filename):
                pass

            @classmethod
            def load(cls, filename):
                return cls()

            def predict(self, x_pred):
                import numpy as np

                n = len(x_pred) if hasattr(x_pred, "__len__") else 1
                return np.zeros((n, 2))

        model = _NativeModel()
        model.run_id = run_id
        cb = DashAITelemetryCallback(model)
        cb(_sample_step(global_step=7))
        cb(_sample_epoch())

        with session_factory() as db:
            total = db.scalar(
                select(func.count()).select_from(Metric).where(Metric.run_id == run_id)
            )
            assert total == len(STEP_FIELDS) + len(EPOCH_FIELDS)

            # Spot-check one STEP row.
            loss_step = db.execute(
                select(Metric).where(
                    Metric.run_id == run_id,
                    Metric.level == LevelEnum.STEP,
                    Metric.name == "loss",
                )
            ).scalar_one()
            assert loss_step.value == pytest.approx(0.5)
            assert loss_step.step == 7
            assert loss_step.split == SplitEnum.TRAIN

            # Spot-check one EPOCH row.
            best = db.execute(
                select(Metric).where(
                    Metric.run_id == run_id,
                    Metric.level == LevelEnum.EPOCH,
                    Metric.name == "best_loss",
                )
            ).scalar_one()
            assert best.value == pytest.approx(0.45)
            assert best.step == 1  # 1-based (0-based epoch 0)