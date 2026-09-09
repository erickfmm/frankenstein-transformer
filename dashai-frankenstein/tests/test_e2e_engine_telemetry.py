"""End-to-end test: engine training streams telemetry into DashAI.

Runs a tiny ViT classification training through
``src.engine.train_from_config`` (the exact path the DashAI plugin uses) with
a :class:`DashAITelemetryCallback`, and asserts STEP/EPOCH metric rows land
in a real DashAI SQLite database.
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

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum  # noqa: E402
from DashAI.back.models.base_model import BaseModel  # noqa: E402

from dashai_frankenstein.adapters.telemetry import (  # noqa: E402
    STEP_FIELDS,
    DashAITelemetryCallback,
)
from dashai_frankenstein.models.base import _inject_engine_overrides  # noqa: E402

# ---------------------------------------------------------------------------
# DashAI DB fixture (shared shape with test_telemetry.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def dashai_db(tmp_path):
    from DashAI.back.dependencies.database.models import (
        Base, Dataset, ModelSession, Run,
    )
    from DashAI.back.dependencies.database.sqlite_database import setup_sqlite_db
    from kink import di

    config = {"SQLITE_DB_PATH": str(tmp_path / "dashai_e2e.db"), "LOGGING_LEVEL": "ERROR"}
    engine, session_factory = setup_sqlite_db(config)
    Base.metadata.create_all(engine)

    def _container_get(name):
        try:
            return di[name]
        except KeyError:
            return None

    prev_sf, prev_eng = _container_get("session_factory"), _container_get("engine")
    di["session_factory"] = session_factory
    di["engine"] = engine

    with session_factory() as db:
        ds = Dataset(name="t", file_path=str(tmp_path / "t.csv"))
        db.add(ds)
        db.flush()
        ms = ModelSession(
            dataset_id=ds.id, name="e2e-session", task_name="ImageClassificationTask",
            input_columns=["img"], output_columns=["label"], splits="{}",
        )
        db.add(ms)
        db.flush()
        run = Run(
            model_session_id=ms.id, model_name="FrankensteinViTClassifier",
            parameters={}, optimizer_name="adamw", optimizer_parameters={},
            goal_metric="Accuracy", name="e2e-run",
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
# Real DashAI model capturing telemetry through the engine loop
# ---------------------------------------------------------------------------

_E2E_CFG = {
    "model_class": "frankenstein_vit",
    "model": {
        "dims": {
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
    "image": {
        "image_size": {"height": 32, "width": 32},
        "patch_size": 16,
        "in_channels": 3,
        "pos_embedding_type": "learned_1d",
        "cls_token": True,
        "pooling_mode": "cls",
        "num_classes": 3,
    },
    "training": {
        "task": "classification",
        "batch_size": 4,
        "num_epochs": 2,
        "optimizer": {
            "optimizer_class": "adamw",
            "parameters": {"adamw-lr_other": 0.001},
        },
        "csv_log_path": "e2e_metrics.csv",
        "checkpoint_every_n_steps": 100000,
        "gpu_temp_guard_enabled": False,
    },
}


def _dummy_loader(steps: int = 6, num_classes: int = 3, batch: int = 4):
    """Tiny torch DataLoader of dict batches for the vision engine path."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __len__(self):
            return steps * batch

        def __getitem__(self, idx):
            g = torch.Generator().manual_seed(idx)
            return (
                torch.rand(3, 32, 32, generator=g),
                int(torch.randint(0, num_classes, (1,), generator=g)),
            )

    def _collate(batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {"pixel_values": images, "labels": labels}

    return DataLoader(_DS(), batch_size=batch, shuffle=True, collate_fn=_collate)


class _NativeModel(BaseModel):
    DISPLAY_NAME = "E2EModel"

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
        return np.zeros((n, 3))


def test_engine_training_streams_telemetry_to_dashai(dashai_db, tmp_path, monkeypatch):
    """1 real engine training (2 epochs) -> STEP+LAST metric rows in DashAI DB."""
    from sqlalchemy import func, select

    from DashAI.back.dependencies.database.models import Metric
    from src.engine import train_from_config

    session_factory, run_id = dashai_db

    model = _NativeModel()
    model.run_id = run_id
    callback = DashAITelemetryCallback(model, split=SplitEnum.TRAIN, log_every_n_steps=1)

    engine_cfg = _inject_engine_overrides(
        _E2E_CFG, task="classification", num_epochs=2, batch_size=4
    )
    monkeypatch.chdir(tmp_path)  # checkpoints/CSV land in tmp

    result = train_from_config(
        engine_cfg,
        dataset=_dummy_loader(steps=6, num_classes=3, batch=4),
        device="cpu",
        supervisor="off",
        metrics_callback=callback,
    )
    assert result is not None
    assert result.model is not None

    with session_factory() as db:
        step_rows = db.execute(
            select(func.count()).select_from(Metric).where(
                Metric.run_id == run_id, Metric.level == LevelEnum.STEP
            )
        ).scalar()
        epoch_rows = db.execute(
            select(func.count()).select_from(Metric).where(
                Metric.run_id == run_id, Metric.level == LevelEnum.EPOCH
            )
        ).scalar()

        # Every optimizer update emits one row per STEP field: the total is a
        # multiple of len(STEP_FIELDS), and at least one update happened per
        # epoch (12 batches / accumulation cadence >= 2 updates).
        assert step_rows > 0
        assert step_rows % len(STEP_FIELDS) == 0
        distinct_steps = db.execute(
            select(func.count(Metric.step.distinct())).where(
                Metric.run_id == run_id, Metric.level == LevelEnum.STEP
            )
        ).scalar()
        assert step_rows == distinct_steps * len(STEP_FIELDS)
        assert distinct_steps >= 2

        # Epoch fields x 2 epochs.
        assert epoch_rows == 2 * 4

        # Loss values are positive and finite.
        losses = db.execute(
            select(Metric).where(
                Metric.run_id == run_id, Metric.level == LevelEnum.STEP,
                Metric.name == "loss",
            ).order_by(Metric.step)
        ).scalars().all()
        assert len(losses) == distinct_steps
        assert all(loss.value > 0 for loss in losses)

        # lr + gpu telemetry fields present.
        names = {
            row.name
            for row in db.execute(
                select(Metric.name).where(Metric.run_id == run_id, Metric.level == LevelEnum.STEP)
            ).all()
        }
        assert {"loss", "accuracy", "learning_rate", "grad_norm", "step_time_ms",
                "clip_ratio", "effective_batch_size"} <= names