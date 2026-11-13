#!/usr/bin/env python3
"""Checkpoint policy contract tests.

Locks in the four checkpoint knobs of ``training.*``:

* ``save_epoch_checkpoints`` (default ``true``) gates the per-epoch
  ``titan_checkpoint_epoch_<n>_epoch_end.pt`` saves;
* ``checkpoint_every_n_steps: 0`` disables rolling checkpoints (and must not
  raise ``ZeroDivisionError`` in the step loop);
* ``num_best_checkpoints: 0`` disables best checkpoints (and must not raise
  ``IndexError`` on the empty heap);
* every successful run writes exactly one stable ``titan_final.pt`` — the only
  artifact the e2e harness keeps (see ``full_tests/_helpers.py::run_training``).
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

try:
    import torch

    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.training.trainer import TitanTrainer, TrainingConfig

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch-free schema runs
    TORCH_AVAILABLE = False

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TRAINING_SCHEMA = _REPO_ROOT / "src" / "schema" / "_training.yaml"


def _training_properties() -> dict:
    return yaml.safe_load(_TRAINING_SCHEMA.read_text(encoding="utf-8"))["properties"]


def test_schema_declares_save_epoch_checkpoints():
    prop = _training_properties()["save_epoch_checkpoints"]
    assert prop["type"] == "boolean"
    assert prop["default"] is True
    assert prop["title"] and prop["title_es"]
    assert prop["description"] and prop["description_es"]


def test_schema_zero_disables_rolling_and_best():
    props = _training_properties()
    # 0 must be a valid value for both disabling knobs (no minimum constraint).
    for key in ("checkpoint_every_n_steps", "num_best_checkpoints"):
        prop = props[key]
        assert prop["type"] == "integer"
        assert "minimum" not in prop
        assert "0 = disabled" in prop["description"]
        assert "0 = deshabilitado" in prop["description_es"]


def test_defaults_match_training_config():
    props = _training_properties()
    config = TrainingConfig()
    assert config.save_epoch_checkpoints is props["save_epoch_checkpoints"]["default"]
    assert config.checkpoint_every_n_steps == props["checkpoint_every_n_steps"]["default"]
    assert config.num_best_checkpoints == props["num_best_checkpoints"]["default"]


# ---------------------------------------------------------------------------
# Trainer unit tests
# ---------------------------------------------------------------------------


def _tiny_encoder() -> "FrankensteinEncoder":
    config = FrankensteinModelConfig.from_dict(
        {
            "dims": {
                "vocab_size": 64,
                "hidden_size": 32,
                "num_layers": 1,
                "num_loops": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "layer_pattern": ["standard"],
                "mode": "encoder",
            }
        }
    )
    return FrankensteinEncoder(config)


def _make_trainer(tmp_path, **training_overrides) -> "TitanTrainer":
    config = TrainingConfig(
        csv_log_path=str(tmp_path / "metrics.csv"),
        gpu_metrics_backend="none",
        gpu_temp_guard_enabled=False,
        **training_overrides,
    )
    model = _tiny_encoder()
    return TitanTrainer(model, model.config, training_config=config, device="cpu", task="mlm")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_save_checkpoint_accepts_stable_filename(tmp_path):
    trainer = _make_trainer(tmp_path)
    path = tmp_path / "checkpoints"

    returned = trainer.save_checkpoint(3, filename="titan_final.pt", path=str(path))

    final = path / "titan_final.pt"
    assert final.exists()
    assert pathlib.Path(returned).name == "titan_final.pt"
    payload = torch.load(final, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_num_best_checkpoints_zero_disables_best_saves(tmp_path):
    trainer = _make_trainer(tmp_path, num_best_checkpoints=0)
    path = tmp_path / "checkpoints"

    # Must be a no-op (no IndexError on the empty heap, no files written).
    trainer._maybe_save_best_checkpoint(0, 1.23, path=str(path))
    trainer._maybe_save_best_checkpoint(1, 0.5, path=str(path))

    assert trainer.best_checkpoints == []
    assert list(path.glob("titan_best_loss_*.pt")) == []


# ---------------------------------------------------------------------------
# Engine integration tests (tiny ViT + built-in dummy dataset)
# ---------------------------------------------------------------------------


def _vision_config(csv_path: str, training_extra: dict) -> dict:
    training = {
        "task": "classification",
        "batch_size": 16,
        "num_epochs": 1,
        "dataloader_workers": 0,
        "gradient_accumulation_steps": 1,
        "gpu_temp_guard_enabled": False,
        "enable_block_grad_norms": False,
        "csv_log_path": csv_path,
        "optimizer": {
            "optimizer_class": "adamw",
            "parameters": {"adamw-lr_other": 0.01},
        },
    }
    training.update(training_extra)
    return {
        "model_class": "frankenstein_vit",
        "model": {
            "dims": {
                "vocab_size": 64,
                "hidden_size": 32,
                "num_layers": 1,
                "num_loops": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "layer_pattern": ["standard"],
                "mode": "encoder",
            },
            "norm": {"type": "layer_norm"},
            "embedding": {
                "factorized": {"enabled": False},
                "conv": {"enabled": False},
            },
            "moe": {"enabled": False},
            "bitnet": {"enabled": False},
            "ffn": {"hidden_size": 64, "activation": "silu"},
        },
        "image": {
            "image_size": {"height": 16, "width": 16},
            "patch_size": 8,
            "in_channels": 3,
            "pos_embedding_type": "learned_1d",
            "cls_token": True,
            "pooling_mode": "cls",
            "num_classes": 5,
        },
        "vision_dataset": {
            "dataset_name": "",
            "rescale": {"height": 16, "width": 16},
        },
        "training": training,
    }


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_engine_e2e_policy_writes_only_final(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.engine import train_from_config

    train_from_config(
        _vision_config(
            str(tmp_path / "metrics.csv"),
            {
                "save_epoch_checkpoints": False,
                "checkpoint_every_n_steps": 0,
                "num_best_checkpoints": 0,
            },
        ),
        device="cpu",
        supervisor="off",
        config_path_hint=str(tmp_path),
    )

    checkpoints = sorted(p.name for p in (tmp_path / "checkpoints").glob("*.pt"))
    assert checkpoints == ["titan_final.pt"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_engine_epoch_checkpoints_flag_writes_epoch_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.engine import train_from_config

    train_from_config(
        _vision_config(
            str(tmp_path / "metrics.csv"),
            {
                "save_epoch_checkpoints": True,
                "checkpoint_every_n_steps": 0,
                "num_best_checkpoints": 0,
            },
        ),
        device="cpu",
        supervisor="off",
        config_path_hint=str(tmp_path),
    )

    checkpoints = sorted(p.name for p in (tmp_path / "checkpoints").glob("*.pt"))
    assert checkpoints == ["titan_checkpoint_epoch_0_epoch_end.pt", "titan_final.pt"]
