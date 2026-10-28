#!/usr/bin/env python3
"""End-to-end smoke tests for the audio training dispatch.

Runs one epoch of :func:`src.engine.train_from_config` for ``asr`` (CTC) and
``audio_classification`` with the synthetic :class:`DummyAudioDataset`, tiny
dims and CPU only. Also checks the ``training.asr.loss`` override resolves onto
the flattened model config.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.engine import train_from_config  # noqa: E402
from src.model.frankenstein_audio import FrankensteinAudio  # noqa: E402
from src.training.config_loader import load_training_config  # noqa: E402


def _audio_config(task: str, *, loss: str = "ctc", tmp_path=None) -> dict:
    config = {
        "model_class": "frankenstein_audio",
        "model": {
            "dims": {
                "vocab_size": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "num_loops": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "layer_pattern": ["conformer_attn"],
                "mode": "encoder",
            },
            "norm": {"type": "layer_norm"},
            "embedding": {"factorized": {"enabled": False}},
            "attention": {"conformer": {"kernel_size": 4, "macaron": True}},
            "moe": {"enabled": False},
            "bitnet": {"enabled": False},
            "ffn": {"hidden_size": 64, "activation": "silu"},
            "positional_encoding": "sinusoidal_absolute",
        },
        "audio": {
            "feature": "logmel",
            "sample_rate": 16000,
            "n_mels": 16,
            "conv_channels": 4,
            "num_classes": 4,
            "pooling_mode": "gap",
            "asr_loss": loss,
            "asr_decoder_dim": 16,
            "asr_decoder_layers": 1,
            "spec_augment": {"enabled": True, "time_masks": 2},
        },
        "audio_dataset": {
            "dataset_name": "",
            "split": "train",
            "audio_column": "audio",
            "text_column": "text",
            "label_column": "label",
            "max_duration_sec": 2.0,
            "resample": True,
        },
        "training": {
            "task": task,
            "batch_size": 4,
            "num_epochs": 1,
            "dataloader_workers": 0,
            "gradient_accumulation_steps": 1,
            "gpu_temp_guard_enabled": False,
            "enable_block_grad_norms": False,
            "csv_log_path": str(tmp_path / "metrics.csv") if tmp_path else "metrics.csv",
            "optimizer": {
                "optimizer_class": "adamw",
                "parameters": {"adamw-lr_other": 0.01},
            },
        },
    }
    if task == "asr" and loss == "transducer":
        config["training"]["asr"] = {"loss": "transducer"}
    return config


class TestAudioTrainingDispatch:
    def test_asr_ctc_one_epoch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = train_from_config(
            _audio_config("asr", tmp_path=tmp_path),
            device="cpu",
            supervisor="off",
            config_path_hint=str(tmp_path),
        )
        assert isinstance(result.model, FrankensteinAudio)
        assert result.model.config.mode == "encoder"
        assert result.model.asr_loss == "ctc"
        assert result.final_epoch == 1
        assert result.best_loss is not None
        assert torch.isfinite(torch.tensor(float(result.best_loss)))

    def test_asr_transducer_one_epoch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = train_from_config(
            _audio_config("asr", loss="transducer", tmp_path=tmp_path),
            device="cpu",
            supervisor="off",
            config_path_hint=str(tmp_path),
        )
        assert isinstance(result.model, FrankensteinAudio)
        assert result.model.asr_loss == "transducer"
        assert result.model.transducer is not None

    def test_audio_classification_one_epoch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = train_from_config(
            _audio_config("audio_classification", tmp_path=tmp_path),
            device="cpu",
            supervisor="off",
            config_path_hint=str(tmp_path),
        )
        assert isinstance(result.model, FrankensteinAudio)
        assert result.final_epoch == 1


class TestAsrLossOverride:
    def test_training_asr_loss_overrides_audio_asr_loss(self, tmp_path):
        import yaml

        config = _audio_config("asr", loss="ctc", tmp_path=tmp_path)
        config["training"]["asr"] = {"loss": "transducer"}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        loaded = load_training_config(str(path))
        assert loaded.task == "asr"
        assert loaded.model_config.asr_loss == "transducer"

    def test_invalid_training_asr_loss_rejected(self, tmp_path):
        import yaml

        config = _audio_config("asr", tmp_path=tmp_path)
        config["training"]["asr"] = {"loss": "warp"}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="training.asr.loss"):
            load_training_config(str(path))

    def test_audio_task_requires_audio_model_class(self, tmp_path):
        import yaml

        config = _audio_config("asr", tmp_path=tmp_path)
        config["model_class"] = "frankenstein_text_encoder"
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="frankenstein_audio"):
            load_training_config(str(path))
