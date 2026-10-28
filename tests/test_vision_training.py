#!/usr/bin/env python3
"""Engine vision-dispatch smoke test (``train_from_config`` + DummyImageDataset).

``engine._train_vision`` builds a :class:`DummyImageDataset` when the config
has no ``vision_dataset.dataset_name``/``dataset_dir``; that built-in dataset
must be accepted by the pre-built-dataset guard (it is a ``torch`` Dataset, not
an ``Iterable``). This regression test trains one epoch of a tiny
``frankenstein_vit`` with the Conformer mixer on the synthetic data.
"""

from __future__ import annotations

import pytest

try:
    import torch
    from src.engine import train_from_config
    from src.model.frankenstein_vit import FrankensteinViT

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard test dependency
    TORCH_AVAILABLE = False


def _vision_config(csv_path: str) -> dict:
    """Return a tiny ViT classification config with the built-in dataset."""
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
                "layer_pattern": ["conformer_attn"],
                "mode": "encoder",
            },
            "norm": {"type": "layer_norm"},
            "embedding": {
                "factorized": {"enabled": False},
                "conv": {"enabled": False},
            },
            "attention": {
                "conformer": {"kernel_size": 4, "macaron": True, "conv_norm": "batch"},
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
        "training": {
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
        },
    }


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_vision_training_accepts_builtin_dummy_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = train_from_config(
        _vision_config(str(tmp_path / "metrics.csv")),
        device="cpu",
        supervisor="off",
        config_path_hint=str(tmp_path),
    )

    assert isinstance(result.model, FrankensteinViT)
    assert result.final_epoch == 1
    assert result.best_loss is not None
    assert torch.isfinite(torch.tensor(float(result.best_loss)))
