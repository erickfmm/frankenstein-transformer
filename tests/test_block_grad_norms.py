"""Tests for canonical parameter-block categories and per-block grad norms.

The block set mirrors the website parameter estimator
(``ft-param-estimator.js``): embeddings, pos, attention, ffn, norm, head,
mhc, router, residual, other. These tests pin the categorizer, the trainer's
CSV schema, and the L2 aggregation used by ``enable_block_grad_norms``.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys
import types

import pytest
import yaml

from src.utils.param_categories import BLOCK_CATEGORIES, categorize_param

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TRAINING_SCHEMA = _REPO_ROOT / "src" / "schema" / "_training.yaml"

try:
    import torch
    import torch.nn.functional as F
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.training.trainer import TitanTrainer, TrainingConfig

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch-free schema runs
    TORCH_AVAILABLE = False


def test_block_categories_match_estimator_breakdown():
    assert BLOCK_CATEGORIES == (
        "embeddings",
        "pos",
        "attention",
        "ffn",
        "norm",
        "head",
        "mhc",
        "router",
        "residual",
        "other",
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Embeddings / patch embedding (encoder, decoder ``backbone.``, ViT)
        # and the audio waveform frontend (conv subsampling + projection).
        ("emb.weight", "embeddings"),
        ("backbone.emb.weight", "embeddings"),
        ("patch_embed.proj.weight", "embeddings"),
        ("frontend.subsample.0.weight", "embeddings"),
        ("frontend.proj.bias", "embeddings"),
        ("backbone.frontend.subsample.2.weight", "embeddings"),
        # Positional encodings and the ViT cls/mask tokens.
        ("pos_encoder.pos_embed", "pos"),
        ("backbone.pos_encoder.bam.theta_alpha", "pos"),
        ("backbone.pos_encoder.ssmax.s", "pos"),
        ("cls_token", "pos"),
        ("mask_token", "pos"),
        # Attention mixers, including their internal norms.
        ("layers.0.mixer.q_proj.weight", "attention"),
        ("backbone.layers.2.mixer.retn.norm.weight", "attention"),
        ("layers.0.mixer.conv.depthwise.weight", "attention"),
        ("backbone.layers.1.mixer.ff_in.weight", "attention"),
        ("backbone.layers.1.mixer.norm_ffn1.weight", "attention"),
        # Standalone block norms.
        ("layers.0.norm1.weight", "norm"),
        ("layers.0.norm2.weight", "norm"),
        ("final_norm.weight", "norm"),
        ("backbone.layers.1.norm1.bias", "norm"),
        # FFN bodies and MoE experts.
        ("layers.0.ffn.up_proj.weight", "ffn"),
        ("layers.0.experts.1.down_proj.weight", "ffn"),
        # MoE router and Mixture-of-Depths depth router.
        ("layers.0.router.weight", "router"),
        ("layers.0.depth_router.weight", "router"),
        # mHC / Hyperloop.
        ("mhc_in_proj.weight", "mhc"),
        ("backbone.mhc_out_proj.bias", "mhc"),
        ("mhc.hyperloop.loops.0.proj.weight", "mhc"),
        # AttnRes residual modules.
        ("residual.query_weight", "residual"),
        ("backbone.residual.query_weight", "residual"),
        # Task/output heads.
        ("head.weight", "head"),
        ("cls_head.weight", "head"),
        ("classification_head.bias", "head"),
        ("patch_pred_head.weight", "head"),
        ("seg_head.weight", "head"),
        ("seg_upsampler.0.weight", "head"),
        # Audio heads: classification, CTC/vocabulary projection is ``head``
        # and the RNN-Transducer (prediction LSTM + joint network) is a head too.
        ("audio_classification_head.weight", "head"),
        ("backbone.audio_classification_head.bias", "head"),
        ("transducer.embed.weight", "head"),
        ("transducer.rnn.weight_ih_l0", "head"),
        ("transducer.joint_in.weight", "head"),
        ("transducer.joint_out.bias", "head"),
        # Unknown parameters fall back to the catch-all.
        ("some_unknown.weight", "other"),
    ],
)
def test_categorize_param(name, expected):
    assert categorize_param(name) == expected


def test_training_schema_declares_wandb_keys():
    properties = yaml.safe_load(_TRAINING_SCHEMA.read_text(encoding="utf-8"))["properties"]
    assert properties["wandb_project"]["default"] is None
    assert set(properties["wandb_project"]["type"]) == {"string", "null"}
    assert properties["wandb_mode"]["default"] == "online"
    assert set(properties["wandb_mode"]["enum"]) == {"online", "offline", "disabled"}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_training_config_wandb_defaults():
    config = TrainingConfig()
    assert config.wandb_project is None
    assert config.wandb_mode == "online"
    assert config.enable_block_grad_norms is True


def _tiny_encoder(**overrides) -> "FrankensteinEncoder":
    """Build a tiny encoder that exercises every non-``other`` block."""
    defaults = dict(
        vocab_size=64,
        hidden_size=48,
        num_layers=2,
        num_loops=1,
        num_heads=6,
        retention_heads=6,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=True,
        ffn_hidden_size=96,
        ffn_activation="gelu",
        positional_encoding="learned_absolute",
        mode="encoder",
        residual_type="block_attn",
        block_attn_num_blocks=2,
        use_mhc=True,
        mhc_expansion_rate=2,
    )
    defaults.update(overrides)
    return FrankensteinEncoder(FrankensteinModelConfig(**defaults))


def _make_trainer(tmp_path, model, **training_overrides) -> "TitanTrainer":
    config = TrainingConfig(
        csv_log_path=str(tmp_path / "metrics.csv"),
        gpu_metrics_backend="none",
        gpu_temp_guard_enabled=False,
        **training_overrides,
    )
    return TitanTrainer(model, model.config, training_config=config, device="cpu", task="mlm")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_every_model_param_maps_to_a_known_block():
    model = _tiny_encoder()
    seen = set()
    for name, _param in model.named_parameters():
        category = categorize_param(name)
        assert category in BLOCK_CATEGORIES, name
        seen.add(category)
    # The fixture enables MoE, mHC, learned PE and block residual attention.
    assert seen == set(BLOCK_CATEGORIES) - {"other"}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_collect_block_grad_norms_matches_manual_l2(tmp_path):
    model = _tiny_encoder()
    trainer = _make_trainer(tmp_path, model)

    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    loss = F.cross_entropy(model(input_ids).view(-1, 64), labels.view(-1))
    loss.backward()

    collected = trainer._collect_block_grad_norms()

    manual_sq = {category: 0.0 for category in BLOCK_CATEGORIES}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        manual_sq[categorize_param(name)] += float(param.grad.detach().float().pow(2).sum().item())
    manual = {key: math.sqrt(value) for key, value in manual_sq.items()}

    assert collected.keys() == manual.keys()
    for category in BLOCK_CATEGORIES:
        assert collected[category] == pytest.approx(manual[category], rel=1e-6, abs=1e-8)
    # Gradients must actually flow through every logged block.
    for category in ("embeddings", "pos", "attention", "ffn", "norm", "head", "mhc", "router", "residual"):
        assert collected[category] > 0.0, category


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_collect_block_grad_norms_disabled_returns_zeros(tmp_path):
    model = _tiny_encoder()
    trainer = _make_trainer(tmp_path, model, enable_block_grad_norms=False)

    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    F.cross_entropy(model(input_ids).view(-1, 64), labels.view(-1)).backward()

    assert all(value == 0.0 for value in trainer._collect_block_grad_norms().values())


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_csv_schema_contains_every_block_column(tmp_path):
    model = _tiny_encoder()
    trainer = _make_trainer(tmp_path, model)

    columns = trainer._get_csv_columns()
    for category in BLOCK_CATEGORIES:
        assert f"grad_norm_{category}" in columns
    # The legacy extra buckets are gone: exactly one column per block.
    assert len([column for column in columns if column.startswith("grad_norm_")]) == len(
        BLOCK_CATEGORIES
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_log_step_writes_full_block_row(tmp_path):
    model = _tiny_encoder()
    trainer = _make_trainer(tmp_path, model)

    trainer._init_csv_logger()
    try:
        trainer._log_step_to_csv(epoch=0, step=0, loss=1.0, accuracy=0.0, lr=1e-3, grad_norm=1.0)
    finally:
        trainer.close()

    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == trainer._get_csv_columns()
    assert len(rows[1]) == len(rows[0])
    header = {column: value for column, value in zip(rows[0], rows[1])}
    for category in BLOCK_CATEGORIES:
        assert f"grad_norm_{category}" in header


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_wandb_disabled_when_project_unset(tmp_path):
    model = _tiny_encoder()
    trainer = _make_trainer(tmp_path, model)

    assert trainer._ensure_wandb_run() is None
    # A second call stays a no-op without attempting another import.
    assert trainer._ensure_wandb_run() is None


def _fake_wandb(monkeypatch):
    """Install a recording ``wandb`` stub and return its call log."""
    calls = {"init": [], "log": [], "finish": 0}
    fake = types.ModuleType("wandb")
    fake.init = lambda **kwargs: calls["init"].append(kwargs)
    fake.log = lambda payload, step=None: calls["log"].append((payload, step))
    fake.finish = lambda: calls.__setitem__("finish", calls["finish"] + 1)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return calls


def _log_params():
    return dict(
        loss=1.0,
        accuracy=0.5,
        lr=1e-4,
        grad_norm=2.0,
        scaler_scale=1.0,
        telemetry={
            "gpu_temp_c": 60.0,
            "gpu_power_w": 100.0,
            "gpu_util_pct": 90.0,
            "gpu_mem_used_mib": 1024.0,
        },
        step_time_ms=1.5,
        tokens_per_sec=2048.0,
        clip_ratio=0.4,
        effective_batch_size=8,
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_wandb_logs_every_block_when_enabled(tmp_path, monkeypatch):
    calls = _fake_wandb(monkeypatch)
    model = _tiny_encoder()
    trainer = _make_trainer(tmp_path, model, wandb_project="unit-test", wandb_mode="offline")

    trainer._log_step_to_wandb(grad_buckets=trainer._get_default_block_grad_norms(), **_log_params())

    assert len(calls["init"]) == 1
    assert calls["init"][0]["project"] == "unit-test"
    assert calls["init"][0]["mode"] == "offline"
    assert len(calls["log"]) == 1
    payload, step = calls["log"][0]
    assert step == int(trainer.global_step)
    assert {f"grad_norm/{block}" for block in BLOCK_CATEGORIES} <= set(payload)
    assert {"loss", "learning_rate", "grad_norm", "clip_ratio"} <= set(payload)

    trainer.close()
    assert calls["finish"] == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_wandb_omits_blocks_when_disabled(tmp_path, monkeypatch):
    calls = _fake_wandb(monkeypatch)
    model = _tiny_encoder()
    trainer = _make_trainer(
        tmp_path, model, wandb_project="unit-test", enable_block_grad_norms=False
    )

    trainer._log_step_to_wandb(grad_buckets=trainer._get_default_block_grad_norms(), **_log_params())

    payload, _step = calls["log"][0]
    assert not any(key.startswith("grad_norm/") for key in payload)
    assert "loss" in payload
