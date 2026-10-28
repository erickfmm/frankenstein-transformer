#!/usr/bin/env python3
"""Parameter-category gates for :class:`FrankensteinAudio` + ``conformer_attn``.

The canonical taxonomy (:data:`src.utils.param_categories.BLOCK_CATEGORIES`)
feeds three consumers: the per-block ``grad_norm_<block>`` training logs, the
optimizer parameter groups and the website parameter estimator's breakdown.
The audio model class must therefore leave nothing in the ``other`` bucket:

* ``frontend.*`` (mel / spectrogram / raw conv subsampling + projection) →
  ``embeddings``,
* ``mixer.*`` (including the Conformer macaron FFN, the MHSA projections and
  the convolution module) → ``attention``,
* ``head`` (CTC / vocabulary projection) and ``audio_classification_head`` /
  ``transducer`` (prediction LSTM + joint network) → ``head``,
* ``cls_token`` → ``pos``.

These tests build tiny CPU models and check the category arithmetic against
``sum(p.numel())``; they do not run a forward pass.
"""

from __future__ import annotations

import pytest

from src.utils.param_categories import BLOCK_CATEGORIES, categorize_param

try:
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_audio import FrankensteinAudio

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch-free schema runs
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")


def _audio_config(**overrides) -> "FrankensteinModelConfig":
    """Return a tiny ``conformer_attn`` audio config for fast CPU tests."""
    params = dict(
        vocab_size=48,
        hidden_size=32,
        num_layers=2,
        num_loops=1,
        num_heads=4,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=["conformer_attn", "standard_attn"],
        use_moe=False,
        ffn_hidden_size=64,
        ffn_activation="silu",
        positional_encoding="sinusoidal_absolute",
        audio_n_mels=32,
        audio_conv_channels=8,
        audio_sample_rate=16000,
        audio_window_ms=25.0,
        audio_stride_ms=10.0,
        audio_n_fft=400,
        num_classes=3,
        pooling_mode="gap",
        asr_loss="ctc",
        asr_decoder_dim=16,
        asr_decoder_layers=1,
    )
    params.update(overrides)
    return FrankensteinModelConfig(**params)


def _category_counts(model) -> dict:
    """Per-category parameter counts, mirroring ``param_count_check.py``."""
    counts = {category: 0 for category in BLOCK_CATEGORIES}
    for name, param in model.named_parameters():
        counts[categorize_param(name)] += param.numel()
    return counts


AUDIO_VARIANTS = {
    "asr_ctc_gap": dict(),
    "asr_transducer": dict(asr_loss="transducer"),
    "classification_cls": dict(pooling_mode="cls"),
    "raw_eeg": dict(audio_feature="raw", audio_in_channels=8),
}


@pytest.mark.parametrize("variant", sorted(AUDIO_VARIANTS))
def test_audio_categories_cover_every_parameter(variant):
    model = FrankensteinAudio(_audio_config(**AUDIO_VARIANTS[variant]))
    counts = _category_counts(model)

    total = sum(param.numel() for param in model.parameters())
    assert sum(counts.values()) == total
    # Every audio parameter lands in a meaningful block: no catch-all leftovers.
    assert counts["other"] == 0, [
        name for name, _ in model.named_parameters() if categorize_param(name) == "other"
    ]
    assert counts["embeddings"] > 0  # frontend conv subsampling + projection
    assert counts["attention"] > 0  # conformer macaron FFN + MHSA + conv module
    assert counts["ffn"] > 0  # shared HybridLayer FFN (second macaron step)
    assert counts["norm"] > 0
    assert counts["head"] > 0  # CTC/vocab projection (+ transducer when enabled)
    # ``pos`` only exists on the pooled classification path (learned [CLS]).
    has_cls = any(name == "cls_token" for name, _ in model.named_parameters())
    assert has_cls == (AUDIO_VARIANTS[variant].get("pooling_mode") == "cls")
    assert (counts["pos"] > 0) == has_cls


def test_audio_optimizer_groups_match_block_totals():
    model = FrankensteinAudio(_audio_config(pooling_mode="cls", asr_loss="transducer"))
    block_counts = _category_counts(model)

    groups = {group["name"]: group["params"] for group in model.configure_optimizers()}
    assert set(groups) == {
        "embeddings",
        "pos",
        "attention",
        "ffn",
        "norms",
        "head",
        "mhc",
        "router",
        "residual",
        "other",
    }

    def _numel(params) -> int:
        return sum(int(param.numel()) for param in params)

    assert _numel(groups["embeddings"]) == block_counts["embeddings"]
    assert _numel(groups["pos"]) == block_counts["pos"]
    assert _numel(groups["attention"]) == block_counts["attention"]
    assert _numel(groups["ffn"]) == block_counts["ffn"]
    assert _numel(groups["norms"]) == block_counts["norm"]
    assert _numel(groups["head"]) == block_counts["head"]
    for empty_group in ("mhc", "router", "residual", "other"):
        assert groups[empty_group] == []


def test_conformer_mixer_parameters_are_attention():
    model = FrankensteinAudio(_audio_config())
    mixer_names = [name for name, _ in model.named_parameters() if ".mixer." in name]
    assert mixer_names, "the tiny config must contain a conformer mixer"
    for name in mixer_names:
        assert categorize_param(name) == "attention", name
    # The Conformer-specific submodules must be visible through the mixer path.
    assert any(".mixer.conv." in name for name in mixer_names)
    assert any(".mixer.q_proj." in name for name in mixer_names)


def test_frontend_buffers_are_not_parameters():
    """The Hann window and mel filterbank must not inflate ``embeddings``."""
    model = FrankensteinAudio(_audio_config())
    buffer_names = [name for name, _ in model.named_buffers()]
    assert any(name.startswith("frontend.") for name in buffer_names)
    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any(name in parameter_names for name in buffer_names)
