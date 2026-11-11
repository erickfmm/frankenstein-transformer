"""Forward/backward smoke test for every trainable mixer at harness dims.

Regression gate for experiment 2's 164 wasted e2e runs: ``cca``,
``ccgqa``, ``msa`` and ``ssog`` crashed on the full sweep
(``hidden_size=96``, ``num_heads=8``, ``seq_len=64``, model-wide RoPE) while
the unit suite only covered 29 of the schema mixers and the KV-cache parity
suite only parametrizes ``IMPLEMENTED_MIXERS``. This module builds one tiny
encoder per schema enum value with exactly the e2e harness geometry and runs
a forward + backward pass, so a mixer that cannot be constructed or trained
fails here in seconds instead of after hours of subprocess compute.

Covered regressions:

* ``cca`` — default ``latent_rank`` must resolve to an even per-head
  latent dim (``96 // 4 = 24`` is odd per head with 8 heads).
* ``cca``/``ccgqa`` — rotary PEs must rotate the *latent* q/k at
  ``latent_head_dim`` width, not the model-wide ``hidden_size / num_heads``.
* ``msa`` — ``topk_blocks`` must be clamped when the sequence yields
  fewer blocks than requested (``seq_len=64`` < ``block_size=128``).
* ``ssog`` — the text grid must match the padded sequence length.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

from src.model.config import FrankensteinModelConfig, merge_model_blocks
from src.model.frankenstein_encoder import FrankensteinEncoder
from src.utils.schema_loader import resolve_schema

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Training-free sparse heuristics (D5): train through the dense path but the
#: documented smoke contract runs them in eval mode, like the e2e harness.
TRAINING_FREE = {"fasa", "sparge"}

#: e2e harness geometry (``full_tests/run_e2e.py`` base config): these exact
#: dims produced the experiment-2 failures, so do not "simplify" them.
HARNESS_DIMS = {
    "dims": {
        "vocab_size": 100,
        "hidden_size": 96,
        "num_layers": 1,
        "num_loops": 1,
        "num_heads": 8,
        "num_kv_heads": 4,
        "dropout": 0.0,
        "layer_pattern": ["standard"],
        "mode": "encoder",
    },
    "attention": {
        "retnet": {"heads": 8},
        "ode": {"solver": "rk4", "steps": 1},
    },
    "norm": {"type": "layer_norm"},
    "bitnet": {"enabled": False},
    "moe": {"enabled": False, "num_experts": 2, "top_k": 1},
    "mixture_of_depths": {"enabled": False},
    "ffn": {"hidden_size": 96, "activation": "gelu"},
}


def _schema_mixers() -> list:
    schema = resolve_schema(REPO_ROOT / "src" / "schema.yaml")
    enum_values = (
        schema["properties"]["model"]["properties"]["dims"]["properties"][
            "layer_pattern"
        ]["items"]["enum"]
    )
    return sorted(enum_values)


def _build_config(layer_pattern, positional_encoding="rope", seq_len=64):
    overrides = {
        "dims": {"layer_pattern": list(layer_pattern)},
        "positional_encoding": positional_encoding,
    }
    if "ssog" in layer_pattern:
        overrides["attention"] = {"ssog": {"grid_h": 1, "grid_w": seq_len}}
    return FrankensteinModelConfig.from_dict(merge_model_blocks(HARNESS_DIMS, overrides))


@pytest.mark.parametrize("mixer", _schema_mixers())
def test_mixer_forward_backward_at_harness_dims(mixer):
    """Every schema mixer builds, forward-passes and backprops at 96/8/64."""
    config = _build_config([mixer])
    model = FrankensteinEncoder(config)
    x = torch.randint(0, config.dims.vocab_size, (2, 64))
    if mixer in TRAINING_FREE:
        model.eval()
        with torch.no_grad():
            y = model(x)
    else:
        y = model(x)
        y.sum().backward()
    assert y.shape == (2, 64, config.dims.vocab_size)


@pytest.mark.parametrize("pe", ["rope", "hope", "sinusoidal_rotary"])
@pytest.mark.parametrize("mixer", ["cca", "ccgqa"])
def test_latent_mixers_support_every_rotary_pe(mixer, pe):
    """Rotary PEs must be applied through a latent-sized encoder.

    The shared model-wide encoder is built at ``hidden_size / num_heads``; the
    latent q/k are narrower, so feeding them through the shared module raised
    ``RuntimeError: The size of tensor a (3) must match the size of tensor
    b (6)`` (experiment 2, 41 runs per mixer).
    """
    config = _build_config([mixer], positional_encoding=pe)
    model = FrankensteinEncoder(config)
    x = torch.randint(0, config.dims.vocab_size, (2, 64))
    y = model(x)
    assert y.shape == (2, 64, config.dims.vocab_size)


@pytest.mark.parametrize("pe", ["alibi", "bam", "nope", "none"])
@pytest.mark.parametrize("mixer", ["cca", "ccgqa"])
def test_latent_mixers_support_additive_pes(mixer, pe):
    """Score-bias PEs (no q/k rotation) keep working through the shared encoder."""
    config = _build_config([mixer], positional_encoding=pe)
    model = FrankensteinEncoder(config)
    x = torch.randint(0, config.dims.vocab_size, (2, 64))
    y = model(x)
    assert y.shape == (2, 64, config.dims.vocab_size)


@pytest.mark.parametrize("mixer", ["cca", "ccgqa"])
def test_latent_mixers_reject_pape_with_clear_error(mixer):
    """PaPE augments q/k from the full-width hidden state: explicit error, not a shape crash."""
    config = _build_config([mixer], positional_encoding="pape")
    with pytest.raises(ValueError, match="not supported with cca"):
        FrankensteinEncoder(config)


def test_msa_short_sequence_clamps_topk():
    """``seq_len < msa_block_size`` yields one block: top-k must clamp, not crash.

    Defaults are ``msa_block_size=128`` / ``msa_topk_blocks=16``; a 6-token
    input produced ``RuntimeError: selected index k out of range``.
    """
    from src.model.attention.params import mixer_params

    config = _build_config(["msa"], seq_len=6)
    msa = mixer_params(config, "msa")
    assert msa.block_size == 128
    assert msa.topk_blocks == 16
    model = FrankensteinEncoder(config)
    x = torch.randint(0, config.dims.vocab_size, (2, 6))
    y = model(x)
    assert y.shape == (2, 6, config.dims.vocab_size)


@pytest.mark.parametrize(
    "hidden,heads",
    [(96, 8), (512, 8), (48, 6), (64, 4), (128, 16)],
)
def test_cca_ccgqa_defaults_are_always_buildable(hidden, heads):
    """Resolved defaults give divisible ranks and even per-head latent dims."""
    config = _build_config(["cca", "ccgqa"])
    # Re-resolve from scratch at the new geometry (nested shape).
    config.dims.hidden_size = hidden
    config.dims.num_heads = heads
    config.attention.setdefault("cca", {}).pop("latent_rank", None)
    ccgqa = config.attention.setdefault("ccgqa", {})
    for key in ("query_latent_rank", "kv_latent_rank", "num_kv_heads"):
        ccgqa.pop(key, None)
    config.__post_init__()

    assert config.attention["cca"]["latent_rank"] % heads == 0
    assert (config.attention["cca"]["latent_rank"] // heads) % 2 == 0
    assert heads % config.attention["ccgqa"]["num_kv_heads"] == 0
    q_dh = config.attention["ccgqa"]["query_latent_rank"] // heads
    kv_dh = config.attention["ccgqa"]["kv_latent_rank"] // config.attention["ccgqa"]["num_kv_heads"]
    assert q_dh == kv_dh
    assert q_dh % 2 == 0
    assert config.attention["ccgqa"]["kv_latent_rank"] <= config.attention["ccgqa"]["query_latent_rank"]

    from src.model.attention.latent.cca import (
        CCAAttention,
        CCGQAAttention,
    )

    CCAAttention(config)
    CCGQAAttention(config)


def test_cca_latent_pe_is_sized_to_latent_head_dim():
    """The mixer's rotary encoder pairs ``latent_head_dim // 2`` dims, not ``head_dim // 2``."""
    from src.model.embeddings.rope import RoPE

    config = _build_config(["cca"], positional_encoding="rope")
    model = FrankensteinEncoder(config)
    layer = model.layers[0]
    mixer = layer.mixer
    assert isinstance(mixer.latent_pe, RoPE)
    assert mixer.latent_pe.pair_dim == mixer.latent_head_dim // 2
    assert mixer.latent_pe.pair_dim != config.dims.hidden_size // config.dims.num_heads // 2
