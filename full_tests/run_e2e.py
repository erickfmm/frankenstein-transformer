"""Exhaustive end-to-end smoke tests for frankenstein-transformer.

This script drives the real CLI with tiny synthetic data and a tiny on-the-fly
SentencePiece tokenizer, exercising:

* every trainable attention mixer in unordered pairs,
* every optimizer,
* every normalization type,
* every model-level positional encoding (rope, hope, nope, alibi, pape, pape_efficient,
  pape_ri, sinusoidal_absolute, sinusoidal_rotary, learned_absolute, none),
* every ViT `pos_embedding_type` across all 3 vision tasks (classification,
  patch_prediction, segmentation),
* major transversal toggles (BitNet, MoE, MoD, mHC, embeddings, residuals/AttnRes, etc.),
* encoder/MLM and decoder/causal-LM tasks,
* vision tasks (frankenstein_vit: patch_prediction, classification, segmentation),
* audio tasks (frankenstein_audio: asr with CTC and RNN-Transducer,
  audio_classification, and `feature: raw` EEG-style signals) using
  `DummyAudioDataset` only (no network access),
* real-data combos (`--category realdata`): every task family trained against
  small HuggingFace Hub datasets — MLM/causal-LM on Spanish poetry, text
  classification, SBERT triplets, ViT segmentation/classification/patch
  prediction, and audio ASR/classification. These combos download from the
  Hub (models + corpora), so run them with a larger per-combo timeout
  (e.g. `--timeout 1800`) because first-time downloads can exceed the
  default,
* deploy, inference, transformers and BitNet-GGUF export smoke tests (audio
  checkpoints support `standard`/`quantized` deploy; Transformers export must
  be rejected with the documented message).

It is intentionally NOT collected by ``pytest tests/`` (it lives in ``full_tests/``
at the repository root) and is meant to be left running for several hours.

Run from the repo root with the uv project env:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py

To run only a subset:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --category optimizers --limit 3

To run only the real-data (HuggingFace Hub) combos — downloads the base model,
tokenizers and corpora on first use, so allow a generous per-combo timeout:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --category realdata --timeout 1800

To run on a specific device (default is cpu):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --device cuda

To enable the GPU thermal guard during training (with optional thresholds):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --device cuda --gpu-temp-guard

To resume from existing tmp/results directory (skip cleanup):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --resume
"""
from __future__ import annotations

import argparse
import itertools
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Make the helpers importable whether this file is run as a script or module.
_FULL_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FULL_TESTS_DIR.parent
sys.path.insert(0, str(_FULL_TESTS_DIR))

import _helpers as helpers  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical lists from the schema
# ---------------------------------------------------------------------------
# fasa_attn and sparge_attn are training-free/eval-only; skip them here.
ATTENTIONS = [
    "retnet", "retnet_attn", "mamba", "ode", "titan_attn", "standard_attn",
    "sigmoid_attn", "sparse_transformer_attn", "longformer_attn", "bigbird_attn",
    "sparsek_attn", "nsa_attn", "gla_attn", "deltanet_attn", "gated_deltanet_attn",
    "gated_deltanet2_attn", "hgrn2_attn", "fox_attn", "gated_softmax_attn",
    "engram_attn", "gqa_attn", "mla_attn", "gqla_attn", "mlra_attn",
    "tucker_attn", "iha_attn", "gta_attn", "mtla_attn", "cca_attn", "ccgqa_attn",
    "msa_attn", "sparda_attn", "kda_attn", "gma_attn", "ssog_attn",
    "falcon1_attn", "falcon2_attn", "falcon3_attn", "falcon1a_attn",
    "falcon2a_attn", "falcon3a_attn",
    # Convolution-augmented attention (arXiv:2005.08100): macaron FFN₁ + MHSA
    # + convolution module. Trains densely in both encoder and decoder modes
    # (causal padding in the decoder), so it belongs in the pair sweep too.
    "conformer_attn",
]

# Model-wide positional encodings (src/schema/_model/_positional_encoding.yaml).
# Each is exercised on a fixed baseline layer_pattern — no cross product with mixers.
POSITIONAL_ENCODINGS = [
    "rope", "hope", "nope", "alibi", "bam", "pape", "pape_efficient", "pape_ri",
    "sinusoidal_absolute", "sinusoidal_rotary", "learned_absolute", "none",
]

# ViT image positional embedding types (src/schema/_image.yaml pos_embedding_type enum).
# Exercised across all 3 vision tasks — no cross product with mixers.
VISION_PE_TYPES = [
    "learned_1d", "none", "learned_absolute", "sinusoidal_absolute",
    "sinusoidal_rotary", "pape", "pape_efficient", "pape_ri",
    "rope", "hope", "nope", "alibi", "bam",
]

OPTIMIZERS = [
    "sgd_momentum", "adamw", "adafactor", "galore_adamw", "prodigy", "lion",
    "sophia", "muon", "turbo_muon", "radam", "adan", "adopt", "ademamix",
    "mars_adamw", "cautious_adamw", "lamb", "schedulefree_adamw", "shampoo",
    "soap", "anon", "apollo", "apollo_mini", "q_apollo",
]

NORMS = ["layer_norm", "dynamic_tanh", "derf", "rms_norm", "prms_norm", "flash_norm"]


# ---------------------------------------------------------------------------
# Real-data combos (HuggingFace Hub datasets)
# ---------------------------------------------------------------------------
# NLP tasks driven by a ``text_dataset`` block require an explicit HF
# tokenizer source (the engine falls back to the legacy toy SPM otherwise),
# and the tokenizer's vocab size is injected into ``model.dims.vocab_size``
# automatically — the configured dims.vocab_size stays a harmless placeholder.
REALDATA_TEXT_TOKENIZER = "dccuchile/bert-base-spanish-wwm-cased"

# SBERT training hard-requires a top-level ``base_model`` (``_train_sbert`` in
# ``src/engine.py``); BETO matches the shipped ``es_sbert_*`` examples.
REALDATA_SBERT_BASE_MODEL = "dccuchile/bert-base-spanish-wwm-cased"


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def _deep_update(base: dict, updates: dict) -> dict:
    """Return a new dict merging ``updates`` into ``base`` recursively.

    Scalar/list values in ``updates`` win over ``base``; nested dicts merge.
    """
    out = dict(base)
    for key, value in updates.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_example_config(name: str, **overrides: Any) -> dict:
    """Load ``configs/examples/<name>.yaml`` and apply deep overrides.

    Some features (the Conformer ViT, the audio model class/tasks) are already
    described by schema-validated example presets. Reusing them here keeps the
    harness honest about the shipped examples; only the data source and the
    step budget are overridden so each case stays a fast offline CPU smoke
    test. Any overrides are written into the run's ``config.yaml``, which is
    the harness' standard temporary-config mechanism.

    Args:
        name: Example file stem (without ``.yaml``).
        **overrides: Recursive overrides merged into the loaded config.

    Returns:
        The example config as a dict, with ``overrides`` applied.
    """
    path = _PROJECT_ROOT / "configs" / "examples" / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if overrides:
        config = _deep_update(config, overrides)
    return config


def _base_config(vocab_size: int, num_layers: int, num_loops: int) -> Tuple[dict, dict]:
    model = helpers.default_model_cfg(
        vocab_size=vocab_size, num_layers=num_layers, num_loops=num_loops
    )
    training = helpers.default_training_cfg(
        helpers.ensure_toy_parquet(), batch_size=1, num_epochs=1
    )
    return model, training


def _ensure_ssog_grid(model: dict, training: dict) -> dict:
    """Give ``ssog_attn`` layers a 1D grid matching the training ``max_length``.

    SSOG requires ``seq_len == grid_h * grid_w``. The toy MLM corpus is padded
    and truncated to exactly ``max_length`` tokens, so a degenerate
    ``1 x max_length`` grid fits the NLP runs (vision runs derive the patch
    grid automatically and must NOT call this).
    """
    pattern = model.get("dims", {}).get("layer_pattern", [])
    if "ssog_attn" not in pattern:
        return model
    max_length = int(training.get("max_length", 32))
    attention = dict(model.get("attention") or {})
    ssog_block = dict(attention.get("ssog") or {})
    ssog_block.setdefault("grid_h", 1)
    ssog_block.setdefault("grid_w", max_length)
    attention["ssog"] = ssog_block
    model = dict(model)
    model["attention"] = attention
    return model


def _with_optimizer(training: dict, optimizer_class: str) -> dict:
    """Build a minimal per-group parameter block for the given optimizer class."""
    prefix = f"{optimizer_class}-"
    params: Dict[str, Any] = {
        f"{prefix}lr_embeddings": 1e-4,
        f"{prefix}lr_norms": 1e-4,
        f"{prefix}lr_attention": 1e-4,
        f"{prefix}lr_other": 1e-4,
        f"{prefix}wd_embeddings": 0.0,
        f"{prefix}wd_norms": 0.0,
        f"{prefix}wd_attention": 0.0,
        f"{prefix}wd_other": 0.0,
    }
    # Adam-style families also accept betas/eps per group.  Providing them is
    # harmless for non-Adam optimizers because the factory simply ignores keys
    # it does not understand, while it satisfies the schema's minProperties.
    if optimizer_class in {
        "adamw", "galore_adamw", "prodigy", "radam", "adan", "adopt", "ademamix",
        "mars_adamw", "cautious_adamw", "schedulefree_adamw", "soap", "lion",
    }:
        for group in ["embeddings", "norms", "attention", "other"]:
            params[f"{prefix}betas_{group}"] = [0.9, 0.999]
            params[f"{prefix}eps_{group}"] = 1e-8
    # A few optimizers have documented specific globals that are safer to set.
    if optimizer_class == "prodigy":
        params[f"{prefix}d_coef"] = 0.8
    if optimizer_class in {"muon", "turbo_muon"}:
        params[f"{prefix}momentum"] = 0.9
        params[f"{prefix}ns_steps"] = 5
    if optimizer_class == "sophia":
        params[f"{prefix}rho"] = 0.05
        params[f"{prefix}update_k"] = 10
    if optimizer_class == "cautious_adamw":
        params[f"{prefix}cautious_clip"] = 1.0
    if optimizer_class == "anon":
        params[f"{prefix}gamma"] = 0.5
    if optimizer_class == "galore_adamw":
        params[f"{prefix}rank"] = 8
        params[f"{prefix}update_proj_gap"] = 50
    if optimizer_class in {"apollo", "apollo_mini", "q_apollo"}:
        params[f"{prefix}rank"] = 8
        params[f"{prefix}update_proj_gap"] = 50
        params[f"{prefix}scale"] = 1.0
        params[f"{prefix}proj_type"] = "std"

    training = training.copy()
    training["optimizer"] = {
        "optimizer_class": optimizer_class,
        "parameters": params,
    }
    return training


def make_attention_pair_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for a, b in itertools.combinations(ATTENTIONS, 2):
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {
            "dims": {
                "layer_pattern": [a, b],
            },
        })
        model = _ensure_ssog_grid(model, training)
        combo_id = f"attn_{a}__{b}"
        configs.append((combo_id, {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_single_attention_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for a in ATTENTIONS:
        model, training = _base_config(vocab_size, num_layers=1, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": [a]}})
        model = _ensure_ssog_grid(model, training)
        configs.append((f"attn_single_{a}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_optimizer_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for opt in OPTIMIZERS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        training = _with_optimizer(training, opt)
        configs.append((f"opt_{opt}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_norm_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for norm in NORMS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        norm_block: Dict[str, Any] = {"type": norm}
        if norm == "prms_norm":
            norm_block["partial_ratio"] = 0.5
        if norm == "flash_norm":
            norm_block["flashnorm_partial_ratio"] = 0.0
        model = _deep_update(model, {"norm": norm_block})
        configs.append((f"norm_{norm}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_positional_encoding_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """One config per model-wide ``positional_encoding`` value.

    Uses a fixed baseline ``[standard_attn, titan_attn]`` pattern — no cross
    product with mixers.  Only the schema-defined ``positional_encoding`` and
    ``positional_encoding_parameters`` keys are touched.
    """
    configs: List[Tuple[str, dict]] = []
    for pe in POSITIONAL_ENCODINGS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        overrides: Dict[str, Any] = {"positional_encoding": pe}
        # PaPE family shares the ``pape`` parameters sub-object (num_parabolas,
        # num_positions, rotation_invariant).  Provide sane toy values.
        if pe in {"pape", "pape_efficient", "pape_ri"}:
            overrides["positional_encoding_parameters"] = {
                "pape": {"num_parabolas": 4, "num_positions": 1, "rotation_invariant": pe == "pape_ri"},
            }
        if pe == "sinusoidal_absolute":
            overrides["positional_encoding_parameters"] = {
                "sinusoidal": {"max_len": 64, "base": 10000.0, "scale": 1.0},
            }
        if pe == "learned_absolute":
            overrides["positional_encoding_parameters"] = {
                "learned": {"max_len": 64, "init_std": 0.02},
            }
        if pe == "bam":
            overrides["positional_encoding_parameters"] = {
                "bam": {"learn_mu": False, "theta_init": 0.0, "mu_init": 0.0, "eps": 1e-5},
            }
        if pe in {"rope", "hope"}:
            overrides["positional_encoding_parameters"] = {
                pe: {"base": 10000.0, "scaling": 1.0} if pe == "rope" else {"base": 10000.0, "damping": 0.01},
            }
        model = _deep_update(model, overrides)
        configs.append((f"pe_{pe}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_transversal_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []

    def add(cid: str, overrides: dict) -> None:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        model = _deep_update(model, overrides)
        model = _ensure_ssog_grid(model, training)
        configs.append((cid, {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))

    add("trans_bitnet_on", {"use_bitnet": True})
    add("trans_bitnet_routers", {"use_bitnet": True, "bitnet_routers": True, "use_moe": True, "num_experts": 2, "top_k_experts": 1})
    add("trans_bitnet_conv", {"use_bitnet": True, "use_bitnet_conv": True, "embedding": {"conv": {"enabled": True, "kernel": 3}}})
    add("trans_factorized_emb", {"embedding": {"factorized": {"enabled": True, "dim": 32}}})
    add("trans_embedding_conv", {"embedding": {"conv": {"enabled": True, "kernel": 3}}})
    add("trans_moe", {"use_moe": True, "num_experts": 2, "top_k_experts": 1})
    add("trans_mod", {"use_mixture_of_depths": True, "mixture_of_depths_capacity_ratio": 0.5})
    add("trans_residuals_none", {"residuals": {"type": "none"}})
    add("trans_residuals_full_attn", {"residuals": {"type": "full_attn", "full_attn": {"init_query_zero": True, "use_rmsnorm_keys": True}}})
    add("trans_residuals_block_attn", {"residuals": {"type": "block_attn", "block_attn": {"num_blocks": 2, "init_query_zero": True, "use_rmsnorm_keys": True}}})
    add("trans_pos_rope", {"attention": {"titan": {"positional_encoding": "rope"}}})
    add("trans_ssmax", {"use_ssmax": True, "ssmax_s_init": 1.0})
    # Fixed numerics (arXiv:2610.31642): cancellation-free softmax and
    # zero-sum-gradient hard-target cross-entropy.
    add("trans_fixed_softmax", {"fixed_softmax": True})
    add("trans_fixed_ce", {"fixed_cross_entropy": True})
    add("trans_fixed_both", {"fixed_softmax": True, "fixed_cross_entropy": True})
    # SSMax rescales logits (larger margins) and BitNet perturbs weights: the
    # interaction is exactly where the precision cliff matters.
    add("trans_fixed_softmax_ssmax_bitnet", {"fixed_softmax": True, "use_ssmax": True, "ssmax_s_init": 1.0, "use_bitnet": True})
    add("trans_num_loops_2", {"dims": {"num_layers": 1, "num_loops": 2, "layer_pattern": ["standard_attn"]}})
    add("trans_ffn_relu", {"ffn_activation": "relu"})
    add("trans_ffn_swiglu", {"ffn_activation": "swiglu"})
    add("trans_mhc", {
        "mhc": {"enabled": True, "expansion_rate": 2, "sinkhorn_iters": 5, "gating_init": 0.01, "checkpoint": False, "full_prec_under_bitnet": True},
    })
    # SSOG transversals: fixed (content-blind) field, and BitNet-wired SSOG.
    add("trans_ssog_fixed_field", {
        "dims": {"layer_pattern": ["ssog_attn", "ssog_attn"]},
        "attention": {"ssog": {"lookat": False}},
    })
    add("trans_ssog_bitnet", {
        "dims": {"layer_pattern": ["ssog_attn", "ssog_attn"]},
        "use_bitnet": True,
    })
    return configs


def make_task_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []

    # Encoder MLM (default path)
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
    configs.append(("task_mlm_encoder", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))

    # Decoder causal LM
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"mode": "decoder", "layer_pattern": ["standard_attn", "titan_attn"]}})
    training = training.copy()
    training["task"] = "causal_lm"
    configs.append(("task_causal_lm_decoder", {"model_class": "frankenstein_text_decoder", "model": model, "training": training}))

    # Decoder causal LM with the Conformer mixer: the convolution module must
    # switch to causal (left) padding in ``mode: decoder`` so training is
    # streaming-safe.
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {
        "dims": {"mode": "decoder", "layer_pattern": ["conformer_attn", "standard_attn"]},
        "attention": {"conformer": {"kernel_size": 8, "macaron": True, "conv_norm": "batch"}},
    })
    training = training.copy()
    training["task"] = "causal_lm"
    configs.append(("task_causal_lm_conformer", {"model_class": "frankenstein_text_decoder", "model": model, "training": training}))

    # Decoder causal LM with fixed numerics: exercises the ignore_index=-100
    # path of the fixed cross-entropy and the decoder softmax/sampling plumbing.
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"mode": "decoder", "layer_pattern": ["standard_attn", "titan_attn"]}})
    model = _deep_update(model, {"fixed_softmax": True, "fixed_cross_entropy": True})
    training = training.copy()
    training["task"] = "causal_lm"
    configs.append(("task_causal_lm_fixed_numerics", {"model_class": "frankenstein_text_decoder", "model": model, "training": training}))
    return configs


def make_batch_size_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for bs in [1, 2]:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        training = training.copy()
        training["batch_size"] = bs
        configs.append((f"batch_size_{bs}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def _vit_model(vocab_size: int) -> dict:
    """Minimal ``model:`` block for a frankenstein_vit (Vision Transformer)."""
    return {
        "dims": {
            "hidden_size": 64, "num_layers": 2, "num_heads": 4, "num_loops": 1,
            "layer_pattern": ["standard_attn"], "mode": "encoder", "dropout": 0.0,
            "vocab_size": vocab_size,
        },
        "norm": {"type": "layer_norm"},
        "embedding": {},
        "attention": {},
        "use_moe": False,
        "use_bitnet": False,
        "ffn_activation": "gelu",
        "ffn_hidden_size": 256,
    }


def _vit_training(task: str) -> dict:
    """Minimal ``training:`` block for a frankenstein_vit task."""
    return {
        "task": task,
        "batch_size": 2,
        "num_epochs": 1,
        "optimizer": {
            "optimizer_class": "adamw",
            "parameters": {
                "adamw-lr_embeddings": 1e-4,
                "adamw-lr_norms": 1e-4,
                "adamw-lr_attention": 1e-4,
                "adamw-lr_other": 1e-4,
                "adamw-wd_embeddings": 0.0,
                "adamw-wd_norms": 0.0,
                "adamw-wd_attention": 0.0,
                "adamw-wd_other": 0.0,
            },
        },
        task: {"batch_size": 2, "num_epochs": 1, "learning_rate": 1e-4},
    }


def make_vision_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Build frankenstein_vit (Vision Transformer) e2e configs for all 3 tasks."""
    configs: List[Tuple[str, dict]] = []

    base_image = {
        "image_size": {"height": 32, "width": 32},
        "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
        "cls_token": True, "pooling_mode": "cls",
    }
    base_dataset = {"rescale": {"height": 32, "width": 32}}

    # Classification
    image = dict(base_image, num_classes=10)
    configs.append(("vit_classification", {
        "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("classification"),
    }))

    # Patch prediction (autosupervised)
    image = dict(base_image, mask_ratio=0.5, mask_token_strategy="bert",
                 prediction_target="mean_color_3bit")
    configs.append(("vit_patch_prediction", {
        "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("patch_prediction"),
    }))

    # Segmentation (pixel head)
    image = dict(base_image, seg_head_type="pixel", num_seg_classes=5)
    configs.append(("vit_segmentation_pixel", {
        "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("segmentation"),
    }))

    # Fixed numerics (arXiv:2610.31642) on the two CE+softmax vision paths:
    # classification CE and segmentation CE + Dice softmax.
    fixed_vit = _deep_update(_vit_model(vocab_size), {
        "fixed_softmax": True, "fixed_cross_entropy": True,
    })
    image = dict(base_image, num_classes=10)
    configs.append(("vit_classification_fixed_numerics", {
        "model_class": "frankenstein_vit", "model": fixed_vit,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("classification"),
    }))
    image = dict(base_image, seg_head_type="pixel", num_seg_classes=5)
    configs.append(("vit_segmentation_fixed_numerics", {
        "model_class": "frankenstein_vit", "model": fixed_vit,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("segmentation"),
    }))

    # SSOG variants — the Gaussian field needs the raw patch raster, so no
    # [CLS] token (gap pooling instead). The grid auto-derives from the image:
    # (32/16)x(32/16) = 2x2.
    ssog_image = {
        "image_size": {"height": 32, "width": 32},
        "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
        "cls_token": False, "pooling_mode": "gap",
    }
    ssog_model = _deep_update(_vit_model(vocab_size), {
        "dims": {"layer_pattern": ["ssog_attn"]},
        "attention": {"ssog": {"num_atoms": 2, "max_offset": 1.0}},
    })

    image = dict(ssog_image, num_classes=10)
    configs.append(("vit_classification_ssog", {
        "model_class": "frankenstein_vit", "model": ssog_model,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("classification"),
    }))

    image = dict(ssog_image, mask_ratio=0.5, mask_token_strategy="bert",
                 prediction_target="mean_color_3bit")
    configs.append(("vit_patch_prediction_ssog", {
        "model_class": "frankenstein_vit", "model": ssog_model,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("patch_prediction"),
    }))

    image = dict(ssog_image, seg_head_type="pixel", num_seg_classes=5)
    configs.append(("vit_segmentation_pixel_ssog", {
        "model_class": "frankenstein_vit", "model": ssog_model,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("segmentation"),
    }))

    # Conformer mixer inside the ViT: the patch raster is treated as the
    # sequence axis for the convolution module. Uses the canonical example
    # preset (`configs/examples/vit_conformer.yaml`), shrunk to a tiny batch.
    configs.append(("vit_conformer", _load_example_config(
        "vit_conformer",
        training={"batch_size": 4, "num_epochs": 1, "dataloader_workers": 0},
    )))

    return configs


def make_vision_pe_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Sweep every ViT ``pos_embedding_type`` across all 3 vision tasks.

    No cross product with mixers — each combo is one (task, PE) pair using the
    default ``standard_attn`` ViT layer pattern.  Mirrors the strategy of
    ``make_positional_encoding_configs`` but for the vision image block.
    """
    configs: List[Tuple[str, dict]] = []
    base_image = {
        "image_size": {"height": 32, "width": 32},
        "patch_size": 16, "in_channels": 3, "cls_token": True, "pooling_mode": "cls",
    }
    base_dataset = {"rescale": {"height": 32, "width": 32}}

    for task, image_extras in [
        ("classification", {"num_classes": 10}),
        ("patch_prediction", {"mask_ratio": 0.5, "mask_token_strategy": "bert",
                             "prediction_target": "mean_color_3bit"}),
        ("segmentation", {"seg_head_type": "pixel", "num_seg_classes": 5}),
    ]:
        for pe in VISION_PE_TYPES:
            image = dict(base_image, pos_embedding_type=pe, **image_extras)
            configs.append((f"vit_pe_{task}_{pe}", {
                "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
                "image": image, "vision_dataset": base_dataset,
                "training": _vit_training(task),
            }))
    return configs


def make_audio_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Build ``frankenstein_audio`` cases for ASR, classification and raw EEG.

    All four cases reuse the canonical example presets
    (``configs/examples/audio_*.yaml``) with two offline/tiny overrides:

    * ``audio_dataset.dataset_name``/``dataset_dir`` are cleared, so the
      engine falls back to :class:`DummyAudioDataset` (synthetic waves and
      transcripts) and never touches the network;
    * the training block is reduced to a single optimizer step
      (``batch_size=16`` over 16 dummy examples, ``gradient_accumulation_steps=1``,
      ``num_epochs=1``). The accumulation override is required: with the
      examples' default of 4 no step completes in a one-batch epoch and no
      checkpoint is written for the deploy smoke.

    Args:
        vocab_size: Unused (the audio presets carry their own ``vocab_size``);
            accepted so every config maker has the same signature.

    Returns:
        List of ``(combo_id, config_dict)`` tuples.
    """
    del vocab_size  # audio presets define their own vocabulary size

    offline_dataset = {"dataset_name": "", "dataset_dir": ""}
    tiny_training = {
        "batch_size": 16,
        "num_epochs": 1,
        "dataloader_workers": 0,
        "gradient_accumulation_steps": 1,
    }

    def audio_case(combo_id: str, example: str, **overrides: Any) -> Tuple[str, dict]:
        config = _load_example_config(
            example,
            audio_dataset=offline_dataset,
            training=tiny_training,
            **overrides,
        )
        return combo_id, config

    return [
        audio_case("audio_asr_ctc", "audio_asr_conformer"),
        audio_case("audio_asr_transducer", "audio_asr_transducer"),
        audio_case("audio_classification", "audio_classification_conformer"),
        audio_case("audio_eeg_classification", "audio_eeg_classification"),
    ]


def make_tokenizer_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Tokenizer source modes: HF-repo load vs train-from-dataset (all algorithms).

    Exercises the top-level ``tokenizer`` block on the CUSTOM (non-base_model)
    path:

    * ``source: hf_repo`` — load a pretrained tokenizer from a repo id; the
      vocab size is injected into ``model.dims.vocab_size``.
    * ``source: train_from_dataset`` — train a NEW tokenizer from the toy
      parquet corpus (``text_dataset`` block) with each of the 4 algorithms
      (bpe, wordpiece, wordlevel, unigram).
    """
    configs: List[Tuple[str, dict]] = []

    for algo in ["bpe", "wordpiece", "wordlevel", "unigram"]:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "standard_attn"]}})
        training = training.copy()
        training["task"] = "mlm"
        tok_training: Dict[str, Any] = {
            "algorithm": algo,
            "min_frequency": 1,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            "save_dir": "trained_tokenizer",
        }
        if algo != "unigram":
            tok_training["vocab_size"] = vocab_size
        configs.append((f"tokenizer_train_{algo}", {
            "model_class": "frankenstein_text_encoder",
            "model": model,
            "tokenizer": {
                "source": "train_from_dataset",
                "training": tok_training,
            },
            "text_dataset": {
                "data_dir": str(helpers.DATA_DIR),
                "text_column": "text",
                "max_samples": 200,
            },
            "training": training,
        }))

    return configs


# ---------------------------------------------------------------------------
# Real-data combos (HuggingFace Hub datasets)
# ---------------------------------------------------------------------------
def make_real_dataset_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Build one real-data combo per task family, against HF Hub datasets.

    Every combo exercises the *actual* dataset field of its task:

    * ``text_dataset.dataset_name`` — MLM, causal LM and text classification.
      The text column of ``adleme94/borges`` is ``poem`` (not ``text``), and
      ``C-L-V/mini_corpus_sentiment_es`` carries string labels
      (``negativo``/``neutral``/``positivo`` → 3 classes via
      ``model.classification_head``).
    * ``vision_dataset.dataset_name`` — the teeth X-ray repo stores the
      segmentation mask in the ``label`` column (an Image), so
      ``segmentation_column: label`` is required; the breast-cancer repo is a
      2-class ClassLabel and also feeds the patch-prediction combo.
    * ``audio_dataset.dataset_name`` — ``astroza/chilean-jokes...`` only has a
      ``test`` split; ``danielquillanroxas/EMOTION_VOICES`` has 7 emotion
      classes (``audio.num_classes: 7``).
    * ``training.sbert.dataset_name`` — Spanish QQP triplets; SBERT training
      hard-requires a top-level ``base_model``.

    All combos are tiny (1 epoch, small batches, ``max_samples`` caps) but
    they download from the HuggingFace Hub — use a generous ``--timeout``.

    Args:
        vocab_size: Toy vocab size used only for the harness' tokenizer copy;
            the engine injects the real tokenizer vocab into the NLP models.

    Returns:
        List of ``(combo_id, config_dict)`` tuples.
    """
    configs: List[Tuple[str, dict]] = []

    hf_tokenizer = {"source": "hf_repo", "name_or_path": REALDATA_TEXT_TOKENIZER}

    # --- MLM on Spanish poetry (Borges) -----------------------------------
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
    training = _deep_update(training, {"task": "mlm", "batch_size": 2, "max_length": 64})
    configs.append(("realdata_mlm_borges", {
        "model_class": "frankenstein_text_encoder",
        "model": model,
        "tokenizer": hf_tokenizer,
        "text_dataset": {
            "dataset_name": "adleme94/borges",
            "split": "train",
            "text_column": "poem",
            "max_samples": 32,
        },
        "training": training,
    }))

    # --- Autoregressive causal LM on the same corpus -----------------------
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {
        "dims": {"mode": "decoder", "layer_pattern": ["standard_attn", "titan_attn"]},
    })
    training = _deep_update(training, {"task": "causal_lm", "batch_size": 2, "max_length": 64})
    configs.append(("realdata_causal_lm_borges", {
        "model_class": "frankenstein_text_decoder",
        "model": model,
        "tokenizer": hf_tokenizer,
        "text_dataset": {
            "dataset_name": "adleme94/borges",
            "split": "train",
            "text_column": "poem",
            "max_samples": 32,
        },
        "training": training,
    }))

    # --- Supervised text classification (3 string labels) ------------------
    model, training = _base_config(vocab_size, num_layers=1, num_loops=1)
    model = _deep_update(model, {
        "dims": {"layer_pattern": ["standard_attn"]},
        "classification_head": {"enabled": True, "num_labels": 3, "pooling_mode": "cls"},
    })
    training = _deep_update(training, {"task": "text_classification", "batch_size": 2, "max_length": 64})
    configs.append(("realdata_text_classification_mini_corpus", {
        "model_class": "frankenstein_text_encoder",
        "model": model,
        "tokenizer": hf_tokenizer,
        "text_dataset": {
            "dataset_name": "C-L-V/mini_corpus_sentiment_es",
            "split": "train",
            "text_column": "text",
            "label_column": "label",
            "use_labels": True,
            "max_samples": 50,
        },
        "training": training,
    }))

    # --- ViT segmentation: teeth panoramic X-rays --------------------------
    # The mask lives in the ``label`` column (an Image), not the default
    # ``segmentation_map`` — override segmentation_column. Binary masks.
    configs.append(("realdata_vit_segmentation_teeth", {
        "model_class": "frankenstein_vit",
        "model": _vit_model(vocab_size),
        "image": {
            "image_size": {"height": 64, "width": 64},
            "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
            "cls_token": True, "pooling_mode": "cls",
            "seg_head_type": "pixel", "num_seg_classes": 2,
        },
        "vision_dataset": {
            "dataset_name": "SerdarHelli/SegmentationOfTeethPanoramicXRayImages",
            "split": "train",
            "image_column": "image",
            "segmentation_column": "label",
        },
        "training": _vit_training("segmentation"),
    }))

    # --- ViT classification: breast cancer ultrasound (2 classes) ----------
    configs.append(("realdata_vit_classification_breastcancer", {
        "model_class": "frankenstein_vit",
        "model": _vit_model(vocab_size),
        "image": {
            "image_size": {"height": 64, "width": 64},
            "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
            "cls_token": True, "pooling_mode": "cls",
            "num_classes": 2,
        },
        "vision_dataset": {
            "dataset_name": "as-cle-bert/breastcanc-ultrasound-class",
            "split": "train",
            "image_column": "image",
            "label_column": "label",
        },
        "training": _vit_training("classification"),
    }))

    # --- ViT patch prediction: autosupervised on the same corpus -----------
    configs.append(("realdata_vit_patch_prediction_breastcancer", {
        "model_class": "frankenstein_vit",
        "model": _vit_model(vocab_size),
        "image": {
            "image_size": {"height": 64, "width": 64},
            "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
            "cls_token": True, "pooling_mode": "cls",
            "mask_ratio": 0.5, "mask_token_strategy": "bert",
            "prediction_target": "mean_color_3bit",
        },
        "vision_dataset": {
            "dataset_name": "as-cle-bert/breastcanc-ultrasound-class",
            "split": "train",
            "image_column": "image",
        },
        "training": _vit_training("patch_prediction"),
    }))

    # --- Audio ASR: Chilean jokes (only a `test` split exists) -------------
    tiny_training = {
        "batch_size": 4,
        "num_epochs": 1,
        "dataloader_workers": 0,
        "gradient_accumulation_steps": 1,
    }
    configs.append(("realdata_audio_asr_vina", _load_example_config(
        "audio_asr_conformer",
        audio_dataset={
            "dataset_name": "astroza/chilean-jokes-festival-de-vina",
            "split": "test",
            "audio_column": "audio",
            "text_column": "text",
            "max_duration_sec": 10.0,
            "resample": True,
        },
        training=tiny_training,
    )))

    # --- Audio classification: 7 emotion classes ---------------------------
    configs.append(("realdata_audio_classification_emotion_voices", _load_example_config(
        "audio_classification_conformer",
        audio={"num_classes": 7},
        audio_dataset={
            "dataset_name": "danielquillanroxas/EMOTION_VOICES",
            "split": "train",
            "audio_column": "audio",
            "label_column": "label",
            "max_duration_sec": 10.0,
            "resample": True,
        },
        training=tiny_training,
    )))

    # --- SBERT triplets (requires a top-level base_model) ------------------
    # The dataset's NEGs column is a stringified list; the trainer parses it
    # via ast.literal_eval (see tests/test_sbert_triplets_stringified_*).
    configs.append(("realdata_sbert_qqp_triplets_es", {
        "base_model": REALDATA_SBERT_BASE_MODEL,
        # Ignored by the loader when base_model is set; present only so the
        # harness' run_training helper finds dims.vocab_size for its toy-SPM
        # copy.
        "model": {"dims": {"vocab_size": vocab_size}},
        "training": {
            "task": "sbert",
            "sbert": {
                "dataset_name": "erickfmm/embedding-data__QQP_triplets__translated_to_spanish",
                "dataset_type": "triplets",
                "columns": {
                    "query": "Q_traducida",
                    "positive": "POS_traducida",
                    "negatives": "NEGs_traducidas",
                },
                "output_dir": "./output/sbert_qqp_triplets_es",
                "batch_size": 16,
                "num_epochs": 1,
                "learning_rate": 2.0e-5,
                "max_train_samples": 64,
                "max_eval_samples": 16,
                "max_length": 128,
                "pooling_mode": "mean",
                "use_amp": False,
                "trust_remote_code": False,
            },
        },
    }))

    return configs


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exhaustive end-to-end smoke tests for frankenstein-transformer.",
    )
    parser.add_argument(
        "--category",
        choices=["all", "attn", "opt", "norm", "pe", "transversal", "task", "batch_size", "vision", "vision_pe", "audio", "tokenizer", "realdata", "deploy"],
        default="all",
        help="Which sweep to run (default: all).",
    )
    parser.add_argument(
        "--logging-level",
        choices=["none", "info", "debug", "only-errors"],
        default="info",
        help="How much progress logging to emit to stdout (default: info).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N training combos (useful for a quick smoke run).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-combo training timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--skip-attn-pairs",
        action="store_true",
        help="Skip the expensive attention-pair sweep.",
    )
    parser.add_argument(
        "--skip-singles",
        action="store_true",
        help="Skip the single-attention sweep.",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip deploy/infer/export smoke tests.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=helpers.TOY_VOCAB_SIZE,
        help="Toy tokenizer vocab size (default: 256).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run training/deploy/infer on (default: cpu).",
    )
    parser.add_argument(
        "--gpu-temp-guard",
        action="store_true",
        help="Enable the GPU thermal guard during training (only meaningful on cuda).",
    )
    parser.add_argument(
        "--gpu-temp-pause-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) at which training pauses (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-resume-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) at which training resumes (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-critical-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) considered critical (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-poll-interval-seconds",
        type=float,
        default=None,
        help="GPU temp polling interval in seconds (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-checkpoint-grace-seconds",
        type=float,
        default=None,
        help="Grace period in seconds before a thermal checkpoint (default: CLI default).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing tmp/results directory instead of cleaning up first.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    log_level = {
        "none": logging.CRITICAL + 10,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "only-errors": logging.ERROR,
    }[args.logging_level]
    helpers.setup_logging(log_level)

    # Clean up tmp directory unless --resume is specified
    if not args.resume and helpers.TMP_DIR.exists():
        logging.info("Cleaning up tmp directory: %s", helpers.TMP_DIR)
        shutil.rmtree(helpers.TMP_DIR)

    helpers.TMP_DIR.mkdir(parents=True, exist_ok=True)
    helpers.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    helpers.write_sitecustomize(helpers.TMP_DIR)
    tokenizer_path = helpers.ensure_toy_tokenizer(args.vocab_size)
    parquet_path = helpers.ensure_toy_parquet()
    runner, env_extra = helpers.resolve_runner()

    logging.info("=" * 60)
    logging.info("Frankenstein exhaustive end-to-end smoke tests")
    logging.info("Tokenizer: %s", tokenizer_path)
    logging.info("Parquet:   %s", parquet_path)
    logging.info("Runner:    %s", " ".join(runner))
    logging.info("=" * 60)

    combos: List[Tuple[str, dict]] = []

    if args.category in {"all", "attn"}:
        if not args.skip_attn_pairs:
            combos.extend(make_attention_pair_configs(args.vocab_size))
        if not args.skip_singles:
            combos.extend(make_single_attention_configs(args.vocab_size))
    if args.category in {"all", "opt"}:
        combos.extend(make_optimizer_configs(args.vocab_size))
    if args.category in {"all", "norm"}:
        combos.extend(make_norm_configs(args.vocab_size))
    if args.category in {"all", "pe"}:
        combos.extend(make_positional_encoding_configs(args.vocab_size))
    if args.category in {"all", "transversal"}:
        combos.extend(make_transversal_configs(args.vocab_size))
    if args.category in {"all", "task"}:
        combos.extend(make_task_configs(args.vocab_size))
    if args.category in {"all", "batch_size"}:
        combos.extend(make_batch_size_configs(args.vocab_size))
    if args.category in {"all", "vision"}:
        combos.extend(make_vision_configs(args.vocab_size))
    if args.category in {"all", "vision_pe"}:
        combos.extend(make_vision_pe_configs(args.vocab_size))
    if args.category in {"all", "audio"}:
        combos.extend(make_audio_configs(args.vocab_size))
    if args.category in {"all", "tokenizer"}:
        combos.extend(make_tokenizer_configs(args.vocab_size))
    if args.category in {"all", "realdata"}:
        combos.extend(make_real_dataset_configs(args.vocab_size))
        if args.timeout <= 600:
            logging.warning(
                "realdata combos download models/corpora from the HuggingFace "
                "Hub; consider --timeout 1800 (current: %ds)", args.timeout,
            )

    if args.limit is not None:
        combos = combos[: args.limit]

    logging.info("Planning to run %d training combos", len(combos))
    logging.debug("Category=%s limit=%s skip_attn_pairs=%s skip_singles=%s skip_deploy=%s",
                  args.category, args.limit, args.skip_attn_pairs, args.skip_singles, args.skip_deploy)

    results: List[helpers.RunResult] = []
    successful_training: helpers.RunResult | None = None

    train_kwargs = dict(
        runner=runner,
        env_extra=env_extra,
        timeout=args.timeout,
        device=args.device,
        gpu_temp_guard=args.gpu_temp_guard,
        gpu_temp_pause_threshold_c=args.gpu_temp_pause_threshold_c,
        gpu_temp_resume_threshold_c=args.gpu_temp_resume_threshold_c,
        gpu_temp_critical_threshold_c=args.gpu_temp_critical_threshold_c,
        gpu_temp_poll_interval_seconds=args.gpu_temp_poll_interval_seconds,
        gpu_temp_checkpoint_grace_seconds=args.gpu_temp_checkpoint_grace_seconds,
    )

    def write_results_incremental() -> None:
        """Write results to disk immediately (for crash recovery)."""
        helpers.write_results(results)

    def find_metrics_csv(combo_id: str) -> Optional[str]:
        """Find the training_metrics.csv file for a given combo."""
        run_dir = helpers.RUNS_DIR / combo_id
        metrics_path = run_dir / "training_metrics.csv"
        if metrics_path.exists():
            return str(metrics_path)
        return None

    for idx, (combo_id, config) in enumerate(combos, start=1):
        logging.info("[%d/%d] Running %s", idx, len(combos), combo_id)
        result = helpers.run_training(
            config=config,
            combo_id=combo_id,
            batch_size=config["training"].get("batch_size", 1),
            **train_kwargs,
        )
        # Capture metrics CSV path if training produced one
        if result.status in ("OK", "GRAD_EXPLODED"):
            metrics_path = find_metrics_csv(combo_id)
            if metrics_path:
                result.metrics_path = metrics_path
                logging.info("  Metrics saved to: %s", metrics_path)
        results.append(result)
        logging.info("[%d/%d] %s -> %s (%.1fs) %s", idx, len(combos), combo_id, result.status, result.duration_sec, result.notes)

        # Write results incrementally after each combo
        write_results_incremental()

        if result.status == "OK" and successful_training is None:
            successful_training = result

    # Deploy / infer / export smoke tests, using the first successful training
    # run. Audio checkpoints follow the dedicated path below: the text `infer`
    # CLI does not apply to them and Transformers export must be rejected.
    def _is_audio_combo(combo_id: str) -> bool:
        return combo_id.startswith("audio_") or combo_id.startswith("realdata_audio_")

    if (
        not args.skip_deploy
        and successful_training is not None
        and not _is_audio_combo(successful_training.combo_id)
    ):
        checkpoint_path = Path(successful_training.checkpoint_path) if successful_training.checkpoint_path else None
        if checkpoint_path and checkpoint_path.exists():
            base_id = successful_training.combo_id
            vocab_size = args.vocab_size

            # standard deploy
            deploy_std_dir = helpers.TMP_DIR / "deploy" / f"{base_id}_standard"
            logging.info("Running deploy (standard) %s -> %s", base_id, deploy_std_dir)
            res = helpers.run_deploy(checkpoint_path, deploy_std_dir, f"{base_id}_deploy_std", runner, env_extra, fmt="standard", device=args.device)
            results.append(res)
            write_results_incremental()
            logging.info("%s -> %s (%.1fs) %s", res.combo_id, res.status, res.duration_sec, res.notes)
            if res.status == "OK":
                helpers.copy_tokenizer_to_deploy_dir(
                    deploy_std_dir, vocab_size,
                    run_dir=helpers.RUNS_DIR / base_id,
                )
                results.append(helpers.run_infer(deploy_std_dir, f"{base_id}_infer", runner, env_extra, device=args.device))
                write_results_incremental()
                logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

            # quantized deploy
            deploy_q_dir = helpers.TMP_DIR / "deploy" / f"{base_id}_quantized"
            logging.info("Running deploy (quantized) %s -> %s", base_id, deploy_q_dir)
            results.append(helpers.run_deploy(checkpoint_path, deploy_q_dir, f"{base_id}_deploy_quantized", runner, env_extra, fmt="quantized", device=args.device))
            write_results_incremental()
            logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

            # transformers-export (needs the original YAML)
            yaml_path = helpers.RUNS_DIR / base_id / "config.yaml"
            if yaml_path.exists():
                export_dir = helpers.TMP_DIR / "transformers-export" / base_id
                logging.info("Running transformers-export %s -> %s", base_id, export_dir)
                results.append(helpers.run_transformers_export(checkpoint_path, yaml_path, export_dir, f"{base_id}_export", runner, env_extra))
                write_results_incremental()
                logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

    # Audio deploy smoke: `standard`/`quantized` must load the audio checkpoint
    # and run the validation forward pass. There is no text-style `infer` for
    # audio, no GGUF export, and the Transformers exporter must reject audio
    # models with the documented message.
    if not args.skip_deploy and args.category in {"all", "audio", "realdata"}:
        audio_result = next(
            (
                result
                for result in results
                if _is_audio_combo(result.combo_id)
                and result.status == "OK"
                and result.checkpoint_path
                and Path(result.checkpoint_path).exists()
            ),
            None,
        )
        if audio_result is None:
            logging.warning(
                "No successful audio training run with a checkpoint; skipping the audio deploy smoke"
            )
        else:
            audio_checkpoint = Path(audio_result.checkpoint_path)
            audio_id = audio_result.combo_id
            for fmt in ("standard", "quantized"):
                audio_deploy_dir = helpers.TMP_DIR / "deploy" / f"{audio_id}_{fmt}"
                logging.info(
                    "Running audio deploy (%s) %s -> %s", fmt, audio_id, audio_deploy_dir
                )
                results.append(helpers.run_deploy(
                    audio_checkpoint,
                    audio_deploy_dir,
                    f"{audio_id}_deploy_{fmt}",
                    runner,
                    env_extra,
                    fmt=fmt,
                    device=args.device,
                ))
                write_results_incremental()
                logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

            audio_yaml_path = helpers.RUNS_DIR / audio_id / "config.yaml"
            if audio_yaml_path.exists():
                export_dir = helpers.TMP_DIR / "transformers-export" / f"{audio_id}_rejected"
                logging.info(
                    "Running transformers-export (expected rejection) %s -> %s",
                    audio_id,
                    export_dir,
                )
                export_result = helpers.run_transformers_export(
                    audio_checkpoint,
                    audio_yaml_path,
                    export_dir,
                    f"{audio_id}_export_expected_reject",
                    runner,
                    env_extra,
                )
                combined = f"{export_result.stdout}\n{export_result.stderr}".lower()
                rejected_as_documented = (
                    export_result.status == "FAILED"
                    and "frankenstein_audio" in combined
                    and "not currently exportable" in combined
                )
                if rejected_as_documented:
                    export_result.status = "OK"
                    export_result.notes = (
                        "audio transformers export rejected with the documented message"
                    )
                else:
                    export_result.status = "FAILED"
                    export_result.notes = (
                        "audio transformers export was not rejected as documented "
                        f"(returncode={export_result.returncode})"
                    )
                results.append(export_result)
                write_results_incremental()
                logging.info("%s -> %s (%.1fs) %s", export_result.combo_id, export_result.status, export_result.duration_sec, export_result.notes)

    # Dedicated BitNet+standard_attn training + bitnet-gguf export, if not already covered.
    if not args.skip_deploy and (args.category in {"all", "transversal"}):
        model, training = _base_config(args.vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {
            "dims": {"layer_pattern": ["standard_attn", "standard_attn"]},
            "use_bitnet": True,
        })
        bitnet_id = "bitnet_std_attn_for_gguf"
        logging.info("Running dedicated %s training", bitnet_id)
        bitnet_result = helpers.run_training(
            config={"model_class": "frankenstein_text_encoder", "model": model, "training": training},
            combo_id=bitnet_id,
            batch_size=1,
            **train_kwargs,
        )
        # Capture metrics for bitnet training too
        if bitnet_result.status in ("OK", "GRAD_EXPLODED"):
            metrics_path = find_metrics_csv(bitnet_id)
            if metrics_path:
                bitnet_result.metrics_path = metrics_path
                logging.info("  Metrics saved to: %s", metrics_path)
        results.append(bitnet_result)
        write_results_incremental()
        if bitnet_result.status == "OK" and bitnet_result.checkpoint_path:
            yaml_path = helpers.RUNS_DIR / bitnet_id / "config.yaml"
            gguf_path = helpers.TMP_DIR / "bitnet-gguf" / f"{bitnet_id}.gguf"
            logging.info("Running bitnet-gguf %s -> %s", bitnet_id, gguf_path)
            results.append(helpers.run_bitnet_gguf(
                Path(bitnet_result.checkpoint_path),
                yaml_path,
                gguf_path,
                f"{bitnet_id}_gguf",
                runner,
                env_extra,
            ))
            write_results_incremental()
            logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

    # Final write (ensures everything is flushed)
    helpers.write_results(results)
    return helpers.print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
