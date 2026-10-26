#!/usr/bin/env python3
"""Flat inference-time runtime configuration.

:class:`InferenceConfig` is the flat, code-side representation of the
schema ``inference:`` YAML block (source of truth:
``src/schema/_inference.yaml``). Like ``training.*`` it is runtime
configuration and is deliberately **not** part of
:class:`~src.model.config.FrankensteinModelConfig` (architectural config).

Precedence (D2): **CLI flags > YAML ``inference:`` > schema defaults**.
Every argparse flag uses ``default=None`` and
:func:`apply_cli_overrides` patches only explicitly-passed keys
(leaf-level shallow merge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["InferenceConfig", "apply_cli_overrides", "inference_config_to_dict"]


@dataclass
class InferenceConfig:
    """Resolved inference-time configuration (generation + KV cache).

    Attributes:
        max_new_tokens: Maximum tokens to generate. Default 64.
        temperature: Sampling temperature. Default 1.0.
        top_k: Top-k filtering (0 disables). Default 50.
        top_p: Nucleus mass (1.0 disables). Default 1.0.
        greedy: Argmax decoding override. Default False.
        kv_cache_enabled: Opt-in incremental decoding (D4). Default False.
        kv_quant_method: One of ``none|int8|int4_kivi``. Default ``none``.
        kv_k_bits: Key bit width (int4_kivi). Default 4.
        kv_v_bits: Value bit width (int4_kivi). Default 4.
        kv_group_size: Tokens per quantization group (int4_kivi). Default 64.
        kv_protected_window: Recent full-precision tokens (int4_kivi). Default 32.
        kv_compress_method: One of ``none|window_sinks``. Default ``none``.
        kv_num_sinks: Pinned leading positions. Default 4.
        kv_window: Sliding window size. Default 512.
        kv_budget: Scalar token budget per layer (None = unlimited).
        kv_per_layer_budgets: Per-layer budget vector (overrides scalar).
        kv_per_mixer: Per-mixer override dicts (``{"mixer": str, "enabled": bool}``).
    """

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    greedy: bool = False
    kv_cache_enabled: bool = False
    kv_quant_method: str = "none"
    kv_k_bits: int = 4
    kv_v_bits: int = 4
    kv_group_size: int = 64
    kv_protected_window: int = 32
    kv_compress_method: str = "none"
    kv_num_sinks: int = 4
    kv_window: int = 512
    kv_budget: Optional[int] = None
    kv_per_layer_budgets: List[int] = field(default_factory=list)
    kv_per_mixer: List[Dict[str, Any]] = field(default_factory=list)


def apply_cli_overrides(cfg: InferenceConfig, args) -> InferenceConfig:
    """Patch ``cfg`` with explicitly-passed CLI flags (CLI > YAML > default, D2).

    Every flag must use ``argparse`` ``default=None`` so "not provided" is
    distinguishable from a real value. Only the keys present on ``args``
    (and not ``None``/``False``-from-absence) are applied.

    Args:
        cfg: The :class:`InferenceConfig` loaded from YAML/defaults.
        args: Parsed argparse namespace carrying the inference flags.

    Returns:
        The patched :class:`InferenceConfig` (same instance).
    """
    mapping = {
        "max_new_tokens": "max_new_tokens",
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "greedy": "greedy",
        "kv_quant": "kv_quant_method",
        "kv_group_size": "kv_group_size",
        "kv_protected_window": "kv_protected_window",
        "kv_compress": "kv_compress_method",
        "kv_num_sinks": "kv_num_sinks",
        "kv_window": "kv_window",
    }
    for arg_name, cfg_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, cfg_name, value)
    # Boolean flags: store_true with default None means "explicitly passed".
    kv_flag = getattr(args, "kv_cache", None)
    if kv_flag is not None:
        cfg.kv_cache_enabled = bool(kv_flag)
    return cfg


def inference_config_to_dict(cfg: InferenceConfig) -> Dict[str, Any]:
    """Serialize to the nested ``inference:`` shape stored in ``config.json``."""
    return {
        "generation": {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "greedy": cfg.greedy,
        },
        "kv_cache": {
            "enabled": cfg.kv_cache_enabled,
            "quantization": {
                "method": cfg.kv_quant_method,
                "k_bits": cfg.kv_k_bits,
                "v_bits": cfg.kv_v_bits,
                "group_size": cfg.kv_group_size,
                "protected_window": cfg.kv_protected_window,
            },
            "compression": {
                "method": cfg.kv_compress_method,
                "num_sinks": cfg.kv_num_sinks,
                "window": cfg.kv_window,
                "budget": cfg.kv_budget,
                "per_layer_budgets": list(cfg.kv_per_layer_budgets),
            },
            "per_mixer": [dict(e) for e in cfg.kv_per_mixer],
        },
    }
