"""Plugin authoring API.

Plugins register declarative component specs through these helpers — the exact
same path the built-in specs use. A plugin module typically calls one of the
``register_*`` functions at import time; the loader imports it before schema
assembly, so the plugin's names and parameter fragments validate strictly and
appear in the Streamlit builder automatically.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from src.registry.spec import (
    ActivationSpec,
    CacheSpec,
    EmbeddingSpec,
    MixerSpec,
    NormSpec,
    OptimizerSpec,
    ParamBinding,
    PESpec,
    ResidualSpec,
)
from src.registry import register_spec

__all__ = [
    "register_mixer",
    "register_optimizer",
    "register_norm",
    "register_activation",
    "register_residual",
    "register_posenc",
    "register_embedding",
]


def register_mixer(
    name: str,
    builder: Any,
    description: str,
    description_es: str,
    *,
    params_schema: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    references: Sequence[str] = (),
    training_free: bool = False,
    needs_pos_encoder: bool = True,
    cache: Optional[CacheSpec] = None,
    cache_implemented: bool = False,
    qk_norm_supported: bool = False,
    qk_clip_supported: bool = False,
    aliases: Sequence[str] = (),
) -> None:
    """Register a sequence mixer.

    Args:
        name: Registry name (snake_case).
        builder: ``"module:attribute"`` or zero-argument callable.
        description: English one-row description.
        description_es: Spanish one-row description.
        params_schema: Inline JSON-Schema fragment for ``attention.<name>``.
        defaults: Runtime parameter defaults.
        references: arXiv identifiers.
        training_free: Trains as dense attention, sparse only at eval.
        needs_pos_encoder: Default for the per-mixer ``use_pe`` flag.
        cache: KV-cache capability (``None`` = never cached).
        cache_implemented: Incremental branch implemented and parity-tested.
        qk_norm_supported: Supports forward-time QK normalization / QuacK.
        qk_clip_supported: Forms per-head logits (supports QK-Clip).
        aliases: Alternative names.
    """
    register_spec(
        MixerSpec(
            name=name,
            builder=builder,
            description=description,
            description_es=description_es,
            params_schema=params_schema,
            defaults=dict(defaults or {}),
            references=tuple(references),
            training_free=training_free,
            needs_pos_encoder=needs_pos_encoder,
            cache=cache,
            cache_implemented=cache_implemented,
            qk_norm_supported=qk_norm_supported,
            qk_clip_supported=qk_clip_supported,
            aliases=tuple(aliases),
        )
    )


def register_optimizer(
    name: str,
    builder: Any,
    description: str,
    description_es: str,
    *,
    param_bindings: Optional[Mapping[str, ParamBinding]] = None,
    supports_shared_per_group_suffixes: bool = True,
    group_policy: str = "adaptive",
    closure_based: bool = False,
    sbert_allowed: bool = True,
    requires_grad_accum_one: bool = False,
    custom_build: Optional[Callable[..., Any]] = None,
    references: Sequence[str] = (),
) -> None:
    """Register an optimizer component."""
    register_spec(
        OptimizerSpec(
            name=name,
            builder=builder,
            description=description,
            description_es=description_es,
            references=tuple(references),
            param_bindings=dict(param_bindings or {}),
            supports_shared_per_group_suffixes=supports_shared_per_group_suffixes,
            group_policy=group_policy,
            closure_based=closure_based,
            sbert_allowed=sbert_allowed,
            requires_grad_accum_one=requires_grad_accum_one,
            custom_build=custom_build,
        )
    )


def register_norm(
    name: str,
    builder: Any,
    description: str,
    description_es: str,
    *,
    params_schema: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    supports_ffn_fusion: bool = False,
    reads_partial_ratio: bool = False,
    references: Sequence[str] = (),
) -> None:
    """Register a normalization component."""
    register_spec(
        NormSpec(
            name=name,
            builder=builder,
            description=description,
            description_es=description_es,
            references=tuple(references),
            params_schema=params_schema,
            defaults=dict(defaults or {}),
            supports_ffn_fusion=supports_ffn_fusion,
            reads_partial_ratio=reads_partial_ratio,
        )
    )


def register_activation(
    name: str,
    builder: Any,
    description: str,
    description_es: str,
    *,
    is_glu: bool = False,
    references: Sequence[str] = (),
) -> None:
    """Register an activation component."""
    register_spec(
        ActivationSpec(
            name=name,
            builder=builder,
            description=description,
            description_es=description_es,
            references=tuple(references),
            is_glu=is_glu,
        )
    )


def register_residual(
    name: str,
    builder: Any,
    description: str,
    description_es: str,
    *,
    references: Sequence[str] = (),
) -> None:
    """Register a residual-connection strategy."""
    register_spec(
        ResidualSpec(
            name=name,
            builder=builder,
            description=description,
            description_es=description_es,
            references=tuple(references),
        )
    )


def register_posenc(
    name: str,
    builder: Any,
    description: str,
    description_es: str,
    *,
    params_schema: Optional[Mapping[str, Any]] = None,
    params_key: str = "",
    aliases: Sequence[str] = (),
    references: Sequence[str] = (),
) -> None:
    """Register a positional encoding."""
    register_spec(
        PESpec(
            name=name,
            builder=builder,
            description=description,
            description_es=description_es,
            references=tuple(references),
            params_schema=params_schema,
            params_key=params_key,
            aliases=tuple(aliases),
        )
    )


def register_embedding(
    name: str,
    builder: Any,
    description: str,
    description_es: str,
    *,
    references: Sequence[str] = (),
) -> None:
    """Register a token-embedding variant."""
    register_spec(
        EmbeddingSpec(
            name=name,
            builder=builder,
            description=description,
            description_es=description_es,
            references=tuple(references),
        )
    )
