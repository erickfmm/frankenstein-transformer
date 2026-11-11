"""Positional-encoding and token-embedding factories.

Both dispatch through the component registry: the twelve
``model.positional_encoding`` values are
:class:`~src.registry.spec.PESpec` entries and the three
``model.embedding.type`` values are
:class:`~src.registry.spec.EmbeddingSpec` entries. Params live in the nested
``positional_encoding_parameters`` / ``embedding`` blocks.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch.nn as nn

from src.registry import (
    ComponentFamily,
    ensure_builtins,
    registry,
    resolve_builder,
)

__all__ = ["build_pos_encoder", "build_token_embedding"]


def _pe_params(config: Any, key: str) -> Mapping[str, Any]:
    """Return one encoding's parameter mapping."""
    parameters = getattr(config, "positional_encoding_parameters", None) or {}
    block = parameters.get(key) if isinstance(parameters, Mapping) else None
    return block if isinstance(block, Mapping) else {}


def _build_nope(config: Any, params) -> nn.Module:
    """Build the no-positional-encoding identity module."""
    from .nope import NoPE

    return NoPE()


def _build_rope(config: Any, params) -> nn.Module:
    """Build RoPE."""
    from .rope import RoPE

    return RoPE(
        int(config.head_dim),
        base=float(params.get("base", 10_000.0)),
        scaling=float(params.get("scaling", 1.0)),
    )


def _build_hope(config: Any, params) -> nn.Module:
    """Build HoPE."""
    from .hope import HoPE

    return HoPE(
        int(config.head_dim),
        base=float(params.get("base", 10_000.0)),
        damping=float(params.get("damping", 0.01)),
    )


def _build_alibi(config: Any, params) -> nn.Module:
    """Build ALiBi."""
    from .alibi import ALiBi

    return ALiBi(int(params.get("num_heads", config.dims.num_heads)))


def _build_bam(config: Any, params) -> nn.Module:
    """Build BAM."""
    from .bam import BAM

    return BAM(
        int(config.dims.num_heads),
        learn_mu=bool(params.get("learn_mu", False)),
        theta_init=float(params.get("theta_init", 0.0)),
        mu_init=float(params.get("mu_init", 0.0)),
        eps=float(params.get("eps", 1e-5)),
    )


def _build_pape(config: Any, params) -> nn.Module:
    """Build PaPE."""
    from .pape import PaPE

    return PaPE(
        int(config.dims.hidden_size),
        int(config.dims.num_heads),
        int(config.head_dim),
        num_parabolas=int(params.get("num_parabolas", 4)),
        num_positions=int(params.get("num_positions", 1)),
    )


def _build_pape_efficient(config: Any, params) -> nn.Module:
    """Build efficient PaPE."""
    from .pape_efficient import PaPEEfficient

    return PaPEEfficient(
        int(config.dims.hidden_size),
        int(config.dims.num_heads),
        int(config.head_dim),
        num_parabolas=int(params.get("num_parabolas", 4)),
        num_positions=int(params.get("num_positions", 1)),
    )


def _build_pape_ri(config: Any, params) -> nn.Module:
    """Build rotation-invariant PaPE."""
    from .pape_ri import PaPERI

    return PaPERI(
        int(config.dims.hidden_size),
        int(config.dims.num_heads),
        int(config.head_dim),
        num_positions=int(params.get("num_positions", 2)),
    )


def _build_sinusoidal_absolute(config: Any, params) -> nn.Module:
    """Build the fixed sinusoidal absolute encoding."""
    from .sinusoidal import SinusoidalAbsolute

    return SinusoidalAbsolute(
        int(config.dims.hidden_size),
        max_len=int(params.get("max_len", 512)),
        base=float(params.get("base", 10_000.0)),
        scale=float(params.get("scale", 1.0)),
    )


def _build_sinusoidal_rotary(config: Any, params) -> nn.Module:
    """Build the sinusoidal rotary encoding."""
    from .sinusoidal import SinusoidalRotary

    return SinusoidalRotary(
        int(config.head_dim),
        max_len=int(params.get("max_len", 512)),
        base=float(params.get("base", 10_000.0)),
        scale=float(params.get("scale", 1.0)),
    )


def _build_learned_absolute(config: Any, params) -> nn.Module:
    """Build the learnable absolute encoding."""
    from .learned_absolute import LearnedAbsolutePE

    return LearnedAbsolutePE(
        int(config.dims.hidden_size),
        max_len=int(params.get("max_len", 512)),
        init_std=float(params.get("init_std", 0.02)),
    )


def build_pos_encoder(config: Any) -> Optional[nn.Module]:
    """Build the positional-encoding module selected by the config.

    Args:
        config: Nested model configuration.

    Returns:
        A positional-encoding ``nn.Module`` (the ``none``/``nope`` aliases
        resolve to the identity module).

    Raises:
        UnknownComponentError: If ``positional_encoding`` is not registered.
    """
    ensure_builtins()
    spec = registry.get(
        ComponentFamily.POSITIONAL_ENCODING,
        registry.resolve_name(ComponentFamily.POSITIONAL_ENCODING, config.positional_encoding),
    )
    params_key = str(getattr(spec, "params_key", "") or "")
    params = _pe_params(config, params_key) if params_key else {}
    module = resolve_builder(spec.builder)(config, params)

    # Scalable Softmax is transversal: attach it to every encoding.
    if getattr(config.ssmax, "enabled", False):
        from .ssmax import SSMax

        module.ssmax = SSMax(
            int(config.dims.num_heads), s_init=float(getattr(config.ssmax, "s_init", 1.0))
        )
    else:
        module.ssmax = None
    return module


def _build_embedding_standard(config: Any, params) -> nn.Module:
    """Build a standard token embedding table."""
    return nn.Embedding(int(config.dims.vocab_size), int(config.dims.hidden_size))


def _build_embedding_factorized(config: Any, params) -> nn.Module:
    """Build a factorized (low-rank) token embedding."""
    from .factorized_embedding import FactorizedEmbedding

    return FactorizedEmbedding(config)


def _build_embedding_dynamic(config: Any, params) -> nn.Module:
    """Build a dynamic (hash) token embedding."""
    from .dynamic_embedding import DynamicEmbedding

    return DynamicEmbedding(config)


def build_token_embedding(config: Any) -> nn.Module:
    """Build the token embedding selected by ``config.embedding.type``.

    Args:
        config: Nested model configuration.

    Returns:
        The token-embedding module.

    Raises:
        UnknownComponentError: If ``embedding.type`` is not registered.
    """
    ensure_builtins()
    spec = registry.get(ComponentFamily.EMBEDDING, str(config.embedding.type))
    return resolve_builder(spec.builder)(config, {})
