"""Factory for selecting a residual-connection module from config.

Dispatches through the component registry: every ``model.residuals.type``
value is a :class:`~src.registry.spec.ResidualSpec` whose builder receives the
config. The factory then finishes the construction (stream count and the
fixed-softmax opt-in) exactly as before.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.registry import (
    ComponentFamily,
    ensure_builtins,
    registry,
    resolve_builder,
)

from .base import ResidualBase
from .block_attn_res import BlockAttentionResidual
from .full_attn_res import FullAttentionResidual
from .no_residual import NoResidual
from .standard import StandardResidual

__all__ = ["build_residual", "RESIDUAL_TYPES"]

RESIDUAL_TYPES: Iterable[str] = ()
"""Supported residual-type strings (derived from the registry)."""


def _residuals(config: Any) -> Any:
    """Return the nested ``residuals`` block (with defaults when absent)."""
    block = getattr(config, "residuals", None)
    if block is None:
        from src.model.config import ResidualsConfig

        return ResidualsConfig()
    return block


def _build_standard(config: Any, params) -> ResidualBase:
    """Build the classical residual connection."""
    return StandardResidual(hidden_size=int(config.hidden_size))


def _build_none(config: Any, params) -> ResidualBase:
    """Build the no-residual ablation."""
    return NoResidual(hidden_size=int(config.hidden_size))


def _build_full_attn(config: Any, params) -> ResidualBase:
    """Build Full AttnRes."""
    block = _residuals(config)
    return FullAttentionResidual(
        hidden_size=int(config.hidden_size),
        num_layers=int(config.num_layers) * int(config.num_loops),
        init_query_zero=bool(block.full_attn.init_query_zero),
        use_rmsnorm_keys=bool(block.full_attn.use_rmsnorm_keys),
        mhc_stream_mode=str(block.mhc_stream_mode),
        gradient_checkpoint=bool(block.gradient_checkpoint),
    )


def _build_block_attn(config: Any, params) -> ResidualBase:
    """Build Block AttnRes."""
    block = _residuals(config)
    return BlockAttentionResidual(
        hidden_size=int(config.hidden_size),
        num_layers=int(config.num_layers) * int(config.num_loops),
        num_blocks=int(block.block_attn.num_blocks),
        init_query_zero=bool(block.block_attn.init_query_zero),
        use_rmsnorm_keys=bool(block.block_attn.use_rmsnorm_keys),
        mhc_stream_mode=str(block.mhc_stream_mode),
        gradient_checkpoint=bool(block.gradient_checkpoint),
    )


def build_residual(config: Any) -> ResidualBase:
    """Build the configured residual-connection module.

    Args:
        config: Nested model configuration.

    Returns:
        The configured :class:`ResidualBase` subclass with the right number
        of streams and pre-allocated layer query vectors.

    Raises:
        UnknownComponentError: If ``residuals.type`` is not registered.
    """
    ensure_builtins()
    global RESIDUAL_TYPES
    RESIDUAL_TYPES = registry.names(ComponentFamily.RESIDUAL)
    block = _residuals(config)
    residual_type = str(block.type).lower()
    spec = registry.get(ComponentFamily.RESIDUAL, residual_type)
    module = resolve_builder(spec.builder)(config, {})
    streams = int(config.mhc_expansion_rate) if bool(getattr(config, "use_mhc", False)) else 1
    module.set_streams(streams)
    module.use_fixed_softmax = bool(getattr(config, "fixed_softmax", False))
    return module
