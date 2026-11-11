"""Normalization factory.

Dispatches through the component registry: every ``model.norm.type`` value is
a :class:`~src.registry.spec.NormSpec` whose builder receives the config and
its bound parameters. Adding a normalization only adds a spec.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.registry import (
    ComponentFamily,
    bind_params,
    ensure_builtins,
    registry,
    resolve_builder,
)

from .derf import Derf
from .dynamic_tanh import DynamicTanhNorm
from .flash import FlashNorm
from .rms import RMSNorm

__all__ = ["get_norm", "norm_params", "is_norm_fusable"]


def norm_params(config: Any) -> dict:
    """Return the ``norm`` block parameters for binding.

    Args:
        config: Model configuration (nested shape).

    Returns:
        Mapping with ``partial_ratio`` and ``flashnorm_partial_ratio``.
    """
    block = getattr(config, "norm", None)
    return {
        "partial_ratio": getattr(block, "partial_ratio", 0.0625),
        "flashnorm_partial_ratio": getattr(block, "flashnorm_partial_ratio", 0.0),
    }


def is_norm_fusable(config: Any) -> bool:
    """Return whether the configured norm supports FFN fusion.

    Args:
        config: Model configuration (nested shape).

    Returns:
        ``True`` when the norm's spec declares ``supports_ffn_fusion``.
    """
    ensure_builtins()
    spec = registry.get(ComponentFamily.NORM, str(config.norm.type))
    return bool(getattr(spec, "supports_ffn_fusion", False))


def _build_layer_norm(config: Any, params) -> nn.Module:
    """Build a standard :class:`nn.LayerNorm`."""
    return nn.LayerNorm(config.dims.hidden_size)


def _build_dynamic_tanh(config: Any, params) -> nn.Module:
    """Build a DynamicTanhNorm."""
    return DynamicTanhNorm(config.dims.hidden_size)


def _build_derf(config: Any, params) -> nn.Module:
    """Build a Derf normalization."""
    return Derf(config.dims.hidden_size)


def _build_rms_norm(config: Any, params) -> nn.Module:
    """Build a full RMSNorm."""
    return RMSNorm(config.dims.hidden_size)


def _build_prms_norm(config: Any, params) -> nn.Module:
    """Build a partial RMSNorm."""
    return RMSNorm(config.dims.hidden_size, partial_ratio=float(params.partial_ratio))


def _build_flash_norm(config: Any, params) -> nn.Module:
    """Build a (possibly partial) FlashNorm."""
    return FlashNorm(
        config.dims.hidden_size, partial_ratio=float(params.flashnorm_partial_ratio)
    )


def _declared_norm_params(spec: Any, config: Any) -> dict:
    """Return only the norm-block parameters declared by the spec.

    Args:
        spec: The normalization spec.
        config: Model configuration.

    Returns:
        The subset of :func:`norm_params` declared by the spec's fragment.
    """
    from src.registry.fragments import allowed_param_keys

    allowed = set(allowed_param_keys(spec))
    return {key: value for key, value in norm_params(config).items() if key in allowed}


def get_norm(config: Any) -> nn.Module:
    """Return the normalization module selected by ``config.norm.type``.

    Args:
        config: Model configuration (nested shape).

    Returns:
        A normalization ``nn.Module``.

    Raises:
        UnknownComponentError: If ``norm.type`` is not registered.
    """
    ensure_builtins()
    spec = registry.maybe_get(ComponentFamily.NORM, str(config.norm.type))
    if spec is None:
        import logging

        logging.getLogger("frankenstein.config").warning(
            "Unknown norm type %r; falling back to layer_norm.", config.norm.type
        )
        return nn.LayerNorm(config.dims.hidden_size)
    params = bind_params(spec, _declared_norm_params(spec, config))
    return resolve_builder(spec.builder)(config, params)
