"""Activation function factory.

Dispatches through the component registry: every ``model.ffn.activation``
value is an :class:`~src.registry.spec.ActivationSpec`. Elementwise builders
live in :mod:`.builders`; GLU variants are gated FFN units handled by
:class:`~.glu.GatedFFN`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from src.registry import (
    ComponentFamily,
    ensure_builtins,
    registry,
    resolve_builder,
)

__all__ = [
    "get_activation",
    "ELEMENTWISE_ACTIVATIONS",
    "ALL_ACTIVATIONS",
    "GLU_VARIANTS",
]


def _activation_params(config: Any) -> Dict[str, Any]:
    """Extract the nested ``ffn.activation_config`` mapping.

    Args:
        config: Model configuration (nested shape).

    Returns:
        The activation configuration mapping (empty when absent).
    """
    block = getattr(config, "ffn", None)
    nested = getattr(block, "activation_config", None)
    if isinstance(nested, dict):
        return dict(nested)
    return {}


def _activation_name(config: Any) -> str:
    """Return the lower-cased activation name from the config."""
    block = getattr(config, "ffn", None)
    return str(getattr(block, "activation", "silu")).lower()


def get_activation(config: Any, dim: Optional[int] = None) -> nn.Module:
    """Return the activation module selected by ``config.ffn.activation``.

    Args:
        config: Model configuration (nested shape).
        dim: Feature dimension the activation operates on (the FFN
            intermediate width). Defaults to ``config.dims.hidden_size``.

    Returns:
        An :class:`nn.Module` implementing the requested activation.

    Raises:
        ValueError: If the activation is a GLU variant (callers must build the
            gated FFN separately).
        UnknownComponentError: If the activation is not registered.
    """
    ensure_builtins()
    name = _activation_name(config)
    spec = registry.get(ComponentFamily.ACTIVATION, name)
    if bool(getattr(spec, "is_glu", False)):
        raise ValueError(
            f"ffn_activation={name!r} is a gated FFN variant; build it via "
            f"make_gated_ffn() instead of get_activation()."
        )
    return resolve_builder(spec.builder)(config, _activation_params(config), dim)


def _refresh_views() -> None:
    """Refresh the derived enum views from the registry."""
    global ELEMENTWISE_ACTIVATIONS, ALL_ACTIVATIONS, GLU_VARIANTS
    ensure_builtins()
    GLU_VARIANTS = registry.property_set(ComponentFamily.ACTIVATION, "is_glu")
    ALL_ACTIVATIONS = frozenset(registry.names(ComponentFamily.ACTIVATION))
    ELEMENTWISE_ACTIVATIONS = ALL_ACTIVATIONS - GLU_VARIANTS


GLU_VARIANTS: frozenset = frozenset()
"""Gated FFN variants (``swiglu``, ``geglu``, ``reglu``)."""

ELEMENTWISE_ACTIVATIONS: frozenset = frozenset()
"""Elementwise activation names."""

ALL_ACTIVATIONS: frozenset = frozenset()
"""Every accepted ``ffn.activation`` value."""

_refresh_views()
