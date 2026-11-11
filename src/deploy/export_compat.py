"""Helpers shared by the HF and GGUF exporters.

The exporters consume the **nested** ``model:`` block directly
(``model.dims.hidden_size``, ``model.bitnet.enabled``, …). This module only
centralises the conversion of that mapping into a runtime
:class:`~src.model.config.FrankensteinModelConfig`.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["model_config_from"]


def model_config_from(model_config: Mapping[str, Any]):
    """Build a nested configuration from a raw ``model:`` mapping.

    Args:
        model_config: Nested ``model:`` block.

    Returns:
        A :class:`~src.model.config.FrankensteinModelConfig` instance.
    """
    from src.model.config import FrankensteinModelConfig

    return FrankensteinModelConfig.from_dict(dict(model_config or {}))
