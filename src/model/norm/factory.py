"""Normalization factory.

Selects the appropriate normalization module from the model configuration,
mirroring the optimizer factory pattern in ``src/model/optimizer/factory.py``.
"""

from __future__ import annotations

import torch.nn as nn

from .derf import Derf
from .dynamic_tanh import DynamicTanhNorm
from .flash import FlashNorm
from .rms import RMSNorm


def get_norm(config):
    """Factory function that returns a normalization module based on config.

    Selects among ``LayerNorm``, ``DynamicTanhNorm``, ``Derf``, ``RMSNorm``,
    ``pRMSNorm`` (partial RMSNorm), and ``FlashNorm`` (weightless RMSNorm
    with optional partial-RMS composition) based on the ``norm_type`` field
    in the configuration.

    Args:
        config: Model configuration object with attributes ``norm_type``
            (one of ``"layer_norm"``, ``"dynamic_tanh"``, ``"derf"``,
            ``"rms_norm"``, ``"prms_norm"``, ``"flash_norm"``) and
            ``hidden_size``. When ``norm_type == "prms_norm"``, the optional
            ``prms_partial_ratio`` attribute (default ``0.0625``) controls
            the fraction of dimensions used for RMS estimation. When
            ``norm_type == "flash_norm"``, the optional
            ``flashnorm_partial_ratio`` attribute (default ``0.0``)
            activates the partial-RMS variant of FlashNorm.

    Returns:
        A normalization ``nn.Module`` instance appropriate for the requested
        ``norm_type``.

    Raises:
        AttributeError: If ``config`` does not have ``norm_type`` or
            ``hidden_size`` attributes.
    """
    if config.norm_type == "dynamic_tanh":
        return DynamicTanhNorm(config.hidden_size)
    if config.norm_type == "derf":
        return Derf(config.hidden_size)
    if config.norm_type == "rms_norm":
        return RMSNorm(config.hidden_size)
    if config.norm_type == "prms_norm":
        partial_ratio = float(getattr(config, "prms_partial_ratio", 0.0625))
        return RMSNorm(config.hidden_size, partial_ratio=partial_ratio)
    if config.norm_type == "flash_norm":
        partial_ratio = float(getattr(config, "flashnorm_partial_ratio", 0.0))
        return FlashNorm(config.hidden_size, partial_ratio=partial_ratio)
    return nn.LayerNorm(config.hidden_size)
