#!/usr/bin/env python3
"""Backward-compatibility shim.

The original monolithic ``frankenstein_model.py`` has been split by class into:

* :mod:`src.model.config` — :class:`FrankensteinModelConfig`
* :mod:`src.model.hybrid_layer` — :class:`HybridLayer`
* :mod:`src.model.frankenstein_encoder` — :class:`FrankensteinEncoder`
  (formerly ``FrankensteinTransformer``; kept as an alias here)
* :mod:`src.model.frankenstein_decoder` — :class:`FrankensteinDecoder`

This module re-exports all public names so existing imports of the form
``from src.model.frankenstein_model import X`` continue to work.
"""

from __future__ import annotations

from .config import FrankensteinModelConfig
from .hybrid_layer import HybridLayer
from .frankenstein_encoder import FrankensteinEncoder
from .frankenstein_decoder import FrankensteinDecoder

# Backward-compatible alias: the encoder was previously named
# ``FrankensteinTransformer``.
FrankensteinTransformer = FrankensteinEncoder

__all__ = [
    "FrankensteinModelConfig",
    "HybridLayer",
    "FrankensteinEncoder",
    "FrankensteinTransformer",
    "FrankensteinDecoder",
]
