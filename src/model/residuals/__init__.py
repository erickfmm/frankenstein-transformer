"""Residual connection modules for Frankenstein Transformer.

This package hosts the four residual-connection strategies described in
the Attention Residuals paper (arXiv:2603.15031) plus the original
standard residual. The residual module owns the cross-layer state
required by Full/Block Attention Residuals and is injected into
:class:`HybridLayer` and the encoder/decoder wrappers.

Variants:
    - :class:`NoResidual`: drops the skip connection entirely (experimental).
    - :class:`StandardResidual`: fixed unit-weight sum ``h_l = h_{l-1} + f_l``.
    - :class:`FullAttentionResidual`: softmax attention over **all** previous
      layer outputs, parameterised by a learned per-layer query vector.
    - :class:`BlockAttentionResidual`: groups layers into ``N`` blocks; within
      each block a standard partial sum is accumulated and across blocks
      attention is applied over block representations (paper Alg. 1).

All variants expose the same call signature ``(layer_idx, layer_output)`` and
manage their own cross-layer state. The :func:`build_residual` factory
selects the right class from :class:`FrankensteinModelConfig`.
"""

from __future__ import annotations

from .base import ResidualBase
from .block_attn_res import BlockAttentionResidual
from .factory import build_residual
from .full_attn_res import FullAttentionResidual
from .no_residual import NoResidual
from .standard import StandardResidual

__all__ = [
    "ResidualBase",
    "StandardResidual",
    "NoResidual",
    "FullAttentionResidual",
    "BlockAttentionResidual",
    "build_residual",
]
