"""Mamba placeholder mixer: a single dense linear projection.

The ``mamba`` layer type is the stateless per-token linear placeholder used by
hybrid patterns (no SSM parameters, no recurrent state). It is registered
through the component registry like every other mixer so ``HybridLayer`` has
no hard-coded special case.
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from .common import BitLinear

__all__ = ["build_mamba"]


def build_mamba(config: Any, pos_encoder: Optional[nn.Module] = None, params=None) -> nn.Module:
    """Build the stateless mamba placeholder projection.

    Args:
        config: Nested model configuration.
        pos_encoder: Shared positional encoder (unused).
        params: Bound mixer parameters (unused).

    Returns:
        A ``hidden_size × hidden_size`` linear projection (BitLinear under
        BitNet).
    """
    proj_cls = BitLinear if bool(config.bitnet.enabled) else nn.Linear
    return proj_cls(int(config.dims.hidden_size), int(config.dims.hidden_size))
