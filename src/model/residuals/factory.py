"""Factory for selecting a residual-connection module from config.

The :func:`build_residual` factory reads the residual-related fields on
:class:`FrankensteinModelConfig` and returns the matching module:

    residual_type == "standard"  →  :class:`StandardResidual`
    residual_type == "none"      →  :class:`NoResidual`
    residual_type == "full_attn" →  :class:`FullAttentionResidual`
    residual_type == "block_attn"→  :class:`BlockAttentionResidual`

The factory is called once by :class:`FrankensteinEncoder` after the
``use_mhc`` flag has been resolved so the residual module can be
configured with the right number of streams.
"""

from __future__ import annotations

from ..config import FrankensteinModelConfig
from .base import ResidualBase
from .block_attn_res import BlockAttentionResidual
from .full_attn_res import FullAttentionResidual
from .no_residual import NoResidual
from .standard import StandardResidual

# Supported residual-type strings (mirrors the schema enum).
RESIDUAL_TYPES = ("standard", "none", "full_attn", "block_attn")


def build_residual(config: FrankensteinModelConfig) -> ResidualBase:
    """Build the configured residual-connection module.

    Args:
        config: :class:`FrankensteinModelConfig` with the residual
            fields populated. The factory reads
            ``residual_type`` and the per-variant flags.

    Returns:
        An instance of one of :class:`StandardResidual`,
        :class:`NoResidual`, :class:`FullAttentionResidual`, or
        :class:`BlockAttentionResidual` with the right number of
        streams and pre-allocated layer query vectors.

    Raises:
        ValueError: If ``residual_type`` is not one of the supported
            values.
    """
    residual_type = str(getattr(config, "residual_type", "standard")).lower()
    if residual_type not in RESIDUAL_TYPES:
        raise ValueError(
            f"residual_type must be one of {RESIDUAL_TYPES}, got {residual_type!r}"
        )

    num_layers = int(config.num_layers) * int(config.num_loops)
    hidden_size = int(config.hidden_size)
    n_streams = (
        int(config.mhc_expansion_rate) if bool(getattr(config, "use_mhc", False)) else 1
    )

    if residual_type == "standard":
        mod = StandardResidual(hidden_size=hidden_size)
    elif residual_type == "none":
        mod = NoResidual(hidden_size=hidden_size)
    elif residual_type == "full_attn":
        mod = FullAttentionResidual(
            hidden_size=hidden_size,
            num_layers=num_layers,
            init_query_zero=bool(getattr(config, "full_attn_init_query_zero", True)),
            use_rmsnorm_keys=bool(getattr(config, "full_attn_use_rmsnorm_keys", True)),
            mhc_stream_mode=str(
                getattr(config, "attnres_mhc_stream_mode", "independent")
            ),
            gradient_checkpoint=bool(
                getattr(config, "attnres_gradient_checkpoint", False)
            ),
        )
    elif residual_type == "block_attn":
        mod = BlockAttentionResidual(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_blocks=int(getattr(config, "block_attn_num_blocks", 8)),
            init_query_zero=bool(getattr(config, "block_attn_init_query_zero", True)),
            use_rmsnorm_keys=bool(
                getattr(config, "block_attn_use_rmsnorm_keys", True)
            ),
            mhc_stream_mode=str(
                getattr(config, "attnres_mhc_stream_mode", "independent")
            ),
            gradient_checkpoint=bool(
                getattr(config, "attnres_gradient_checkpoint", False)
            ),
        )
    else:  # pragma: no cover — exhaustive above
        raise ValueError(f"Unhandled residual_type: {residual_type!r}")

    mod.set_streams(n_streams)
    return mod


__all__ = ["build_residual", "RESIDUAL_TYPES"]
