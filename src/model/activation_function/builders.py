"""Construction functions for every elementwise activation.

Generated once from the former ``get_activation`` if/elif chain: one
function per registry activation. ``get_activation`` dispatches through
the component registry and calls the matching function.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch.nn as nn

from .common import (
    Arctan,
    Elliott,
    GELU,
    GELUTanh,
    Identity,
    Mish,
    ReLU,
    SiLU,
    Sigmoid,
    Softplus,
    Softsign,
    Swish,
    Tanh,
)
from .exponential import (
    CELU,
    EELU,
    ELU,
    ELiSH,
    FELU,
    HardELiSH,
    MPELU,
    PDELU,
    PELU,
    PREU,
    SELU,
    SoftExp,
)
from .learnable import Maxout, RationalActivation, SwishTrainable
from .rectified import (
    AbsReLU,
    BReLU,
    DisReLU,
    HardSwish,
    Hexpo,
    LeakyReLU,
    LiSHT,
    NLReLU,
    PenalizedTanh,
    PReLU,
    ReLU6,
    VReLU,
)


def _feature_dim(config: Any, dim: Optional[int]) -> int:
    """Resolve the feature dimension an activation operates on."""
    if dim is not None:
        return int(dim)
    return int(getattr(config.dims, "hidden_size", 1))


def build_silu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``silu`` activation."""
    return SiLU()


def build_gelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``gelu`` activation."""
    return GELU()


def build_gelu_tanh(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``gelu_tanh`` activation."""
    return GELUTanh()


def build_relu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``relu`` activation."""
    return ReLU()


def build_sigmoid(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``sigmoid`` activation."""
    return Sigmoid()


def build_tanh(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``tanh`` activation."""
    return Tanh()


def build_arctan(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``arctan`` activation."""
    return Arctan()


def build_softsign(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``softsign`` activation."""
    return Softsign()


def build_elliott(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``elliott`` activation."""
    return Elliott()


def build_identity(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``identity`` activation."""
    return Identity()


def build_softplus(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``softplus`` activation."""
    return Softplus()


def build_mish(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``mish`` activation."""
    return Mish()


def build_leaky_relu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``leaky_relu`` activation."""
    return LeakyReLU(float(params.get("leaky_relu_slope", 0.01)))


def build_relu6(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``relu6`` activation."""
    return ReLU6()


def build_hardswish(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``hardswish`` activation."""
    return HardSwish()


def build_prelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``prelu`` activation."""
    return PReLU(_feature_dim(config, dim), init=float(params.get("prelu_init", 0.25)))


def build_abs_relu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``abs_relu`` activation."""
    return AbsReLU()


def build_nl_relu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``nl_relu`` activation."""
    return NLReLU()


def build_brelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``brelu`` activation."""
    return BReLU()


def build_vrelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``vrelu`` activation."""
    return VReLU()


def build_hexpo(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``hexpo`` activation."""
    return Hexpo()


def build_ptanh(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``ptanh`` activation."""
    return PenalizedTanh()


def build_dis_relu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``dis_relu`` activation."""
    return DisReLU()


def build_lisht(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``lisht`` activation."""
    return LiSHT()


def build_elu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``elu`` activation."""
    return ELU(float(params.get("elu_alpha", 1.0)))


def build_selu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``selu`` activation."""
    return SELU()


def build_celu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``celu`` activation."""
    return CELU(float(params.get("celu_alpha", 1.0)))


def build_pelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``pelu`` activation."""
    return PELU(
        _feature_dim(config, dim), alpha_init=float(params.get("pelu_alpha", 1.0))
    )


def build_mpelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``mpelu`` activation."""
    return MPELU(
        _feature_dim(config, dim),
        alpha_init=float(params.get("mpelu_alpha", 1.0)),
        beta_init=float(params.get("mpelu_beta", 1.0)),
    )


def build_felu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``felu`` activation."""
    return FELU(
        _feature_dim(config, dim), alpha_init=float(params.get("felu_alpha", 1.0))
    )


def build_eelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``eelu`` activation."""
    return EELU(
        _feature_dim(config, dim),
        alpha_init=float(params.get("eelu_alpha", 1.0)),
        beta_init=float(params.get("eelu_beta", 1.0)),
    )


def build_pdelu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``pdelu`` activation."""
    return PDELU(
        _feature_dim(config, dim), alpha_init=float(params.get("pdelu_alpha", 1.0))
    )


def build_preu(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``preu`` activation."""
    return PREU(
        _feature_dim(config, dim),
        alpha_init=float(params.get("preu_alpha", 1.0)),
        beta_init=float(params.get("preu_beta", 1.0)),
    )


def build_softexp(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``softexp`` activation."""
    return SoftExp(
        _feature_dim(config, dim), alpha_init=float(params.get("softexp_alpha", 0.0))
    )


def build_elish(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``elish`` activation."""
    return ELiSH()


def build_hardelish(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``hardelish`` activation."""
    return HardELiSH()


def build_swish(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``swish`` activation."""
    return Swish(float(params.get("swish_beta", 1.0)))


def build_swish_trainable(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``swish_trainable`` activation."""
    return SwishTrainable(
        dim=_feature_dim(config, dim), beta_init=float(params.get("swish_beta", 1.0))
    )


def build_maxout(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``maxout`` activation."""
    return Maxout(
        _feature_dim(config, dim), num_pieces=int(params.get("maxout_pieces", 2))
    )


def build_raf(
    config: Any, params: Mapping[str, Any], dim: Optional[int] = None
) -> nn.Module:
    """Build the ``raf`` activation."""
    return RationalActivation(
        degrees=tuple(params.get("raf_degrees", [5, 4])),
        version=str(params.get("raf_version", "A")),
        approx_func=str(params.get("raf_approx_func", "gelu")),
        trainable=bool(params.get("raf_trainable", True)),
        input_scaling=bool(params.get("raf_input_scaling", False)),
    )
