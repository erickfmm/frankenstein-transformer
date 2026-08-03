"""Activation function factory.

Selects the appropriate activation module from the model configuration,
mirroring the normalization factory pattern in
``src/model/norm/factory.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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

# GLU variants are gated FFN units, not elementwise activations; kept separate
# so the factory and the schema both recognize them.
GLU_VARIANTS = {"swiglu", "geglu", "reglu"}


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    """Read a config attribute or key, returning ``default`` if absent."""
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def _activation_params(config: Any) -> Dict[str, Any]:
    """Extract the nested ``ffn_activation_config`` dict from ``config``."""
    nested = _cfg_get(config, "ffn_activation_config", None)
    if nested is None:
        return {}
    if isinstance(nested, dict):
        return nested
    return {}


def get_activation(config: Any, dim: Optional[int] = None) -> nn.Module:
    """Factory function that returns an activation module based on config.

    Dispatches on ``config.ffn_activation``. Elementwise activations return a
    ready-to-call :class:`nn.Module`. GLU variants (``"swiglu"``,
    ``"geglu"``, ``"reglu"``) are *not* built here (they require linear
    projections); callers should check :data:`GLU_VARIANTS` and build a
    :class:`~.glu.GatedFFN` instead.

    Args:
        config: Model configuration object exposing ``ffn_activation``
            (string) and an optional ``ffn_activation_config`` mapping with
            keys such as ``prelu_init``, ``elu_alpha``, ``swish_beta``,
            ``leaky_relu_slope``, and the RAF keys ``raf_degrees``,
            ``raf_version``, ``raf_approx_func``, ``raf_trainable``,
            ``raf_input_scaling``.
        dim: Feature dimension that the activation operates on. When ``None``,
            defaults to ``config.hidden_size`` (or ``config.dim``). For FFN
            activations this must be the *intermediate* dimension
            (``ffn_hidden_size``), since the activation runs after the
            up-projection. Default: ``None``.

    Returns:
        An :class:`nn.Module` implementing the requested activation.

    Raises:
        ValueError: If ``ffn_activation`` is a GLU variant (callers must build
            the gated FFN separately) or is otherwise not recognized.
    """
    name = str(_cfg_get(config, "ffn_activation", "silu")).lower()
    p = _activation_params(config)

    def feature_dim(default: int = 1) -> int:
        if dim is not None:
            return int(dim)
        return int(_cfg_get(config, "hidden_size", _cfg_get(config, "dim", default)))

    if name in GLU_VARIANTS:
        raise ValueError(
            f"ffn_activation={name!r} is a gated FFN variant; build it via "
            f"make_gated_ffn() instead of get_activation()."
        )

    # ---- common / classical ----
    if name == "silu":
        return SiLU()
    if name == "gelu":
        return GELU()
    if name == "gelu_tanh":
        return GELUTanh()
    if name == "relu":
        return ReLU()
    if name == "sigmoid":
        return Sigmoid()
    if name == "tanh":
        return Tanh()
    if name == "arctan":
        return Arctan()
    if name == "softsign":
        return Softsign()
    if name == "elliott":
        return Elliott()
    if name == "identity":
        return Identity()
    if name == "softplus":
        return Softplus()
    if name == "mish":
        return Mish()

    # ---- rectified family ----
    if name == "leaky_relu":
        return LeakyReLU(float(p.get("leaky_relu_slope", 0.01)))
    if name == "relu6":
        return ReLU6()
    if name == "hardswish":
        return HardSwish()
    if name == "prelu":
        return PReLU(feature_dim(), init=float(p.get("prelu_init", 0.25)))
    if name == "abs_relu":
        return AbsReLU()
    if name == "nl_relu":
        return NLReLU()
    if name == "brelu":
        return BReLU()
    if name == "vrelu":
        return VReLU()
    if name == "hexpo":
        return Hexpo()
    if name == "ptanh":
        return PenalizedTanh()
    if name == "dis_relu":
        return DisReLU()
    if name == "lisht":
        return LiSHT()

    # ---- exponential / ELU family ----
    if name == "elu":
        return ELU(float(p.get("elu_alpha", 1.0)))
    if name == "selu":
        return SELU()
    if name == "celu":
        return CELU(float(p.get("celu_alpha", 1.0)))
    if name == "pelu":
        return PELU(feature_dim(), alpha_init=float(p.get("pelu_alpha", 1.0)))
    if name == "mpelu":
        return MPELU(
            feature_dim(),
            alpha_init=float(p.get("mpelu_alpha", 1.0)),
            beta_init=float(p.get("mpelu_beta", 1.0)),
        )
    if name == "felu":
        return FELU(feature_dim(), alpha_init=float(p.get("felu_alpha", 1.0)))
    if name == "eelu":
        return EELU(
            feature_dim(),
            alpha_init=float(p.get("eelu_alpha", 1.0)),
            beta_init=float(p.get("eelu_beta", 1.0)),
        )
    if name == "pdelu":
        return PDELU(feature_dim(), alpha_init=float(p.get("pdelu_alpha", 1.0)))
    if name == "preu":
        return PREU(
            feature_dim(),
            alpha_init=float(p.get("preu_alpha", 1.0)),
            beta_init=float(p.get("preu_beta", 1.0)),
        )
    if name == "softexp":
        return SoftExp(feature_dim(), alpha_init=float(p.get("softexp_alpha", 0.0)))
    if name == "elish":
        return ELiSH()
    if name == "hardelish":
        return HardELiSH()

    # ---- learnable / adaptive family ----
    if name == "swish":
        return Swish(float(p.get("swish_beta", 1.0)))
    if name == "swish_trainable":
        return SwishTrainable(dim=feature_dim(), beta_init=float(p.get("swish_beta", 1.0)))
    if name == "maxout":
        return Maxout(feature_dim(), num_pieces=int(p.get("maxout_pieces", 2)))
    if name == "raf":
        return RationalActivation(
            degrees=tuple(p.get("raf_degrees", [5, 4])),
            version=str(p.get("raf_version", "A")),
            approx_func=str(p.get("raf_approx_func", "gelu")),
            trainable=bool(p.get("raf_trainable", True)),
            input_scaling=bool(p.get("raf_input_scaling", False)),
        )

    raise ValueError(
        f"Unknown ffn_activation {name!r}. Valid elementwise activations: "
        f"{sorted(ELEMENTWISE_ACTIVATIONS)}. GLU FFN variants: "
        f"{sorted(GLU_VARIANTS)}."
    )


# Canonical enum of all recognized ``ffn_activation`` names (source of truth
# mirrored in the schema ``enum`` field).
ELEMENTWISE_ACTIVATIONS = {
    "silu", "gelu", "gelu_tanh", "relu", "sigmoid", "tanh", "arctan",
    "softsign", "elliott", "identity", "softplus", "mish",
    "leaky_relu", "relu6", "hardswish", "prelu", "abs_relu", "nl_relu",
    "brelu", "vrelu", "hexpo", "ptanh", "dis_relu", "lisht",
    "elu", "selu", "celu", "pelu", "mpelu", "felu", "eelu", "pdelu",
    "preu", "softexp", "elish", "hardelish",
    "swish", "swish_trainable", "maxout", "raf",
}

ALL_ACTIVATIONS = ELEMENTWISE_ACTIVATIONS | GLU_VARIANTS


__all__ = [
    "get_activation",
    "ELEMENTWISE_ACTIVATIONS",
    "ALL_ACTIVATIONS",
    "GLU_VARIANTS",
]
