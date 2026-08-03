"""Activation functions and factory.

Groups the activation functions implemented across the survey of Dubey et al.
(arXiv:2109.14545), the systematic overview of Lederer (arXiv:2101.09957), and
the Rational Activation Function of Fang et al. (arXiv:2208.14111). Provides
the :func:`get_activation` factory that selects an elementwise activation from
the model configuration, mirroring the normalization factory in
``src/model/norm``.

GLU variants (SwiGLU/GEGLU/ReGLU) are gated FFN units, not elementwise
activations; they are built via :func:`make_gated_ffn`.
"""

from __future__ import annotations

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
from .factory import (
    ALL_ACTIVATIONS,
    ELEMENTWISE_ACTIVATIONS,
    GLU_VARIANTS,
    get_activation,
)
from .glu import GatedFFN, make_gated_ffn
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

__all__ = [
    # factory
    "get_activation",
    "ELEMENTWISE_ACTIVATIONS",
    "ALL_ACTIVATIONS",
    "GLU_VARIANTS",
    # common
    "Sigmoid",
    "Tanh",
    "Arctan",
    "Softsign",
    "Elliott",
    "Identity",
    "Softplus",
    "Mish",
    "GELU",
    "GELUTanh",
    "ReLU",
    "SiLU",
    "Swish",
    # rectified
    "LeakyReLU",
    "ReLU6",
    "HardSwish",
    "PReLU",
    "AbsReLU",
    "NLReLU",
    "BReLU",
    "VReLU",
    "Hexpo",
    "PenalizedTanh",
    "DisReLU",
    "LiSHT",
    # exponential
    "ELU",
    "SELU",
    "CELU",
    "PELU",
    "MPELU",
    "FELU",
    "EELU",
    "PDELU",
    "PREU",
    "SoftExp",
    "ELiSH",
    "HardELiSH",
    # learnable
    "RationalActivation",
    "SwishTrainable",
    "Maxout",
    # glu
    "GatedFFN",
    "make_gated_ffn",
]
