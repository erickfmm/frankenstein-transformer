"""Embedding and positional encoding modules.

Groups the token embedding (factorized) and the positional encodings
(RoPE / HoPE / NoPE / ALiBi / BAM / PaPE / sinusoidal / learned) that operate on
hidden representations, plus the transversal Scalable Softmax (SSMax) logit
rescale.
"""

from __future__ import annotations

from .alibi import ALiBi
from .bam import BAM
from .factorized_embedding import FactorizedEmbedding
from .factory import build_pos_encoder
from .hope import HoPE
from .learned_absolute import LearnedAbsolutePE
from .nope import NoPE
from .pape import PaPE
from .pape_efficient import PaPEEfficient
from .pape_ri import PaPERI
from .rope import RoPE
from .sinusoidal import SinusoidalAbsolute, SinusoidalRotary
from .ssmax import SSMax

__all__ = [
    "ALiBi",
    "BAM",
    "FactorizedEmbedding",
    "HoPE",
    "LearnedAbsolutePE",
    "NoPE",
    "PaPE",
    "PaPEEfficient",
    "PaPERI",
    "RoPE",
    "SSMax",
    "SinusoidalAbsolute",
    "SinusoidalRotary",
    "build_pos_encoder",
]