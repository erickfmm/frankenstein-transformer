"""Embedding and positional encoding modules.

Groups the token embedding (factorized) and the positional encodings
(RoPE / HoPE / NoPE / ALiBi / PaPE / sinusoidal / learned) that operate on
hidden representations.
"""

from __future__ import annotations

from .alibi import ALiBi
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

__all__ = [
    "ALiBi",
    "FactorizedEmbedding",
    "HoPE",
    "LearnedAbsolutePE",
    "NoPE",
    "PaPE",
    "PaPEEfficient",
    "PaPERI",
    "RoPE",
    "SinusoidalAbsolute",
    "SinusoidalRotary",
    "build_pos_encoder",
]