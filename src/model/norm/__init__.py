"""Normalization modules and factory.

Groups the custom normalization layers (``DynamicTanhNorm``, ``Derf``,
``RMSNorm``, ``FlashNorm``), the query–key normalization / attention-logit
stabilization module (:mod:`src.model.norm.qk`), and the ``get_norm`` factory
function that selects one from the model configuration.
"""

from __future__ import annotations

from .derf import Derf
from .dynamic_tanh import DynamicTanhNorm
from .factory import get_norm
from .flash import FlashNorm, FlashNormBitLinear, FlashNormLinear, fold_norm_weights
from .qk import (
    FORWARD_METHODS,
    QK_CLIP_MIXERS,
    QK_NORM_METHODS,
    QK_NORM_MIXERS,
    QKNorm,
    QuacKController,
    TRAINING_METHODS,
    apply_qk_clip,
    attach_qk_norm,
    make_qk_norm,
    record_qk_max_logits,
)
from .rms import RMSNorm

__all__ = [
    "DynamicTanhNorm",
    "Derf",
    "RMSNorm",
    "FlashNorm",
    "FlashNormLinear",
    "FlashNormBitLinear",
    "fold_norm_weights",
    "get_norm",
    "QKNorm",
    "QuacKController",
    "apply_qk_clip",
    "attach_qk_norm",
    "make_qk_norm",
    "record_qk_max_logits",
    "QK_NORM_METHODS",
    "QK_NORM_MIXERS",
    "QK_CLIP_MIXERS",
    "FORWARD_METHODS",
    "TRAINING_METHODS",
]
