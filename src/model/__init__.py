from src.registry import ensure_builtins

from .config import FrankensteinModelConfig
from .hybrid_layer import HybridLayer
from .frankenstein_encoder import FrankensteinEncoder
from .frankenstein_decoder import FrankensteinDecoder
from .frankenstein_vit import FrankensteinViT
from .frankenstein_audio import FrankensteinAudio
from .residuals import (
    BlockAttentionResidual,
    FullAttentionResidual,
    NoResidual,
    ResidualBase,
    StandardResidual,
    build_residual,
)

__all__ = [
    "FrankensteinModelConfig",
    "HybridLayer",
    "FrankensteinEncoder",
    "FrankensteinDecoder",
    "FrankensteinViT",
    "FrankensteinAudio",
    "ResidualBase",
    "StandardResidual",
    "NoResidual",
    "FullAttentionResidual",
    "BlockAttentionResidual",
    "build_residual",
]

# Register every built-in component spec as a first-party plugin.
# Importing the model package is the canonical entry point for training,
# deployment and the test suite, so this guarantees the registry is populated.
ensure_builtins()
