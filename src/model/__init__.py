from .config import FrankensteinModelConfig
from .hybrid_layer import HybridLayer
from .frankenstein_encoder import FrankensteinEncoder
from .frankenstein_decoder import FrankensteinDecoder

# Backward-compatible alias: the encoder was previously named
# ``FrankensteinTransformer``.
FrankensteinTransformer = FrankensteinEncoder

__all__ = [
    "FrankensteinModelConfig",
    "HybridLayer",
    "FrankensteinEncoder",
    "FrankensteinTransformer",
    "FrankensteinDecoder",
]
