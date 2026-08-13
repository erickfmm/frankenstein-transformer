"""dashai-frankenstein — DashAI plugin for Frankenstein Transformer.

Registers Frankenstein model classes (encoder, decoder, ViT) as DashAI
components discovered via the ``dashai.plugins`` entry-points group.
"""
from dashai_frankenstein.models.mlm import FrankensteinMLMModel  # noqa: F401

__all__ = ["FrankensteinMLMModel"]
__version__ = "0.1.0"
