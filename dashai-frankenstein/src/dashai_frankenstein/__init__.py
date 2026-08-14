"""dashai-frankenstein — DashAI plugin for Frankenstein Transformer.

Registers Frankenstein model classes (encoder, decoder, ViT classifier/segmenter)
and a SegmentationTask as DashAI components, discovered via the
``dashai.plugins`` entry-points group.
"""
from dashai_frankenstein.models.decoder import FrankensteinDecoderModel  # noqa: F401
from dashai_frankenstein.models.mlm import FrankensteinMLMModel  # noqa: F401
from dashai_frankenstein.models.vit_classifier import FrankensteinViTClassifier  # noqa: F401
from dashai_frankenstein.models.vit_segmenter import FrankensteinViTSegmenter  # noqa: F401
from dashai_frankenstein.tasks.segmentation import SegmentationTask  # noqa: F401

__all__ = [
    "FrankensteinMLMModel",
    "FrankensteinDecoderModel",
    "FrankensteinViTClassifier",
    "FrankensteinViTSegmenter",
    "SegmentationTask",
]
__version__ = "0.2.0"
