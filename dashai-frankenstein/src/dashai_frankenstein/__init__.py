"""dashai-frankenstein — DashAI plugin for Frankenstein Transformer.

Registers Frankenstein model classes (encoder, decoder, ViT
classifier/segmenter, MLM/causal-LM pretrainers) and the SegmentationTask /
MaskedLanguageModelingTask / CausalLMPretrainingTask DashAI tasks,
discovered via the ``dashai.plugins`` entry-points group.
"""
from dashai_frankenstein.models.causal_lm_model import FrankensteinCausalLMModel  # noqa: F401
from dashai_frankenstein.models.decoder import FrankensteinDecoderModel  # noqa: F401
from dashai_frankenstein.models.mlm import FrankensteinMLMModel  # noqa: F401
from dashai_frankenstein.models.pretrainer import FrankensteinPretrainer  # noqa: F401
from dashai_frankenstein.models.vit_classifier import FrankensteinViTClassifier  # noqa: F401
from dashai_frankenstein.models.vit_segmenter import FrankensteinViTSegmenter  # noqa: F401
from dashai_frankenstein.tasks.causal_lm_task import CausalLMPretrainingTask  # noqa: F401
from dashai_frankenstein.tasks.mlm_task import MaskedLanguageModelingTask  # noqa: F401
from dashai_frankenstein.tasks.segmentation import SegmentationTask  # noqa: F401

__all__ = [
    "FrankensteinMLMModel",
    "FrankensteinDecoderModel",
    "FrankensteinPretrainer",
    "FrankensteinCausalLMModel",
    "FrankensteinViTClassifier",
    "FrankensteinViTSegmenter",
    "MaskedLanguageModelingTask",
    "CausalLMPretrainingTask",
    "SegmentationTask",
]
__version__ = "0.3.0"