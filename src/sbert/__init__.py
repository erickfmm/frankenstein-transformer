"""
SBERT v2 Module for Frankenstein Transformer

This module provides fine-tuning and inference capabilities for
Sentence-BERT models based on Frankenstein v2 architecture.
"""

from .inference_sbert import SBERTInference, SimilarityResult
from .train_sbert import FrankensteinSentenceTransformer, SBERTTrainer

__all__ = [
    'SBERTInference',
    'SimilarityResult',
    'FrankensteinSentenceTransformer',
    'SBERTTrainer'
]

__version__ = '2.0.0'
