"""
SBERT v2 Module for TORMENTED-BERT-Frankenstein

This module provides fine-tuning and inference capabilities for
Sentence-BERT models based on TORMENTED-BERT v2 architecture.
"""

from .inference_sbert import SBERTInference, SimilarityResult
from .train_sbert import TormentedBertSentenceTransformer, SBERTTrainer

__all__ = [
    'SBERTInference',
    'SimilarityResult',
    'TormentedBertSentenceTransformer',
    'SBERTTrainer'
]

__version__ = '2.0.0'
