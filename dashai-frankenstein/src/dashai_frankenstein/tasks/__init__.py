"""Frankenstein DashAI task components."""
from dashai_frankenstein.tasks.causal_lm_task import CausalLMPretrainingTask  # noqa: F401
from dashai_frankenstein.tasks.segmentation import SegmentationTask  # noqa: F401

__all__ = ["CausalLMPretrainingTask", "SegmentationTask"]