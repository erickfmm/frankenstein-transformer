"""Frankenstein DashAI model components.

The package re-exports the component classes so they can be referenced from the
``dashai.plugins`` entry points as ``dashai_frankenstein:ClassName``.
"""
from dashai_frankenstein.models.mlm import FrankensteinMLMModel  # noqa: F401

__all__ = ["FrankensteinMLMModel"]
