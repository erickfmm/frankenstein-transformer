"""Normalize the ``config`` payload stored inside training checkpoints.

Trainers persist ``checkpoint["config"] = self.config``, i.e. a live
:class:`~src.model.config.FrankensteinModelConfig` dataclass (nested config,
decision D15). Older checkpoints stored the plain ``model:`` mapping instead,
and the Transformers exporter historically accepted both. Centralizing the
conversion here keeps ``deploy`` and ``transformers_export`` from re-deriving
it (a mismatch caused every e2e deploy to fail with
``'FrankensteinModelConfig' object has no attribute 'get'``).

The module deliberately imports nothing beyond the standard library: it is
used by torch-heavy entrypoints that must not pay for (or depend on)
``transformers``.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

__all__ = ["as_plain_dict"]


def as_plain_dict(value: Any) -> Dict[str, Any]:
    """Return ``value`` as a plain ``dict`` for schema-style lookups.

    Args:
        value: A checkpoint/config payload: ``None``, a mapping, a dataclass
            instance (what the trainer writes) or any object exposing
            ``__dict__``.

    Returns:
        A shallow copy of the payload as a ``dict``; ``{}`` for ``None``.

    Raises:
        TypeError: If the payload cannot be interpreted as a mapping.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported config payload type: {type(value)!r}")
