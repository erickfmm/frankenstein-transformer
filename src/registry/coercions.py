"""Torch-free coercion helpers for declarative parameter bindings.

These mirror the semantics of :mod:`src.model.optimizer.base` (``to_float``,
``to_int``, ``to_betas``) without importing torch, so optimizer specs can be
imported by the registry and schema assembly in torch-free contexts.
"""

from __future__ import annotations

from typing import Any, Tuple

__all__ = [
    "as_float",
    "as_int",
    "as_bool",
    "as_lower_str",
    "as_betas",
    "as_optional_int",
]


def as_float(value: Any, default: Any) -> float:
    """Cast to float, falling back to ``default``.

    Args:
        value: Raw value (may be ``None`` or uncoercible).
        default: Fallback value.

    Returns:
        The float value or the fallback.
    """
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_int(value: Any, default: Any) -> int:
    """Cast to int, falling back to ``default``.

    Args:
        value: Raw value.
        default: Fallback value.

    Returns:
        The int value or the fallback.
    """
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def as_optional_int(value: Any, default: Any = None) -> Any:
    """Cast to int, preserving an explicit absence as ``None``.

    Args:
        value: Raw value.
        default: Fallback (``None`` by default).

    Returns:
        The int value, ``None`` when absent, or the fallback.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: Any = False) -> bool:
    """Cast to bool with YAML-friendly string parsing.

    Args:
        value: Raw value.
        default: Fallback value.

    Returns:
        The boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(default)


def as_lower_str(value: Any, default: Any = "") -> str:
    """Cast to a stripped, lower-cased string.

    Args:
        value: Raw value.
        default: Fallback value.

    Returns:
        The normalized string.
    """
    if value is None:
        return str(default).strip().lower()
    return str(value).strip().lower()


def as_betas(value: Any, default: Tuple[float, float] = (0.9, 0.95)) -> Tuple[float, float]:
    """Cast to a ``(beta1, beta2)`` pair.

    Args:
        value: Raw value, expected to be a two-element sequence.
        default: Fallback pair.

    Returns:
        A two-element tuple of floats.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return default
    return default
