"""Downgrade unknown-key schema violations to logged warnings.

The JSON-Schema modules under ``src/schema/**`` keep
``additionalProperties: false`` so the schema remains the executable
contract and documents every known configuration key. User configs may
still carry keys the schema does not know *yet* (the schema evolves on its
own schedule) or plain typos. Rather than aborting the run, those
``additionalProperties`` violations are logged as ``WARNING`` records and
ignored; every other violation (bad enum, wrong type, failing conditional
rule, ...) stays a hard error.

The schema itself stays strict: the in-repo presets (``configs/*.yaml``
and ``configs/examples/*.yaml``) must still validate cleanly under
``tests/test_schema_consistency.py``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Type

__all__ = [
    "classify_schema_errors",
    "filter_dataclass_kwargs",
    "warn_extra_keys",
]

LOGGER = logging.getLogger("frankenstein.config")

#: ``jsonschema`` reports an unknown key through this validator keyword.
_EXTRA_KEY_VALIDATOR = "additionalProperties"


def _location(error: Any) -> str:
    """Return the JSON path of *error* as ``a.b.c`` (``<root>`` when empty)."""
    path = ".".join(str(part) for part in getattr(error, "absolute_path", ()))
    return path or "<root>"


def classify_schema_errors(
    errors: Iterable[Any],
) -> Tuple[List[Any], List[Any]]:
    """Split jsonschema errors into ``(hard_errors, extra_key_errors)``.

    Args:
        errors: Iterable of ``jsonschema.ValidationError`` objects.

    Returns:
        A tuple of two lists: violations that must still fail the load, and
        ``additionalProperties`` violations (unknown keys) that callers may
        downgrade to warnings.
    """
    hard: List[Any] = []
    extra_keys: List[Any] = []
    for error in errors:
        if getattr(error, "validator", None) == _EXTRA_KEY_VALIDATOR:
            extra_keys.append(error)
        else:
            hard.append(error)
    return hard, extra_keys


def warn_extra_keys(
    errors: Sequence[Any],
    *,
    source: str,
    prefix: str = "",
) -> None:
    """Log unknown-key schema violations as ``WARNING`` records.

    Args:
        errors: ``additionalProperties`` violations returned by
            :func:`classify_schema_errors`.
        source: Human-readable config identifier (usually the file path).
        prefix: Optional path prefix (``"inference"`` for the runtime block).
    """
    for error in errors:
        location = _location(error)
        if prefix:
            location = prefix if location == "<root>" else f"{prefix}.{location}"
        LOGGER.warning(
            "[schema] %s: %s: unknown key(s) ignored (not declared in "
            "src/schema/**): %s",
            source,
            location,
            getattr(error, "message", error),
        )


def filter_dataclass_kwargs(
    cls: Type[Any],
    kwargs: Dict[str, Any],
    *,
    source: str,
    warn: bool = True,
) -> Dict[str, Any]:
    """Drop mapping keys that are not fields of dataclass *cls*.

    Used as the last line of defence before ``cls(**kwargs)``: schema
    validation already reports unknown keys, but the flattener also feeds
    legacy (flat) documents and checkpoint dicts that bypass the schema.

    Args:
        cls: Target dataclass type.
        kwargs: Candidate keyword arguments.
        source: Human-readable config identifier used in warnings.
        warn: Emit one warning per dropped key. Set to ``False`` when the
            caller already reported the same keys through schema validation.

    Returns:
        A new mapping containing only dataclass field names.
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(key for key in kwargs if key not in field_names)
    if warn:
        for key in unknown:
            LOGGER.warning(
                "[schema] %s: unknown key '%s' ignored (not a %s field and "
                "not declared in src/schema/**)",
                source,
                key,
                cls.__name__,
            )
    if not unknown:
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in field_names}
