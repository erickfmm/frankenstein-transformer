"""Pre-launch validation of the Frankenstein training config (one-line JSON).

The Frankenstein JSON Schema (``src/schema.yaml`` + ``src/schema/*.yaml``) is the
single source of truth: it enforces ``additionalProperties: false`` and the enum
ranges. The config loader (:func:`src.training.config_loader.load_training_config`)
adds cross-component constraints (``hidden_size % num_heads``, BitNet flags,
optimizer presence, task/model_class compatibility, …).

This module runs both checks on a raw JSON string (single-line, as accepted by
the DashAI single-line text field) and raises a single :class:`ValueError` with
a concatenated, user-readable message on failure.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

import yaml


def _frankenstein_schema_path() -> str:
    """Return the absolute path to the Frankenstein ``src/schema.yaml``.

    Resolves it relative to the imported ``src`` package so it works both in
    editable (``-e``) and installed mode.

    Returns
    -------
    str
        Absolute path to ``schema.yaml``.

    Raises
    ------
    RuntimeError
        If the Frankenstein ``src`` package cannot be imported.
    """
    try:
        import src  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "frankenstein-transformer is not installed (cannot import 'src'). "
            "Install it before using the dashai-frankenstein plugin."
        ) from exc
    return os.path.join(os.path.dirname(os.path.abspath(src.__file__)), "schema.yaml")


def validate_json(json_text: str) -> None:
    """Validate a Frankenstein training config (one-line JSON) against schema + loader.

    Runs three stages, each prefixed in the error message so the user knows
    which check failed:

    1. ``json.loads`` — catches JSON syntax errors (the DashAI field is
       single-line, so the config is expected as a one-line JSON string).
    2. ``jsonschema.validate`` against the resolved Frankenstein JSON Schema —
       catches ``additionalProperties: false`` and enum violations.
    3. ``load_training_config`` on a temp file — catches cross-component
       constraints (``hidden_size % num_heads``, ``num_kv_heads`` divides
       ``num_heads``, BitNet flags, ffn activation, optimizer presence,
       task/model_class compatibility, ``frankensteindecoder`` forces
       ``mode: decoder``, vision tasks require ``frankenstein_vit``, …).

    Parameters
    ----------
    json_text : str
        A full Frankenstein training config as a single-line JSON string.

    Raises
    ------
    ValueError
        If any stage fails, with a message identifying the failing stage.
    """
    if not json_text or not str(json_text).strip():
        raise ValueError(
            "frankenstein_json is empty — provide a one-line JSON config."
        )

    text = str(json_text)

    # Stage 1 — JSON syntax.
    try:
        parsed: Dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"[JSON parse] {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"[JSON parse] Top-level JSON node must be an object, got "
            f"{type(parsed).__name__}."
        )

    # Stage 2 — JSON Schema (additionalProperties:false + enums).
    try:
        from src.utils.schema_loader import resolve_schema  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "frankenstein-transformer is not installed (cannot import "
            "'src.utils.schema_loader')."
        ) from exc

    import jsonschema

    schema_path = _frankenstein_schema_path()
    schema = resolve_schema(schema_path)
    try:
        jsonschema.validate(instance=parsed, schema=schema)
    except jsonschema.ValidationError as exc:
        loc = ".".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ValueError(f"[JSON Schema] {loc}: {exc.message}") from exc

    # Stage 3 — config loader (cross-component constraints).
    try:
        from src.training.config_loader import load_training_config  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "frankenstein-transformer is not installed (cannot import "
            "'src.training.config_loader')."
        ) from exc

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="dashai_frank_val_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(parsed, handle, sort_keys=False)
        try:
            load_training_config(tmp_path)
        except ValueError as exc:
            raise ValueError(f"[Config loader] {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — surface any loader error
            raise ValueError(f"[Config loader] {exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass