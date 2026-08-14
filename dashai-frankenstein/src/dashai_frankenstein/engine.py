"""Thin facade over the Frankenstein engine API.

This wraps :mod:`src.engine` (the Phase-0 non-CLI engine in
``frankenstein-transformer``) so the DashAI adapter components do not import
Frankenstein internals directly. All Frankenstein interaction — model
construction, checkpoint save/load, and config parsing — goes through here.

The user-facing field carries a **single-line JSON** string (the DashAI form
field is single-line). It is parsed with ``json.loads`` here, then re-serialized
to a ``.yaml`` temp file consumed by Frankenstein's ``load_training_config``
(which reads YAML). JSON is a subset of YAML, so the schema and loader are
agnostic to the input format.

If Frankenstein is not installed, the import fails loudly with a clear message.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from src.engine import (  # type: ignore
        SUPPORTED_DEVICE_CHOICES,
        TrainResult,
        build_model,
        load_checkpoint,
        resolve_torch_device,
        save_checkpoint,
    )
    from src.training.config_loader import load_training_config  # type: ignore
    from src.training.trainer import TrainingConfig  # type: ignore
    from src.model.config import FrankensteinModelConfig  # type: ignore
    from src.utils.schema_loader import resolve_schema  # type: ignore
except ImportError as exc:  # pragma: no cover - depends on install env
    raise ImportError(
        "frankenstein-transformer is required by the dashai-frankenstein plugin. "
        "Install it with `pip install frankenstein-transformer>=1.1.0`."
    ) from exc


__all__ = [
    "SUPPORTED_DEVICE_CHOICES",
    "TrainResult",
    "TrainingConfig",
    "FrankensteinModelConfig",
    "build_model",
    "build_model_from_json",
    "resolve_tokenizer",
    "load_checkpoint",
    "resolve_torch_device",
    "save_checkpoint",
    "load_training_config",
    "resolve_schema",
    "resolve_device",
    "validate_training_json",
]


def resolve_device(label: str) -> str:
    """Map a DashAI/Frankenstein device label to a torch device string.

    Accepts DashAI labels (``"GPU"``/``"CPU"``) and Frankenstein/runtime
    labels (``"auto"``, ``"cuda"``, ``"cpu"``, ``"mps"``). ``"auto"`` selects
    CUDA if available, then MPS, falling back to CPU (via
    :func:`src.utils.device.resolve_torch_device`).

    Parameters
    ----------
    label : str
        Device label (``"GPU"``, ``"CPU"``, ``"auto"``, ``"cuda"``, ``"mps"``).

    Returns
    -------
    str
        A torch device string accepted by the Frankenstein engine
        (``"cuda"``, ``"cpu"``, or ``"mps"``).
    """
    label = str(label or "auto").strip().lower()
    if label in {"gpu", "cuda"} or label.startswith("gpu"):
        # DashAI enumerates "GPU 0: <name> - ..." for multi-GPU hosts.
        return "cuda"
    if label in {"cpu"}:
        return "cpu"
    # ``auto``, ``cuda``, ``mps`` — delegate to Frankenstein's resolver,
    # which checks availability and raises ValueError if unsupported.
    return resolve_torch_device(label)


def validate_training_json(json_text: str) -> None:
    """Validate a Frankenstein training config (one-line JSON) before launch.

    Thin facade over :func:`dashai_frankenstein.validate.validate_json` so the
    engine remains the single entry point for Frankenstein interaction.

    Parameters
    ----------
    json_text : str
        A full Frankenstein training config as a single-line JSON string.

    Raises
    ------
    ValueError
        If the JSON fails schema or config-loader validation.
    """
    from dashai_frankenstein.validate import validate_json

    validate_json(json_text)


def build_model_from_json(
    frankenstein_json: str,
    *,
    model_class_override: Optional[str] = None,
    num_labels: Optional[int] = None,
    tokenizer_name_or_path: Optional[str] = None,
    vocab_size_override: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any]:
    """Build a Frankenstein model + config + tokenizer from a JSON payload.

    The JSON is parsed, merged with any overrides, written to a ``.yaml`` temp
    file, and validated by Frankenstein's own ``load_training_config`` (so the
    Frankenstein schema stays the source of truth). When ``num_labels`` is
    provided the encoder classification head (Strategy A) is enabled. When
    ``vocab_size_override`` is provided it is injected as
    ``model.dims.vocab_size`` so the embedding matches the tokenizer
    (Frankenstein constraint: vocab_size must match the tokenizer's vocab).
    Arbitrary nested keys can be injected via ``overrides`` (e.g. ViT
    ``model.num_classes`` / ``model.image_height``).

    Parameters
    ----------
    frankenstein_json : str
        A full Frankenstein training config as a single-line JSON string.
    model_class_override : str, optional
        Force a ``model_class`` (e.g. ``"frankensteindecoder"``).
    num_labels : int, optional
        Number of classes for the encoder classification head.
    tokenizer_name_or_path : str, optional
        Injected as ``tokenizer.name_or_path`` when the JSON omits it.
    vocab_size_override : int, optional
        Injected as ``model.dims.vocab_size`` to match the resolved tokenizer.
    overrides : dict, optional
        Nested dict deep-merged into the parsed JSON before validation. Use
        dotted-free nested form, e.g. ``{"model": {"num_classes": 10}}``.

    Returns
    -------
    tuple
        ``(model, loaded_config, tokenizer)`` where ``loaded_config`` is a
        :class:`LoadedTrainingConfig` and ``tokenizer`` may be ``None`` for
        vision/decoder paths that build lazily.
    """
    import json
    import os
    import tempfile

    parsed: Dict[str, Any]
    if isinstance(frankenstein_json, dict):
        parsed = dict(frankenstein_json)
    else:
        parsed = json.loads(frankenstein_json) or {}

    if model_class_override:
        parsed["model_class"] = model_class_override
    if tokenizer_name_or_path and "tokenizer" not in parsed:
        parsed["tokenizer"] = {"name_or_path": tokenizer_name_or_path}
    if vocab_size_override is not None:
        model_block = parsed.setdefault("model", {})
        dims_block = model_block.setdefault("dims", {})
        dims_block["vocab_size"] = int(vocab_size_override)
    if overrides:
        _deep_merge(parsed, overrides)

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="dashai_frank_")
    try:
        import yaml

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(parsed, handle, sort_keys=False)
        loaded = load_training_config(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    model = build_model(loaded.model_class, loaded.model_config, num_labels=num_labels)

    # The tokenizer is resolved by the model component (HF AutoTokenizer from the
    # JSON's tokenizer.name_or_path / base_model), not here — keeps the facade
    # focused on model construction and the vocab constraint.
    tokenizer: Any = None

    return model, loaded, tokenizer


def resolve_tokenizer(loaded: Any) -> Any:
    """Resolve an HF tokenizer for a loaded Frankenstein config.

    Prefers ``tokenizer.name_or_path``, then ``base_model``. Returns ``None``
    if neither is set (vision paths).

    Parameters
    ----------
    loaded : LoadedTrainingConfig
        The validated Frankenstein config.

    Returns
    -------
    transformers.PreTrainedTokenizer or None
    """
    from transformers import AutoTokenizer

    tok_cfg = loaded.tokenizer_config or {}
    name = str(tok_cfg.get("name_or_path", "")).strip()
    if not name and loaded.base_model:
        name = str(loaded.base_model)
    if not name:
        return None
    trust_remote_code = bool(tok_cfg.get("trust_remote_code", False))
    use_fast = bool(tok_cfg.get("use_fast", True))
    tokenizer = AutoTokenizer.from_pretrained(
        name, use_fast=use_fast, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``extra`` into ``base`` (in place); returns ``base``.

    Lists are replaced (not extended) to keep behavior predictable.

    Parameters
    ----------
    base : dict
        Target dict (mutated).
    extra : dict
        Source dict whose values win on conflict.

    Returns
    -------
    dict
        The merged ``base`` dict.
    """
    for key, value in extra.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base