"""Thin facade over the Frankenstein engine API.

This wraps :mod:`src.engine` (the Phase-0 non-CLI engine in
``frankenstein-transformer``) so the DashAI adapter components do not import
Frankenstein internals directly. All Frankenstein interaction — model
construction, checkpoint save/load, and YAML parsing — goes through here.

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
    "build_model_from_yaml",
    "resolve_tokenizer",
    "load_checkpoint",
    "resolve_torch_device",
    "save_checkpoint",
    "load_training_config",
    "resolve_device",
]


def resolve_device(label: str) -> str:
    """Map a DashAI device label (``"GPU"``/``"CPU"``) to a torch device string.

    Parameters
    ----------
    label : str
        DashAI device label.

    Returns
    -------
    str
        A torch device string accepted by the Frankenstein engine.
    """
    label = str(label or "").strip().lower()
    if label in {"gpu", "cuda"}:
        return "cuda"
    if label.startswith("gpu"):
        # DashAI enumerates "GPU 0: <name> - ..." for multi-GPU hosts.
        return "cuda"
    return "cpu"


def build_model_from_yaml(
    frankenstein_yaml: str,
    *,
    model_class_override: Optional[str] = None,
    num_labels: Optional[int] = None,
    tokenizer_name_or_path: Optional[str] = None,
    vocab_size_override: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Build a Frankenstein model + config + tokenizer from a YAML payload.

    The YAML is written to a temp file and validated by Frankenstein's own
    ``load_training_config`` (so the Frankenstein schema stays the source of
    truth). When ``num_labels`` is provided the encoder classification head
    (Strategy A) is enabled. When ``vocab_size_override`` is provided it is
    injected as ``model.dims.vocab_size`` so the embedding matches the tokenizer
    (Frankenstein constraint: vocab_size must match the tokenizer's vocab).

    Parameters
    ----------
    frankenstein_yaml : str
        A full Frankenstein training YAML document.
    model_class_override : str, optional
        Force a ``model_class`` (e.g. ``"frankensteindecoder"``).
    num_labels : int, optional
        Number of classes for the encoder classification head.
    tokenizer_name_or_path : str, optional
        Injected as ``tokenizer.name_or_path`` when the YAML omits it.
    vocab_size_override : int, optional
        Injected as ``model.dims.vocab_size`` to match the resolved tokenizer.

    Returns
    -------
    tuple
        ``(model, loaded_config, tokenizer)`` where ``loaded_config`` is a
        :class:`LoadedTrainingConfig` and ``tokenizer`` may be ``None`` for
        vision/decoder paths that build lazily.
    """
    import os
    import tempfile

    parsed: Dict[str, Any]
    if isinstance(frankenstein_yaml, dict):
        parsed = dict(frankenstein_yaml)
    else:
        import yaml

        parsed = yaml.safe_load(frankenstein_yaml) or {}

    if model_class_override:
        parsed["model_class"] = model_class_override
    if tokenizer_name_or_path and "tokenizer" not in parsed:
        parsed["tokenizer"] = {"name_or_path": tokenizer_name_or_path}
    if vocab_size_override is not None:
        model_block = parsed.setdefault("model", {})
        dims_block = model_block.setdefault("dims", {})
        dims_block["vocab_size"] = int(vocab_size_override)

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
    # YAML's tokenizer.name_or_path / base_model), not here — keeps the facade
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
