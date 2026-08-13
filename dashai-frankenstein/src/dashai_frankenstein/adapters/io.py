"""IO adapter: DashAI run directory <-> Frankenstein checkpoint bundle.

Each DashAI run persists artifacts under a run-specific directory. This module
bridges the DashAI ``save(filename)``/``load(filename)`` contract to the
Frankenstein engine's :func:`save_checkpoint`/:func:`load_checkpoint`, which
write a self-contained bundle (``model.pt`` + ``config.yaml`` + ``tokenizer/``
+ ``dashai_meta.json``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


def save_run(
    filename: str,
    model: Any,
    loaded_config: Any,
    tokenizer: Any,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a Frankenstein model bundle into the DashAI run directory.

    Parameters
    ----------
    filename : str
        DashAI run directory (created if missing).
    model : torch.nn.Module
        The trained Frankenstein model.
    loaded_config : LoadedTrainingConfig
        The validated Frankenstein config (round-tripped as config.yaml).
    tokenizer : Any
        Tokenizer persisted alongside the weights.
    extra : dict, optional
        Extra metadata (e.g. ``{"num_labels": N, "label_encodings": {...}}``).

    Returns
    -------
    str
        Path to the saved ``model.pt``.
    """
    from dashai_frankenstein.engine import save_checkpoint

    os.makedirs(filename, exist_ok=True)
    path = save_checkpoint(filename, model, loaded_config, tokenizer, extra=extra)

    # Drop a small DashAI-facing marker so the load path can sanity-check.
    marker = {"plugin": "dashai-frankenstein", "version": "0.1.0"}
    with open(os.path.join(filename, "plugin_marker.json"), "w", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2)
    return path


def load_run(filename: str) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Reload a bundle saved by :func:`save_run`.

    Rebuilds the ``nn.Module`` from ``config.yaml`` via the Frankenstein engine
    and restores weights. ``extra`` (including ``num_labels``) is applied to the
    config before building so the classification head is reconstructed.

    Returns
    -------
    tuple
        ``(model, loaded_config, tokenizer, extra)``.
    """
    from dashai_frankenstein.engine import load_checkpoint

    return load_checkpoint(filename)
