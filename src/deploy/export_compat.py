"""Flat-name view over the nested model configuration for export code.

The HF/GGUF exporters were written against the historical flat configuration
vocabulary (``use_bitnet``, ``hidden_size``, ``top_k_experts``, …). This
adapter materialises that vocabulary from a nested config **without** touching
the runtime configuration pipeline: it is a read-only serialization view for
the export metadata, not a second config path.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = ["flat_config_view", "model_config_from"]


def model_config_from(model_config: Mapping[str, Any]):
    """Build a nested configuration from a raw ``model:`` mapping.

    Args:
        model_config: Nested (or legacy flat) model block.

    Returns:
        A :class:`~src.model.config.FrankensteinModelConfig` instance.
    """
    from src.model.config import FrankensteinModelConfig

    return FrankensteinModelConfig.from_dict(dict(model_config or {}))


def flat_config_view(model_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the historical flat-key view of a model configuration.

    Args:
        model_config: Nested (or legacy flat) model block.

    Returns:
        A dict with the flat names the exporters consume.
    """
    from src.model.config import flat_name_set

    nested_markers = {
        "dims", "norm", "embedding", "attention", "positional_encoding",
        "positional_encoding_parameters", "moe", "bitnet",
        "mixture_of_depths", "ssmax", "numerics", "ffn",
        "classification_head", "mhc", "residuals",
    }
    if not hasattr(model_config, "to_jsonable"):
        if isinstance(model_config, Mapping) and not (nested_markers & set(model_config)):
            # Already flat (export metadata fragments): pass through without
            # cross-validating a partial dict against the nested config.
            return {key: value for key, value in model_config.items()}
        config = model_config_from(model_config)
    else:
        config = model_config
    view: Dict[str, Any] = {}
    import dataclasses

    for name in flat_name_set():
        try:
            value = getattr(config, name)
        except AttributeError:  # pragma: no cover - defensive
            continue
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        view[name] = value
    view["token_vocab"] = getattr(config, "token_vocab", None)
    view["num_classes"] = int(config.num_classes)
    view["pooling_mode"] = str(config.pooling_mode)
    view["model_class"] = getattr(config, "model_class", None)
    return view
