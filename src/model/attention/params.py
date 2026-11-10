"""Per-mixer parameter binding helpers.

Mixers receive a :class:`~src.registry.binding.ComponentParams` namespace with
their knobs (bound from ``model.attention.<mixer>`` through the mixer's
schema fragment). When a mixer is constructed directly in tests or by third
party code without ``params``, :func:`mixer_params` derives them from the
config's nested ``attention`` mapping so there is no silent flat namespace.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["mixer_params", "use_pe_for"]


def mixer_params(
    config: Any,
    name: str,
    params: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Return the bound parameter namespace for a mixer.

    Args:
        config: Model configuration (nested shape).
        name: Mixer registry name (or alias).
        params: Already-bound parameters, returned unchanged when provided.

    Returns:
        A :class:`~src.registry.binding.ComponentParams` namespace.
    """
    if params is not None:
        return params
    from src.registry import ComponentFamily, bind_params, ensure_builtins, registry

    ensure_builtins()
    canonical = registry.resolve_name(ComponentFamily.MIXER, name)
    spec = registry.get(ComponentFamily.MIXER, canonical)
    attention = getattr(config, "attention", None)
    raw: Mapping[str, Any] = {}
    if isinstance(attention, Mapping):
        raw = attention.get(canonical) or attention.get(name) or {}
    return bind_params(spec, raw)


def use_pe_for(config: Any, name: str, default: bool = True) -> bool:
    """Return whether a mixer consumes the model-wide positional encoding.

    Args:
        config: Model configuration (nested shape).
        name: Mixer registry name (or alias).
        default: Fallback when neither the config nor the spec decides.

    Returns:
        ``True`` when the mixer should apply the positional encoding.
    """
    from src.registry import ComponentFamily, ensure_builtins, registry

    ensure_builtins()
    canonical = registry.resolve_name(ComponentFamily.MIXER, name)
    parameters = getattr(config, "positional_encoding_parameters", None)
    if isinstance(parameters, Mapping):
        flags = parameters.get("use_pe") or {}
        key = f"{canonical}_use_pe"
        if isinstance(flags, Mapping) and flags.get(key) is not None:
            return bool(flags[key])
    spec = registry.maybe_get(ComponentFamily.MIXER, canonical)
    if spec is not None:
        return bool(getattr(spec, "needs_pos_encoder", default))
    return bool(default)
