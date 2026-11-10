"""Component registry and declarative plugin API.

Public surface:

* :data:`registry` — process-global :class:`ComponentRegistry`.
* :data:`register_spec` / :func:`register` — registration entry points.
* :class:`ComponentSpec` and the per-family spec dataclasses.
* :func:`bind_params` / :class:`ComponentParams` — parameter binding.
* :func:`ensure_builtins` — register every built-in component (idempotent).

The package itself is torch-free: builders are lazy references resolved on
first use.
"""

from __future__ import annotations

from .binding import ComponentParams, bind_params, coerce_value
from .builtin import ensure_builtins, reset_builtins
from .errors import (
    ComponentError,
    DuplicateComponentError,
    InvalidSpecError,
    PluginLoadError,
    UnknownComponentError,
)
from .registry import ComponentRegistry, register_spec, registry, resolve_builder
from .spec import (
    SPEC_CLASSES,
    ActivationSpec,
    CacheSpec,
    ComponentFamily,
    ComponentSpec,
    EmbeddingSpec,
    MixerSpec,
    NormSpec,
    OptimizerSpec,
    PESpec,
    ParamBinding,
    ResidualSpec,
)

register = register_spec
"""Alias kept for the plugin authoring API."""

__all__ = [
    "registry",
    "register",
    "register_spec",
    "resolve_builder",
    "ComponentRegistry",
    "ComponentSpec",
    "ComponentFamily",
    "MixerSpec",
    "OptimizerSpec",
    "NormSpec",
    "ActivationSpec",
    "ResidualSpec",
    "PESpec",
    "EmbeddingSpec",
    "CacheSpec",
    "ParamBinding",
    "SPEC_CLASSES",
    "ComponentParams",
    "bind_params",
    "coerce_value",
    "ensure_builtins",
    "reset_builtins",
    "ComponentError",
    "DuplicateComponentError",
    "UnknownComponentError",
    "InvalidSpecError",
    "PluginLoadError",
]
