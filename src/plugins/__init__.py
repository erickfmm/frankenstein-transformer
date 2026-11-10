"""Plugin authoring API and layered loader."""

from __future__ import annotations

from .api import (
    register_activation,
    register_embedding,
    register_mixer,
    register_norm,
    register_optimizer,
    register_posenc,
    register_residual,
)
from .loader import load_entry_point_plugins, load_plugin_paths, load_plugins

__all__ = [
    "load_plugins",
    "load_plugin_paths",
    "load_entry_point_plugins",
    "register_mixer",
    "register_optimizer",
    "register_norm",
    "register_activation",
    "register_residual",
    "register_posenc",
    "register_embedding",
]
