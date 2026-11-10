"""Exception hierarchy for the component registry and plugin loader."""

from __future__ import annotations

__all__ = [
    "ComponentError",
    "DuplicateComponentError",
    "UnknownComponentError",
    "InvalidSpecError",
    "PluginLoadError",
]


class ComponentError(ValueError):
    """Base class for every registry / plugin error."""


class DuplicateComponentError(ComponentError):
    """A different spec was registered under an existing (family, name) key."""


class UnknownComponentError(ComponentError):
    """A requested component name is not registered in its family."""


class InvalidSpecError(ComponentError):
    """A component spec is malformed."""


class PluginLoadError(ComponentError):
    """A config-declared plugin module could not be imported or registered."""
