"""Process-global component registry.

The registry is the single source of truth for every pluggable component.
It never imports torch or any ``src.model`` package: builders are lazy
``"module:attribute"`` references resolved on first use, so schema assembly,
plugin discovery and documentation generation stay import-cheap and
CPU/torch-free.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple

from .errors import DuplicateComponentError, InvalidSpecError, UnknownComponentError
from .spec import ComponentFamily, ComponentSpec

__all__ = ["ComponentRegistry", "registry", "register_spec", "resolve_builder"]


@functools.lru_cache(maxsize=None)
def _import_builder(target: str) -> Any:
    """Resolve a ``"module:attribute"`` builder reference (cached).

    Args:
        target: Import string, e.g.
            ``"src.model.optimizer.muon2:Muon2"``. Dotted attributes are
            supported after the colon.

    Returns:
        The referenced object.
    """
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    if not attribute:
        return module
    resolved: Any = module
    for part in attribute.split("."):
        resolved = getattr(resolved, part)
    return resolved


def resolve_builder(target: Any) -> Any:
    """Return the callable behind a spec ``builder`` field.

    Args:
        target: A ``"module:attribute"`` string or a ready callable.

    Returns:
        The resolved callable.

    Raises:
        ImportError: If the module/attribute cannot be imported.
    """
    if isinstance(target, str):
        return _import_builder(target)
    return target


def _builder_key(builder: Any) -> str:
    """Return a stable key for a builder reference.

    Args:
        builder: ``"module:attribute"`` string or callable.

    Returns:
        The reference string.
    """
    if isinstance(builder, str):
        return builder
    return (
        f"{getattr(builder, '__module__', '')}:"
        f"{getattr(builder, '__qualname__', repr(builder))}"
    )


def _fingerprint(spec: ComponentSpec) -> tuple:
    """Return a stable fingerprint used for idempotent re-registration.

    Args:
        spec: Component spec.

    Returns:
        A hashable tuple capturing everything that affects behavior.
    """
    return (
        spec.name,
        spec.family,
        _builder_key(spec.builder),
        repr(spec.params_schema),
        repr(dict(spec.defaults)),
        spec.description,
        spec.description_es,
        tuple(spec.references),
        getattr(spec, "training_free", None),
        getattr(spec, "needs_pos_encoder", None),
        repr(getattr(spec, "cache", None)),
        getattr(spec, "cache_implemented", None),
        getattr(spec, "qk_norm_supported", None),
        getattr(spec, "qk_clip_supported", None),
        tuple(getattr(spec, "aliases", ()) or ()),
        repr(sorted(getattr(spec, "param_bindings", {}) or {})),
        getattr(spec, "group_policy", None),
        getattr(spec, "is_glu", None),
        getattr(spec, "supports_ffn_fusion", None),
        getattr(spec, "params_key", None),
    )


class ComponentRegistry:
    """Per-family spec maps with duplicate detection and derived views."""

    def __init__(self) -> None:
        """Initialize an empty registry with one map per family."""
        self._specs: Dict[ComponentFamily, Dict[str, ComponentSpec]] = {
            family: {} for family in ComponentFamily
        }

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, spec: ComponentSpec) -> None:
        """Register one component spec.

        Re-registering an identical spec is idempotent (module reimports under
        pytest / Streamlit reloads). A *different* spec under an existing
        ``(family, name)`` key raises :class:`DuplicateComponentError`.

        Args:
            spec: The declarative component record.

        Raises:
            InvalidSpecError: If ``spec`` is not a ComponentSpec.
            DuplicateComponentError: If a different spec already uses the key.
        """
        if not isinstance(spec, ComponentSpec):
            raise InvalidSpecError(
                f"register() expects a ComponentSpec, got {type(spec).__name__}"
            )
        bucket = self._specs[spec.family]
        existing = bucket.get(spec.name)
        if existing is not None:
            if _fingerprint(existing) == _fingerprint(spec):
                return
            raise DuplicateComponentError(
                f"Component '{spec.name}' is already registered in family "
                f"'{spec.family.value}'. A plugin may not shadow a built-in "
                "or another plugin; choose a unique name."
            )
        bucket[spec.name] = spec

    def register_all(self, specs: Iterable[ComponentSpec]) -> None:
        """Register every spec in ``specs``.

        Args:
            specs: Iterable of component specs.
        """
        for spec in specs:
            self.register(spec)

    def unregister(self, family: ComponentFamily, name: str) -> None:
        """Remove a component (test utility).

        Args:
            family: Component family.
            name: Registry key.
        """
        self._specs[family].pop(name, None)

    def clear(self) -> None:
        """Remove every registered component (test utility)."""
        for bucket in self._specs.values():
            bucket.clear()

    # ------------------------------------------------------------------
    # querying
    # ------------------------------------------------------------------
    def get(self, family: ComponentFamily, name: str) -> ComponentSpec:
        """Return the spec registered under ``(family, name)``.

        Args:
            family: Component family.
            name: Registry key.

        Returns:
            The registered spec.

        Raises:
            UnknownComponentError: If the name is not registered; the message
                lists every known name and hints at missing plugins.
        """
        try:
            return self._specs[family][name]
        except KeyError:
            known = self.names(family)
            hint = (
                " If this is a plugin component, make sure the plugin is "
                "installed (entry-point group 'frankenstein.plugins') or "
                "listed under the top-level `plugins:` key."
            )
            raise UnknownComponentError(
                f"Unknown {family.value} '{name}'. Known values: "
                f"{list(known)}.{hint}"
            ) from None

    def maybe_get(self, family: ComponentFamily, name: str) -> ComponentSpec | None:
        """Return the spec or ``None`` when unregistered.

        Args:
            family: Component family.
            name: Registry key.

        Returns:
            The spec, or ``None``.
        """
        return self._specs[family].get(name)

    def names(self, family: ComponentFamily) -> Tuple[str, ...]:
        """Return sorted registry names for a family.

        Args:
            family: Component family.

        Returns:
            A sorted tuple of component names.
        """
        return tuple(sorted(self._specs[family]))

    def specs(self, family: ComponentFamily) -> Tuple[ComponentSpec, ...]:
        """Return every spec in a family, sorted by name.

        Args:
            family: Component family.

        Returns:
            A tuple of specs ordered by registry name.
        """
        return tuple(self._specs[family][name] for name in self.names(family))

    def property_set(self, family: ComponentFamily, attribute: str) -> frozenset:
        """Return the names whose spec has a truthy ``attribute``.

        Used to derive every capability set that used to be hand-maintained
        (``TRAINING_FREE_LAYERS``, ``QK_NORM_MIXERS``, ``IMPLEMENTED_MIXERS``…).

        Args:
            family: Component family.
            attribute: Spec attribute to test.

        Returns:
            A frozenset of component names.
        """
        return frozenset(
            spec.name
            for spec in self._specs[family].values()
            if bool(getattr(spec, attribute, False))
        )

    def alias_index(self, family: ComponentFamily) -> Mapping[str, str]:
        """Map alias names to canonical registry names.

        Args:
            family: Component family.

        Returns:
            A dict of ``alias -> canonical name`` for every spec exposing an
            ``aliases`` field.
        """
        index: Dict[str, str] = {}
        for spec in self._specs[family].values():
            for alias in getattr(spec, "aliases", ()) or ():
                index[alias] = spec.name
        return index

    def resolve_name(self, family: ComponentFamily, name: str) -> str:
        """Resolve a name or alias to its canonical registry name.

        Args:
            family: Component family.
            name: Registry key or alias.

        Returns:
            The canonical registry key.

        Raises:
            UnknownComponentError: If neither the name nor an alias matches.
        """
        normalized = str(name).strip().lower()
        if normalized in self._specs[family]:
            return normalized
        alias = self.alias_index(family).get(normalized)
        if alias is not None:
            return alias
        # Reuse get() for the informative error message.
        self.get(family, normalized)
        raise AssertionError("unreachable")  # pragma: no cover

    def __iter__(self) -> Iterator[Tuple[ComponentFamily, ComponentSpec]]:
        """Iterate over every registered ``(family, spec)`` pair."""
        for family in ComponentFamily:
            for name in self.names(family):
                yield family, self._specs[family][name]


registry = ComponentRegistry()
"""Process-global registry singleton."""

register_spec = registry.register
"""Convenience alias for :meth:`ComponentRegistry.register`."""
