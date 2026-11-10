"""Coercion and binding of component parameters.

:func:`bind_params` turns a raw YAML parameters dict into a
:class:`ComponentParams` namespace using the component spec's schema fragment
for allowed keys and types, and its runtime defaults for absent values. This
one helper replaces the per-family ``getattr(config, "<flat_field>", default)``
ladders and the optimizer factory's hand-written coercion branches.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping

from .errors import ComponentError
from .fragments import allowed_param_keys, fragment_property, param_defaults

__all__ = ["ComponentParams", "bind_params", "coerce_value"]


def _to_bool(value: Any, default: Any) -> bool:
    """Coerce a value to bool with YAML-friendly string parsing.

    Args:
        value: Raw value.
        default: Fallback when conversion is impossible.

    Returns:
        The boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(default)


def coerce_value(value: Any, schema: Mapping[str, Any], default: Any) -> Any:
    """Coerce ``value`` to the JSON-Schema type declared by ``schema``.

    Args:
        value: Raw value (may be ``None``).
        schema: The parameter's JSON-Schema sub-object.
        default: Fallback used when ``value`` is ``None`` or uncoercible.

    Returns:
        The coerced value.
    """
    if value is None:
        return default
    declared = schema.get("type")
    if isinstance(declared, (list, tuple)):
        non_null = [entry for entry in declared if entry != "null"]
        declared = non_null[0] if non_null else "null"
    if declared == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if declared == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if declared == "boolean":
        return _to_bool(value, default)
    if declared == "string":
        return str(value)
    if declared == "array":
        if isinstance(value, (list, tuple)):
            return list(value)
        return default
    if declared == "object":
        if isinstance(value, Mapping):
            return dict(value)
        return default
    return value


class ComponentParams(Mapping):
    """Attribute-access view over one component's bound parameters.

    Missing parameters raise :class:`AttributeError` (so
    ``getattr(params, "key", default)`` keeps working), while the mapping
    protocol exposes the full dict.
    """

    __slots__ = ("name", "_values")

    def __init__(self, name: str, values: Mapping[str, Any]) -> None:
        """Initialize the namespace.

        Args:
            name: Component registry name.
            values: Coerced parameter values.
        """
        self.name = name
        self._values: Dict[str, Any] = dict(values)

    def __getattr__(self, key: str) -> Any:
        """Return a bound parameter by name.

        Args:
            key: Parameter name.

        Returns:
            The parameter value.

        Raises:
            AttributeError: If the parameter is not bound.
        """
        try:
            return self._values[key]
        except KeyError:
            raise AttributeError(
                f"Component '{self.name}' has no parameter {key!r}; "
                f"bound: {sorted(self._values)}"
            ) from None

    def __getitem__(self, key: str) -> Any:
        """Return a bound parameter by name (mapping protocol)."""
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over parameter names."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._values)

    def value(self, key: str, default: Any = None) -> Any:
        """Return a parameter, treating an explicit ``None`` as absent.

        Nullable parameters (optional ranks, dimensions) keep their YAML
        ``null`` through binding; consumers that need a concrete fallback use
        this helper instead of ``getattr``.

        Args:
            key: Parameter name.
            default: Fallback used when the parameter is missing or ``None``.

        Returns:
            The parameter value or the fallback.
        """
        value = self._values.get(key, default)
        return default if value is None else value

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the bound values."""
        return dict(self._values)

    def __repr__(self) -> str:
        """Return a compact representation."""
        return f"ComponentParams(name={self.name!r}, values={self._values!r})"


def bind_params(spec, raw: Mapping[str, Any] | None, *, strict: bool = True) -> ComponentParams:
    """Bind a raw parameters dict to a component spec.

    Args:
        spec: The component spec (schema fragment + defaults).
        raw: Raw parameters from the YAML block (``None`` or ``{}`` allowed).
        strict: When ``True`` (default), keys absent from the fragment raise
            :class:`ComponentError`. The assembled schema already rejects
            them at load time; strict binding is the second line of defense
            for programmatic construction.

    Returns:
        A :class:`ComponentParams` with defaults applied and values coerced.

    Raises:
        ComponentError: If ``raw`` contains unknown keys (strict mode).
    """
    raw = dict(raw or {})
    allowed = set(allowed_param_keys(spec))
    if strict:
        unknown = sorted(key for key in raw if key not in allowed)
        if unknown:
            raise ComponentError(
                f"Unknown parameter(s) for component '{spec.name}': {unknown}. "
                f"Allowed: {sorted(allowed)}"
            )
    defaults = param_defaults(spec)
    values: Dict[str, Any] = {}
    for key in allowed:
        schema = fragment_property(spec, key)
        values[key] = coerce_value(raw.get(key), schema, defaults.get(key))
    return ComponentParams(spec.name, values)
