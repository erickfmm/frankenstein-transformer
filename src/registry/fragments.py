"""Schema-fragment loading for component specs.

A spec's ``params_schema`` is either an inline JSON-Schema mapping or a
``"schema/_model/<file>.yaml#/json/pointer"`` reference resolved relative to
the ``src/`` directory. Built-in components keep their hand-authored YAML
fragments (they are already the documentation contract); plugins use inline
mappings.
"""

from __future__ import annotations

import functools
import pathlib
from typing import Any, Dict, Mapping, Tuple

import yaml

from .errors import InvalidSpecError

__all__ = [
    "SRC_ROOT",
    "load_fragment_ref",
    "resolve_params_schema",
    "allowed_param_keys",
    "param_defaults",
    "fragment_property",
]

SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent
"""The ``src/`` directory; fragment references are resolved relative to it."""


def _navigate(document: Any, pointer: str) -> Any:
    """Navigate a document along a slash-separated JSON pointer.

    Args:
        document: The loaded YAML document.
        pointer: Slash-separated path without the leading ``#``.

    Returns:
        The referenced sub-node.

    Raises:
        InvalidSpecError: If the pointer does not resolve.
    """
    target = document
    for part in pointer.split("/"):
        if part == "":
            continue
        try:
            if isinstance(target, dict):
                target = target[part]
            elif isinstance(target, list):
                target = target[int(part)]
            else:
                raise KeyError(part)
        except (KeyError, IndexError, ValueError) as exc:
            raise InvalidSpecError(
                f"Fragment pointer '#/{pointer}' does not resolve"
            ) from exc
    return target


@functools.lru_cache(maxsize=None)
def load_fragment_ref(ref: str) -> Mapping[str, Any]:
    """Load and cache a ``"path#/pointer"`` schema fragment.

    Args:
        ref: Fragment reference such as
            ``"schema/_model/_attention_latent.yaml#/properties/mla"``.

    Returns:
        The referenced mapping.

    Raises:
        InvalidSpecError: If the file or pointer is invalid, or the fragment
            is not a mapping.
    """
    file_part, _, pointer = ref.partition("#")
    path = SRC_ROOT / file_part
    if not path.is_file():
        raise InvalidSpecError(f"Schema fragment file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    node: Any = _navigate(document, pointer) if pointer else document
    if not isinstance(node, Mapping):
        raise InvalidSpecError(f"Schema fragment {ref!r} is not a mapping")
    return node


def resolve_params_schema(spec) -> Mapping[str, Any]:
    """Return a spec's parameter schema as a mapping.

    Args:
        spec: A :class:`~src.registry.spec.ComponentSpec`.

    Returns:
        The inline mapping, the loaded fragment, or an empty mapping.
    """
    fragment = getattr(spec, "params_schema", None)
    if fragment is None:
        return {}
    if isinstance(fragment, str):
        return load_fragment_ref(fragment)
    if isinstance(fragment, Mapping):
        return fragment
    raise InvalidSpecError(
        f"Component {getattr(spec, 'name', '<unknown>')!r}: params_schema must "
        "be a mapping, a fragment reference string, or None"
    )


def allowed_param_keys(spec) -> Tuple[str, ...]:
    """Return the parameter names a component accepts.

    Args:
        spec: A component spec.

    Returns:
        A sorted tuple of parameter keys declared by the fragment.
    """
    properties = resolve_params_schema(spec).get("properties") or {}
    return tuple(sorted(properties))


def fragment_property(spec, key: str) -> Mapping[str, Any]:
    """Return one parameter's schema sub-object.

    Args:
        spec: A component spec.
        key: Parameter name.

    Returns:
        The parameter schema mapping (empty when undeclared).
    """
    return (resolve_params_schema(spec).get("properties") or {}).get(key) or {}


def param_defaults(spec) -> Dict[str, Any]:
    """Return runtime defaults for a component's parameters.

    Fragment-declared defaults are the base; ``spec.defaults`` overrides them
    (the two are cross-checked by ``tests/test_schema_consistency.py``).

    Args:
        spec: A component spec.

    Returns:
        A dict of parameter name -> default value. Parameters without any
        declared default map to ``None``.
    """
    defaults: Dict[str, Any] = {key: None for key in allowed_param_keys(spec)}
    for key, entry in (resolve_params_schema(spec).get("properties") or {}).items():
        if isinstance(entry, Mapping) and "default" in entry:
            defaults[key] = entry["default"]
    defaults.update(dict(getattr(spec, "defaults", {}) or {}))
    return defaults
