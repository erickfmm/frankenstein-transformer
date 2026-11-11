"""Optimizer factory: registry-driven parameter routing and construction.

The optimizer registry is declarative: every optimizer ships an
:class:`~src.registry.spec.OptimizerSpec` (name, lazy builder, per-suffix
:class:`~src.registry.spec.ParamBinding` map, group policy and capability
flags). This module contains no per-optimizer branches — adding an optimizer
never touches it.

``OPTIMIZER_REGISTRY`` and ``OPTIMIZER_ALLOWED_KEYS`` remain as derived,
read-only views over the registry for convenience and documentation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Dict, List

from torch.optim import Optimizer

from src.registry import ComponentFamily, registry, resolve_builder
from src.registry.spec import OptimizerSpec

from .base import (
    GROUP_NAMES,
    ensure_no_unknown_parameters,
    extract_prefixed_parameters,
    parse_group_value,
    to_betas,
    to_float,
    to_int,
    with_named_groups,
)

__all__ = [
    "OPTIMIZER_REGISTRY",
    "OPTIMIZER_ALLOWED_KEYS",
    "build_optimizer",
]


def _common_per_group_keys() -> frozenset[str]:
    """Return the shared per-group suffix names accepted by every optimizer."""
    return frozenset(
        f"{stem}_{group}"
        for stem in ("lr", "wd", "betas", "eps")
        for group in GROUP_NAMES
    )


COMMON_PER_GROUP_KEYS = _common_per_group_keys()
"""Shared ``lr_*``/``wd_*``/``betas_*``/``eps_*`` suffix names."""


def allowed_keys(spec: OptimizerSpec) -> frozenset[str]:
    """Return the full set of accepted suffixes for one optimizer.

    Args:
        spec: The optimizer spec.

    Returns:
        Shared per-group suffixes (when supported) plus the optimizer's own
        constructor bindings.
    """
    keys = set(spec.param_bindings)
    if spec.supports_shared_per_group_suffixes:
        keys |= COMMON_PER_GROUP_KEYS
    return frozenset(keys)


class _RegistryView(Mapping):
    """Read-only ``name -> implementation`` mapping derived from the registry."""

    def __init__(self, family: ComponentFamily) -> None:
        """Initialize the view over one component family."""
        self._family = family

    def __getitem__(self, name: str) -> Any:
        """Resolve and return the component implementation for ``name``."""
        from src.registry import ensure_builtins

        ensure_builtins()
        return resolve_builder(registry.get(self._family, name).builder)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registry names."""
        from src.registry import ensure_builtins

        ensure_builtins()
        return iter(registry.names(self._family))

    def __len__(self) -> int:
        """Return the number of registered components."""
        from src.registry import ensure_builtins

        ensure_builtins()
        return len(registry.names(self._family))


class _AllowedKeysView(Mapping):
    """Read-only ``name -> allowed suffix set`` mapping derived from specs."""

    def __init__(self, family: ComponentFamily) -> None:
        """Initialize the view over one component family."""
        self._family = family

    def __getitem__(self, name: str) -> frozenset[str]:
        """Return the allowed suffixes for ``name``."""
        from src.registry import ensure_builtins

        ensure_builtins()
        spec = registry.get(self._family, name)
        return allowed_keys(spec)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registry names."""
        from src.registry import ensure_builtins

        ensure_builtins()
        return iter(registry.names(self._family))

    def __len__(self) -> int:
        """Return the number of registered components."""
        from src.registry import ensure_builtins

        ensure_builtins()
        return len(registry.names(self._family))


OPTIMIZER_REGISTRY: Mapping[str, Any] = _RegistryView(ComponentFamily.OPTIMIZER)
"""Derived ``optimizer name -> Optimizer subclass`` mapping."""

OPTIMIZER_ALLOWED_KEYS: Mapping[str, frozenset[str]] = _AllowedKeysView(
    ComponentFamily.OPTIMIZER
)
"""Derived ``optimizer name -> allowed suffix names`` mapping."""


def _build_param_groups_for_optimizer(
    spec: OptimizerSpec,
    base_param_groups: List[Dict[str, Any]],
    scoped_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert raw model parameter groups into optimizer-ready groups.

    Args:
        spec: The optimizer spec (its ``group_policy`` decides whether
            per-group ``betas``/``eps`` are routed).
        base_param_groups: Raw parameter groups from the model.
        scoped_params: Optimizer-scoped hyperparameters (prefix stripped).

    Returns:
        Parameter-group dicts suitable for an ``Optimizer`` constructor.
    """
    result: List[Dict[str, Any]] = []
    adaptive = spec.group_policy != "sgd"
    for group in with_named_groups(base_param_groups):
        group_name = str(group["name"])
        converted = {
            "params": group["params"],
            "lr": to_float(
                parse_group_value(scoped_params, "lr", group_name, group.get("lr", 1e-3)),
                1e-3,
            ),
            "weight_decay": to_float(
                parse_group_value(
                    scoped_params, "wd", group_name, group.get("weight_decay", 0.0)
                ),
                0.0,
            ),
        }
        if adaptive:
            converted["betas"] = to_betas(
                parse_group_value(
                    scoped_params, "betas", group_name, group.get("betas", (0.9, 0.95))
                )
            )
            converted["eps"] = to_float(
                parse_group_value(scoped_params, "eps", group_name, group.get("eps", 1e-8)),
                1e-8,
            )
        converted["name"] = group_name
        if "param_names" in group:
            # Per-parameter names (aligned with ``params``) let optimizers such
            # as per-head Muon classify projections; harmless otherwise.
            converted["param_names"] = group["param_names"]
        result.append(converted)
    return result


def build_optimizer(
    optimizer_class: str,
    param_groups: List[Dict[str, Any]],
    parameters: Dict[str, Any],
) -> Optimizer:
    """Construct an optimizer instance from a configuration-driven spec.

    Args:
        optimizer_class: The optimizer identifier from the YAML config
            (e.g. ``"adamw"``, ``"muon2"``).
        param_groups: Raw parameter groups from the model's
            ``configure_optimizers`` method.
        parameters: The optimizer parameters mapping (keys are
            ``"<optimizer>-<suffix>"``).

    Returns:
        A configured ``torch.optim.Optimizer`` instance.

    Raises:
        UnknownComponentError: If ``optimizer_class`` is not registered.
        ValueError: If unrecognized scoped parameters are present.
    """
    from src.registry import ensure_builtins

    ensure_builtins()
    optimizer_name = str(optimizer_class or "").strip().lower()
    spec = registry.get(ComponentFamily.OPTIMIZER, optimizer_name)

    scoped = extract_prefixed_parameters(spec.name, parameters or {})
    ensure_no_unknown_parameters(spec.name, scoped, allowed_keys(spec))

    converted_groups = _build_param_groups_for_optimizer(spec, param_groups, scoped)
    kwargs = {
        key: binding.coerce(scoped.get(key), binding.default)
        for key, binding in spec.param_bindings.items()
    }
    if spec.custom_build is not None:
        return spec.custom_build(spec, converted_groups, scoped, kwargs)
    return resolve_builder(spec.builder)(converted_groups, **kwargs)
