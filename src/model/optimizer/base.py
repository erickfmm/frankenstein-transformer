"""Parameter group routing infrastructure for the optimizer factory.

Provides the canonical parameter group taxonomy (GROUP_NAMES), utility functions
for type-safe extraction of optimizer hyperparameters from flat configuration
dictionaries, and helpers for annotating parameter groups with human-readable
names. All optimizer implementations in this package consume parameter groups
produced by the routines defined here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

GROUP_NAMES: Tuple[str, ...] = (
    "embeddings",
    "norms",
    "attention",
    "other",
)
"""Canonical parameter group taxonomy used by all optimizers.

Each group name corresponds to a distinct architectural component of the
TormentedBERT model. The factory routes per-group hyperparameters (lr, wd,
betas, eps) by matching these names against prefixed keys in the training
configuration.

Group semantics:
    embeddings: Token and position embedding matrices.
    norms: LayerNorm, DynamicTanh, or Derf normalization parameters.
    attention: Attention projection weights (Q, K, V, O) and recurrent /
        state-space / ODE mixer parameters (retnet, mamba, ode, titan, ...).
    other: All remaining parameters not assigned to a specific group.
"""


def to_float(value: Any, default: float) -> float:
    """Safely cast a configuration value to float, falling back to a default.

    Args:
        value: The raw value extracted from the configuration dictionary.
            May be None, a numeric type, or an unconvertible string.
        default: The fallback value returned when conversion fails.

    Returns:
        The float representation of `value`, or `default` if conversion
        is not possible.
    """
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_int(value: Any, default: int) -> int:
    """Safely cast a configuration value to int, falling back to a default.

    Args:
        value: The raw value extracted from the configuration dictionary.
        default: The fallback value returned when conversion fails.

    Returns:
        The int representation of `value`, or `default` if conversion
        is not possible.
    """
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def to_betas(value: Any, default: Tuple[float, float] = (0.9, 0.95)) -> Tuple[float, float]:
    """Safely cast a configuration value to a (beta1, beta2) tuple.

    Args:
        value: The raw value, expected to be a 2-element list or tuple.
        default: Fallback (beta1, beta2) pair returned when conversion fails.

    Returns:
        A ``(beta1, beta2)`` tuple of floats, or `default` if the input
        is not a valid 2-element sequence.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def extract_prefixed_parameters(prefix: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Extract optimizer-scoped keys from a flat configuration dictionary.

    Configuration keys follow the convention ``<optimizer>-<param>``
    (e.g. ``adamw-lr_embeddings``). This function strips the optimizer
    prefix and returns a dictionary of bare parameter names.

    Args:
        prefix: The optimizer class name (e.g. ``"adamw"``).
        parameters: The full flat configuration dictionary, which may
            contain keys for multiple optimizers and global settings.

    Returns:
        A dictionary mapping bare parameter names (without the
        ``<prefix>-`` segment) to their raw values. Returns an empty
        dict if `parameters` is None or empty.
    """
    result: Dict[str, Any] = {}
    expected_prefix = f"{prefix}-"
    for key, value in (parameters or {}).items():
        if key.startswith(expected_prefix):
            result[key[len(expected_prefix):]] = value
    return result


def ensure_no_unknown_parameters(
    optimizer_name: str,
    scoped_params: Dict[str, Any],
    allowed_keys: Iterable[str],
) -> None:
    """Validate that all scoped parameters are recognized by the optimizer.

    Raises a ``ValueError`` listing any keys present in `scoped_params`
    that are not in the `allowed_keys` whitelist. This prevents silent
    ignoring of misspelled or unsupported configuration keys.

    Args:
        optimizer_name: Human-readable optimizer identifier used in the
            error message.
        scoped_params: Dictionary of optimizer-scoped parameters (output
            of :func:`extract_prefixed_parameters`).
        allowed_keys: Iterable of permitted bare parameter names for this
            optimizer.

    Raises:
        ValueError: If one or more keys in `scoped_params` are not in
            `allowed_keys`.
    """
    allowed = set(allowed_keys)
    unknown = sorted(k for k in scoped_params if k not in allowed)
    if unknown:
        raise ValueError(
            f"Unknown parameters for optimizer '{optimizer_name}': {unknown}. "
            f"Allowed keys: {sorted(allowed)}"
        )


def parse_group_value(scoped_params: Dict[str, Any], key_stem: str, group_name: str, default: Any) -> Any:
    """Look up a per-group hyperparameter with a fallback default.

    Constructs the key ``<key_stem>_<group_name>`` (e.g. ``lr_embeddings``)
    and returns its value from `scoped_params` if present, otherwise
    returns `default`.

    Args:
        scoped_params: Dictionary of optimizer-scoped parameters.
        key_stem: The base parameter name (e.g. ``"lr"``, ``"wd"``,
            ``"betas"``, ``"eps"``).
        group_name: One of the :data:`GROUP_NAMES` values.
        default: Fallback value returned when the key is absent.

    Returns:
        The per-group parameter value, or `default`.
    """
    key = f"{key_stem}_{group_name}"
    if key in scoped_params:
        return scoped_params[key]
    return default


def with_named_groups(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every parameter group carries a ``"name"`` key.

    Groups that lack an explicit name are assigned ``"other"``, matching
    the catch-all entry in :data:`GROUP_NAMES`.

    Args:
        param_groups: List of parameter group dictionaries as produced
            by the model's ``configure_optimizers`` method.

    Returns:
        A new list of parameter group dictionaries where every entry
        has a ``"name"`` key. The original list is not mutated.
    """
    named_groups: List[Dict[str, Any]] = []
    for group in param_groups:
        copied = dict(group)
        if "name" not in copied:
            copied["name"] = "other"
        named_groups.append(copied)
    return named_groups
