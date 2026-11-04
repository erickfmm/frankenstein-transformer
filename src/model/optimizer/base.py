"""Parameter group routing infrastructure for the optimizer factory.

Provides the canonical parameter group taxonomy (GROUP_NAMES), utility functions
for type-safe extraction of optimizer hyperparameters from flat configuration
dictionaries, and helpers for annotating parameter groups with human-readable
names. All optimizer implementations in this package consume parameter groups
produced by the routines defined here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import torch

from src.utils.param_categories import categorize_param

GROUP_NAMES: Tuple[str, ...] = (
    "embeddings",
    "pos",
    "attention",
    "ffn",
    "norms",
    "head",
    "mhc",
    "router",
    "residual",
    "other",
)
"""Canonical parameter group taxonomy used by all optimizers.

The names mirror :data:`src.utils.param_categories.BLOCK_CATEGORIES` — the
single taxonomy shared by the website parameter estimator, the per-block
gradient-norm logs (``grad_norm_<block>``) and optimizer group routing. The
only intentional difference is ``norm`` → ``norms`` (plural), kept for
backward compatibility with existing ``<optimizer>-lr_norms`` keys and
presets.

Each group name corresponds to a distinct architectural component of the
Frankenstein model. The factory routes per-group hyperparameters (lr, wd,
betas, eps) by matching these names against prefixed keys in the training
configuration.

Group semantics:
    embeddings: Token / patch embedding lookup tables.
    pos: Learned positional encodings, cls/mask tokens.
    attention: Mixers and everything nested inside them (projection weights,
        QK-norms, recurrent / state-space / ODE parameters such as retnet,
        mamba or titan).
    ffn: Feed-forward bodies and MoE experts.
    norms: Standalone block normalization parameters (LayerNorm, DynamicTanh,
        Derf, RMSNorm families).
    head: Task / output heads (LM head, classification and segmentation
        heads).
    mhc: Manifold-constrained hyper-connections and Hyperloop parameters.
    router: MoE and Mixture-of-Depths routers.
    residual: AttnRes residual modules.
    other: All remaining parameters not assigned to a specific group.
"""

BLOCK_TO_GROUP: Dict[str, str] = {
    "embeddings": "embeddings",
    "pos": "pos",
    "attention": "attention",
    "ffn": "ffn",
    "norm": "norms",
    "head": "head",
    "mhc": "mhc",
    "router": "router",
    "residual": "residual",
    "other": "other",
}
"""Maps :data:`BLOCK_CATEGORIES` entries to :data:`GROUP_NAMES` entries.

``categorize_param`` returns ``norm`` while the optimizer-facing group name is
the historical plural ``norms``; every other category maps to itself.
"""

DEFAULT_GROUP_LR: Dict[str, float] = {
    "embeddings": 1e-6,
    "pos": 2e-6,
    "attention": 3e-6,
    "ffn": 2e-6,
    "norms": 5e-6,
    "head": 2e-6,
    "mhc": 3e-6,
    "router": 2e-6,
    "residual": 2e-6,
    "other": 2e-6,
}
"""Default learning rate per group, used when no scoped override is given."""

DEFAULT_GROUP_WD: Dict[str, float] = {
    "embeddings": 0.01,
    "pos": 0.01,
    "attention": 0.01,
    "ffn": 0.01,
    "norms": 0.001,
    "head": 0.01,
    "mhc": 0.01,
    "router": 0.01,
    "residual": 0.01,
    "other": 0.01,
}
"""Default weight decay per group, used when no scoped override is given."""

ADAMW_FALLBACK_GROUPS: frozenset[str] = frozenset({"embeddings", "pos", "head"})
"""Groups routed to the AdamW fallback by the Muon-family optimizers.

The Muon recipe applies orthogonalized updates only to the hidden-layer
matrices; embedding tables, learned positional encodings and output heads
are trained with AdamW instead. These groups therefore bypass the
Newton-Schulz path in :class:`~src.model.optimizer.muon.Muon` and
:class:`~src.model.optimizer.turbo_muon.TurboMuon` and receive decoupled
AdamW updates. Parameters outside these groups that are not 2-D matrices
(biases, norm scales, convolution kernels) also take the fallback.
"""


def adamw_fallback_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: Dict[str, Any],
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    """Apply one decoupled-AdamW step to a single parameter.

    Mirrors :class:`torch.optim.AdamW`: first/second-moment EMAs with bias
    correction, a decoupled decay applied to the parameter before the
    adaptive update, and the epsilon added after the bias correction of the
    second moment. Used by the Muon-family optimizers as the fallback for
    parameters that are not hidden-layer 2-D matrices.

    Args:
        param: The parameter tensor to update in place.
        grad: The parameter's gradient.
        state: The optimizer's per-parameter state dictionary. Populated
            with ``exp_avg``, ``exp_avg_sq`` and ``step`` on first use.
        lr: Learning rate for this parameter group.
        betas: ``(beta1, beta2)`` moment coefficients.
        eps: Numerical-stability term for the adaptive denominator.
        weight_decay: Decoupled weight-decay coefficient.
    """
    beta1, beta2 = float(betas[0]), float(betas[1])

    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(param)
        state["exp_avg_sq"] = torch.zeros_like(param)
        state["step"] = 0

    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    state["step"] += 1
    step = state["step"]

    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1**step
    bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)

    if weight_decay > 0:
        param.mul_(1.0 - lr * weight_decay)

    denom = exp_avg_sq.sqrt() / bias_correction2_sqrt + eps
    param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


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


def categorize_param_groups(
    named_parameters: Iterable[Tuple[str, Any]],
    *,
    drop_empty: bool = True,
) -> Dict[str, List[Any]]:
    """Group model parameters with the canonical block categorizer.

    Uses :func:`src.utils.param_categories.categorize_param` — the same
    taxonomy as the website parameter estimator and the ``grad_norm_<block>``
    training logs — and maps every ``BLOCK_CATEGORIES`` entry onto its
    :data:`GROUP_NAMES` counterpart (``norm`` → ``norms``). Parameters that
    do not require gradients are skipped.

    Args:
        named_parameters: Iterable of ``(name, parameter)`` pairs, typically
            ``model.named_parameters()``.
        drop_empty: When ``True`` (default) groups without parameters are
            omitted, so the result can be passed straight to
            :func:`build_optimizer`. When ``False`` every :data:`GROUP_NAMES`
            entry is present, possibly with an empty parameter list.

    Returns:
        A dictionary keyed by :data:`GROUP_NAMES` entries whose values are
        lists of parameters.

    Raises:
        KeyError: If the categorizer returns a block that has no group
            mapping, which would indicate a taxonomy drift between
            ``param_categories.BLOCK_CATEGORIES`` and :data:`GROUP_NAMES`.
    """
    groups: Dict[str, List[Any]] = {name: [] for name in GROUP_NAMES}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        groups[BLOCK_TO_GROUP[categorize_param(name)]].append(param)
    if drop_empty:
        return {name: params for name, params in groups.items() if params}
    return groups
