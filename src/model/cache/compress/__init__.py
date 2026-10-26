"""KV-cache compression / eviction policies.

Eviction is a **read-time** transform (literature-mandated layering:
Quest arXiv:2406.10774, StreamingLLM arXiv:2309.17453): policies operate
on a slot across decode steps via ``on_append`` / ``on_step_end`` and are
composable with write-time quantization codecs (MiniCache x quant,
arXiv:2405.14366).

Registry (v1):

* ``none`` — keep everything (default).
* ``window_sinks`` — StreamingLLM attention sinks + sliding window.

Future policies stay enum-ready but are NOT implemented in v1; requesting
them raises a ``ValueError`` from :func:`build_cache_policy` at
generation start (a configuration error, never a mid-decode failure):

* ``h2o`` — attention-mass accumulation + lowest-mass eviction (H2O,
  arXiv:2306.14048).
* ``snapkv_prefill`` — one-shot post-prefill compression using look-ahead
  statistics (SnapKV, arXiv:2404.14469).
* ``pyramid`` — static per-layer budgets, deeper layers get less
  (PyramidKV, arXiv:2406.02069; CAKE arXiv:2503.12491; D2O arXiv:2406.13035).
* ``sparsek`` — learned scorer + top-k retention (SparseK,
  arXiv:2406.16747); only applies to ``sparsek_attn`` layers (D14).

Constraints (plan §10): policies apply only to position-indexed slots
(``KVSlot`` / ``RingSlot`` / ``LatentKVSlot``); ``StateSlot``\\ s are
constant-size; ``mtla_attn`` slots are excluded (D13).
"""

from __future__ import annotations

from typing import List, Optional

from .none import NonePolicy
from .window_sinks import WindowSinksPolicy

__all__ = ["NonePolicy", "WindowSinksPolicy", "CACHE_POLICIES", "build_cache_policy"]

#: Policy registry (v1).
CACHE_POLICIES = {
    "none": NonePolicy,
    "window_sinks": WindowSinksPolicy,
}


def build_cache_policy(
    method: Optional[str],
    num_sinks: int = 4,
    window: int = 512,
    budget: Optional[int] = None,
    per_layer_budgets: Optional[List[int]] = None,
):
    """Build a cache policy from resolved inference configuration.

    Args:
        method: One of :data:`CACHE_POLICIES` keys or ``None``.
        num_sinks: Pinned leading positions (``window_sinks``).
        window: Sliding window size (``window_sinks``). A scalar ``budget``
            (when given) overrides it.
        budget: Optional scalar token budget per layer (``None`` =
            unlimited).
        per_layer_budgets: Optional per-layer budget vector; takes
            precedence over ``window``/``budget`` when non-empty.

    Returns:
        A policy instance, or ``None`` when ``method`` is ``None``/``"none"``.

    Raises:
        ValueError: If ``method`` is not a registered policy.
    """
    if method is None or str(method).lower() == "none":
        return None
    key = str(method).lower()
    if key == "window_sinks":
        effective_window = int(budget) if budget is not None else int(window)
        return WindowSinksPolicy(
            num_sinks=num_sinks,
            window=effective_window,
            per_layer_budgets=per_layer_budgets or None,
        )
    raise ValueError(
        f"Unknown kv-cache compression method {method!r}; "
        f"supported: {sorted(CACHE_POLICIES)}"
    )
