"""Declarative per-mixer cache capability registry.

``CACHE_SPECS`` is a **derived view** over the mixer registry: every
:class:`~src.registry.spec.MixerSpec` declares its ``cache`` capability
(a :class:`~src.registry.spec.CacheSpec`), and this module re-exports it
for the runtime (:class:`~src.model.hybrid_layer.HybridLayer`), the
``config_loader`` validation and the KV-cache test suites. The registry is
the single source of truth for the per-mixer capability (D13).

Rules (see the KV-cache integration plan, §5.4):

* The registry is declarative: ``HybridLayer`` asks it whether a layer may
  cache and which slot kinds it needs.
* A layer caches only when ``supported`` is True **and** the mixer key is
  in :data:`IMPLEMENTED_MIXERS` (the subset whose incremental branches are
  actually implemented and parity-tested). Anything else falls back to the
  legacy full-recompute path — **inference never fails because of the
  cache** (D5: no exceptions, ever).
* ``config_loader`` consults this registry to emit warnings when
  ``kv_cache.enabled`` is true and the ``layer_pattern`` contains
  non-cacheable mixers.

Compatibility classes (plan §6.1):

* (i)  classic per-token K/V attention → ``KVSlot``
* (ii) compressed/latent attention → ``LatentKVSlot``
* (iii) fixed-size recurrent state → ``StateSlot``
* (iv) ring / sliding buffer → ``RingSlot``
* (v)  auxiliary FIFO tail → ``ConvTailSlot`` / ``TokenTailSlot``
* (vii) not sequence-attention over past tokens → **never cached**
"""

from __future__ import annotations

from typing import Dict

from src.registry.spec import CacheSpec

__all__ = ["CacheSpec", "CACHE_SPECS", "IMPLEMENTED_MIXERS", "is_cacheable", "spec_for"]


#: Mixers whose incremental (cached) branches are implemented and covered
#: by the parity test suite. Everything else silently falls back to the
#: legacy full-recompute path (D5). Extended each implementation phase:
#: F0 foundations first, then the KV/latent/window family (F1), the
#: recurrent/auxiliary-state family (F2), and policy-based mixers (F4).
IMPLEMENTED_MIXERS = frozenset(
    {
        # ---- F0 foundations ----
        "standard",       # F0
        "gqa",            # F0
        "mamba",               # F0 passthrough (stateless; runs unchanged)
        # ---- F1 dense / softmax ----
        "sigmoid",        # F1 (L1 normalizer re-summed over the cache)
        "ca",             # F1 (row-local moment normalizer, re-summed over the cache)
        "gated_softmax",  # F1
        "xsa",            # F1 (row-local exclusion over the current-step value)
        "memory",         # F1 (standard KV; values built at append time)
        "titan",          # F1
        "fox",            # F1 (cumulative log-forget extras)
        "fpa",            # F1 (range-minimum field extras; exact, Prop. 2.4b)
        # ---- F1 latent family ----
        "mla",            # F1 (latent cached, expand + RoPE per step)
        "gmla",           # F1 (latent cached, NoPE, expand + gate per step)
        "gqla",           # F1
        "mlra",           # F1
        "tucker",         # F1
        "kpa",            # F1 (Kronecker factor streams, expand + RoPE per step)
        "iha",            # F1 (pseudo-mix after cache read)
        "gta",            # F1 (grouped post-RoPE keys + latent values)
        # ---- F1 sparse family ----
        "swa",            # F1 (ring retention == attention pattern)
        "fastconformer",  # F1 (ring retention == window + pinned globals)
        "longformer",     # F1 (pattern over positions)
        "bigbird",        # F1 (seeded random links, R2)
        "sparse_transformer",  # F1 (position-based patterns)
        "fasa",           # F1 (row-local top-k over cached RoPE keys)
        "lsa",            # F2 (bucket cache over immutable complete cluster blocks)
        # ---- F2 recurrent / state family ----
        "retnet",              # F2 (parallel == recurrent form)
        "gla",            # F2
        "deltanet",       # F2
        "gated_deltanet", # F2
        "gated_deltanet2",# F2
        "hgrn2",          # F2
        "kda",            # F2
        "gma",            # F2 (prefix cumsum state)
    }
)


def _derive_cache_specs() -> Dict[str, CacheSpec]:
    """Build the ``mixer_key -> CacheSpec`` map from the component registry.

    Returns:
        One :class:`CacheSpec` per built-in mixer that declares one (every
        built-in mixer does; plugins opt in by declaring ``cache=`` on their
        own spec at import time).
    """
    from src.registry import ComponentFamily, ensure_builtins, registry

    ensure_builtins()
    return {
        spec.name: spec.cache
        for spec in registry.specs(ComponentFamily.MIXER)
        if spec.cache is not None
    }


#: ``mixer_key -> CacheSpec``, derived from the registry (single source of truth).
CACHE_SPECS: Dict[str, CacheSpec] = _derive_cache_specs()


def spec_for(mixer_key: str) -> CacheSpec:
    """Return the :class:`CacheSpec` for ``mixer_key`` (unknown keys are
    treated as unsupported — never raise, D5)."""
    return CACHE_SPECS.get(mixer_key, CacheSpec(supported=False, notes="unknown mixer key"))


def is_cacheable(mixer_key: str) -> bool:
    """True when the mixer both admits a cache (per its paper) and has an
    implemented, parity-tested incremental branch."""
    spec = CACHE_SPECS.get(mixer_key)
    if spec is None or not spec.supported:
        return False
    if mixer_key == "engram":
        # Engram consumes only the token tail, which the manager always
        # provides when caching; it never touches activation slots.
        return True
    return mixer_key in IMPLEMENTED_MIXERS
