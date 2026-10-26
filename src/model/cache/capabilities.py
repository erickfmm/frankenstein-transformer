"""Declarative per-mixer cache capability registry.

``CACHE_SPECS`` is the single declarative source consumed by
:class:`~src.model.hybrid_layer.HybridLayer` at runtime and by
``config_loader`` for static validation. One entry per schema
``layer_pattern`` enum value (44 total: 43 mixer classes + the ``mamba``
linear placeholder).

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

from dataclasses import dataclass, field
from typing import Dict, Tuple

__all__ = ["CacheSpec", "CACHE_SPECS", "IMPLEMENTED_MIXERS", "is_cacheable", "spec_for"]


@dataclass(frozen=True)
class CacheSpec:
    """Static cache capability description for one mixer key.

    Attributes:
        supported: Whether the mixer's source formulation admits a cache
            at all. ``False`` ⇒ always runs uncached (with a warning from
            ``config_loader`` when the cache is enabled).
        slot_type: Primary slot kind (``""`` when not supported).
        slot_types: Additional slot kinds (e.g. ``("state", "conv_tail")``
            for falcon-style mixers with a causal short conv).
        extras: Per-token metadata the mixer will store (documentation).
        notes: Paper-based rationale, surfaced in ``config_loader``
            warnings and docs.
    """

    supported: bool
    slot_type: str = ""
    slot_types: Tuple[str, ...] = ()
    extras: Tuple[str, ...] = ()
    notes: str = ""


#: Mixers whose incremental (cached) branches are implemented and covered
#: by the parity test suite. Everything else silently falls back to the
#: legacy full-recompute path (D5). Extended each implementation phase:
#: F0 foundations first, then the KV/latent/window family (F1), the
#: recurrent/auxiliary-state family (F2), and policy-based mixers (F4).
IMPLEMENTED_MIXERS = frozenset(
    {
        # ---- F0 foundations ----
        "standard_attn",       # F0
        "gqa_attn",            # F0
        "mamba",               # F0 passthrough (stateless; runs unchanged)
        # ---- F1 dense / softmax ----
        "sigmoid_attn",        # F1 (L1 normalizer re-summed over the cache)
        "gated_softmax_attn",  # F1
        "titan_attn",          # F1
        "fox_attn",            # F1 (cumulative log-forget extras)
        # ---- F1 latent family ----
        "mla_attn",            # F1 (latent cached, expand + RoPE per step)
        "gqla_attn",           # F1
        "mlra_attn",           # F1
        "tucker_attn",         # F1
        "iha_attn",            # F1 (pseudo-mix after cache read)
        "gta_attn",            # F1 (grouped post-RoPE keys + latent values)
        # ---- F1 sparse family ----
        "swa_attn",            # F1 (ring retention == attention pattern)
        "longformer_attn",     # F1 (pattern over positions)
        "bigbird_attn",        # F1 (seeded random links, R2)
        "sparse_transformer_attn",  # F1 (position-based patterns)
        "fasa_attn",           # F1 (row-local top-k over cached RoPE keys)
        # ---- F2 recurrent / state family ----
        "retnet",              # F2 (parallel == recurrent form)
        "retnet_attn",         # F2
        "gla_attn",            # F2
        "deltanet_attn",       # F2
        "gated_deltanet_attn", # F2
        "gated_deltanet2_attn",# F2
        "hgrn2_attn",          # F2
        "kda_attn",            # F2
        "gma_attn",            # F2 (prefix cumsum state)
    }
)


#: mixer_key -> CacheSpec. One entry per schema enum value (44).
CACHE_SPECS: Dict[str, CacheSpec] = {
    # ---- Dense / softmax family (plan §6.2) ----
    "standard_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="MHA (arXiv:1706.03762); the 2017 paper never describes the "
        "cache (later folklore), but the recipe is the canonical KV append.",
    ),
    "gqa_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="GQA (arXiv:2305.13245) — the paper IS the quality-vs-cache-size "
        "tradeoff; most cache-ready mixer, smallest KV per token.",
    ),
    "sigmoid_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="Sigmoid attention (arXiv:2409.04431); not shift-invariant — the "
        "L1 normalizer must be re-summed over the full cache each step.",
    ),
    "gated_softmax_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="Gated attention (arXiv:2505.06708); the gate depends only on "
        "the current token — nothing extra to cache.",
    ),
    "titan_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="Titans (arXiv:2501.00663); this repo implements only the "
        "attention window (the neural-memory module is not implemented).",
    ),
    "fox_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        extras=("cumulative_log_forget",),
        notes="FoX (arXiv:2503.02130); caches (K, V, c_j) with c_j the "
        "cumulative log-forget per token/head; D_ij = c_i - c_j is "
        "pairwise-relative.",
    ),
    "ode": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="ODE Transformer (arXiv:2203.09176); KV taken from the "
        "converged z(1) state; K/V of the new token recomputed at every "
        "solver iteration with shared weights.",
    ),
    # ---- Latent family (plan §6.3) ----
    "mla_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        extras=("k_rope",),
        notes="Latent MHA (arXiv:2506.09342; origin MLA/DeepSeek-V2 "
        "arXiv:2405.04434) — 'only the latent is cached'. This repo's "
        "variant applies RoPE to the decompressed q,k, so DeepSeek's "
        "absorb trick is unavailable; expand + RoPE every step.",
    ),
    "gqla_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        notes="GQLA (arXiv:2605.15250); two algebraically equivalent decode "
        "paths over one weight set; slot layout follows gqla_decode_path.",
    ),
    "mlra_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        extras=("k_rope",),
        notes="MLRA (arXiv:2603.02188); RoPE key decoupled; total cache is "
        "MLA-sized.",
    ),
    "tucker_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        notes="Tucker attention (arXiv:2603.30033); factorized K/V cached "
        "when ranks < hidden.",
    ),
    "iha_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="IHA (arXiv:2602.21371); pseudo-K/V are static learned mixings "
        "computed on the fly — only the real per-token K/V is cached.",
    ),
    "gta_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        extras=("grouped_keys", "latent_values"),
        notes="GTA (arXiv:2506.17286); one K per head-group (shared map) + "
        "per-head latent values — heterogeneous slot.",
    ),
    "mtla_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        extras=("position_ranges",),
        notes="MTLA (arXiv:2505.13544); the only method compressing the "
        "cache along time via a hyper-network merge at stride boundaries. "
        "Compression policies never run on MTLA slots (D13).",
    ),
    "cca_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        slot_types=("latent_kv", "conv_tail"),
        notes="CCA (arXiv:2510.04476); attention inside the compressed "
        "latent + causal-conv tail and previous shifted value.",
    ),
    "ccgqa_attn": CacheSpec(
        supported=True,
        slot_type="latent_kv",
        slot_types=("latent_kv", "conv_tail"),
        notes="CCGQA (arXiv:2510.04476); CCA recipe + GQA head sharing "
        "inside the latent.",
    ),
    "gma_attn": CacheSpec(
        supported=True,
        slot_type="state",
        extras=("value_accumulator", "normalizer_accumulator"),
        notes="Gaussian Mixture Attention (arXiv:2606.18283); state-only — "
        "causal variant defined by prefix-restricted cumulative sums; not "
        "QK^T, only rotation PEs apply.",
    ),
    # ---- Sparse family (plan §6.4) ----
    "longformer_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        extras=("global_tokens",),
        notes="Longformer (arXiv:2004.05150); window+globals pattern evaluated "
        "over positions; unbounded KV slot since global tokens may sit "
        "anywhere (no safe eviction).",
    ),
    "swa_attn": CacheSpec(
        supported=True,
        slot_type="ring",
        notes="Sliding window + attention sinks (arXiv:2608.28444, built on "
        "StreamingLLM arXiv:2309.17453) — the paper IS a cache-eviction "
        "recipe; removing sinks collapses performance. Ring retention "
        "policy equals the attention pattern.",
    ),
    "bigbird_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        extras=("random_links",),
        notes="BigBird (arXiv:2007.14062); random links are seeded per "
        "(logical_layer_idx, absolute query position) or parity fails (R2).",
    ),
    "sparse_transformer_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="Sparse Transformers (arXiv:1904.10509); strided/fixed patterns "
        "evaluated position-based (B5); stride from the TOTAL length.",
    ),
    "sparsek_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="SparseK (arXiv:2406.16747) — implemented as a CachePolicy "
        "over a KVSlot (D14): learned scorer + top-k retention with "
        "permanent within-generation eviction.",
    ),
    "nsa_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        extras=("block_statistics",),
        notes="NSA (arXiv:2502.11089); v1 runs UNCACHED (falls back, D5): the "
        "compress-branch block statistics are chunk-level means, not "
        "row-local, so exact incremental decode needs the full three-branch "
        "design (F2).",
    ),
    "msa_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="MiniMax Sparse Attention (arXiv:2606.13392); v1 runs UNCACHED "
        "(falls back, D5): per-GQA block selection consumes chunk-level "
        "block statistics; exact decode needs query-local selection (F2).",
    ),
    "sparda_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="SparDA (arXiv:2606.04511); v1 runs UNCACHED (falls back, D5): "
        "block selection uses chunk-level statistics; the cross-layer "
        "Forecast prefetch is F4-grade work.",
    ),
    "sparge_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="SpargeAttn (arXiv:2502.18137); inference method by definition "
        "— the cache stays full (it skips compute, not memory). v1 runs "
        "UNCACHED (falls back, D5): stage-1 block means depend on query "
        "block padding, so exact decode needs block-aligned caching (F2). "
        "Eval-only.",
    ),
    "fasa_attn": CacheSpec(
        supported=True,
        slot_type="kv",
        notes="FASA (arXiv:2602.03152); decode-time selection over cached "
        "RoPE'd keys via dominant frequency-chunk proxies. Eval-only.",
    ),
    # ---- Recurrent / linear family (plan §6.5) ----
    "retnet": CacheSpec(
        supported=True,
        slot_type="state",
        notes="RetNet (arXiv:2307.08621); S_n = gamma S_{n-1} + K_n^T V_n "
        "per head with multi-scale decay; call shape S2 (mixer replaces x).",
    ),
    "retnet_attn": CacheSpec(
        supported=True,
        slot_type="state",
        notes="RetNet thin wrapper (arXiv:2307.08621).",
    ),
    "gla_attn": CacheSpec(
        supported=True,
        slot_type="state",
        notes="GLA (arXiv:2312.06635); data-dependent diagonal gate — decode "
        "is strictly token-recurrent.",
    ),
    "deltanet_attn": CacheSpec(
        supported=True,
        slot_type="state",
        notes="DeltaNet (arXiv:2406.06484); the delta rule overwrites in "
        "place (exact retrieval).",
    ),
    "gated_deltanet_attn": CacheSpec(
        supported=True,
        slot_type="state",
        notes="Gated DeltaNet (arXiv:2412.06464); add a ConvTailSlot when a "
        "causal short conv is configured.",
    ),
    "gated_deltanet2_attn": CacheSpec(
        supported=True,
        slot_type="state",
        notes="Gated DeltaNet-2 (arXiv:2605.22791); 'replaces the unbounded "
        "cache... fixed-size recurrent state... constant memory'.",
    ),
    "hgrn2_attn": CacheSpec(
        supported=True,
        slot_type="state",
        notes="HGRN2 (arXiv:2404.07904); linear RNN, O(1)/step; the gate "
        "lower bound prevents full erasure.",
    ),
    "kda_attn": CacheSpec(
        supported=True,
        slot_type="state",
        notes="KDA / Kimi Linear (arXiv:2510.26692); channel-wise decay; "
        "repo uses the plain recurrence (no DPLR kernel).",
    ),
    "falcon1_attn": CacheSpec(
        supported=True,
        slot_type="state",
        slot_types=("state", "conv_tail"),
        extras=("rehearsal_fifo",),
        notes="Falcon-1 (arXiv:2608.27763); read-after-write prefix shift; "
        "causal short conv => ConvTailSlot; boundary sentinels at step 0.",
    ),
    "falcon2_attn": CacheSpec(
        supported=True,
        slot_type="state",
        slot_types=("state", "conv_tail"),
        extras=("rehearsal_fifo",),
        notes="Falcon-2 (arXiv:2608.27763).",
    ),
    "falcon3_attn": CacheSpec(
        supported=True,
        slot_type="state",
        slot_types=("state", "conv_tail"),
        extras=("rehearsal_fifo",),
        notes="Falcon-3 (arXiv:2608.27763); the sliding window bounds the "
        "rehearsal batch — exact continuation needs a FIFO of the last "
        "B-1 pairs.",
    ),
    "falcon1a_attn": CacheSpec(
        supported=True,
        slot_type="state",
        slot_types=("state", "conv_tail"),
        notes="Falcon-1A additive variant (arXiv:2608.27763).",
    ),
    "falcon2a_attn": CacheSpec(
        supported=True,
        slot_type="state",
        slot_types=("state", "conv_tail"),
        notes="Falcon-2A additive variant (arXiv:2608.27763).",
    ),
    "falcon3a_attn": CacheSpec(
        supported=True,
        slot_type="state",
        slot_types=("state", "conv_tail"),
        extras=("rehearsal_fifo",),
        notes="Falcon-3A additive variant (arXiv:2608.27763).",
    ),
    # ---- Other (plan §6.6) ----
    "engram_attn": CacheSpec(
        supported=True,
        slot_type="token_tail",
        notes="Engram (arXiv:2601.07372); O(1) n-gram-suffix lookup — only "
        "the last max_ngram-1 token ids are required (TokenTailSlot).",
    ),
    "mamba": CacheSpec(
        supported=False,
        notes="Stateless per-token linear placeholder — inherently "
        "incremental; nothing to cache. Verified to run unchanged in the "
        "cached loop.",
    ),
    "ssog_attn": CacheSpec(
        supported=False,
        notes="SSOG (pisoni.ai blog + GitHub; no arXiv): bidirectional "
        "vision-grid mixer with no causal formulation in the source — no "
        "cache can be defined. Never cached; runs uncached.",
    ),
}


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
    if mixer_key == "engram_attn":
        # Engram consumes only the token tail, which the manager always
        # provides when caching; it never touches activation slots.
        return True
    return mixer_key in IMPLEMENTED_MIXERS
