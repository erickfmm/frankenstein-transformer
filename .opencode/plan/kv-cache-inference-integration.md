# Plan: KV Cache Integration for the Inference Module

**Repo**: `frankenstein-transformer` (schema-first, config-driven transformer toolkit)
**Status**: Approved plan, not yet implemented
**Date**: 2026-10-26
**Audience**: An implementation agent with **no prior context** from the planning conversation. Everything needed is in this document.

> **Source-of-truth note**: All `file:line` references were verified against the working tree on 2026-10-26. If the code drifted since then, re-verify the referenced lines before editing. ArXiv IDs were verified against arXiv/alphaXiv; the one exception is `ssog_attn`, whose source is a blog + GitHub repo (no arXiv paper exists).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Repository primer](#2-repository-primer)
3. [Current state diagnosis](#3-current-state-diagnosis)
4. [Strategy selection](#4-strategy-selection)
5. [Target architecture](#5-target-architecture)
6. [Per-mixer compatibility matrix](#6-per-mixer-compatibility-matrix)
7. [Mechanism interactions](#7-mechanism-interactions)
8. [Positional-encoding changes](#8-positional-encoding-changes)
9. [KV-cache quantization](#9-kv-cache-quantization)
10. [KV-cache compression / eviction](#10-kv-cache-compression--eviction)
11. [Configuration design](#11-configuration-design)
12. [File-by-file change plan](#12-file-by-file-change-plan)
13. [Testing plan](#13-testing-plan)
14. [Implementation phases and acceptance criteria](#14-implementation-phases-and-acceptance-criteria)
15. [Risks and mitigations](#15-risks-and-mitigations)
16. [Decision log](#16-decision-log)
17. [Bibliography](#17-bibliography)
18. [Appendix A — current code behavior reference](#appendix-a--current-code-behavior-reference)
19. [Appendix B — signature diffs](#appendix-b--signature-diffs)

---

## 1. Executive summary

The toolkit currently has **zero KV-cache infrastructure**: the only autoregressive decode loop
(`FrankensteinDecoder.generate()`) re-feeds the entire sequence at every step (O(S²) per token),
recurrent mixers zero their state at every `forward()`, and no positional encoder accepts a
position offset.

This plan adds a **central `FrankensteinCache` manager** (HF `past_key_values`-style, vLLM-inspired
slot abstraction) that is created by the inference script, threaded through
`decoder → encoder → HybridLayer → mixer` as an optional keyword argument, and backed by **typed
per-layer slots** whose write/read paths host **pluggable quantization codecs** and **pluggable
compression policies**. Every one of the 44 dispatchable layer types is covered by a declarative
capability registry (`CacheSpec`); mixers that cannot cache (per their source papers) simply run
uncached — **inference never fails because of the cache**.

Confirmed decisions (see §16 for the full log):

- **D1** — Central cache manager + typed slots + per-mixer declarative capability spec (not per-mixer internal state).
- **D2** — Configuration lives in a new schema module `inference:` (source of truth) **plus** CLI flags that override YAML presets key-by-key (precedence: CLI > YAML > schema default).
- **D3** — Initial quantization codecs: `none`, `int8` (asymmetric K per-channel / V per-token), `int4_kivi` (group-wise + protected full-precision window). All other methods are future plugins behind the `CacheCodec` interface.
- **D4** — Opt-in activation: `inference.kv_cache.enabled` (default `false`). Existing presets and tests are unaffected.
- **D5** — Unsupported mixer + cache enabled ⇒ silent fallback to the existing full-recompute path (optionally a warning from `config_loader`); never an exception.

Implementation is split into 6 phases (F0–F5, §14): foundations → KV/latent/window mixers (18) →
recurrent-state and auxiliary-state mixers (21) → quantization → compression → ecosystem (HF export,
docs, examples).

---

## 2. Repository primer

### 2.1 What the repo is

A research toolkit for transformer experiments. Model architecture is **fully config-driven**:

- YAML presets are validated against a JSON-Schema with `additionalProperties: false` at every level
  (root: `src/schema.yaml`; model sub-modules: `src/schema/_model/_*.yaml`).
- The nested model tree is flattened into one flat dataclass `FrankensteinModelConfig`
  (`src/model/config.py:92+`) by `src/utils/config_flatten.py`; per-mixer keys are renamed to flat
  `<mixer>_<leaf>` fields via `_ATTENTION_MIXER_RENAMES` (`config_flatten.py:72-174`).
- Each layer gets its mixer from `layer_pattern[i % len(layer_pattern)]`
  (`src/model/frankenstein_encoder.py:116-121`) — **different layers may use completely different
  attention mechanisms**. The whole stack can be re-run `num_loops` times; a global
  `logical_layer_idx` counter (0 .. `num_layers * num_loops - 1`) identifies each logical layer and
  is already threaded into every mixer call.
- 44 dispatchable layer types: 43 real mixer classes in a registry (`src/model/hybrid_layer.py:175-219`)
  plus a `mamba` key mapped to a plain linear placeholder (`hybrid_layer.py:221-222`, documented as a
  known limitation in `docs/architecture.md` §14).
- Two mixers are **eval-only**: `HybridLayer.TRAINING_FREE_LAYERS = {"fasa_attn", "sparge_attn"}`
  (`hybrid_layer.py:114`, enforced at `hybrid_layer.py:360-363` and `444-447`).
- Model classes: `FrankensteinEncoder` (`src/model/frankenstein_encoder.py`),
  `FrankensteinDecoder` (wraps the encoder with `mode="decoder"`, `frankenstein_decoder.py:108-109`),
  `FrankensteinViT` (`src/model/frankenstein_vit.py`). Forced modes (hard rule): the decoder forces
  `mode: decoder`, the ViT forces `mode: encoder`; `training.task: causal_lm` requires the decoder.
- BitNet is **on by default** (`bitnet.enabled: true`): every projection is a `BitLinear`
  (ternary weights, int8 per-token activations, STE) instead of `nn.Linear`
  (`src/model/attention/common.py:60-96`; class selection at `hybrid_layer.py:159-160`).

### 2.2 Environment & commands (from AGENTS.md — must be respected)

```bash
# CPU-first development and testing (D7). NEVER require CUDA in tests.
uv sync --extra cpu --extra train
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
uv run --extra cpu --extra train python -m pytest tests/test_kv_cache_mixers.py -v   # single file
```

- Style: `from __future__ import annotations`, absolute `src.` imports, Google-style docstrings, no linter configured — match surrounding code.
- Gates that must stay green (AGENTS.md): `test_schema_consistency.py` (presets validate, mixer enum ↔ bilingual tables, optimizer prefixes, schema-default ↔ dataclass pairs), `test_yaml_examples.py`, `test_schema_attention_layers.py`, optimizer tests, `test_common_modules.py`, `test_naming_conventions.py` (no `frankenstein_model.py`), etc.
- Every new example YAML under `configs/examples/` is auto-globbed and schema-smoke-tested.
- Schema docs are bilingual: every property needs `description`/`description_es`, `title`/`title_es`.

### 2.3 Hard rules relevant to this plan

1. Schema is the contract — every new YAML key must exist in `src/schema/**`.
2. Invariants: `hidden_size % num_heads == 0`, `num_kv_heads` divides `num_heads` (GQA) — the cache layout derives from these.
3. `fasa_attn` / `sparge_attn` are eval-only — never in training presets; they are *inference* methods by design, so they interact *positively* with the cache.
4. mHC and Mixture-of-Depths are mutually exclusive (`hybrid_layer.py:141-146`).
5. Optimizer keys are prefixed — irrelevant here but do not break the prefix checker.
6. `docs/architecture.md` is the canonical implementation spec — this plan adds a section there (F5).

---

## 3. Current state diagnosis

### 3.1 The only autoregressive path

`FrankensteinDecoder.generate()` (`src/model/frankenstein_decoder.py:129-166`):

- `@torch.inference_mode()`; loops `max_new_tokens` times calling `self.forward(input_ids)` on the
  **entire** sequence; applies temperature and top-k sampling (`torch.multinomial`); appends the
  sampled token. Cost: **O(S²) per generated token**. Batched on the `B` dim, no padding handling.
- `generate()` is **not wired to any CLI entry point**. The CLI `infer` subcommand
  (`src/cli.py:256-282`) only routes to SBERT inference (`src/sbert/inference_sbert.py:114-150`) or
  MLM single-pass prediction (`src/deploy/inference.py::main`, `:152-173`).
- `FrankensteinDecoder.forward(self, input_ids) -> (B, S, vocab)` (`frankenstein_decoder.py:113-127`)
  — **no** `attention_mask`, `past_key_values`, `use_cache`, or `position_ids` parameter.

### 3.2 Mixer uniform signature and dispatch call shapes

Every mixer module has constructor signature `(config, pos_encoder=None)`. Canonical forward:

```python
forward(x: torch.Tensor,                       # (B, S, hidden)
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None) -> torch.Tensor      # (B, S, hidden)
```

`HybridLayer` dispatches with **three different call shapes** (`hybrid_layer.py:365-372`, repeated in
`_forward_dense_mhc` at `:452-464`):

| Shape | Mixers | Call |
|---|---|---|
| S1 | `mamba` | `x = x + self.mixer(x)` (no kwargs; linear placeholder) |
| S2 | `ode`, `retnet` | `x = self.mixer(x)` — the mixer **replaces** `x` wholesale (no residual added inside the layer) |
| S3 | `engram_attn` | `self.mixer(x, input_ids=..., logical_layer_idx=..., pos_encoder=...)` — **requires `input_ids`**; returns `zeros_like(x)` when absent (`engram.py:314-315`) |
| S4 (default) | all remaining 39 | `self.mixer(x, logical_layer_idx=..., pos_encoder=...)` |

Causal masks are built **inside each mixer from `seq_len` only**, with absolute positions `0..S-1`.
No mixer takes or returns a mask, cache, or state object.

### 3.3 Blockers inventory (must all be fixed)

| # | Blocker | Evidence | Fix (this plan) |
|---|---|---|---|
| B1 | `RoPE.forward` always positions from 0: `pos = torch.arange(seq_len) * scaling` | `src/model/embeddings/rope.py:79` (forward `:60-99`); same for `hope.py`, `sinusoidal_rotary` | `position_offset` parameter (§8) |
| B2 | `ALiBi.bias(seq_len, device, dtype)` builds the full `(1, H, S, S)` matrix from position 0 | `src/model/embeddings/alibi.py:84-110`; same style for `bam` | `bias(q_len, kv_len, q_offset)` (§8) |
| B3 | Recurrent mixers zero-init their state inside `forward()` — nothing survives across calls | e.g. `src/model/attention/gated/gla_attn.py:155-162` (`state = torch.zeros(...)`) | `StateSlot` recurrent-state cache (F2) |
| B4 | AttnRes residuals reset their cross-layer buffers at the start of every forward | `frankenstein_encoder.py:213-216` (`reset_state()`, `set_embedding(x)`) | incremental accumulation mode (F2) |
| B5 | Sparse-pattern masks regenerated from absolute `seq_len` each call; BigBird's random links are re-drawn per call | `src/model/attention/sparse/*.py`; `bigbird_attn.py:153` | pattern builders accept `q_len/kv_len/q_offset`; BigBird seeds its pattern (F1) |
| B6 | `ssmax` score rescale `* s·ln(S)` depends on total sequence length | `src/model/attention/common.py:317-322` | use accumulated `kv_len` (§8) |
| B7 | HF export swallows cache kwargs silently | `src/deploy/transformers_export.py` — `FrankensteinForCausalLM.forward(input_ids, labels, **_)` discards extras (`:163-196`) | real `past_key_values` support (F5) |
| B8 | MoD routing would fragment caches at prefill | `hybrid_layer.py:502-567`; short-circuit `capacity >= seq_len` at `:552-553` | force dense when caching (§7.3) |

### 3.4 Existing cache-related code

Grep for `past_key_values|use_cache|kv_cache|incremental|cache` in `src/` returns **no functional
hits** — only docstring prose (mixers describing what their papers' runtimes would store) and
unrelated hits (`torch.cuda.empty_cache` in `trainer.py:1886`, HF dataset caches). Everything in
this plan is new code.

### 3.5 Inference/deploy plumbing today

| File | Role |
|---|---|
| `src/deploy/inference.py` | `FrankensteinInference`: loads `config.json` → `FrankensteinModelConfig(**flatten_model_dict(...))` (`:93-106`); picks Encoder/Decoder/ViT by `config.mode` (`:108-160`); `predict`/`predict_masked`/`batch_predict`/`benchmark` — single full forward only (`:288`) |
| `src/deploy/deploy.py` | `ModelDeployer`: checkpoint → artifacts (`config.json` + `model*.pt`); formats `standard` / `quantized` (BitNet ternary) |
| `src/deploy/quantization.py` | BitNet weight packing only (`BitLinear`/`BitConv1d`); everything else stored fp |
| `src/deploy/transformers_export.py` | HF export; `FrankensteinForMaskedLM` (`:129-160`), `FrankensteinForCausalLM` (`:163-196`) |
| `src/deploy/bitnet_gguf_export.py` | best-effort GGUF (`i2_s`); `--check` gate accepts standard_attn-only hybrids |
| `src/engine.py` | framework-neutral `build_model` (`:195-250`) and `load_checkpoint` (`:1295-1357`) |
| `src/cli.py` | `infer` (`:256-282`), `deploy` (`:235-254`) — argparse flags only; no generation flags |

---

## 4. Strategy selection

### 4.1 Alternatives considered

| Strategy | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| A. Cache inside each mixer, invoked by the inference script | Each attention module owns its cache tensors; the script pokes them | Locality of computation | Ambiguous layer identity under `num_loops`/Hyperloop/MoD; quantization logic duplicated ×43; impossible to inspect, truncate, or serialize; the script would need per-mixer knowledge | ❌ Rejected |
| **B. Central CacheManager + typed slots + declarative per-mixer capability spec** | One `FrankensteinCache` object (HF `past_key_values`-style) passed down as optional kwarg; slots are typed per state class; codecs/policies live in the slot layer | Single identity key `(logical_layer_idx, slot_kind)`; quantization/compression orthogonal to mixer math; uniform fallback; serializable (future prefix caching); script only orchestrates | Requires additive forward-signature changes across 44 dispatch targets | ✅ **Selected (D1)** |
| C. Cache at the generation-loop level, outside the model | Loop manages tensors only | Zero model changes | Impossible: K/V and recurrent states are produced inside mixers; cannot cover PE offsets or cross-layer mechanisms | ❌ Rejected |

This matches how production systems solve it: HF `past_key_values`/`Cache` classes (per-layer
tuples → static/quantized variants), vLLM block tables (paged, content-agnostic), and hybrid
SSM+attention frameworks (Jamba arXiv:2403.19887, Griffin arXiv:2402.19427) that keep KV tensors and
recurrent states in one scheduler. The literature also pins the layering: quantization/normalization
are **write-time transforms** (KIVI arXiv:2402.02750, KVQuant arXiv:2401.18079), while sparsity and
eviction are **read-time transforms** (Quest arXiv:2406.10774, StreamingLLM arXiv:2309.17453); real
systems keep them orthogonal — this plan exposes both, per mixer.

### 4.2 Data flow after integration

```
CLI / script
  cfg = load_inference_config(yaml)         # schema-validated InferenceConfig
  cfg = apply_cli_overrides(cfg, argv)      # CLI > YAML > defaults (D2)
  cache = FrankensteinCache.new(model, cfg) # owns slots, token tail, cache_position
  y   = decoder.generate(input_ids, cache=cache, ...)      # prefill + decode loop
      └─ decoder.forward(x, past_key_values=cache, use_cache=True, cache_position=t)
           └─ HybridLayer.forward(..., cache=cache, cache_position=t)
                ├─ MoD gate (dense forced at prefill when caching; §7.3)
                ├─ mHC path (unchanged; per-token stream)
                └─ mixer.forward(x, logical_layer_idx=i, pos_encoder=pe,
                                 cache=cache.slot(i), cache_position=t)
                     ├─ supported:  slot.append(k,v[,extras]);  K,V = slot.read()
                     └─ unsupported: legacy path (cache is None)   # D5, never raises
```

---

## 5. Target architecture

### 5.1 New package layout

```
src/model/cache/
├── __init__.py        # public exports: FrankensteinCache, CacheSpec, slot/codec/policy classes
├── base.py            # protocols: CacheSlot, CacheCodec, CachePolicy; SlotMeta dataclass
├── manager.py         # FrankensteinCache: slot dict keyed by logical_layer_idx, token tail,
│                      #   cache_position, truncate(), device moves, force-dense-prefill flag
├── slots.py           # KVSlot, LatentKVSlot, StateSlot, RingSlot, ConvTailSlot, TokenTailSlot
├── capabilities.py    # CACHE_SPECS: mixer_key -> CacheSpec (single declarative registry)
├── hooks.py           # shared helpers used by mixers: get_or_create_slot, append_and_read_kv,
│                      #   build_causal_mask(kv_len, q_len, q_offset), pe-offset plumbing
├── quant/
│   ├── __init__.py    # build_cache_codec(name, params) -> CacheCodec  (registry)
│   ├── none.py        # identity (fp16/fp32 passthrough)
│   ├── int8.py        # symmetric int8, asymmetric axes (K: channel, V: token)
│   └── kivi4.py       # group-wise int4 KIVI-style with protected fp window
│   # future: int2_kivi.py, palu_lowrank.py, commvq.py, qjl_sketch.py, block_gtq.py
└── compress/
    ├── __init__.py    # build_cache_policy(name, params) -> CachePolicy (registry)
    ├── none.py
    └── window_sinks.py# StreamingLLM-style: pinned sinks + sliding window
    # future: h2o.py, snapkv_prefill.py, pyramid.py, sparsek.py
```

### 5.2 Core protocols (`src/model/cache/base.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, runtime_checkable
import torch

QAxis = Literal["channel", "token"]  # quantization granularity axis
#   KIVI (arXiv:2402.02750): keys have persistent per-CHANNEL outliers (RoPE structure),
#   values have per-TOKEN outliers -> K quantized per channel(-group), V per token.


@dataclass
class SlotMeta:
    """Static per-slot description, fixed at slot creation from config + mixer."""
    mixer_key: str
    logical_layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    pe_type: str                 # "rope" | "hope" | "alibi" | ... (visibility for codecs; see §9.6)
    rope_scaling: float = 1.0    # keys are stored POST-RoPE; scaling versioned here (§16, D8)


@runtime_checkable
class CacheCodec(Protocol):
    """Lossless ('none') or lossy transform applied at slot WRITE and READ time.

    Contract: encode_write/decode_read are pure w.r.t. the slot; update() appends
    already-encoded rows. Codecs must never raise on unknown tensors -- on any
    internal error they fall back to identity and record a one-time warning."""
    name: str
    def encode_write(self, name: str, tensor: torch.Tensor, axis: QAxis, meta: SlotMeta) -> Any: ...
    def decode_read(self, stored: Any, meta: SlotMeta) -> torch.Tensor: ...
    def update(self, stored: Any, new_stored: Any, meta: SlotMeta) -> Any: ...


@runtime_checkable
class CachePolicy(Protocol):
    """Eviction / compression policy operating on a slot across decode steps."""
    name: str
    def on_append(self, slot: "CacheSlot", cache_position: int) -> None: ...
    def on_step_end(self, slot: "CacheSlot", cache_position: int) -> None: ...
    def budget(self, logical_layer_idx: int) -> Optional[int]:
        """Max retained tokens for this layer; None = unlimited."""


@runtime_checkable
class CacheSlot(Protocol):
    kind: str                       # "kv" | "latent_kv" | "state" | "ring" | "conv_tail" | "token_tail"
    meta: SlotMeta
    length: int                     # number of cached positions (0 for state slots)
    def append(self, **tensors: torch.Tensor) -> None: ...
    def read(self) -> Dict[str, Any]          # dequantized views for the mixer
    def truncate(self, num_tokens: int) -> None: ...   # cheap rollback (LayerSkip-style)
    def to(self, device: torch.device) -> None: ...
```

### 5.3 Slot types (`src/model/cache/slots.py`)

| Slot | `kind` | Stores | Used by | Notes |
|---|---|---|---|---|
| `KVSlot` | `"kv"` | `k: (B, H_kv, T, d)`, `v: (B, H_kv, T, d)` **post-RoPE**, `positions: (T,)` absolute int64, optional `extras: dict[str, Tensor]` | all quadratic softmax mixers | `extras` carry per-token metadata: FOX cumulative log-forget `c_j` `(B, H, T)`, FoX-Pro previous `(k̃, ṽ)`, MTLA merged-slot position ranges, NSA block statistics, BigBird selected block ids |
| `LatentKVSlot` | `"latent_kv"` | `c_kv: (B, T, r_kv)` (+ optional decoupled `k_rope: (B, H_kv, T, d_rope)`), `positions` | `mla_attn`, `gqla_attn` (absorb path), `mlra_attn`, `tucker_attn` (ranked), `gta_attn` variant, `cca_attn`/`ccgqa_attn` | heterogeneous shapes per mixer; quantizer treats `c_kv` per-token and `k_rope` per-channel |
| `StateSlot` | `"state"` | one recurrent matrix `S: (B, H, d_k, d_v)` (+ optional gate accumulator tensors), updated **in place** per token | `retnet(_attn)`, `gla_attn`, `deltanet_attn`, `gated_deltanet(_2)_attn`, `hgrn2_attn`, `kda_attn`, falcon ×6, `gma_attn` | constant memory; never quantized in F3 (D12); `truncate` is a no-op (states are not position-indexed) |
| `RingSlot` | `"ring"` | fixed-capacity K/V ring buffer with head/tail pointers, `positions`, pinned sink region | `swa_attn`, `longformer_attn`, `bigbird_attn`, `sparse_transformer_attn` (bounded mode), any mixer + `window_sinks` policy | position remapping on wrap is **not needed for correctness** because `positions` metadata is stored; Mistral-style unrolled re-rotation is unnecessary (keys are stored post-RoPE) |
| `ConvTailSlot` | `"conv_tail"` | last `kernel − 1` pre-conv rows (FIFO), small and constant | falcon ×6, `cca_attn`/`ccgqa_attn`, gated-deltanet family when a causal short conv is present | compositional: a mixer can own one `KVSlot`/`StateSlot` **and** one `ConvTailSlot` |
| `TokenTailSlot` | `"token_tail"` | last `n` token ids `(int64)` | `engram_attn` (n-gram hash needs the previous `max_ngram − 1` ids), generic consumers | fed from the generation loop's token stream, not from activations |

**Slot read/write with quantization (KVSlot example)**

```python
class KVSlot:
    def append(self, *, k, v, positions, extras=None):
        self._k_blocks.append(codec.encode_write("k", k, axis="channel", meta))  # int8/int4
        self._v_blocks.append(codec.encode_write("v", v, axis="token",   meta))
        self._positions = torch.cat([self._positions, positions])
        policy.on_append(self, cache_position)
    def read(self):
        K = torch.cat([codec.decode_read(b, meta) for b in self._k_blocks], dim=2)
        V = torch.cat([codec.decode_read(b, meta) for b in self._v_blocks], dim=2)
        return {"k": K, "v": V, "positions": self._positions, "extras": self._extras}
```

v1 keeps this simple (decode+concat per step). A documented future optimization is a persistent
decoded buffer invalidated on append (and, later, fused quantized-attention kernels à la KIVI/QJL);
do **not** build kernels in v1.

### 5.4 Capability registry (`src/model/cache/capabilities.py`)

```python
@dataclass(frozen=True)
class CacheSpec:
    supported: bool
    slot_type: str                  # "" when not supported
    slot_types: Tuple[str, ...] = ()# a mixer may need several slots (e.g. state + conv_tail)
    extras: Tuple[str, ...] = ()    # per-token metadata the mixer will store (documentation)
    notes: str = ""                 # paper-based rationale, surfaced in config_loader warnings

CACHE_SPECS: Dict[str, CacheSpec] = { ... }   # one entry per schema enum value (44), see §6
```

Rules:

- The registry is the **single declarative source** consumed by `HybridLayer` at runtime and by
  `config_loader` for static validation.
- `HybridLayer.slot_for(logical_layer_idx)` returns `None` when `spec.supported is False` or a
  `per_mixer` override disables the mixer ⇒ mixer receives `cache=None` and runs the legacy path.
  **No exceptions, ever (D5).**
- A consistency test asserts `set(CACHE_SPECS) == set(schema enum) == set(mixer_registry keys) ∪ {"mamba"}`.

### 5.5 Forward plumbing (all changes additive; see Appendix B for exact diffs)

- **Mixers**: `forward(x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)`.
  Legacy path is bit-identical when `cache is None`.
- **`HybridLayer.forward`** gains `cache=None, cache_position=None`; resolves the slot via
  `capabilities` and passes it to the mixer respecting the four call shapes (S1 skips entirely;
  S2 passes `cache` only from F2 when the mixer supports state slots; S3 also receives the token
  tail via `input_ids`-adjacent plumbing).
- **`FrankensteinEncoder.forward(input_ids, past_key_values=None, use_cache=False,
  cache_position=None, return_cache=False)`** → `(logits, cache)` when caching. Handles: absolute-PE
  `pos_encoder.add` only on new tokens with offset; AttnRes incremental mode; mHC unchanged.
- **`FrankensteinDecoder.forward`** mirrors the encoder signature.
- **`FrankensteinDecoder.generate(...)`** rewritten as prefill + decode:

```python
@torch.inference_mode()
def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=50, top_p=1.0,
             greedy=False, past_key_values=None, use_cache=False):
    # 1) PREFILL: one full forward over the prompt, building the cache.
    logits, cache = self.forward(input_ids, past_key_values=past_key_values,
                                 use_cache=use_cache, return_cache=use_cache)
    next_logits = logits[:, -1, :]
    tokens = []
    for step in range(max_new_tokens):
        t = sample(next_logits, temperature, top_k, top_p, greedy)   # greedy ⇒ argmax
        tokens.append(t)
        if step == max_new_tokens - 1:
            break
        cache_position = cache.next_position()                        # absolute index to write
        logits, cache = self.forward(t, past_key_values=cache, use_cache=use_cache,
                                     cache_position=cache_position, return_cache=True)
        next_logits = logits[:, -1, :]
    return torch.cat([input_ids, torch.stack(tokens, dim=1)], dim=1)
```

- When `use_cache=False` (default), `generate()` keeps the legacy O(S²) full-recompute loop —
  **bit-identical behavior to today** (D4).

### 5.6 Causal masking in cached decode

- Prefill: unchanged `triu` causal masks over the prompt.
- Decode (`q_len == 1`): no mask needed for softmax mixers (a single query attends to all cached
  keys). Sparse patterns evaluate their pattern against `kv_positions` metadata (B5 fix): a cached
  key is attended iff `pattern(q_pos, kv_pos)` holds. Pattern builders gain optional
  `(q_len, kv_len, q_offset, kv_positions)` parameters; defaults reproduce today's behavior.

---

## 6. Per-mixer compatibility matrix

### 6.1 State-class legend

| Class | Meaning | Cache representation |
|---|---|---|
| (i) | Classic per-token K/V attention | `KVSlot` |
| (ii) | Compressed / latent attention | `LatentKVSlot` |
| (iii) | Fixed-size recurrent state | `StateSlot` |
| (iv) | Ring / sliding buffer (± attention sinks) | `RingSlot` |
| (v) | Auxiliary FIFO tail (causal conv, last pairs, cumulative scalars) | `ConvTailSlot` / extras |
| (vi) | Paper describes no cache; full recompute is the only documented mode | fallback, cache possible in principle |
| (vii) | Not sequence-attention over past tokens | **never cached** |

### 6.2 Dense / softmax family (quadratic QKᵀ)

| Key | Class (file) | Paper | State | Cache in paper? | Incremental-decode recipe | Gotchas / notes | Phase |
|---|---|---|---|---|---|---|---|
| `standard_attn` | `StandardAttention` (`attention/standard.py:23`) | Attention Is All You Need, arXiv:1706.03762 | (i) | **No** — the 2017 paper never describes the cache (later folklore) | Append `K_t, V_t`; O(1) score per cached key | Docstring already claims cache support (`:32`) — aspirational today | **F0** |
| `gqa_attn` | `GroupedQueryAttention` (`attention/grouped_query_attention.py:24`) | GQA, arXiv:2305.13245 (MQA arXiv:1911.02150) | (i) | **Yes** — the paper *is* the quality-vs-cache-size tradeoff | Cache `num_kv_heads` heads/token; `repeat_interleave` query heads onto groups | Most cache-ready mixer; smallest KV per token | **F0** |
| `sigmoid_attn` | `SigmoidAttention` (`attention/sigmoid.py:25`) | Sigmoid Self-Attention, arXiv:2409.04431 | (i) | **No** — kernels only (FlashSigmoid) | Identical to standard; per query, accumulate sigmoid logits **and** the L1 denominator over the *full* cached prefix | Sigmoid is not shift-invariant: the normalizer `Σ_j σ(s_ij)` must be re-summed over the whole cache each step; **no online-softmax max trick** | F1 |
| `gated_softmax_attn` | `GatedSoftmaxAttention` (`attention/gated/gated_softmax_attn.py:33`) | Gated Attention, arXiv:2505.06708 | (i) | **No** | Standard KV cache | Gate `σ(XW_g)` is applied to the SDPA **output** and depends only on the current token ⇒ nothing extra to cache | F1 |
| `titan_attn` | `TitanAttention` (`attention/titan.py:25`) | Titans, arXiv:2501.00663 | (i) — window | **Yes** (state-only for the *memory* module) | Standard KV cache over the attention window | Repo implements **only the attention part**; the neural-memory (fast-weight) module is not implemented — treated as plain (i); HoPE/RoPE via PE layer | F1 |
| `fox_attn` | `ForgettingAttention` (`attention/gated/fox_attn.py:34`) | Forgetting Transformer (FoX), arXiv:2503.02130 | (i)+(v) | **Yes (partial)** — §3 hardware-aware: store cumulative log-forget sums | Cache `(K_j, V_j, c_j)` with `c_i = Σ_{l≤i} log f_l` per token/head; logit bias `D_ij = c_i − c_j`; decode = row of `D` against cached `c_j` | `D` is pairwise-relative ⇒ must cache one scalar per token/head; FoX-Pro KV-shift additionally needs previous `(k̃, ṽ)`; paper leaves forget-gate-based eviction as **future work** | F1 |
| `ode` | `ODEAttentionBlock` (`attention/ode.py:117`) | ODE Transformer, arXiv:2203.09176 (ACL 2022; repo docstring credits "Zhang et al. 2021", arXiv lists Li et al.) | (i), final solver step | **No** — NMT paper, no cache discussion | Standard KV cache taken from the **converged `z(1)`** state | K/V of the new token are recomputed at every Euler/RK4 iteration; weights shared across steps | F2 |

### 6.3 Latent family (`src/model/attention/latent/`, 10 classes)

| Key | Class (file) | Paper | State | Cache in paper? | Incremental-decode recipe | Gotchas / notes | Phase |
|---|---|---|---|---|---|---|---|
| `mla_attn` | `MLAAttention` (`latent/mla_attn.py:41`) | Latent MHA for Small LMs, arXiv:2506.09342 — **origin: MLA / DeepSeek-V2, arXiv:2405.04434** | (ii) | **Yes** — "only the latent is cached"; 45 % KV reduction at `r = d/2` | Cache `c_KV = W_DKV x` (r_kv/token); decode: `k = W_UK c_KV`, `v = W_UV c_KV` | **This repo's variant applies RoPE to the decompressed q,k** ⇒ DeepSeek's MQA-absorb trick (absorb `W_UK` into `W_q`) is *unavailable*; expand + RoPE every step. DeepSeek's own MLA caches `(c_KV, k_rope)` with a decoupled RoPE key to keep absorption valid — do not confuse the two designs | F1 |
| `gqla_attn` | `GQLAAttention` (`latent/gqla_attn.py:41`) | GQLA, arXiv:2605.15250 | (ii) | **Yes** — two explicit decode paths over one weight set | `gqla_decode_path: "mqa_absorb"`: cache joint latent, fold `W_UK/W_UV` into `W_q/W_o` (MLA-style); `"gqa"`: expand latent to `num_groups` full K/V heads | Paths are algebraically equivalent; choice is a hardware/roofline decision; repo already records the path — the slot layout follows it | F1 |
| `mlra_attn` | `MLRAAttention` (`latent/mlra_attn.py:36`) | MLRA, arXiv:2603.02188 | (ii) | **Yes** — 4 disjoint latent sub-heads, partitionable | Absorb per-block NoPE up-projections into queries; RoPE key decoupled (separate) | Win is TP **sharding** (each device loads 1/4), not compression; total cache ≈ MLA-sized | F1 |
| `tucker_attn` | `TuckerAttention` (`latent/tucker_attn.py:39`) | Tucker Attention, arXiv:2603.30033 | (ii) if ranks < d, else (i) | **Partial** — cache named as bottleneck; no explicit decode algorithm | Cache factorized/latent K,V when `key_rank`/`value_rank < hidden` | Paper focuses on parameter count (≈10× fewer params); cache size follows the ranks | F1 |
| `iha_attn` | `IHAAttention` (`latent/iha_attn.py:42`) | Interleaved Head Attention, arXiv:2602.21371 | (i) | **No** — expressivity paper, no cache section | Cache the H real per-token K,V (MHA-sized) | Pseudo-K/V are *static learned mixings* (parameters) — computed on the fly, **not** cached; P² pseudo-patterns cost extra compute per step | F1 |
| `gta_attn` | `GTAAttention` (`latent/gta_attn.py:39`) | GTA, arXiv:2506.17286 | (ii) | **Yes** — "shared attention map… decreasing the key cache size"; KV −70 %, 2× e2e | Cache one K per head-*group* (shared map) + latent `c_V` decoded by a silu MLP | **Heterogeneous slot**: grouped keys + per-head latent values | F1 |
| `mtla_attn` | `MTLAAttention` (`latent/mtla_attn.py:38`) | MTLA, arXiv:2505.13544 | (ii) | **Yes** — "during inference the KV cache has fewer [entries] than the input"; 5.3× decode, 8.3× memory | At each stride boundary a hyper-network merges `merge_factor` adjacent cache vectors into one slot; queries attend via the **stride-aware causal mask** | The only method compressing the cache along *time*; merged slots carry **position ranges**; schedule is dynamic (hyper-net). Parity between train/infer schedules is a named risk (§15) — implement with equivalence tests; on failure ⇒ mark unsupported and fall back (D5) | F1 |
| `cca_attn` | `CCAAttention` (`latent/cca_attn.py:96`) | CCA, arXiv:2510.04476 | (ii)+(v) | **Yes** — attention runs entirely inside the compressed latent | Cache compressed latent K,V **plus** causal-conv tail (last `kernel−1` packed q/k latents) and the previous shifted value | RoPE applied *in the latent* (no decoupled RoPE key needed, unlike MLA); two convs (causal seq + head-wise channel) add small carry state | F2 |
| `ccgqa_attn` | `CCGQAAttention` (`latent/cca_attn.py:316`) | CCGQA, arXiv:2510.04476 | (ii)+(v) | **Yes** — claims 8× KV compression at no quality loss | CCA recipe + GQA head sharing inside the latent (`C₂/C₁ = H/G`); decoupled Q `C₁` vs KV `C₂` compression | Same conv-tail states as CCA | F2 |
| `gma_attn` | `GaussianMixtureAttention` (`latent/gma_attn.py:82`) | Gaussian Mixture Attention, arXiv:2606.18283 | (iii) | **State-only** — causal variant defined by prefix-restricted cumulative sums | `Ṽ_t = Ṽ_{t−1} + γ^K_t v_tᵀ`; `Z_t = Z_{t−1} + γ^K_t`; `o_t = Γ^Q_t Ṽ_t / (Γ^Q_t Z_t + ε)` — K-slot accumulators per head | Not QKᵀ; only rotation PEs apply (no score-bias PE possible, `gma_attn.py:100-102`); authors note causal GMA still trails optimized SDPA | F2 |

### 6.4 Sparse family (`src/model/attention/sparse/`, 10 classes)

| Key | Class (file) | Paper | State | Cache in paper? | Incremental-decode recipe | Gotchas / notes | Phase |
|---|---|---|---|---|---|---|---|
| `sparse_transformer_attn` | `SparseTransformerAttention` (`sparse/sparse_transformer_attn.py:29`) | Sparse Transformers, arXiv:1904.10509 | (i) | **Yes (implicit)** — sampling section caches prior activations; per-step attention touches O(√n) keys | Standard K/V cache; each query attends its strided/fixed pattern positions | "Summary columns" are the *last position of each √n block* — a token's role changes when a block completes ⇒ pattern evaluation must be position-based (B5 fix) | F1 |
| `longformer_attn` | `LongformerAttention` (`sparse/longformer_attn.py:26`) | Longformer, arXiv:2004.05150 | (iv) | **No** — encoder-only tasks; no decode described | (If forced) ring window `w` + pinned global-token K/V | Globals are task-specific indices; paper's dilated windows complicate a single ring buffer | F1 |
| `swa_attn` | `SlidingWindowAttention` (`sparse/swa_attn.py:33`) | "Sliding Window Beats Linear Attention", arXiv:2608.28444 (**built on** StreamingLLM, arXiv:2309.17453) | (iv) | **Yes** — the paper *is* a cache-eviction recipe: "keeps only the first few tokens (attention sinks) and a sliding window of recent tokens" | Pin first `swa_num_sinks` (paper: s=4) sink K/V; ring-buffer the last `w − s` K/V; nothing else | Removing sinks collapses performance catastrophically (StreamingLLM); training-free mask change only | F1 |
| `bigbird_attn` | `BigBirdAttention` (`sparse/bigbird_attn.py:28`) | BigBird, arXiv:2007.14062 | (iv) | **No** — encoder benchmarks only | Ring window + pinned globals + **seeded** random links | Random links currently re-drawn per forward ⇒ must become seed-reproducible (seed from `logical_layer_idx` at slot creation) or stored as slot extras; else parity with full forward fails | F1 |
| `sparsek_attn` | `SparseKAttention` (`sparse/sparsek_attn.py:95`) | SparseK, arXiv:2406.16747 | (v) — policy | **Yes** — "linear time complexity and constant memory footprint during generation" | Learned `score_net` scores cached KV; differentiable top-k keeps exactly k survivor pairs | Eviction is **permanent within a generation**; best modeled as a `CachePolicy` operating on a `KVSlot`, not a distinct slot | **F4** (as policy) |
| `nsa_attn` | `NSAAttention` (`sparse/nsa_attn.py:31`) | Native Sparse Attention, arXiv:2502.11089 | (i)+(ii)+(iv) | **Yes** — hardware-aligned decode (11.6× at 64k) | Maintain **three caches**: (a) block-compressed KV (mean-pooled per block), (b) fine-grained KV for query-selected blocks, (c) sliding window; learned gates `g` blend branches | Selection is query- and head-dependent ⇒ fine-grained cache cannot be evicted. Phased: **F1 window-branch-only cached decode; F2 full three-branch** | F1 (partial) → F2 (full) |
| `msa_attn` | `MSAAttention` (`sparse/msa_attn.py:50`) | MiniMax Sparse Attention, arXiv:2606.13392 | (i) | **Yes** — GQA cache + Index branch; 7.6× decode at 1M | Standard grouped KV cache; per step, the detached index head scores blocks (block size 128, exp-free), top-k per GQA group, local block forced in | Index-branch input is detached (`no_grad`, `:254-260`); trained via KL alignment aux loss only | F1 |
| `sparda_attn` | `SparDAAttention` (`sparse/sparda_attn.py:44`) | SparDA, arXiv:2606.04511 | (i) + prefetch metadata | **Yes** — decode-focused: full KV (offloadable to CPU); Forecast of layer ℓ predicts blocks for layer ℓ+1 → PCIe prefetch overlap | Keep full KV; run Forecast → top-k block ids for next layer; fetch while current layer computes | Forecast is cross-layer lookahead (<0.5 % params); F1 covers plain cached decode; prefetch optimization is F4-grade work | F1 (basic) → F4 (prefetch) |
| `sparge_attn` ⚠️ eval-only | `SpargeAttention` (`sparse/sparge_attn.py:34`) | SpargeAttn, arXiv:2502.18137 | (i) | **Yes** — inference method by definition; the cache stays **full** (it skips compute, not memory) | Two-stage block pruning over the cached K/V: pre-softmax block scores → skip; softmax-aware online filter → skip more | Training-free; rides on quantized KV (SageAttention-style); online-softmax rescaling carried across kept blocks; diagonal always preserved | F1 (read-only over cache) |
| `fasa_attn` ⚠️ eval-only | `FASAAttention` (`sparse/fasa_attn.py:35`) | FASA, arXiv:2602.03152 | (i) + dynamic top-k | **Yes** — decode-time selection: per-head dominant RoPE-frequency chunks give a "computationally free" importance proxy | Cache (RoPE'd) keys; per query, dot products restricted to the head's dominant frequency-chunk dimensions rank tokens; attend only to top `n_fac` | Importance **must** be recomputed per query over cached RoPE-K (cannot precompute at write); per-head selection; paper: 18.9 % of cache retained → 2.56× speedup, ≈100 % of full-KV quality on LongBench-V1 | F1 (read-only over cache) |

### 6.5 Recurrent / linear family (state-based)

| Key | Class (file) | Paper | State | Cache in paper? | Incremental-decode recipe | Gotchas / notes | Phase |
|---|---|---|---|---|---|---|---|
| `retnet` | `MultiScaleRetention` (`attention/retnet.py:26`) | RetNet, arXiv:2307.08621 | (iii) | **State-only** — the "recurrent" O(1)-per-token paradigm of the paper | `S_n = γ S_{n−1} + K_nᵀ V_n`; `o_n = Q_n S_n` (× decay normalizer) per head | RoPE enters as a complex rotation folded into γ (DID form); per-head multi-scale γ; mixer **replaces x** (call shape S2) — state slot plumbed only from F2 | F2 |
| `retnet_attn` | `RetNetAttention` (`attention/gated/retnet_attn.py:26`) | RetNet, arXiv:2307.08621 | (iii) | **State-only** | same as `retnet` (thin wrapper over `MultiScaleRetention`) | | F2 |
| `gla_attn` | `GatedLinearAttention` (`attention/gated/gla_attn.py:32`) | GLA, arXiv:2312.06635 | (iii) | **State-only** — recurrent + chunkwise forms; O(d²) state | `S_t = G_t ⊙ S_{t−1} + v_t k_tᵀ`, diagonal data-dependent gate `G_t` (logsigmoid/16) | Data-dependent gate ⇒ paper's chunkwise merge schedules are **training-only**; decode is strictly token-recurrent | F2 |
| `deltanet_attn` | `DeltaNetAttention` (`attention/gated/deltanet_attn.py:32`) | Parallelizing DeltaNet, arXiv:2406.06484 | (iii) | **State-only** — contribution is parallelization (WY representation) | `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, k L2-normalized | Delta rule *overwrites* in place (exact retrieval, not additive) | F2 |
| `gated_deltanet_attn` | `GatedDeltaNetAttention` (`attention/gated/gated_deltanet_attn.py:32`) | Gated DeltaNet, arXiv:2412.06464 | (iii) | **State-only** — deployed in Qwen3-Next | `S_t = α_t S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` | Production deployments add a causal conv ⇒ check repo for a short conv; if present, add `ConvTailSlot` | F2 |
| `gated_deltanet2_attn` | `GatedDeltaNet2Attention` (`attention/gated/gated_deltanet2_attn.py:41`) | Gated DeltaNet-2, arXiv:2605.22791 | (iii) | **State-only** — abstract: "replaces the unbounded cache… fixed-size recurrent state… constant memory" | `S_t = (I − k_t(b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t(w_t ⊙ v_t)ᵀ` | Two channel-wise gates (erase `b_t`, write `w_t`) + channel decay `D_t`; fp32 log-decay params | F2 |
| `hgrn2_attn` | `HGRN2Attention` (`attention/gated/hgrn2_attn.py:31`) | HGRN2, arXiv:2404.07904 | (iii) | **State-only** — linear RNN, O(1)/step | `S_t = diag(g_t) S_{t−1} + v_t k_tᵀ`, gate `g_t` lower-bounded | Lower bound prevents full erasure (hierarchical memory) | F2 |
| `kda_attn` | `KDAAttention` (`attention/gated/kda_attn.py:51`) | Kimi Delta Attention / Kimi Linear, arXiv:2510.26692 | (iii) | **State-only** for KDA layers; the full Kimi Linear model is a KDA+MLA hybrid (−75 % KV, 6× decode at 1M) | `S_t = (I − β_t k_t k_tᵀ) D_t S_{t−1} + β_t v_t k_tᵀ`, `D_t = diag(α_t)` per **key channel** | Channel-wise decay ⇒ production uses a bespoke DPLR chunkwise kernel; repo is fine with the plain recurrence | F2 |
| `falcon1_attn` … `falcon3a_attn` (6 keys) | `Falcon{1,2,3,1A,2A,3A}Attention` (`attention/gated/falcon_attn.py:344/444/543/724/805/887`, shared `_FalconBase`, fwd `:166-218`) | Fast Weight Attention (Falcon family), arXiv:2608.27763 | (iii)+(v) | **State-only** — "recurrent, masked-parallel, and chunk-parallel forms"; O(d²) state, O(1)/step | `S_t = γ_t S_{t−1} + n_t x_t r_tᵀ` (regression variants 1/2/3) or `n_t x_t v_tᵀ` (additive 1A/2A/3A); read `o_t = S_tᵀ φ(q_t)` | (1) **Read-after-write prefix shift**: training pair is `(φ(k_{t−1}), v_t)` — one-step offset vs DeltaNet; (2) causal short conv on projections ⇒ `ConvTailSlot`; (3) boundary sentinels `x_1 = 0`, `η_1 = 0`, `(α_1, γ_1) = (0, 1)`; (4) fp32 `log1p(−α)` chunk-local renormalization; (5) Falcon-3/3A: sliding window bounds the **rehearsal batch** — exact continuation needs a FIFO of the last `B−1` pairs (`falcon_attn.py:558-559`) | F2 |

### 6.6 Other

| Key | Class (file) | Paper | State | Cache in paper? | Recipe / notes | Phase |
|---|---|---|---|---|---|---|
| `engram_attn` | `EngramLayer` (`attention/engram.py:215`) | Engram: Conditional Memory via Scalable Lookup, arXiv:2601.07372 | (vii) + token tail | **N/A** — not attention over past tokens; O(1) deterministic n-gram-suffix lookup in static learnable tables | Needs only the last `max_ngram − 1` **token ids** (already available to the generation loop); tables are static parameters; repo already passes `input_ids` (call shape S3) | F2 (`TokenTailSlot`) |
| `mamba` | *none* — linear placeholder (`hybrid_layer.py:221-222`) | — | n/a | n/a | Stateless per-token linear ⇒ **inherently incremental**, no cache needed; only verify it runs in the cached loop | F0 (trivial) |
| `ssog_attn` | `SSOGAttention` (`attention/ssog.py:98`) | SSOG — "A Few Gaussians Is All You Need" (pisoni.ai blog + GitHub; **no arXiv**) | (vii) | **N/A** | Learned Gaussian field over relative 2-D grid positions; encoder-only, bidirectional; **no causal formulation exists** — no cache can be defined; raises unless `seq_len == grid_h × grid_w` (`ssog.py:263-272`) | **Never cached**; documented as such |

### 6.7 "Not cacheable" summary (fallback, never a failure — D5)

| Mixer | Reason | Behavior when `kv_cache.enabled` |
|---|---|---|
| `ssog_attn` | Bidirectional vision-grid mixer; no causal formulation exists in the source | Runs uncached; `config_loader` warns |
| `engram_attn` | O(1) table lookup; only a FIFO of token ids required (provided by `TokenTailSlot` in F2) | F0–F1: runs uncached; F2+: runs with token-tail metadata only |
| `mamba` | Stateless linear placeholder; nothing to cache | Runs unchanged (already incremental) |
| PaPE-configured layers (`pape`, `pape_efficient`, `pape_ri`) | PE augmentation is **content-dependent** (`encode_qk(hidden, q, k, positions)`, `common.py:275`); correctness of caching post-PE keys is unproven | Layer marked unsupported in `CACHE_SPECS` **when its `use_pe` flag selects PaPE**; falls back to uncached execution for that layer (documented; revisit after formal verification) |

### 6.8 Phase assignment summary

| Phase | Mixers | Count |
|---|---|---|
| **F0** | `standard_attn`, `gqa_attn` (+ `mamba` passthrough verification) | 2 |
| **F1** | `sigmoid_attn`, `gated_softmax_attn`, `titan_attn`, `fox_attn`, `mla_attn`, `gqla_attn`, `mlra_attn`, `tucker_attn`, `iha_attn`, `gta_attn`, `mtla_attn`, `sparse_transformer_attn`, `longformer_attn`, `swa_attn`, `bigbird_attn`, `msa_attn`, `sparge_attn`, `fasa_attn` (+ NSA window-branch partial, SparDA basic) | 18 (+2 partial) |
| **F2** | `ode`, `retnet`, `retnet_attn`, `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `gated_deltanet2_attn`, `hgrn2_attn`, `kda_attn`, `falcon1_attn`, `falcon2_attn`, `falcon3_attn`, `falcon1a_attn`, `falcon2a_attn`, `falcon3a_attn`, `gma_attn`, `cca_attn`, `ccgqa_attn`, `engram_attn` (token tail), NSA full, SparDA full; + AttnRes incremental + MoD prefill policy | 21 |
| **F3** | Quantization codecs over all cached slots (`none`, `int8`, `int4_kivi`) | — |
| **F4** | Compression policies (`window_sinks`, `h2o`, `snapkv_prefill`, `pyramid`, `sparsek`-as-policy) | — |
| **F5** | HF export, GGUF check, docs, examples, bibliography | — |

Never cached: `ssog_attn`, `mamba` (n/a). Total: 2 + 18 + 21 + 2 = 43 mixers + `mamba` = **44 enum values covered**.

---

## 7. Mechanism interactions

### 7.1 `num_loops` and Hyperloop (`src/model/hyperloop.py`)

- Without hyperloop the encoder re-runs the stack `num_loops` times (`frankenstein_encoder.py:292-294`);
  with hyperloop it runs begin layers once, middle layers `num_loops` times wrapped in hyper-connections
  (`:264-291`; `HyperloopConnections` at `hyperloop.py:59`, expand/collapse `:221-242`,
  `apply_fpre/apply_update` `:244-288`; paper arXiv:2604.21254).
- **Cache identity = `logical_layer_idx` (0 .. `num_layers × num_loops − 1`)**, *not* module identity:
  the same physical layer object executes `num_loops` times with shared weights but distinct logical
  roles. The encoder already threads `logical_layer_idx` into every call (`:247`) — reuse it as the
  slot key. Each logical iteration gets its **own slot**.
- The hyper-connection stream `(B, S, n, C)` is **per-token** state: at decode with `S_new = 1` it
  flows through the same forward without cross-step carry. No extra slot type required.

### 7.2 mHC (`src/model/mhc.py`, arXiv:2512.24880)

- Expands the residual into an n-stream `(B, S, n, C)`; per sub-layer, input-dependent `H_pre`/`H_post`
  and a Sinkhorn-Knopp doubly-stochastic `H_res` (`SinkhornKnoppFunction`, `mhc.py:44-100`); wired per
  sub-layer at `hybrid_layer.py:305-324`, executed in `_forward_dense_mhc` (`:418-489`).
- The mixer is unchanged; the cache stores K/V of the mixer's normalized pre-projection input. The
  n-stream itself needs no cross-step carry (per-token, same argument as §7.1).
- mHC ⇔ MoD mutual exclusion (`hybrid_layer.py:141-146`) already prevents the only conflicting
  combination. **No cache changes required.** Tests must cover `mhc.enabled: true` + cache anyway.

### 7.3 Mixture-of-Depths (`hybrid_layer.py:502-567`)

- Per layer: `capacity = max(1, ceil(S · capacity_ratio))`; `torch.topk` over the sequence selects the
  tokens to process; the rest bypass the mixer. Short-circuit: when `capacity >= seq_len`, the dense
  path runs (`:552-553`).
- **Decode (`S_new = 1`)**: capacity ≥ 1 = seq_len ⇒ MoD degenerates to dense automatically — cached
  decode is safe by construction.
- **Prefill with cache**: routing would produce ragged, token-dependent caches. Decision (**D9**):
  when `kv_cache.enabled`, HybridLayer forces the dense path during prefill
  (`cache_position == 0`-anchored / `q_len > 1`) — i.e., every prompt token is processed at every
  routed layer. This is a documented, deliberate train/inference divergence (it can only *add* K/V,
  never attend to keys that do not exist; the router at decode sees dense-layer statistics).
  A per-mixer ragged cache (VestigeKV-style, arXiv:2609.03949) is explicitly **out of scope** for v1
  and listed as future work.
- Aux loss attribute `last_mixture_of_depths_aux_loss` is unaffected.

### 7.4 AttnRes residuals (`src/model/residuals/`, types `standard|none|full_attn|block_attn`)

- `FullAttentionResidual` runs depth-wise softmax attention over **all previous logical-layer
  outputs**, with learned per-layer pseudo-queries; buffers `v_0 = embedding`, `v_l = layer outputs`;
  the encoder resets the state at every forward start (`frankenstein_encoder.py:213-216`) and applies
  `residual(idx, x)` itself (`:253-254`) — `HybridLayer` deliberately does not (comment at
  `hybrid_layer.py:374-382`). Lifecycle: `register_state(num_layers) → reset_state() → forward(layer_idx, x) → finalize(x)` (`residuals/base.py:33-160`).
- For incremental decoding these buffers must **accumulate across steps** instead of resetting:
  add an incremental mode (enabled when `use_cache`) where `reset_state()` is called once at prefill
  and each decode step appends the new token's per-layer outputs `(B, 1, C)`. `finalize` keeps its
  partial-block folding logic (`block_attn_num_blocks`) operating on the growing buffer.
- This is F2 work; until then, cache + AttnRes combination is disabled with a warning (spec-level
  `supported: False` for residuals, not mixers).

### 7.5 BitNet (`bitnet.enabled` default true; `routers`, `conv`)

- `BitLinear` (`attention/common.py:60-96`): per-forward ternary weight quant (absmean, STE) + int8
  per-token activation quant + input LayerNorm. Stateless per forward. `bake_ternary_weights()`
  (`:98-129`) exists for export.
- **No interaction with the KV cache** (weights/activations, not cache contents). KV quantization
  (§9) composes with it; the existing FP16 deployment guard (`deploy/inference.py:143-154` refuses
  fp16 when `BitLinear` is present) is untouched. Quant-error tests must run with BitNet on and off
  (§13).

### 7.6 `layer_pattern` heterogeneity

- `layer_pattern` may list different mixers per layer; `FrankensteinCache` creates slots lazily from
  the per-layer `CacheSpec`, so a single model can hold `KVSlot`s (layers 0, 2), a `LatentKVSlot`
  (layer 1), `StateSlot`s (layer 3)… natively — the Jamba/Griffin pattern (arXiv:2403.19887 /
  arXiv:2402.19427). Slot creation is driven by `HybridLayer` on first use per `logical_layer_idx`.

### 7.7 Eval-only mixers (`fasa_attn`, `sparge_attn`)

- They are **decode-time selectors** by design: they read a full K/V cache and skip computation.
  They are first-class citizens of the cached path (F1), still gated by
  `TRAINING_FREE_LAYERS` for training (unchanged).

### 7.8 Rollback (future speculative decoding / early exit)

- `FrankensteinCache.truncate(num_tokens)` drops positions beyond a token boundary across all slots
  (cheap: list truncation + length bookkeeping; `StateSlot`s are no-ops). Required by
  LayerSkip-style verification (arXiv:2404.16710). Implement the API in F0, exercise it in tests.

---

## 8. Positional-encoding changes

All in `src/model/attention/common.py` (single choke point) + the PE modules. Defaults preserve
current behavior exactly.

### 8.1 Rotary family (`rope`, `hope`, `sinusoidal_rotary`)

Current: `pos = torch.arange(seq_len) * scaling` (`rope.py:79`) — always starts at 0.

New:

```python
def forward(self, x, logical_layer_idx=0, position_offset: Optional[int | torch.Tensor] = None):
    seq_len = x.shape[1]
    start = 0 if position_offset is None else int(position_offset)
    pos = torch.arange(start, start + seq_len, device=x.device) * self.scaling
    ...  # unchanged rotation math
```

Keys are cached **post-RoPE** (industry standard; Mistral-style buffers). The slot stores the
`rope_scaling` version (SlotMeta): if the schedule changes at runtime (YaRN-style, arXiv:2309.00071),
already-rotated keys are invalid — the manager detects the mismatch and raises a *configuration*
error at generation start (never mid-decode).

### 8.2 Additive score biases (`alibi`, `bam`)

Current: `bias(seq_len, device, dtype)` → full `(1, H, S, S)` from position 0 (`alibi.py:84-110`).

New: `bias(q_len, kv_len, q_offset, device, dtype) -> (1, H, q_len, kv_len)` with

```
bias[h, i, j] = −slope_h · (q_offset + i − j)      for j ≤ q_offset + i   (causal)
```

Current full-square builder is the special case `q_offset = 0, q_len = kv_len = S`. ALiBi/BAM carry
**no positional tensor in the cache** (their virtue: arXiv:2108.12409); in cached decode, `j` ranges
over `kv_positions` metadata, so eviction/evicted gaps produce correct relative distances.

### 8.3 `ssmax`

Score rescale `scores * s · ln(S)` (`common.py:317-322`) depends on total length: use
`ln(kv_len)` with `kv_len = cached + new`. Thread `kv_len` through `apply_pe_to_scores`.

### 8.4 Absolute PEs (`sinusoidal_absolute`, `learned_absolute`)

Added once to embeddings via `pos_encoder.add(x)` (`frankenstein_encoder.py:186-187`). In cached
mode, call with `position_offset` so the new token gets its absolute index. No cache interaction
otherwise.

### 8.5 PaPE family (`pape`, `pape_efficient`, `pape_ri`)

Content-dependent q/k augmentation (`encode_qk(hidden, q, k, positions)`, `common.py:275`).
**Decision (D11)**: mark PaPE-configured layers unsupported in `CACHE_SPECS` (fallback to uncached
execution) until a formal verification proves caching augmented keys is exact. This is per-layer,
driven by the per-mixer `use_pe` flat fields (`model/config.py:488-531`).

### 8.6 Plumbing point

`apply_pe_to_qk(..., position_offset=None)` and `apply_pe_to_scores(..., q_len, kv_len, q_offset)`
(`common.py:236-278` / `:281-324`) — mixers pass through; 39 mixers need **no PE-specific edits**.

---

## 9. KV-cache quantization

### 9.1 Interface and registration

`CacheCodec` (§5.2). Registry in `src/model/cache/quant/__init__.py`:

```python
CACHE_CODECS = {"none": NoneCodec, "int8": Int8Codec, "int4_kivi": Kivi4Codec}
# future (interface-ready, NOT in v1): "int2_kivi", "palu_lowrank", "commvq", "qjl_sketch", "block_gtq"
def build_cache_codec(name: str, params) -> CacheCodec
```

Codecs are **orthogonal to mixers**: the same codec serves `KVSlot`, `LatentKVSlot` (latent +
`k_rope` quantized separately: latent per-token, rope-key per-channel), and (in the future)
compressed slots. **Recurrent states are not quantized in v1** (D12 — the literature is thinnest
there; fallback to fp is automatic).

### 9.2 v1 codecs

**`none`** — fp16/bf16 passthrough (dtype from SlotMeta). Baseline for exactness tests.

**`int8`** — symmetric, asymmetric axes (KIVI, arXiv:2402.02750):

```
K (axis="channel"):  per (head, channel, token-group g):
    s_{h,c,g} = max_t∈g |K[t,h,c]| / 127 ,   K̂ = round(K / s) ∈ [−127, 127]
V (axis="token"):    per (head, token):
    s_{t,h}   = max_c |V[t,h,c]| / 127 ,     V̂ = round(V / s) ∈ [−127, 127]
Dequant: K ≈ K̂ ⊙ s , V ≈ V̂ ⊙ s
```

**`int4_kivi`** — group-wise int4 with protected window (KIVI + StreamingLLM, arXiv:2309.17453;
SKVQ two-region pattern, arXiv:2405.06219):

- Signed 4-bit levels `[-7, 7]`; `group_size` tokens per group along T (default 64);
  granularity axes identical to int8 (K per-channel-per-group, V per-token).
- **Protected region**: the most recent `protected_window` tokens plus `num_sinks` leading tokens
  are kept **full precision** and are never quantized, never evicted (KVQuant dense-and-sparse
  outliers, arXiv:2401.18079; KVSink, arXiv:2508.04257). Reading concatenates
  `[quantized history | fp protected window]`.
- **Append semantics**: a token-by-token decode appends rows to the fp protected window; when it
  exceeds `protected_window`, the oldest chunk that completes a group is quantized and moved to the
  history blocks (this mirrors KIVI's residual window and avoids per-token scale recomputation).
  Prefill quantizes whole groups immediately.

### 9.3 Where hooks live (literature-mandated layering)

- Quantization/normalization/rotation = **write-time** transforms → slot `append`.
- Selection/eviction = **read-time** transforms → policy + mixer read path (Quest-style page
  metadata, arXiv:2406.10774, is future work in `extras`).
- The codec must declare PE compatibility: post-RoPE storage (our default) works with per-channel
  K; CommVQ-style RoPE-commutative codebooks (arXiv:2506.18879) or pre-RoPE storage (KVQuant) are
  future codecs that would change `SlotMeta` semantics — the interface reserves `SlotMeta.pe_type`
  for this.

### 9.4 Budgets are per-layer vectors

Attention mass is pyramidal in depth (PyramidKV, arXiv:2406.02069); bit allocations are per-layer
and K≠V (KVTuner, arXiv:2502.15075; KVQuant sensitivity-weighted). Therefore:
`compression.per_layer_budgets: list[int]` (fallback to scalar `budget`), and
`quantization.k_bits`/`v_bits` are separate. `CachePolicy.budget(logical_layer_idx)` consumes the
vector.

### 9.5 Future codec plugins (design notes only — do not implement)

| Plugin | Paper | Sketch |
|---|---|---|
| `int2_kivi` | KIVI extensions / KVarN, arXiv:2606.03458 | 2-bit + dual-axis variance normalization (Hadamard) at write |
| `palu_lowrank` | Palu, arXiv:2407.21118 | rank-r projection at write, reconstruct at read; per-layer rank budget |
| `commvq` | CommVQ, arXiv:2506.18879 | RoPE-commutative codebooks; attention in codebook space without dequant |
| `qjl_sketch` | QJL, arXiv:2406.03482 | 1-bit JL signs of keys; unbiased inner-product estimator; no reconstructable K |
| `block_gtq` | Block-GTQ, arXiv:2606.24033 | bits allocated per RoPE frequency block by energy |

### 9.6 Rules adopted from the literature (checklist for reviewers)

1. Asymmetric granularity K vs V (KIVI; confirmed by KVTuner, AsymKV-COLING'25).
2. Keys are the hard tensor because of RoPE outlier structure → per-channel K, protected outliers.
3. Query-agnostic storage + (future) query-aware reading; entries carry importance metadata only
   where cheap (VestigeKV shows query-agnostic eviction failures on long-lived caches).
4. Protected slot classes: sinks + recent window stay fp and un-evicted.
5. Quantize-on-write, select-on-read, composable with eviction and cross-layer merging
   (MiniCache × quant, arXiv:2405.14366; EVICPRESS, arXiv:2512.14946).

---

## 10. KV-cache compression / eviction

`CachePolicy` (§5.2), registry in `src/model/cache/compress/__init__.py`.

| Phase | Policy | Basis | Behavior |
|---|---|---|---|
| F4 | `none` | — | Keep everything (default) |
| F4 | `window_sinks` | StreamingLLM arXiv:2309.17453; "Sliding Window Beats Linear Attention" arXiv:2608.28444 | Pin `num_sinks` leading positions; keep last `window` positions; drop the rest. Requires slots to honor `positions` metadata for correct relative distances afterwards |
| F4+ | `h2o` | H2O arXiv:2306.14048 | Accumulate per-token attention mass in slot `extras` during decode; evict lowest mass beyond budget; sinks+window protected |
| F4+ | `snapkv_prefill` | SnapKV arXiv:2404.14469 | One-shot compression of prompt KV after prefill using the look-ahead window's attention statistics |
| F4+ | `pyramid` | PyramidKV arXiv:2406.02069; CAKE arXiv:2503.12491; D2O arXiv:2406.13035 | Static per-layer budgets (deeper layers get less), driven by `per_layer_budgets` |
| F4+ | `sparsek` | SparseK arXiv:2406.16747 | Learned scorer + top-k retention; permanent eviction within a generation; only applies to layers whose mixer is `sparsek_attn` |

Constraints:

- Policies apply **only** to `KVSlot`/`RingSlot`/`LatentKVSlot`. `StateSlot`s are constant-size —
  nothing to evict.
- MTLA manages its own merging schedule internally; external policies must not run on `mtla_attn`
  slots (encode as `CacheSpec.extras` note + runtime check).
- DuoAttention-style per-(layer, head) retrieval/streaming head splits (arXiv:2410.10819) are a
  documented future refinement of `budget()` (head axis), not v1.

---

## 11. Configuration design

### 11.1 New schema module: `src/schema/_inference.yaml`

Top-level section `inference:` in `src/schema.yaml` (optional; `training` remains the only required
section). Full draft (bilingual `description`/`description_es` + `title`/`title_es` required on
every property — abbreviated here for space):

```yaml
$schema: "http://json-schema.org/draft-07/schema#"
title: Inference-time configuration (generation and KV cache)
title_es: Configuración de inferencia (generación y caché KV)
type: object
additionalProperties: false
properties:
  generation:
    type: object
    additionalProperties: false
    properties:
      max_new_tokens: { type: integer, minimum: 1, default: 64 }
      temperature:    { type: number,  minimum: 0, default: 1.0 }
      top_k:          { type: integer, minimum: 0, default: 50 }        # 0 disables top-k
      top_p:          { type: number,  minimum: 0, maximum: 1, default: 1.0 }
      greedy:         { type: boolean, default: false }                  # overrides sampling
  kv_cache:
    type: object
    additionalProperties: false
    properties:
      enabled:        { type: boolean, default: false }                  # D4: opt-in
      quantization:
        type: object
        additionalProperties: false
        properties:
          method:          { enum: [none, int8, int4_kivi], default: none }   # future: int2_kivi, palu_lowrank, commvq, qjl_sketch, block_gtq
          k_bits:          { type: integer, enum: [2, 4, 8], default: 4 }     # int4_kivi only
          v_bits:          { type: integer, enum: [2, 4, 8], default: 4 }     # int4_kivi only
          group_size:      { type: integer, minimum: 16, default: 64 }
          protected_window:{ type: integer, minimum: 0, default: 32 }
      compression:
        type: object
        additionalProperties: false
        properties:
          method:        { enum: [none, window_sinks], default: none }      # future: h2o, snapkv_prefill, pyramid, sparsek
          num_sinks:     { type: integer, minimum: 0, default: 4 }
          window:        { type: integer, minimum: 1, default: 512 }
          budget:        { type: ["integer", "null"], default: null }       # tokens/layer, null = unlimited
          per_layer_budgets: { type: array, items: { type: integer, minimum: 1 }, default: [] }
      per_mixer:
        type: array
        items:
          type: object
          additionalProperties: false
          properties:
            mixer:   { type: string }        # must be a layer_pattern enum value
            enabled: { type: boolean, default: true }
          required: [mixer]
```

Conditional rules (add to `src/schema/_conditional_rules.yaml` + mirror in `config_loader.py`):
- `quantization.method == "int8"` ⇒ ignore `k_bits/v_bits` (fixed 8);
  `== "int4_kivi"` ⇒ uses them; `== "none"` ⇒ all quant sub-keys ignored.
- `compression.method == "window_sinks"` ⇒ `window ≥ protected_window` (else schema error).
- `compression.method != "none"` ⇒ `kv_cache.enabled == true`.
- `kv_cache.enabled == true` and any `layer_pattern` entry with `supported == false` ⇒ **warning**
  from `config_loader` naming the mixers (not an error — D5).

### 11.2 Dataclass: `src/deploy/inference_config.py`

```python
@dataclass
class InferenceConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    greedy: bool = False
    kv_cache_enabled: bool = False
    kv_quant_method: str = "none"
    kv_k_bits: int = 4
    kv_v_bits: int = 4
    kv_group_size: int = 64
    kv_protected_window: int = 32
    kv_compress_method: str = "none"
    kv_num_sinks: int = 4
    kv_window: int = 512
    kv_budget: Optional[int] = None
    kv_per_layer_budgets: List[int] = field(default_factory=list)
    kv_per_mixer: List[Dict[str, Any]] = field(default_factory=list)
```

Note the flat fields: `inference:` is runtime configuration, deliberately **not** part of
`FrankensteinModelConfig` (architectural config). This follows the repo's separation
(`training.*` is runtime too).

### 11.3 Loading and flattening

- `src/utils/config_flatten.py`: add `flatten_inference_dict(inference_data) -> dict` (nested
  `inference:` tree → flat `InferenceConfig` kwargs; pass-through for legacy absent section).
- `src/training/config_loader.py`: `load_inference_config(path_or_dict) -> InferenceConfig`
  (validates the `inference` subtree against `_inference.yaml`, applies conditional rules, emits
  cache-support warnings by consulting `CACHE_SPECS` against `layer_pattern` and the per-mixer
  `use_pe` flags).
- Default-parity test analogous to `_DEFAULT_PAIRS` (`tests/test_schema_consistency.py:99`): add
  `INFERENCE_DEFAULT_PAIRS` mapping every schema default above to the dataclass default; the test
  asserts equality (same mechanism as the model pairs, `:140+`).

### 11.4 CLI flags and override precedence (D2)

`src/cli.py` `infer` gains a generation mode:

```
frankenstein-transformer infer --model <dir> --task causal_lm
    --prompt "..."            [--input file]  [--output file]
    --max-new-tokens N        --temperature T --top-k K --top-p P --greedy
    --kv-cache | --no-kv-cache
    --kv-quant {none,int8,int4_kivi}
    --kv-group-size N         --kv-protected-window N
    --kv-compress {none,window_sinks} --kv-num-sinks N --kv-window N
```

Precedence: **CLI > YAML `inference:` > schema default.** Mechanism: every flag uses
`argparse` `default=None`; `apply_cli_overrides(cfg: InferenceConfig, args) -> InferenceConfig`
sets only the keys explicitly passed (leaf-level, shallow merge). A YAML preset delivered via
`--config` is fully loaded first; flags then patch it. `--task causal_lm` requires the deployed
model to be a decoder (`config.mode == "decoder"`), else a clear CLI error (mirrors the forced-mode
rule for training).

### 11.5 Deploy artifacts

`ModelDeployer` serializes the **resolved** `InferenceConfig` into `config.json`
(`"inference": {...}`); `FrankensteinInference` reads it back. Missing section ⇒ all defaults
(opt-in not active ⇒ legacy behavior). This makes cached inference reproducible without the
original YAML.

---

## 12. File-by-file change plan

Order follows the repo's extension recipes (`docs/architecture.md` §11, esp. §11.8 for config
fields). "NEW" = create.

| # | File | Change | Phase |
|---|---|---|---|
| 1 | `src/model/cache/base.py` | NEW — protocols + `SlotMeta` (§5.2) | F0 |
| 2 | `src/model/cache/slots.py` | NEW — `KVSlot`, `LatentKVSlot`, `StateSlot`, `RingSlot`, `ConvTailSlot`, `TokenTailSlot` | F0 (KV/Latent/Ring/TokenTail), F2 (State, ConvTail) |
| 3 | `src/model/cache/capabilities.py` | NEW — `CacheSpec` + `CACHE_SPECS` for all 44 enum values (contents = §6 matrix) | F0, extended each phase |
| 4 | `src/model/cache/manager.py` | NEW — `FrankensteinCache` (§5.4, §5.6): lazy slot creation, `next_position()`, `advance`, `truncate`, `token_tail`, `to(device)`, `force_dense_prefill` | F0 |
| 5 | `src/model/cache/hooks.py` | NEW — shared mixer helpers: `slot_append_read()`, `build_mask(q_len, kv_len, q_offset, kv_positions)`, offset plumbing shims | F0 |
| 6 | `src/model/attention/common.py` | `apply_pe_to_qk(..., position_offset=None)`; `apply_pe_to_scores(..., q_len=None, kv_len=None, q_offset=None)`; ssmax uses `kv_len` | F0 |
| 7 | `src/model/embeddings/rope.py`, `hope.py`, `sinusoidal_rotary` module | `position_offset` parameter (§8.1) | F0 |
| 8 | `src/model/embeddings/alibi.py`, `bam` module | `bias(q_len, kv_len, q_offset)` (§8.2) | F0 |
| 9 | `src/model/attention/standard.py`, `grouped_query_attention.py` | optional `cache, cache_position` kwargs; `CACHE_SPEC`; append/read via hooks | F0 |
| 10 | `src/model/hybrid_layer.py` | thread `cache/cache_position` through all 4 call shapes; MoD dense-force at prefill when caching; slot resolution; mHC/MoD interlocks untouched | F0, extended F1/F2 |
| 11 | `src/model/frankenstein_encoder.py` | `forward(..., past_key_values=None, use_cache=False, cache_position=None, return_cache=False)`; absolute-PE offset; AttnRes incremental hook (F2) | F0, F2 |
| 12 | `src/model/frankenstein_decoder.py` | mirror forward; rewritten `generate()` (§5.5) with sampling (greedy/top-k/top-p) | F0 |
| 13 | F1 mixer files (18): `sigmoid.py`, `gated/gated_softmax_attn.py`, `titan.py`, `gated/fox_attn.py`, `latent/*.py` (mla, gqla, mlra, tucker, iha, gta, mtla), `sparse/*.py` (sparse_transformer, longformer, swa, bigbird, msa, sparge, fasa) | cached forward branches + pattern builders with `(q_len, kv_len, q_offset, kv_positions)` + BigBird seeding + FOX/MTLA slot extras | F1 |
| 14 | F2 mixer files (21): `ode.py`, `retnet.py`, `gated/retnet_attn.py`, `gated/{gla,deltanet,gated_deltanet,gated_deltanet2,hgrn2,kda}_attn.py`, `gated/falcon_attn.py`, `latent/{gma,cca}_attn.py`, `sparse/nsa_attn.py`, `sparse/sparda_attn.py`, `engram.py` | state slots, conv tails, token tail, NSA 3-branch, accumulators | F2 |
| 15 | `src/model/residuals/base.py`, `full_attn_res.py`, `block_attn_res.py` | incremental accumulation mode (§7.4) | F2 |
| 16 | `src/schema/_inference.yaml` | NEW (§11.1) | F1 |
| 17 | `src/schema.yaml` | `$ref: schema/_inference.yaml`; `inference` optional | F1 |
| 18 | `src/schema/_conditional_rules.yaml` | inference rules (§11.1) | F1 |
| 19 | `src/utils/config_flatten.py` | `flatten_inference_dict` | F1 |
| 20 | `src/deploy/inference_config.py` | NEW — `InferenceConfig` + `apply_cli_overrides` (§11.2, §11.4) | F1 |
| 21 | `src/training/config_loader.py` | `load_inference_config` + warning pass (§11.3) | F1 |
| 22 | `src/deploy/inference.py` | generation API (`generate(prompt, ...)`), cache wiring, benchmark split prefill/decode (tok/s each) | F1 |
| 23 | `src/cli.py` | flags (§11.4), `--task causal_lm` routing, decoder requirement | F1 |
| 24 | `src/model/cache/quant/{__init__,none,int8,kivi4}.py` | NEW (§9) | F3 |
| 25 | `src/model/cache/compress/{__init__,none,window_sinks}.py` | NEW (§10) | F4 |
| 26 | `src/deploy/transformers_export.py` | real `past_key_values`/`use_cache`/`cache_position` in `FrankensteinForCausalLM` (B7) | F5 |
| 27 | `src/deploy/bitnet_gguf_export.py` | `--check` updated for cached hybrids | F5 |
| 28 | `src/deploy/deploy.py` | serialize resolved `InferenceConfig` into `config.json` (§11.5) | F1 |
| 29 | `configs/examples/kv_cache_generate.yaml` | NEW example preset (smoke-tested) | F1 |
| 30 | `configs/README.md` | inference section | F1 |
| 31 | `tests/*` | see §13 | each phase |
| 32 | `docs/architecture.md` | new §7.x "KV cache & incremental decoding" (matrix summary, slot table, config), §8.3 CLI update, §10 deploy update, §14 limitations (PaPE, ssog, ragged-MoD) | F5 |
| 33 | `docs/bibliography/*.bib` | entries for cited papers (§17) | F5 |
| 34 | `AGENTS.md` | note in "Where things live" (`src/model/cache/`) + gate list additions | F5 |

---

## 13. Testing plan

All tests CPU-only (repo rule). Naming follows existing conventions (`tests/test_*.py`).

| ID | Test file | Covers | Key assertions |
|---|---|---|---|
| T1 | `tests/test_kv_cache_slots.py` | all slot types | append/read/truncate/to(device); ring wrap-around with `positions` metadata; `StateSlot` in-place update; conv-tail FIFO order; token-tail contents |
| T2 | `tests/test_kv_cache_codecs.py` | `none`, `int8`, `int4_kivi` | round-trip error bounds on synthetic tensors: int8 max rel-err ≤ 1e-2, int4 ≤ 5e-2 (per-axis); protected window stays exactly fp; group-boundary handling; codec `update()` equals dequant→concat→requant |
| T3 | `tests/test_kv_cache_mixers.py` | **parametrized over every cacheable mixer** | `full_forward(x)` vs `prefill(x[:, :t]) + t-step decode` → `torch.allclose(..., rtol=1e-5, atol=1e-6)` in fp32 with `quantization: none`, BitNet off. Mask/pattern parity included. Per-mixer overrides for known-float-sensitive paths (fox log-sums: fp64 accumulation internally) |
| T4 | `tests/test_kv_cache_fallback.py` | **all 44 enum values** with `kv_cache.enabled=true` | no exception; unsupported mixers produce identical logits with cache on/off; PaPE layer falls back per-layer while other layers cache |
| T5 | `tests/test_kv_cache_mechanisms.py` | `num_loops>1`, Hyperloop on, mHC on, MoD on, AttnRes on (F2), heterogeneous `layer_pattern` | equivalence full vs cached for each; Hyperloop slots distinct per `logical_layer_idx`; MoD dense-forced prefill documented divergence check (prefill outputs equal full-forward outputs when routing disabled) |
| T6 | `tests/test_kv_cache_eval_only.py` | `sparge_attn`, `fasa_attn` | cached decode works; training-mode guard still raises |
| T7 | `tests/test_inference_generate.py` | `generate()` | greedy parity cache vs no-cache (exact); sampling determinism with fixed seed; `truncate()` rollback correctness |
| T8 | `tests/test_kv_cache_quant_e2e.py` | F3 end-to-end | int8: greedy decode within small logit drift vs fp (fixed prompts, threshold asserted); int4_kivi: drift bounded, protected window exact; runs with BitNet on and off |
| T9 | `tests/test_inference_config.py` | config resolution | YAML loading, `flatten_inference_dict`, `INFERENCE_DEFAULT_PAIRS` parity, CLI>YAML>default precedence, conditional rules (window ≥ protected_window; compression⇒enabled), warning emission for unsupported mixers |
| T10 | `tests/test_cli_parser.py` (extend) | new flags parse; defaults None (no override) | |
| T11 | `tests/test_transformers_export.py` (extend) | F5 | exported `FrankensteinForCausalLM.generate` (HF) consumes cache; outputs match local cached decode |
| T12 | gates | `test_schema_consistency.py` extended with inference pairs; `test_yaml_examples.py` auto-picks `kv_cache_generate.yaml`; `test_schema_attention_layers.py` extended to assert `CACHE_SPECS` ↔ enum ↔ registry parity |

Existing decoder test (`test_model_variants.py`) must keep passing **unchanged** (D4 guarantees
default behavior is bit-identical).

---

## 14. Implementation phases and acceptance criteria

| Phase | Scope | Acceptance criteria (all CPU) |
|---|---|---|
| **F0 Foundations** | Items 1–12 of §12 (cache package core, PE offsets, decoder plumbing, `standard_attn`+`gqa_attn` cached, `generate()` opt-in, benchmark) | T1, T3 (2 mixers), T4 (44, fallback), T5 (loops/mHC/MoD), T7 green; existing suite green; `generate(use_cache=False)` byte-identical to legacy |
| **F1 KV/latent/window** | 18 mixers + NSA window-branch + SparDA basic; schema `_inference.yaml` + `InferenceConfig` + flatten + CLI + deploy serialization + example | T3 (F1 mixers), T4, T6, T9, T10 green; `kv_cache_generate.yaml` smoke-tested; CLI round-trip demo |
| **F2 State & aux** | 21 mixers (StateSlot, ConvTailSlot, accumulators, token tail), AttnRes incremental, MoD prefill policy, NSA full | T3 (F2 mixers), T5 (AttnRes), T4 green |
| **F3 Quantization** | `none`/`int8`/`int4_kivi` + budgets | T2, T8 green |
| **F4 Compression** | `window_sinks` (+`h2o`, `snapkv_prefill`, `pyramid`, `sparsek`-as-policy if time allows; else stubs behind enum with `NotImplementedPolicyError`→fallback) | compression tests (extend T2/T8): evicted caches keep correctness on retained region; sinks exact |
| **F5 Ecosystem** | HF export, GGUF check, docs, bibliography, AGENTS.md | T11 green; `docs/architecture.md` published; full suite green |

Suggested PR granularity: one PR per phase (F1 may split into dense/latent/sparse sub-PRs).

---

## 15. Risks and mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | MTLA train/infer merge-schedule parity (stride-aware mask assumes cache shorter than input) | High for `mtla_attn` | Equivalence tests gate the mixer's `supported` flag; on failure ⇒ unsupported + fallback (D5). No silent wrongness |
| R2 | BigBird random links not reproducible across incremental steps | Medium | Seed per `(logical_layer_idx, generation)` at slot creation; store selected blocks as extras; test masks match full path |
| R3 | PaPE cache correctness unproven (content-dependent q/k augmentation) | Medium | D11: unsupported until verified; per-layer fallback; §14 F5 documents the open question |
| R4 | NSA three-branch complexity | Medium | Phased (window branch F1, full F2); each branch tested independently |
| R5 | Numerical drift accumulation over long decodes (quant + BitNet activations) | Medium | T8 drift thresholds; protected window exact; literature: normalize-at-write (KVarN arXiv:2606.03458, OScaR arXiv:2605.19660) listed as future codec hooks |
| R6 | Hyperloop/mHC n-stream state assumed per-token — if a future variant introduces cross-step carry, cached decode silently breaks | Low | T5 asserts equivalence per mechanism; comment at slot-resolution point |
| R7 | MoD forced-dense prefill diverges from training routing statistics | Low | Documented (D9); decode keeps router; ragged cache (VestigeKV, arXiv:2609.03949) is future work |
| R8 | Signature changes break external consumers (HF export, Streamlit GUI) | Low | All kwargs optional with `None` defaults (Appendix B); T11; GUI unaffected (builder-only) |
| R9 | `retnet`/`ode` call shape S2 (mixer replaces x) complicates cache threading | Low | Cache threaded only when mixer's spec supports it; else None (matches current shapes) |
| R10 | Falcon read-after-write prefix shift off-by-one | Medium | Dedicated parity test for all 6 variants incl. sentinels at step 0 (T3) |

---

## 16. Decision log

| # | Decision | Rationale |
|---|---|---|
| D1 | Central `FrankensteinCache` + typed slots + declarative `CacheSpec` (not per-mixer internal caches) | Layer identity under loops/Hyperloop/MoD; orthogonality of quantization/compression; inspectability; uniform fallback. Matches HF/vLLM/Jamba practice |
| D2 | Config: schema `_inference.yaml` as source of truth + CLI leaf-level overrides (CLI > YAML > default) | User decision; keeps schema-first validation while allowing runtime patching of presets |
| D3 | v1 codecs: `none`, `int8`, `int4_kivi` | User decision; validates the codec interface with one lossless + two real lossy implementations at different operating points |
| D4 | Opt-in activation (`kv_cache.enabled: false` default); legacy path bit-identical | User decision; zero risk to presets/tests |
| D5 | Unsupported mixer ⇒ silent per-mixer fallback; `config_loader` warns; never raises | User requirement: "inference must not fail — that mixer simply isn't computed with cache" |
| D6 | Slot key = `logical_layer_idx` | Correct under `num_loops`/Hyperloop (same physical layer, distinct logical roles) |
| D7 | K/V stored **post-RoPE**; `SlotMeta.rope_scaling` versioned | Industry standard; ring re-rotation unnecessary; PE-change invalidation handled at config level |
| D8 | MoD prefill forced dense when caching | Ragged caches out of scope for v1; divergence documented and tested |
| D9 | Recurrent states not quantized in v1 | Thinnest literature; automatic fp fallback |
| D10 | PaPE-configured layers unsupported (per-layer fallback) until verified | Content-dependent PE augmentation; correctness unproven |
| D11 | `_inference.yaml` optional; `training` remains the only required section | Backwards compatibility of every existing preset |
| D12 | `generate()` keeps the legacy loop when `use_cache=False` | Guarantees byte-identical legacy behavior |
| D13 | Compression policies never run on `mtla_attn` slots | MTLA has its own dynamic merging schedule |
| D14 | `sparsek_attn` implemented as a `CachePolicy`, not a slot | Its paper is an eviction policy with permanent within-generation eviction |

---

## 17. Bibliography

All IDs verified against arXiv/alphaXiv (2026-10). Grouped by role in this plan.

### Surveys & systems
- Survey on LLM Acceleration based on KV Cache Management — arXiv:2412.19442
- System-Aware KV Cache Optimization Survey — arXiv:2607.08057
- KV Cache Management for LLM Serving Survey — arXiv:2607.02574
- PagedAttention / vLLM — arXiv:2309.06180 · SGLang/RadixAttention — arXiv:2312.07104 · Hydragen — arXiv:2402.05099 · CacheGen — arXiv:2310.07240 · Sarathi-Serve — arXiv:2403.02310 · Mooncake — arXiv:2407.00079
- Cross-layer sharing: YOCO — arXiv:2405.05254 · CLA — arXiv:2405.12981 · CacheBlend — arXiv:2405.16444

### Positional encoding & cache correctness
- RoPE — arXiv:2104.09864 · ALiBi — arXiv:2108.12409 · MQA — arXiv:1911.02150 · GQA — arXiv:2305.13245
- YaRN — arXiv:2309.00071 · LM-Infinite — arXiv:2308.16137 · Block-GTQ (RoPE-aware bit allocation) — arXiv:2606.24033 · VestigeKV — arXiv:2609.03949

### Quantization
- KIVI — arXiv:2402.02750 · KVQuant — arXiv:2401.18079 · GEAR — arXiv:2403.05527 · Palu — arXiv:2407.21118 · SKVQ — arXiv:2405.06219 · MiKV — arXiv:2402.18096 · QJL — arXiv:2406.03482 · CommVQ — arXiv:2506.18879 · QuaRot — arXiv:2404.00456 · SpinQuant — arXiv:2405.16406 · KVTuner — arXiv:2502.15075 · KVarN — arXiv:2606.03458 · OScaR — arXiv:2605.19660 · TaSQ — arXiv:2610.03027 · KVSink — arXiv:2508.04257

### Compression / eviction / merging
- H2O — arXiv:2306.14048 · Scissorhands — arXiv:2305.17118 · StreamingLLM — arXiv:2309.17453 · FastGen — arXiv:2310.01801 · SnapKV — arXiv:2404.14469 · PyramidKV — arXiv:2406.02069 · Quest — arXiv:2406.10774 · DuoAttention — arXiv:2410.10819 · CAKE — arXiv:2503.12491 · D2O — arXiv:2406.13035 · MiniCache — arXiv:2405.14366 · Look-M — arXiv:2406.18139 · SeKV — arXiv:2606.31145 · CompressKV — arXiv:2606.24467 · EVICPRESS — arXiv:2512.14946

### Non-standard / hybrid state caches
- Mistral 7B (rolling SWA buffer) — arXiv:2310.06825 · Jamba — arXiv:2403.19887 · Zamba — arXiv:2405.16712 · Griffin — arXiv:2402.19427 · Mamba — arXiv:2312.00752 · Mamba-2 — arXiv:2405.21060 · RetNet — arXiv:2307.08621 · GLA — arXiv:2312.06635 · Lightning Attention-2 — arXiv:2401.04658 · BitNet b1.58 — arXiv:2402.17764

### Token routing / skipping
- Mixture-of-Depths — arXiv:2404.02258 · LayerSkip — arXiv:2404.16710 · Mixtral — arXiv:2401.04088 · DeepSeek-V3 — arXiv:2412.19437

### Mixer papers (per §6)
- Dense: 1706.03762 (MHA) · 2305.13245 (GQA) · 2409.04431 (Sigmoid SA) · 2505.06708 (Gated Attention) · 2503.02130 (FoX) · 2501.00663 (Titans) · 2203.09176 (ODE Transformer)
- Latent: 2506.09342 (Latent MHA small LMs) / 2405.04434 (MLA, DeepSeek-V2) · 2605.15250 (GQLA) · 2603.02188 (MLRA) · 2603.30033 (Tucker) · 2602.21371 (IHA) · 2506.17286 (GTA) · 2505.13544 (MTLA) · 2510.04476 (CCA/CCGQA) · 2606.18283 (GMA)
- Sparse: 1904.10509 · 2004.05150 (Longformer) · 2608.28444 (+2309.17453) (SWA+sinks) · 2007.14062 (BigBird) · 2406.16747 (SparseK) · 2502.11089 (NSA) · 2606.13392 (MiniMax SA) · 2606.04511 (SparDA) · 2502.18137 (SpargeAttn) · 2602.03152 (FASA)
- Recurrent/linear: 2307.08621 (RetNet) · 2312.06635 (GLA) · 2406.06484 (DeltaNet) · 2412.06464 (Gated DeltaNet) · 2605.22791 (GDN-2) · 2404.07904 (HGRN2) · 2510.26692 (KDA/Kimi Linear) · 2608.27763 (Falcon)
- Other: 2601.07372 (Engram) · SSOG (pisoni.ai blog + GitHub; no arXiv)

---

## Appendix A — current code behavior reference

Abridged, verified descriptions (line refs valid 2026-10-26):

- **Dispatch shapes** (`hybrid_layer.py:365-372`): `mamba` → `x + mixer(x)`; `ode`/`retnet` → `x = mixer(x)`;
  `engram_attn` → `mixer(x, input_ids=..., logical_layer_idx=..., pos_encoder=...)`; else →
  `mixer(x, logical_layer_idx=..., pos_encoder=...)`. Same in `_forward_dense_mhc` (`:452-464`).
- **Looping** (`frankenstein_encoder.py:264-294`): hyperloop path runs begin layers once, middle
  `num_loops` times; plain path `for _ in range(num_loops)`. `logical_layer_idx` increments across
  loops and is passed to every mixer (`:247`).
- **MoD** (`hybrid_layer.py:500-567`): `capacity = max(1, ceil(S * ratio))`; `topk` per batch over
  sequence; scatter-back; short-circuit dense when `capacity >= seq_len` (`:552-553`).
- **AttnRes reset** (`frankenstein_encoder.py:213-216`): `residual.reset_state()` + `set_embedding(x)`
  at every forward start; layer application at `:253-254`.
- **RoPE positions** (`rope.py:79`): `pos = torch.arange(seq_len) * scaling` — no offset support.
- **generate()** (`frankenstein_decoder.py:129-166`): full-sequence forward per token; temperature ×
  top-k multinomial; no greedy flag.
- **HF export** (`transformers_export.py:163-196`): `forward(input_ids, labels=None, **_)` — extra
  kwargs silently discarded.
- **Schema mixer enum** (`_dims.yaml:297-343`, 44 values) with EN/ES description tables at
  `:193-236`/`:247-290` — any new *layer type* would need all three + registry + tests; this plan
  adds **no new mixer keys**, so those tables are untouched.

## Appendix B — signature diffs (additive only)

```python
# Mixers (S4 shape; S3 adds input_ids already present; S1/S2 unchanged until F2):
- def forward(self, x, logical_layer_idx=None, pos_encoder=None): ...
+ def forward(self, x, logical_layer_idx=None, pos_encoder=None,
+             cache: Optional["CacheSlot"] = None,
+             cache_position: Optional[int] = None): ...

# HybridLayer / Encoder / Decoder:
+ past_key_values: Optional["FrankensteinCache"] = None,
+ use_cache: bool = False,
+ cache_position: Optional[int] = None,
+ return_cache: bool = False,      # encoder/decoder forward only

# PE modules:
+ position_offset: Optional[int] = None          # rope / hope / sinusoidal_rotary / pos_encoder.add
+ def bias(self, q_len, kv_len, q_offset=0, ...) # alibi / bam (replaces bias(seq_len, ...))

# Pattern builders (sparse family):
+ kv_len: Optional[int] = None, q_offset: int = 0, kv_positions: Optional[Tensor] = None
```

All defaults reproduce current behavior exactly; every existing call site, preset, and test is
unaffected until the cache is explicitly enabled (D4).

---

*End of plan.*
