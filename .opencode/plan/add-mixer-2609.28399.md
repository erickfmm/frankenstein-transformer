# Plan: add Memory Attention mixer (arXiv:2609.28399)

> Status: **plan only** — nothing outside `.opencode/plan/` has been created,
> edited or deleted by the planning command. Every change below is described,
> not applied.

## 1. Header

**Paper:** *Memory Attention*, Jiale Kang (Yuanshi Intelligence), arXiv:2609.28399v2
[cs.LG], 14 Oct 2026. Standard attention spends a dense `W_V` projection building
values that often encode content reusable across contexts. Memory Attention (MA)
**removes the value projection entirely** and constructs values by adding a
layer-specific, token-ID-indexed memory table to the contextual keys:
`V = K + Norm(E[s])`, with `E ∈ R^{N×d_v}` per layer and RMSNorm applied per
key/value head. Queries/keys keep the usual projections and RoPE applies only to
the scoring pair; the softmax attention weights are unchanged. The paper trains
MHA/GQA/MQA/gated backbones from scratch on FineWeb-10BT, reporting better
LAMBADA/WikiText perplexity and +0.6–1.2 avg downstream points at matched token
budgets (with N·d_v more parameters per layer), 1.42×/1.16× token efficiency,
and a **conventional KV cache** at inference (normalization folded into the
table, `E[i] ← Norm(E[i])`). It belongs in this repo because it is a trainable
from-scratch sequence mixer with a trivially cacheable dense core, a GQA-style
head structure (`d_v = H_KV·d_h`), and an exact parameter formula — everything
the schema-first registry is built to absorb.

## 2. Paper extract (verbatim)

### 2.1 Forward equations

Background (paper eqs. 1–2):

```
Q = XW_Q,  K = XW_K,  V = XW_V                                              (1)
Attn(X) = softmax( QK^T / sqrt(d_k) + C ) V                                 (2)
```

Memory Attention construction (paper §3.1, eqs. 7–9):

```
M = Norm(E[s]) ∈ R^{T×d_v}                                                  (7)
   where E ∈ R^{N×d_v} (N = vocabulary size, d_v = H_KV · d_h),
   E[s] retrieves the rows indexed by the input tokens, and
   "Norm applies RMSNorm independently to each token vector within each
    key/value head" (paper §3.1).

Q = XW_Q,   K = XW_K,   V = K + M                                           (8)

Q_R = RoPE(Q),   K_R = RoPE(K),   V = K + M                                 (9)
   "Values are constructed from K before this transformation, while Q_R and
    K_R are used to compute attention weights."
```

Decomposition (eqs. 10–12):

```
V = XW_K + Norm(E[s])                     (shared contextual projection
                                           + token-dependent bias)          (10)
O_MA = A(K + M) = A K + A M               (contextual + memory retrieval)  (11)
o^mem_t = Σ_{j≤t} a_{tj} m_{s_j}                                             (12)
```

Inference folding (eqs. 13–14):

```
E[i] = Norm(E[i]),  i = 1,…,N                                               (13)
V = K + E[s]                                                                (14)
```

Cost model (§3.3, Table 1, eqs. 15–16; §3.4 eqs. 17–18; §3.5 eqs. 19–20; §4.4 eq. 21):

```
Standard value projections:      L·d·d_v params          MA: L·(N·d_v + p_norm)
Standard value-constr. FLOPs:    ≈ 6·L·S·d·d_v            MA: O(L·S·d_v)
ΔF_V ≈ L·S·d_v·(2d − 1)                                                     (15)
η_model ≈ L·S·d_v·(2d−1) / (F_common + 2·L·S·d·d_v)                         (16)
P_offloaded = b_w·L·N·d_v  bytes;   D_Offload = b_w·L·S·d_v                  (17)(18)
MA-Recall:  V_{1:T} = K_{1:T} + E[s_{1:T}]                                  (19)
D_Recall = b_w·L·B·R·d_v                                                    (20)
ρ_token(ℓ) = n_Standard(ℓ) / n_MA(ℓ)                                        (21)
```

### 2.2 Algorithm forms (repo-shaped; the paper has no numbered algorithm float)

Parallel / training form (per layer, per head-group):

```
Input: hidden states X ∈ R^{B×T×H}, token IDs s ∈ Z^{B×T}
Q ← XW_Q                      # (B, nH·hd)
K ← XW_K                      # (B, nKv·hd)          — also the value basis
M ← RMSNorm_per_KVhead(E[s])   # (B, T, nKv·hd), γ gains, eps = 1e-6
V ← K + M                     # BEFORE RoPE
Q_R ← RoPE(Q); K_R ← RoPE(K)  # scoring pair only
A ← softmax(Q_R K_R^T / √hd + mask)      # causal (decoder) or full (encoder)
O ← W_O · (A · expand(V))     # expand = repeat_interleave(nH/nKv)
```

Recurrent / decode form (conventional KV cache, paper §4.4: "All three
configurations retain a conventional KV cache"):

```
prefill: compute (K_R, V) for T tokens, append both to the KVSlot
decode (1 new token t):
    k_t ← X_t W_K ; v_t ← k_t + RMSNorm(E[s_t]) ; k_t ← RoPE(k_t)
    append (k_t, v_t); scores over full cache; softmax; aggregate; W_O
```

### 2.3 Parameter inventory (per MA layer)

| Tensor | Shape | Bias | Notes |
|---|---|---|---|
| `q_proj` (`W_Q`) | `H × nH·hd` (= H×H) | none | proj_cls (BitLinear under BitNet) |
| `k_proj` (`W_K`) | `H × nKv·hd` (= H×d_v) | none | shared scoring/value basis (eq. 10) |
| `out_proj` (`W_O`) | `nH·hd × H` (= H×H) | none | |
| `memory` (`E`) | `N × nKv·hd` (= N×d_v) | — | `nn.Embedding(vocab_size, d_v)`; **replaces `W_V`** |
| RMSNorm gains (`γ`) | `nKv·hd` (= d_v) | — | per-KV-head, `p_norm` of Table 1; eps 1e-6 |

**Exact parameter-count formula** (with `hd = H/nH`, `d_v = nKv·hd`,
`N = vocab_size`, all projections bias-free, BitNet-invariant):

```
P_memory_attn = 2·H² + H·d_v + N·d_v + d_v
             = 2·H² + H·(nKv·hd) + N·(nKv·hd) + (nKv·hd)
```

`param_count_check.py` routes every one of these to `attention`
(`src/utils/param_categories.py` matches on the `mixer` path segment —
`layers.i.mixer.memory.weight` → `"mixer" in parts` → `attention`; a module
named `memory` is safe: no `emb/ffn/router/residual/mhc` segment collides).
The JS estimator must use the identical formula.

### 2.4 Hyperparameters / knobs

**None.** The paper defines no MA hyperparameter beyond the architecture:
`N` = `dims.vocab_size`, `H_KV` = `dims.num_kv_heads` (same consumption pattern
as `gqa_attn`, which the schema documents as reading `dims.num_kv_heads`),
`d_h` = `hidden_size / num_heads`, RMSNorm `ε = 1e-6` (paper Appendix A; fixed
constant, not a knob), RoPE base 10 000 (= `rope.base` default). Consequently:
**no** `_attention_memory.yaml`, **no** `_model.yaml` `$ref`, **no**
`_ATTENTION_MIXER_RENAMES` entry, **no** knob fields in
`FrankensteinModelConfig` (only `memory_attn_use_pe`, §3).

### 2.5 State & cache

- **Compatibility class (i)** — classic per-token KV attention → `KVSlot`.
- Cache stores post-RoPE `K_R` and the *constructed* `V` (built once at append
  time from the unrotated key + memory row). Per-step decode cost: one embedding
  row lookup + `d_v` adds + the standard attention over the cache. Nothing
  beyond `KVSlot` is needed; KV quantization codecs (`none/int8/int4_kivi`) apply
  unchanged.
- MA-Offload (CPU/SSD residency, eqs. 17–18) is a systems technique, out of
  scope. MA-Recall (eq. 19: store only `K` + token IDs, reconstruct `V` on the
  fly, ~50 % cache saving) is **analyzed but unmeasured** in the paper — out of
  scope, recorded as an open question.

### 2.6 Complexity

- Training: `O(n²·d)` attention (identical to `standard_attn`) with value
  construction `O(n·d_v)` lookup+norm+add replacing the `O(n·d·d_v)` matmul.
- Inference: `O(n)` per token with KV cache; table storage `O(N·d_v)` per layer
  (capacity, not compute).

### 2.7 Trainability

Fully trainable from scratch (paper trains all backbones from scratch with
AdamW). **Not** training-free → no `TRAINING_FREE_LAYERS` membership, no
`dense_forward` surrogate, no ⚠️ wording anywhere, and it **joins** the e2e
`ATTENTIONS` list.

### 2.8 Positional encoding

Consumes the model-wide PE on the scoring pair only (RoPE on Q/K, eq. 9); values
are built from the **unrotated** keys. `per_mixer_use_pe` default **true**
(softmax-attention-like). `memory_attn_use_pe: bool = True`.

### 2.9 Mode constraints

Decoder/causal is the paper's setting; encoder (bidirectional mask) works
identically to `standard_attn` — both modes supported. **Text-token tasks only**:
the memory lookup requires `input_ids`, which ViT patches and audio frames do not
have (engram precedent: no hard `config_loader` rule; document in the schema
descriptions and annex). MLM / SBERT / text_classification / causal_lm are all
valid.

### 2.10 BibTeX data

Author: Jiale Kang (kangjiale827@gmail.com; work conducted at Yuanshi
Intelligence). Title: *Memory Attention*. arXiv:2609.28399v2 [cs.LG],
submitted 14 Oct 2026. Key: `kang_memory_attention_2026` (pattern
`<firstauthor>_<shortname>_<year>` per `kimi_kda_2025`).

## 3. Decisions (Phase 3)

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `memory_attn` | Paper's short name "Memory Attention"; snake_case + `_attn` suffix; grep confirms no collision in `src/`, `configs/`, `tests/`, `docs/` |
| Class name | `MemoryAttention` | Peer convention (`StandardAttention`, `GroupedQueryAttention`) |
| Family | **New "Memory" family** (user decision) | Attention core is dense softmax but the value mechanism is lookup-based; kept separate from Dense and from engram's "Conditional memory" label (no reclassification of engram — guardrail) |
| Module location | Top level `src/model/attention/memory.py` | Directories are only `gated/ latent/ sparse/` + top level; dense/top-level peers (`standard.py`, `sigmoid.py`, `grouped_query_attention.py`) are top level |
| Family label everywhere | "Memory" / ES "Memoria" | Taxonomy gets a new branch; new `memory-map.mmd` family figure; new `ft-diagram.js` STYLES class (the extra cost of a new family, per user choice) |
| Closest peers | `standard.py` (core + cached-branch shape), `grouped_query_attention.py` (kv-heads handling, `repeat_interleave` groups, cached branch), `engram.py` (the only `input_ids` consumer) | `hybrid_layer.py` line 443 special-cases engram for `input_ids`; MA needs the same |
| Knobs | None (§2.4) | — |
| `use_pe` | `memory_attn_use_pe: bool = True`; schema `use_pe.memory_attn: default true` | RoPE paper, softmax core |
| KV-cache | **Decision 1 (user-confirmed): full classic KV** | Paper §4.4 "All three configurations retain a conventional KV cache"; class (i); incremental branch ≈ GQA's; `IMPLEMENTED_MIXERS` + parity |
| Example preset | `configs/examples/es_arch_memory_adamw.yaml` (encoder MLM, `es_arch_kda_adamw.yaml` shape minus machine-specific local-parquet lines) | Command recipe row 11 |
| Full preset | **Yes (user-approved): `configs/memory_attn.yaml`** — paper reference model L24-D2048 (24 layers, hidden 2048, 32/32 heads, RoPE 10 000, RMSNorm, gated MLP 5632, vocab 32 000, causal LM) | `causal_lm_decoder.yaml` shape; joins the smoke-tested preset set |
| README drift fix | **Approved**: badge "42 across 8 categories" → "46 across 10 categories"; add missing Conformer row | User approval |
| Census bumps | enum 45→46 (45 concrete + `mamba`); new family Memory (1); §7.4 "46 enum values in 10 families"; summary tables "46 … ten families" + new group "Memory (1)"; annex-15 "forty-four dispatchable … Twenty-six parity" → "forty-five … Twenty-seven"; taxonomy root "45 variants" → "46 variants" + new branch; README badge/table | Actual numbers read from the repo (not hardcoded) |

**Hard rules restated:** schema is the contract (`additionalProperties: false`);
`hidden_size % num_heads == 0` and `num_heads % num_kv_heads == 0` (rule 2 —
globally enforced, mirror GQA's constructor validation); `len(layer_pattern) ==
num_layers`; inference never fails because of the cache (D5); papers are the
theory source of truth (D11) — annex formulations must match arXiv:2609.28399
and `paper-es` must translate them faithfully; MA is trainable (rule 5 does not
apply); MA needs `input_ids` like `engram_attn` (rule 6 unaffected — no model
class forces MA in or out).

## 4. File-by-file plan table

Execution order. Every row carries the intended content; anchors refer to the
current `main` @ commit `b1c3711` (clean tree at planning time).

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/attention/memory.py` (new) | `"""Memory Attention…"""` docstring citing *Kang, "Memory Attention", arXiv:2609.28399*; `from __future__ import annotations`; Google-style docstrings; CPU-safe. `class MemoryAttention(nn.Module)` with `__init__(self, config, pos_encoder=None)` mirroring `GroupedQueryAttention.__init__` (lines 40–72): `hidden_size`, `num_heads`, `num_kv_heads = int(getattr(config, "num_kv_heads", 1))`, `head_dim`, `scale`, the three GQA validation `ValueError`s (hidden % heads; `1 <= nKv <= nH`; `nH % nKv`), `num_groups = nH // nKv`, `kv_hidden = nKv·hd`. Projections: `q_proj` H→H, `k_proj` H→`kv_hidden`, `out_proj` H→H, all `proj_cls(H, …, bias=False)` (BitLinear iff `config.use_bitnet`); **no `v_proj`**. `self.memory = nn.Embedding(config.vocab_size, kv_hidden)` (plain, never BitLinear — lookup, not a Linear) + per-KV-head RMSNorm gains `self.memory_norm_weight = nn.Parameter(torch.ones(kv_hidden))`, eps fixed `1e-6`; default `nn.Embedding` init is safe because the per-head RMSNorm bounds each row's contribution. `self.dropout`, `mode`, `pos_encoder`, `pe_type`, `self.use_pe = bool(getattr(config, "memory_attn_use_pe", True))`; `attach_qk_norm(self, config, self.head_dim)`. `forward(self, x, input_ids=None, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)` — **raises `ValueError` when `input_ids is None`** (clear message: text-token mixer). Flow: `q/k` projections → head split → `qk_norm.normalize_q/normalize_k` (GQA pattern) → `m = _rms_per_kv_head(self.memory(input_ids))` reshaped `(B,T,nKv,hd)` → `v = k + m` **before RoPE** → cached branch (below) or legacy: `apply_pe_to_qk` on q/k only, `k/v.repeat_interleave(num_groups, dim=1)`, `scores = q@kᵀ·scale`, `apply_pe_to_scores`, `record_qk_max_logits`, decoder `triu` mask when `mode == "decoder"`, `module_softmax(self, …)` (honors `use_fixed_softmax`), dropout, aggregate, `out_proj`. Cached branch mirrors `grouped_query_attention.py` lines 109–140: `slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None`; `positions_for(cache_position, T, device)`; `q, k = apply_pe_to_qk(…, position_offset=cache_position)`; `kv_decode_append(slot, k, v, positions)`; `K/V.repeat_interleave(num_groups, dim=1)`; scores + `apply_pe_to_scores(kv_positions=…)` + `record_qk_max_logits` + `causal_mask_from_positions` + `module_softmax` + dropout + `out_proj`. Map every equation to a code comment: eq. (7) → lookup+norm helper, (8)/(10) → `v = k + m`, (9) → RoPE pair only, (13)/(14) → note that folding is skipped in v1 (normalize-on-the-fly keeps train/eval identical; folding is a pure eval optimization, listed as follow-up). |
| 2 | `src/model/attention/__init__.py` | Add `from .memory import MemoryAttention` (after the `standard` import, line 19) and `"MemoryAttention"` in `__all__` (after `"StandardAttention"`, line 58). |
| 3 | `src/model/hybrid_layer.py` | (a) Import: `from .attention.memory import MemoryAttention` near line 65. (b) Registry: `"memory_attn": MemoryAttention,` in `mixer_registry` (after `"conformer_attn"`, line 227). (c) **`input_ids` threading** — extend the dispatch in `_forward_dense` (currently `elif self.layer_type == "engram_attn":` line 443) with a sibling branch placed before the generic `elif slot is not None:` (line 445): `elif self.layer_type == "memory_attn": x = self.mixer(x, input_ids=input_ids, logical_layer_idx=logical_layer_idx, pos_encoder=self.pos_encoder, cache=slot, cache_position=cache_position)` — note `slot` may be `None` (legacy path) and the mixer's forward must accept `cache=None`. Keep the engram branch untouched (engram's forward does not accept `cache`). (d) Mirror the same branch in `_forward_dense_mhc` after the engram case (line 558). No `TRAINING_FREE_LAYERS` change. MoD already threads `selected_input_ids` (line 696). |
| 4 | `src/model/config.py` | Add `memory_attn_use_pe: bool = True` to the `use_pe` field block (after `sigmoid_attn_use_pe`, line 595) + its docstring line (after line 284: `memory_attn_use_pe: Whether ``memory_attn`` layers consume the model-wide positional encoding (RoPE on the scoring pair only; values from unrotated keys).`). No other fields (no knobs). |
| 5 | `src/utils/config_flatten.py` | **No change.** `_flatten_pe_parameters` flattens `use_pe` generically (`out[f"{mixer_name}_use_pe"] = flag`) and `_ATTENTION_MIXER_RENAMES` is only for knob sub-objects. |
| 6 | `src/schema/_model/_dims.yaml` | (a) Append `- memory_attn` at the end of `layer_pattern.items.enum` (after `- falcon3a_attn`, line 375 — append-at-end precedent of kda/falcon/msa/sparda). (b) EN table (after the `conformer_attn` row, line 261): ``\| `memory_attn` \| Memory Attention — value projection replaced by layer-specific token-ID memory, V = K + Norm(E\[s\]) (paper: arxiv:2609.28399) \| ✅ \| ✅ \|``. (c) ES table (after `conformer_attn`, line 317): ``\| `memory_attn` \| Memory Attention — proyección de valores reemplazada por memoria token-ID por capa, V = K + Norm(E\[s\]) (paper: arxiv:2609.28399) \| ✅ \| ✅ \|``. No ⚠️ wording (trainable mixer). |
| 7 | *(none)* | No `_attention_memory.yaml` — MA has no knobs (§2.4). |
| 8 | `src/schema/_model.yaml` | No `$ref` (knobless). Append `memory_attn` to the knobless-mixers list in **both** `attention` descriptions: EN line 54 ("…, `kda_attn`, `mamba` do not need a sub-object") → add `memory_attn` before `mamba`; ES lines 60–64 mirror. Optionally add `(paper: arxiv:2609.28399)` to the references line. |
| 9 | `src/schema/_model/_positional_encoding.yaml` | Add after line 410 (`conformer_attn`): `memory_attn: { type: boolean, title: "memory_attn", description: "Model-wide PE for `memory_attn` layers (RoPE on scoring Q/K only; values from unrotated keys) (paper: arxiv:2609.28399).", default: true, description_es: "PE a nivel de modelo para capas `memory_attn` (RoPE solo en Q/K de puntuación; valores desde claves sin rotar) (paper: arxiv:2609.28399)." }` |
| 10 | `src/schema/_model/_norm.yaml` | Append `` `memory_attn` `` to the QK-norm "Supported mixers" list in both languages (EN lines 172–177, ES lines 239–244) — MA forms per-head dot-product logits, so it enters `QK_NORM_MIXERS` and (derived) `QK_CLIP_MIXERS`; `config_loader._validate_qk_norm` reads the set automatically. |
| 11 | `src/model/norm/qk.py` | Add `"memory_attn"` to `QK_NORM_MIXERS` (line 46–71). `QK_CLIP_MIXERS = QK_NORM_MIXERS − {"gma_attn"}` picks it up automatically. |
| 12 | `src/model/cache/capabilities.py` | (a) `CACHE_SPECS["memory_attn"] = CacheSpec(supported=True, slot_type="kv", notes="Memory Attention (arXiv:2609.28399); conventional KV cache (class i): prefill computes V = K + RMSNorm(E[s]) once and appends (K_R, V); decode looks up only the new token's memory row. The paper's MA-Recall (store K + token IDs, reconstruct V, ~50% smaller cache) is analyzed but unmeasured — not implemented.")` (place after `kda_attn`, line 343). (b) Add `"memory_attn",       # F1 (KV with lookup-built values)` to `IMPLEMENTED_MIXERS` (F1 dense/softmax block, line 75). (c) Docstring counts: line 6 "(45 total: 44 mixer classes…" → "(46 total: 45 mixer classes…"; line 106 "One entry per schema enum value (45)" → 46. |
| 13 | `configs/examples/es_arch_memory_adamw.yaml` (new) | `es_arch_kda_adamw.yaml` shape, encoder MLM: `model_class: frankenstein_text_encoder`; `dims`: `vocab_size: 32000`, `hidden_size: 256`, `num_layers: 4`, `num_loops: 1`, `num_heads: 4`, `num_kv_heads: 4` (MHA — paper flagship; d_v = 256), `layer_pattern: [memory_attn]` ×4, `mode: encoder`; `norm.type: layer_norm`; `embedding.conv.enabled: false`; `moe.enabled: false`; `bitnet.enabled: false`; `ffn: {hidden_size: 1024, activation: gelu}`; `positional_encoding: rope` (base default 10000 = paper). Training block = kda's minus `local_parquet_dir`/`prefer_local_cache`/`stream_local_parquet` (machine-specific): `task: mlm`, `seed: 42`, `batch_size: 4`, `max_length: 512`, `mlm_probability: 0.15`, `max_samples: 1000000`, `dataset_batch_size: 25000`, `num_workers: 8`, `cache_dir: ./temp_data/es_arch_memory_adamw_cache`, `gradient_accumulation_steps: 8`, AdamW parameter block copied from kda (lr 1e-6/5e-6/3e-6/2e-6, wd 0.01/0.001/0.01/0.01, betas 0.9/0.95, eps 1e-8), `scheduler_total_steps: 9000`, `csv_log_path: training_metrics_es_arch_memory_adamw.csv`. Must pass `load_training_config` (auto-globbed by `tests/test_yaml_examples.py`). |
| 14 | `configs/memory_attn.yaml` (new, user-approved preset) | Paper reference model, `causal_lm_decoder.yaml` shape: `model_class: frankenstein_text_decoder`; `dims`: `vocab_size: 32000`, `hidden_size: 2048`, `num_layers: 24`, `num_loops: 1`, `num_heads: 32`, `num_kv_heads: 32`, `dropout: 0.0`, `layer_pattern: [memory_attn]` × 24, `mode: decoder`; `norm.type: rms_norm` (paper); `embedding: {factorized: {enabled: false}, conv: {enabled: false}}`; `moe.enabled: false`; `bitnet.enabled: false`; `ffn: {hidden_size: 5632, activation: swiglu}` (paper's gated MLP); `positional_encoding: rope` + `positional_encoding_parameters.rope.base: 10000` (paper). Training: `task: causal_lm`, `seed: 42`, `batch_size: 4`, `max_length: 2048` (paper), `max_samples: 20000000`, `dataset_batch_size: 25000`, `num_workers: 8`, `cache_dir: ./temp_data/v2_dataset_cache`, AdamW block copied from `causal_lm_decoder.yaml`, `scheduler_type: cosine`, `scheduler_warmup_ratio: 0.05` (≈1024/20480 steps), `grad_clip_max_norm: 1.0` (paper), `csv_log_path: training_metrics_memory_attn.csv`. Header comment: reproduction of the L24-D2048 MA model of arXiv:2609.28399 (Table 8 config). |
| 15 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (lines 224–238) with a comment: `# Memory Attention (arXiv:2609.28399): V = K + Norm(E[s]) — trains` `# densely in encoder and decoder modes; conventional KV cache.` `"memory_attn",`. This auto-generates `attn_single_memory_attn` and every `attn_<a>__memory_attn` / `attn_memory_attn__<b>` pair via `make_attention_pair_configs`/`make_single_attention_configs` (lines 734–756). **Not** training-free → it belongs in the list (header comment reserves it for trainable mixers). |
| 16 | `tests/test_schema_attention_layers.py` | (a) Add `"memory_attn"` to the `expected` set (line 18–64). (b) `test_cache_specs_match_layer_pattern_enum` (line 138): docstring "(45)" → "(46)", `assertEqual(len(enum_values), 45)` → `46` (line 149). |
| 17 | `tests/test_kv_cache_fallback.py` | Line 28 docstring "(45 values)" → "(46 values)"; line 62 `assertEqual(len(enum), 45)` → `46`. (`test_every_enum_value_runs_with_cache` iterates the schema enum, so `memory_attn` is exercised automatically — it must run correctly under cache; see §9 assumptions.) |
| 18 | `tests/test_attention_modules.py` (family test) | New `class MemoryAttentionTests(unittest.TestCase)` after `SigmoidAttentionTests` (line 99–126), building a tiny config (mirror `_config` helpers in the file; `num_heads=4`, `num_kv_heads=2`, `vocab_size=64`) and asserting: `test_encoder_output_shape`, `test_decoder_output_shape` (causal path), `test_missing_input_ids_raises` (`ValueError`), `test_value_equals_key_plus_normalized_memory` (manual parity: rebuild `V = K + RMSNorm(E[s])` from the module's own weights and compare), `test_gradient_flows` (incl. `memory.weight` and `memory_norm_weight` receive grads), `test_with_bitnet` (projections are BitLinear, `memory` stays `nn.Embedding`), `test_no_nans_encoder_decoder`, `test_use_pe_off_changes_output` (`memory_attn_use_pe=False`). |
| 19 | `tests/test_bitnet_coverage.py` | Add `"memory_attn"` to the mixer list (after `"gqa_attn"`, line 56). |
| 20 | `tests/test_kv_cache_mixers.py` | **No edit needed**: `MIXERS = sorted(m for m in IMPLEMENTED_MIXERS if m != "mamba")` auto-parametrizes the new mixer (full forward vs prefill+decode parity, greedy-token equality). Verify during implementation that the parity harness's decode path threads `input_ids` (the `FrankensteinDecoder`/`generate` loop passes them; HybridLayer row 3 guarantees delivery) — if the harness builds prompts only, the branch in row 3 already covers it. |
| 21 | `tests/test_qk_norm.py` | Optional coverage: add `MemoryAttention` to `test_forward_methods_run_on_supported_mixers` (line 128) **with** `input_ids=torch.randint(...)` — the loop currently calls `mixer(x)` without IDs, so either add a per-mixer ids fixture or cover MA's QK-norm path inside `MemoryAttentionTests` (preferred, row 18). |
| 22 | `docs/paper/figures/mixers/memory.mmd` (new) | House style (`standard.mmd`): init header `%%{init: {"theme":"neutral","flowchart":{"curve":"basis","nodeSpacing":18,"rankSpacing":26,"useMaxWidth":false}}}%%`, `flowchart TD`; nodes: `x["x<br/>[B,n,H]"]`, `s["token IDs s<br/>[B,n]"]`, `q["Q = Wq·x<br/>[B,n,nH·hd]"]`, `k["K = Wk·x<br/>[B,n,nKv·hd]"]`, `e["E[s]<br/>[B,n,nKv·hd]"]`, `m["M = RMSNorm(E[s])<br/>per KV head"]`, `v["V = K + M<br/>(pre-RoPE)"]`, `qr["Qᵣ = RoPE(Q)"]`, `kr["Kᵣ = RoPE(K)"]`, `sc["scores = QᵣKᵣᵀ/√d<br/>[B,nH,n]"]`, `sm["softmax<br/>(scores)"]`, `a["A<br/>= softmax · V"]`, `o["O = Wo·A<br/>[B,n,H]"]`; edges `x -->|"Wq"| q`, `x -->|"Wk"| k`, `s -->|"lookup"| e`, `e -->|"RMSNorm"| m`, `k -->|"+"| v`, `m -->|"+"| v`, `q -->|"RoPE"| qr`, `k -->|"RoPE"| kr`, two `QᵣKᵣᵀ` edges into `sc`, `sc --> sm --> a`, `v -->|"A·V"| a`, `a -->|"Wo"| o`. No `Wv` node — that absence is the point; note it in the annex caption. |
| 23 | `docs/paper/figures/mixer-taxonomy.mmd` | Root label `"Sequence Mixer Registry (45 variants)"` → `(46 variants)` (line 3). New branch (after the `m` "Conditional memory" chain, line 15): `mm["Memory (1)"] --> mm1["memory_attn"]`, plus `root --> mm` (line 23–31 block) and add `mm` to the `classDef category` class list (line 34). Do **not** touch engram's "Conditional memory" branch. |
| 24 | `docs/paper/figures/memory-map.mmd` (new, family map) | Mirror `dense-map.mmd`: init header with `"wrappingWidth":240`, `flowchart TD`, `subgraph memory["Memory family: lookup-based value capacity, O(n²) mixing"]` with node `ma["Memory Attention<br/>V = K + RMSNorm(E[s])"]` and `end`. Build via `build-figures.sh both` → `memory-map.pdf` + `.svg`. |
| 25 | `docs/bibliography/attention_types.bib` | Append (pattern of `kimi_kda_2025`): `@misc{kang_memory_attention_2026, title={Memory Attention}, author={Jiale Kang}, year={2026}, eprint={2609.28399}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={https://arxiv.org/abs/2609.28399}, publisher={arXiv}}`. |
| 26 | `docs/paper/appendices/annex-2-recurrent-attention.tex` | (a) Intro paragraph (line 4): add "token-indexed memory values" to the paradigm list. (b) New `\subsection{Token-Indexed Memory: Memory Attention}` between "Dense Attention Baselines" (ends line 86) and "Recurrent and Retentive Architectures" (line 88), drafted in §4.1 (below): figure `\fitfigure{\includegraphics{figures/mixers/memory.pdf}}` + `\label{fig:mixer-memory}`, `algorithmic` blocks for eqs. (7)–(9) + (2), inference-folding note (13)–(14), parameter/storage comparison (paper Table 1), strengths/weaknesses `itemize`, `\citep{kang_memory_attention_2026}`, note that RoPE applies only to scoring and that MA-Offload/MA-Recall are analyzed-not-implemented. (c) Dispatch pseudocode (lines 231–257): add `\ElsIf{$m=\texttt{memory\_attn}$} \State $H \gets \text{memory-attention}(H)$` after the `sigmoid_attn` case (lines 240–241). |
| 27 | `docs/paper/sections/04-architecture-taxonomy.tex` | Line 7: "forty-five sequence mixers organized into nine functional categories" → "forty-six … ten functional categories". Itemize (after item 6 "Conditional memory", line 14): `\item \textbf{Token-indexed memory} (1): Memory Attention replaces the dedicated value projection with layer-specific token memory, constructing values as $V = K + \mathrm{Norm}(E[s])$ while keeping the softmax weighting unchanged \citep{kang_memory_attention_2026}.` Caption line 23: "forty-five … nine" → "forty-six … ten" + append "token-indexed memory relocates value capacity into lookup tables". New short subsection + figure after "Dense and Recurrent Families" (lines 27–42): `\subsection{Token-Indexed Memory Family}` with 2–3 sentences (mechanism, no-`W_v`, conventional KV cache) + `\begin{figure}[t] \fitfigure{\includegraphics{figures/memory-map.pdf}} \caption{…} \label{fig:memory-map}`. |
| 28 | `docs/paper/appendices/annex-summary-tables.tex` | Line 4 intro: "45 sequence mixers across nine families" → "46 sequence mixers across ten families". New group after the Latent group (before `\bottomrule`, line 94): `\midrule \multicolumn{6}{c}{\textbf{Memory (1)}} \\ \midrule memory\_attn & Memory & $\mathcal{O}(n^2 d)$ & $\mathcal{O}(n)$ cache & Token-ID memory values $V = K + \mathrm{Norm}(E[s])$; removes the value projection; $+Nd_v$ parameters per layer & \citep{kang_memory_attention_2026} \\`. |
| 29 | `docs/paper/appendices/annex-schema.tex` | New row in `tab:layer-pattern-enum` after `standard\_attn`/`sigmoid\_attn` (lines 227–228): `memory\_attn & Memory Attention — token-ID memory values (V = K + Norm(E[s])) & \checkmark & \checkmark \\`. |
| 30 | `docs/paper/appendices/annex-15-kv-cache.tex` | Line 20: "every one of the forty-four dispatchable layer types. Twenty-six have parity-tested incremental branches" → "forty-five … Twenty-seven". Add a cache-recipe item next to `\item \textbf{Titans.}` (line 163): `\item \textbf{Memory Attention.} Class (i): prefill caches (post-RoPE $K$, $V = K + \mathrm{RMSNorm}(E[s])$); decode looks up only the new token's row, so the slot is byte-identical to standard KV \citep{kang_memory_attention_2026}.` Test-matrix table (line 337): "fallback & all $44$ enum values" → "$45$". |
| 31 | `docs/paper/appendices/annex-12-positional-encodings.tex` | Per-mixer `use_pe` defaults list (lines 210–213): append `\texttt{memory\_attn}` to the **True** item (after `\texttt{engram\_attn}`), with the note "RoPE on scoring Q/K only; values built from unrotated keys". |
| 32 | `docs/paper/sections/01-introduction.tex`, `10-discussion.tex`, `11-conclusion.tex` | "forty-five sequence mixers" → "forty-six" (01 line 14; 10 line 16 "forty-five implemented mixers"; 11 line 8). 01 also says "across nine families" → "ten families". |
| 33 | ES paper: `docs/paper-es/appendices/annex-2-recurrent-attention.tex`, `sections/01-introduccion.tex`, `04-taxonomia-arquitectura.tex`, `10-discusion.tex`, `11-conclusion.tex`, `appendices/annex-summary-tables.tex`, `annex-schema.tex`, `annex-15-cache-kv.tex`, `annex-12-codificaciones-posicionales.tex` | Faithful Spanish translation of rows 26–32: annex-2 gets a subsection mirroring its own heading style (`\subsubsection` directly: "Atención con Memoria (Memory Attention)"), figures via the ES paper's shared `../paper/figures` path (verify `\graphicspath` at implementation time); "cuarenta y cinco" → "cuarenta y seis"; "nueve familias/categorías" → "diez"; summary intro "45 mezcladores de secuencia en nueve familias" → "46 … diez familias"; new group `\multicolumn{6}{c}{\textbf{Memoria (1)}}` + row `memory\_attn & Memoria & $\mathcal{O}(n^2 d)$ & $\mathcal{O}(n)$ caché & Valores de memoria por ID de token $V = K + \mathrm{Norm}(E[s])$; elimina la proyección de valores; $+Nd_v$ parámetros por capa & \citep{kang_memory_attention_2026} \\`; annex-schema ES row after line 228: `memory\_attn & Atención con memoria — valores por ID de token (V = K + Norm(E[s])) & \checkmark & \checkmark \\`; annex-15 ES line 20 "cuarenta y cuatro … Veintiséis" → "cuarenta y cinco … Veintisiete". |
| 34 | `README.md` | (a) Badge line 92: "42 across 8 categories (Dense, GQA, Recurrent, Sparse, Gated, Fast-Weight, Latent, Geometric)" → "46 across 10 categories (Dense, GQA, Recurrent, Sparse, Gated, Fast-Weight, Latent, Geometric, Convolution-augmented, Memory)" (drift fix approved). (b) Mixer Categories table: add `**Memory** (1) | memory_attn | Token-ID memory values: V = K + Norm(E[s]); removes the value projection` and the missing drift-fix row `**Convolution-augmented** (1) | conformer_attn | Macaron FFN₁ + MHSA + depthwise-conv module (arXiv:2005.08100)`. Counts now 2+1+6+9+8+6+10+1+1+1 = 45 → +Memory = 46 ✓. |
| 35 | `docs/architecture.md` | (a) §1 table line 25: "**45** (44 concrete mixers + `mamba`)" → "**46** (45 concrete mixers + `mamba`)" and line 81 "45 mixers" → "46 mixers". (b) §7.4 line 350: "45 enum values in 9 families" → "46 enum values in 10 families"; add row after `conformer_attn` (line 399): `\| memory_attn \| Memory \| dims.num_kv_heads \| yes \|` (subkey mirrors `gqa_attn`'s). (c) §7.13 "Implemented cached branches" list (lines 622–629): append `memory_attn` (26 → 27 mixers); fallback list unchanged (MA caches). (d) One sentence in §7.4 after the `swa_attn` note describing MA citing arXiv:2609.28399. |
| 36 | `configs/README.md` | Line 268 block: add `- Memory blocks: memory_attn — Memory Attention, token-ID memory values V = K + Norm(E[s]) (arXiv:2609.28399).` Also add the new preset to any preset index in the file (grep `kimi_k3`/`conformer` at implementation time; mirror their listing style). |
| 37 | `docs/source/api/model/attention/memory.rst` (new) + `docs/source/api/model/attention/index.rst` | `memory.rst`: `Memory Attention` title + `.. automodule:: src.model.attention.memory` with `:members:`/`:undoc-members:`/`:show-inheritance:` (mirror `standard.rst`). `index.rst` toctree: add `memory` after `sigmoid`. |

### 4.1 Drafted annex subsection (row 26 — EN; ES translates faithfully)

```latex
\subsection{Token-Indexed Memory: Memory Attention}

\begin{figure}[H]
\centering
\fitfigure{\includegraphics{figures/mixers/memory.pdf}}
\caption{Dataflow of Memory Attention: the value projection is removed;
values are the unrotated keys plus a per-KV-head RMSNorm'd row of a
layer-specific token-ID memory table. RoPE applies only to the scoring pair.}
\label{fig:mixer-memory}
\end{figure}

Memory Attention \citep{kang_memory_attention_2026} removes the dedicated value
projection and constructs values from contextual keys plus layer-specific
token memory. With token IDs $s$, vocabulary size $N$ and
$d_v = H_{KV} d_h$:

\begin{algorithmic}[1]
\State \textbf{Input:} hidden states $X \in \mathbb{R}^{T \times d}$, token IDs $s$, table $E \in \mathbb{R}^{N \times d_v}$
\State $M \gets \mathrm{Norm}(E[s]) \in \mathbb{R}^{T \times d_v}$ \Comment{RMSNorm per token vector, per key/value head}
\State $Q \gets X W_Q$, \quad $K \gets X W_K$, \quad $V \gets K + M$ \Comment{no $W_V$; values from unrotated keys}
\State $Q_R \gets \mathrm{RoPE}(Q)$, \quad $K_R \gets \mathrm{RoPE}(K)$ \Comment{rotation for scoring only}
\State \textbf{Output:} $Y = \mathrm{softmax}\!\left(\tfrac{Q_R K_R^\top}{\sqrt{d_k}} + C\right) V$
\end{algorithmic}

The output decomposes as $O_{\mathrm{MA}} = A(K{+}M) = AK + AM$: a contextual
retrieval term over keys and a memory retrieval term aggregated under the same
attention distribution. Across $L$ replaced layers, MA swaps the standard
$L\,d\,d_v$ value-projection parameters for $L(N d_v + p_{\mathrm{norm}})$
memory and normalization parameters --- added capacity accessed by lookup
instead of dense computation. At inference the per-row normalization is folded
into the table ($E[i] \gets \mathrm{Norm}(E[i])$), reducing online value
construction to lookup and addition while retaining a conventional KV cache.
The paper reports improved LAMBADA/WikiText perplexity and average downstream
accuracy across MHA/GQA/MQA backbones at matched token budgets, with
$1.42\times$/$1.16\times$ token efficiency; the comparison includes the
additional memory parameters, so it does not isolate structure from capacity.
MA-Offload (CPU-resident tables with prefetching) and MA-Recall (reconstructing
historical values as $V_{1:T} = K_{1:T} + E[s_{1:T}]$ for a ${\sim}50\%$ smaller
value cache) are analyzed but not implemented here.

\begin{itemize}[leftmargin=*]
    \item \textbf{Training complexity}: $\mathcal{O}(n^2 d)$ attention with
    $\mathcal{O}(n d_v)$ lookup+norm+add value construction (replacing the
    $\mathcal{O}(n\, d\, d_v)$ projection).
    \item \textbf{Inference complexity}: $\mathcal{O}(n)$ per token with a
    conventional KV cache; $N d_v$ extra parameters per layer.
    \item \textbf{Strengths}: Extra capacity without dense-compute growth;
    layer-specific token representations; exact standard-attention aggregation;
    KV-cache-compatible by construction.
    \item \textbf{Weaknesses}: Parameter count grows with the vocabulary
    ($N \gg d$); needs token IDs (text-only); the reported gains are not
    parameter-matched.
\end{itemize}
```

## 5. Ordered execution checklist (implementation time — do not run now)

Hard preconditions: worktree clean (`git status --short` empty, confirmed at
planning time); `uv` extras pinned per AGENTS.md (CPU-first:
`uv run --extra cpu --extra train …`); `pdflatex`+`bibtex` available at
`/usr/bin`; `chromium` available (mermaid); `pandoc` available. Mirror
`erickfmm.github.io/` already cloned next to the repo.

Implementation (rows 1–15):
- [ ] Row 1: `src/model/attention/memory.py`
- [ ] Row 2: exports (`attention/__init__.py`)
- [ ] Row 3: `hybrid_layer.py` (import, registry, `input_ids` branch ×2)
- [ ] Rows 4, 9, 10, 11: `config.py` field; schema `use_pe` entry; `_norm.yaml`
      lists; `QK_NORM_MIXERS`
- [ ] Rows 6, 8: `_dims.yaml` enum + both tables; `_model.yaml` no-knob lists
- [ ] Row 12: `capabilities.py` (`CacheSpec`, `IMPLEMENTED_MIXERS`, docstrings)
- [ ] Rows 13–14: example + preset YAML
- [ ] Row 15: `run_e2e.py` `ATTENTIONS`

Tests (rows 16–21):
- [ ] `test_schema_attention_layers.py` (enum + 46)
- [ ] `test_kv_cache_fallback.py` (46)
- [ ] `test_attention_modules.py` (`MemoryAttentionTests`)
- [ ] `test_bitnet_coverage.py`
- [ ] confirm `test_kv_cache_mixers.py` picks `memory_attn` up (no edit)

```bash
# targeted gates (run at implementation time, not now)
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_attention_modules.py tests/test_bitnet_coverage.py \
  tests/test_kv_cache_mixers.py tests/test_kv_cache_fallback.py \
  tests/test_qk_norm.py \
  -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ \
  --continue-on-collection-errors -v --tb=short -p no:warnings
```

Expected: all green; `test_schema_consistency::test_layer_pattern_enum_matches_documentation`
passes only with both `_dims.yaml` tables updated; `test_cache_specs_match_layer_pattern_enum`
passes with enum=46 and `CACHE_SPECS` covering it.

E2E smoke, incl. targeted real-data training (default `--realonall`):

```bash
# enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F memory_attn
# expected: attn_single_memory_attn plus every attn_<a>__memory_attn /
# attn_memory_attn__<b> pair (45 new combos + 1 solo)

# TARGETED SMOKE — actual training on the REAL Borges corpus (default --realonall;
# first invocation downloads tokenizer/corpus from the HF Hub)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only memory_attn

# minimal real-data run if the pair sweep is too heavy: the solo combo only
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_memory_attn

# offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only memory_attn --no-realonall

# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke (real training box): swap the extra for --extra cu118 --extra train.
```

Expected evidence: every selected combo reports **PASS** with a finite training
loss over real corpus windows, and the run directory keeps the generated
`config.yaml` + metrics for inspection. A NaN/Inf loss, a schema rejection or a
timeout is a **blocker**, not a warning.

Figures (rows 22–24; needs chromium or `PUPPETEER_EXECUTABLE`):

```bash
( cd docs/paper/figures && ./build-figures.sh both )
# expected: mixers/memory.{pdf,svg}, memory-map.{pdf,svg}, refreshed
# mixer-taxonomy.{pdf,svg}; no other diagrams change
```

Docs EN/ES (rows 25–36): bib entry → annex-2 (EN+ES) → section 04 (EN+ES) →
summary tables (EN+ES) → schema annex (EN+ES) → annex-15 (EN+ES) → annex-12
(EN+ES) → sections 01/10/11 (EN+ES) → README/architecture/configs-README.

Sphinx (row 37):

```bash
bash docs/build_docs.sh   # expected: builds cleanly; memory page listed
```

Param ground truth + JS estimator (after rows 13–14 land; see §6 for the
site-side formula):

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py
# expected: "... configs counted, 0 failures"; param_truth.json diff is
# intentional (new example + preset rows)
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01
# expected: ALL PASS (≤1% drift) — only after the ft-param-estimator.js case lands (§6)
```

PDF rebuild (implementation time; mirrors the repo's TeX layout):

```bash
cd docs/paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
rm -f *.aux *.log *.out *.toc *.blg

cd ../paper-es
pdflatex -interaction=nonstopmode paper-es.tex
bibtex paper-es
pdflatex -interaction=nonstopmode paper-es.tex
pdflatex -interaction=nonstopmode paper-es.tex
rm -f *.aux *.log *.out *.toc *.blg
```

Post-PDF verification: no `Fatal error`, no `Citation .* undefined` (the new
`kang_memory_attention_2026` key resolves), `paper.pdf`/`paper-es.pdf` newer
than the edited `.tex`/`.bbl`, and
`git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf`/figure files.

Fallbacks: if `pdflatex`/`bibtex` are unavailable, leave sources updated and
note the rebuild must happen on a TeX machine before the next release (the
website sync copies prebuilt PDFs). If no Chrome is available for mermaid,
commit the `.mmd` sources and render `.pdf`/`.svg` before the paper/website
sync. If the HF Hub is unreachable for the real-data smoke, record the
`--no-realonall` PASS and re-run the real-data pass later.

## 6. Website sync (planned, not executed)

Follow `.opencode/command/sync-website.md`; the state script drives the exact
drift list. Steps for this change:

1. **Bootstrap/refresh** `erickfmm.github.io/` (already cloned at planning
   time); run `bash .opencode/command/sync-website-state.sh` and copy per the
   file map: `src/schema.yaml` + `src/schema/**` (now containing
   `memory_attn`), `configs/*.yaml` + `configs/README.md` (new
   `memory_attn.yaml`), `configs/examples/**` (new
   `es_arch_memory_adamw.yaml`), `docs/paper/**` + `docs/paper-es/**`
   (tex, bbl, pdf, figures incl. `memory.mmd`/`memory.pdf`/`memory.svg` and
   `memory-map.*` + regenerated `mixer-taxonomy.*`), `docs/bibliography/**`
   (new bib entry), `docs/architecture.md`.
2. **Site-only edits** (mirror is the source of truth for these):
   - `ft-diagram.js`:
     - `CATS`: add `memory_attn: { c: 'mem', l: 'Memory Attention' },`
       (new category key `mem`).
     - `SUBSTRUCT`: add a template mirroring the paper's dataflow:
       `x` → `Q = Wq·x`, `K = Wk·x`; `s` (token IDs) → `E[s]` → `M =
       RMSNorm(E[s]) per KV head`; `V = K + M (pre-RoPE)`; `scores =
       QᵣKᵣᵀ/√d`; `A = softmax(scores)`; `O = Wo·(A·V)` (edge labels `Wq`,
       `Wk`, `lookup`, `RMSNorm`, `+`, `QᵣKᵣᵀ`, `softmax`, `Wo`).
     - `STYLES` + `STYLES_PRINT` (new family category): add e.g.
       `mem: 'fill:#58a6ff,stroke:#1f6feb,color:#fff'` (distinct from
       `attn/recur/ssm/ode/sparse/eval/field/fastw/emb/norm/ffn/moe/output`).
   - `ft-param-estimator.js`: add to `mixerParams` after `gqa_attn`:
     ```js
     case 'memory_attn': {
       var Vocab = toNum(pick(m, 'dims.vocab_size', 'vocab_size', 50000), 50000);
       // Q (H*H) + K (H*kv) + O (H*H) + memory table (Vocab*kv)
       // + per-KV-head RMSNorm gains (kv); no W_V.
       return 2 * H * H + H * kv + Vocab * kv + kv;
     }
     ```
     (§2.3's formula verbatim; `kv = hd * nKv` already exists in the switch
     preamble.)
   - `index.html`: grep `standard_attn`/`kda_attn`/`falcon3a` at sync time —
     at planning time no hardcoded mixer list exists (mixers are read from the
     mirrored schema; `standard_attn` appears only as a fallback default,
     which needs no edit). Only extend a list if the grep finds one.
3. **Paper tab HTML rebuild** (`flatten_tex.py --web` + `pandoc` +
   `postprocess_html.py`, en + es) and top-level `paper/en.pdf` / `es.pdf`
   copies (commands in sync-website.md §3).
4. **Validation Gate** (sync-website.md §5, in order):
   `param_count_check.py` → `0 failures` (regenerates `param_truth.json`;
   intentional diff), estimator `--tol 0.01` → `ALL PASS`, JSON round-trip
   script, CLI parse script, and the state-script probes (`schema-refs`,
   `example-yaml`, `cli-flags`, …) all PASS.
5. Leave both repos **uncommitted** until §7.

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` (English, conventional tag, structured body; never
stage `*.aux *.log *.out *.toc *.blg`, caches or mermaid temp files;
`param_truth.json` changes are intentional).

1. **Main repo** — `feat(attention): add Memory Attention mixer (arXiv:2609.28399)`
   Body: module + registry + `input_ids` threading; schema (enum + both tables,
   no-knob lists, `use_pe` entry, QK-norm lists); `QK_NORM_MIXERS`; cache
   decision (CacheSpec + IMPLEMENTED_MIXERS + parity); example + preset; tests
   (schema/fallback/modules/bitnet, auto parity); e2e `ATTENTIONS` + real-data
   smoke result; figures (memory + memory-map + taxonomy, mmd + rendered
   pdf/svg); docs (annex-2 EN/ES, section 04 EN/ES, summary tables EN/ES,
   schema annex EN/ES, annex-15 EN/ES, annex-12 EN/ES, sections 01/10/11 EN/ES,
   bibliography, README + architecture + configs/README census incl. approved
   badge/conformer drift fix, Sphinx rst); rebuilt PDFs + `.bbl`.
   Stage exactly:
   ```
   src/model/attention/memory.py
   src/model/attention/__init__.py
   src/model/hybrid_layer.py
   src/model/config.py
   src/model/norm/qk.py
   src/model/cache/capabilities.py
   src/schema/_model/_dims.yaml
   src/schema/_model.yaml
   src/schema/_model/_positional_encoding.yaml
   src/schema/_model/_norm.yaml
   configs/examples/es_arch_memory_adamw.yaml
   configs/memory_attn.yaml
   full_tests/run_e2e.py
   full_tests/param_truth.json
   tests/test_schema_attention_layers.py
   tests/test_kv_cache_fallback.py
   tests/test_attention_modules.py
   tests/test_bitnet_coverage.py
   docs/paper/figures/mixers/memory.mmd docs/paper/figures/mixers/memory.pdf
   docs/paper/figures/mixers/memory.svg
   docs/paper/figures/memory-map.mmd docs/paper/figures/memory-map.pdf
   docs/paper/figures/memory-map.svg
   docs/paper/figures/mixer-taxonomy.mmd docs/paper/figures/mixer-taxonomy.pdf
   docs/paper/figures/mixer-taxonomy.svg
   docs/bibliography/attention_types.bib
   docs/paper/appendices/annex-2-recurrent-attention.tex
   docs/paper/sections/01-introduction.tex
   docs/paper/sections/04-architecture-taxonomy.tex
   docs/paper/sections/10-discussion.tex
   docs/paper/sections/11-conclusion.tex
   docs/paper/appendices/annex-summary-tables.tex
   docs/paper/appendices/annex-schema.tex
   docs/paper/appendices/annex-15-kv-cache.tex
   docs/paper/appendices/annex-12-positional-encodings.tex
   docs/paper/paper.pdf docs/paper/paper.bbl
   <same set under docs/paper-es/> + docs/paper-es/paper-es.pdf + paper-es.bbl
   README.md docs/architecture.md configs/README.md
   docs/source/api/model/attention/memory.rst
   docs/source/api/model/attention/index.rst
   ```
   Then `git push origin HEAD:main`.
2. **Mirror** — `git -C erickfmm.github.io …` commit
   `docs(site): sync Memory Attention mixer (arXiv:2609.28399)` staging
   `frankenstein-transformer/` copies of: `schema.yaml`, `schema/**`,
   `examples/*.yaml`, `examples/examples/**`, `examples/README.md`,
   `paper/**` (en + es tex/bbl/pdf/figures/HTML), `paper/bibliography/**`,
   `paper/specs/architecture.md`, `ft-diagram.js`, `ft-param-estimator.js`,
   `paper/en.pdf`, `paper/es.pdf`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted pytest families | §5 targeted block | all pass |
| Full suite | `pytest tests/ --continue-on-collection-errors` | all pass (76 files) |
| Schema enum/table consistency | `pytest tests/test_schema_consistency.py -v` | `test_layer_pattern_enum_matches_documentation`, `test_all_presets_validate`, `test_schema_defaults_match_dataclass` green |
| Cache-spec coverage | `pytest tests/test_kv_cache_fallback.py tests/test_schema_attention_layers.py -v` | enum=46 everywhere; `is_cacheable("memory_attn") == True` |
| KV parity | `pytest tests/test_kv_cache_mixers.py -v` | `memory_attn × {rope, nope, alibi}` full-vs-cached parity, greedy-token equality |
| QK-norm capability | `pytest tests/test_qk_norm.py` | green; `memory_attn` accepted in patterns with `qk_norm.enabled` |
| e2e combo enumeration | `run_e2e.py --category attn --list-combos \| grep -F memory_attn` | `attn_single_memory_attn` + 45 pair combos |
| e2e targeted real-data training | `run_e2e.py --category attn --only memory_attn` | all selected combos PASS, finite loss on the real Borges corpus |
| e2e solo fallback | `--only attn_single_memory_attn` | PASS |
| e2e offline fallback | `--only memory_attn --no-realonall` | PASS (recorded if HF Hub down) |
| Figure render | `build-figures.sh both` | `memory.pdf/svg`, `memory-map.pdf/svg`, refreshed `mixer-taxonomy.pdf/svg` |
| Param ground truth | `param_count_check.py` | `… 0 failures`; truth JSON gains the two new configs |
| JS estimator | `param_estimate_check.mjs … --tol 0.01` | `ALL PASS` (formula §2.3/§6) |
| Sphinx | `bash docs/build_docs.sh` | clean build, `memory` page present |
| PDF EN/ES | §5 pdflatex/bibtex block | no Fatal error, no undefined citation, PDFs newer than sources |
| Website probes | `sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, estimator, JSON round-trip all PASS |
| Census sweep | grep "forty-five\|45 sequence\|45 enum\|45 variants\|(45" across README/docs | zero stale occurrences; every surface says 46 / ten families |

## 9. Risks, assumptions and open questions

**Assumptions to re-verify at implementation time** (line numbers drift):

- `FrankensteinDecoder`'s forward/`generate()` cached decode threads
  `input_ids` down to `HybridLayer` (engram relies on the same plumbing; if the
  cached path drops them, row 3's branch still delivers them — verify with the
  parity test, which is the real check).
- `tests/test_kv_cache_mixers.py`'s `_parity` harness feeds prompt IDs the
  decoder can pass through (its `prompt = torch.randint(...)` suggests yes).
- `test_every_enum_value_runs_with_cache` (fallback suite) drives MA under
  cache with IDs; if its fixture omits `input_ids`, extend the fixture call —
  it must not special-case MA out.
- ViT/audio: MA in a vision/audio preset would fail at runtime (no IDs). No
  hard validation is added (engram precedent); confirm nothing in
  `config_loader` needs a new rule (grep found none for engram).
- `es_arch_*` example naming/`csv_log_path` uniqueness vs existing files.
- `docs/paper-es` `\graphicspath` resolves `../paper/figures` (annex-2 ES
  already embeds shared mixer PDFs — verify once).
- The estimator's `kv` preamble variable and `pick()` helper exist in the
  deployed `ft-param-estimator.js` (they do at planning time).

**Risks:**

- **Memory-table parameter blow-up in smoke tests**: `N × d_v` per layer is the
  point of the mixer; the example keeps `vocab 32000 × d_v 256 × 4 layers ≈
  33 M` table params (build-only in the gates; e2e uses the toy vocab). The
  full preset (`32000 × 2048 × 24 ≈ 1.6 B` table params) is schema-smoke +
  param-count territory only — never trained on CPU.
- **Naming collisions**: none found (`memory_attn`, `MemoryAttention`, bib key
  `kang_memory_attention_2026` all free).
- **QK-Clip/QuacK**: MA forms per-head logits → both are safe; the trainer's
  `_qk_clip_active` path uses `record_qk_max_logits`, which the module calls on
  every branch.
- **Chunkwise/simplifications vs paper**: none — the module implements the
  equations exactly; the only deviation is **not** folding RMSNorm at eval
  (folding is mathematically identical but keeps one code path; listed as an
  optional follow-up).
- **New-family census cost**: ten census surfaces gain a new family (taxonomy
  branch, family map, STYLES class, README/architecture/summary/annex-12/15
  counts) — every one enumerated in §4; a missed surface is caught by the §8
  census-sweep grep.

**Open questions (paper-level, not blocking):**

1. **MA-Recall** (eq. 19): would shrink the KV cache ~50 % but is unmeasured in
   the paper. Optional follow-up phase behind `inference.*` config; needs user
   approval and a new slot strategy (store `K` + token-tail IDs). Out of scope.
2. **MA-Offload** (eqs. 17–18): CPU/SSD table residency + prefetch — systems
   work with no repo surface today. Out of scope; documented in the annex.
3. The paper's **"Gate" backbone variant** (Table 2, gated attention
   configuration) is a fourth experimental configuration, not a separate
   mixer; the repo's `gated_softmax_attn` covers gating orthogonally. Not
   planned.
4. The paper has **no parameter-matched control** (its own limitation, stated
   twice) — the annex trade-off table must repeat this honestly.

## 10. Out of scope

No code, schema, preset, test, figure, doc, PDF, mirror or git change was made
by this command. This file (`.opencode/plan/add-mixer-2609.28399.md`) is the
only artifact. Implementation is a separate, explicitly requested step
(switch agents, then execute §5 top-to-bottom).
