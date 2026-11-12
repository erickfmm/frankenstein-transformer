# Plan: add Tensor Product Attention (TPA) mixer (arXiv:2501.06425)

**Status: plan only — nothing outside `.opencode/plan/` was modified.**

Tensor Product Attention (Zhang et al., NeurIPS 2025 Spotlight, arXiv:2501.06425)
attacks the KV-cache memory bottleneck of long-context LLMs. Instead of
projecting each token into full `h × d_h` query/key/value slices, TPA
factorizes each per-token slice into a normalized sum of `R` rank-1 **outer
products of contextual vector factors** (a CP-style tensor decomposition of
the *activations*, not the weights): `Q_t = (1/R_Q) Σ_r a^Q_r(x_t) ⊗ b^Q_r(x_t)`
with `a ∈ R^h`, `b ∈ R^{d_h}`, and likewise for K (rank `R_K`) and V (rank
`R_V`). Attention then proceeds as standard scaled dot-product attention. The
KV cache stores only the factor pairs, `(R_K+R_V)(h+d_h)` floats/token vs
`2 h d_h` for MHA, and RoPE commutes with the factorization (it can be applied
row-wise to the `B` factors, so cached keys are pre-rotated). T6 (TPA +
LLaMA-style SwiGLU blocks) matches or beats MHA/MQA/GQA/MLA on perplexity and
benchmarks. TPA belongs in this repo as a canonical Latent-family
KV-compression mixer: the missing *vector-factor* special case next to `kpa`
(matrix Kronecker factors, whose docstring already names TPA) and `tucker`
(weight-factor decomposition). The user explicitly approved adding TPA as its
own mixer despite `kpa` subsuming it expressively.

---

## 2. Paper extract (arXiv:2501.06425v7)

### Forward equations (paper §3.1–3.3)

Notation: `d_model`, `h` heads, `d_h = d_attn/h` head width, `d_attn = h·d_h`
(usually `= d_model`), ranks `R_Q, R_K, R_V`.

Contextual factorization (eq. 3.1), per token `t` with hidden state `x_t`:

```
Q_t = (1/R_Q) Σ_{r=1}^{R_Q} a^Q_r(x_t) ⊗ b^Q_r(x_t) ∈ R^{h×d_h}
K_t = (1/R_K) Σ_{r=1}^{R_K} a^K_r(x_t) ⊗ b^K_r(x_t)
V_t = (1/R_V) Σ_{r=1}^{R_V} a^V_r(x_t) ⊗ b^V_r(x_t)
a^*_r(x_t) ∈ R^h ,  b^*_r(x_t) ∈ R^{d_h}
```

Latent factor maps (linear, no bias in the reference):

```
a^Q_r(x_t) = W^{aQ}_r x_t ∈ R^h ,   b^Q_r(x_t) = W^{bQ}_r x_t ∈ R^{d_h},
W^{aQ}_r ∈ R^{h×d_model}, W^{bQ}_r ∈ R^{d_h×d_model}   (analogous for K, V)
```

Merged-output form (what we implement — single projection per factor side):

```
a_Q(x_t) = W^{aQ} x_t ∈ R^{R_Q·h},  b_Q(x_t) = W^{bQ} x_t ∈ R^{R_Q·d_h}
A_Q(x_t) ∈ R^{R_Q×h}, B_Q(x_t) ∈ R^{R_Q×d_h}   (reshape, rank-major rows)
Q_t = (1/R_Q) A_Q(x_t)^⊤ B_Q(x_t) ∈ R^{h×d_h}
```

Scaled dot-product attention (eqs. 3.2–3.3), per head `i`:

```
head_i = Softmax( Q_i K_i^⊤ / √d_h ) V_i ,   Q_i, K_i, V_i ∈ R^{T×d_h}
TPA(Q,K,V) = Concat(head_1, …, head_h) W^O ,  W^O ∈ R^{(h·d_h)×d_model}
```

RoPE integration (§3.2, Theorem 3.1): RoPE acts on the head-width axis only, so
it applies row-wise to the `B` factors:

```
B̃_K(x_t) := RoPE_t(B_K(x_t)) = B_K(x_t) T_t        (eq. 3.5, pre-rotation)
RoPE_t(Q_t) = Q_t T_t = (1/R_Q) A_Q(x_t)^⊤ B̃_Q(x_t)  (eq. 3.6)
RoPE relative-position identity preserved:  Q̃_t K̃_s^⊤ = Q_t T_{t−s} K_s^⊤
```

KV cache (§3.3): per past token cache `A_K(x_t) ∈ R^{R_K×h}`,
`B̃_K(x_t) ∈ R^{R_K×d_h}` (pre-rotated), `A_V(x_t) ∈ R^{R_V×h}`,
`B_V(x_t) ∈ R^{R_V×d_h}` — `(R_K+R_V)(h+d_h)` floats/token vs `2 h d_h`.

### Pseudocode

Parallel/training (materialized path — what we implement):

```
A_Q, B_Q = W_aQ x, W_bQ x          # (B,T,R_Q·h), (B,T,R_Q·d_h)
A_K, B_K, A_V, B_V = …             # same for K, V
Q = einsum("btrh,btrd->bthd", A_Q.view(B,T,R_Q,h), B_Q.view(B,T,R_Q,d_h)) / R_Q
K, V = …                           # same, (B,h,T,d_h)
apply PE (RoPE) to materialized q,k        # repo convention (tucker/kpa)
scores = Q K^⊤ / √d_h ; causal mask (decoder)
out = softmax(scores) V → concat heads → W^O
```

Recurrent/decode (FlashTPA, Appendix A/B — specialized, no Q/K/V
materialization; **optional optimization, not implemented in v1**):

```
P[b,q,k,r,s]  = ⟨B_Q[b,q,r,:], B_K[b,k,s,:]⟩
L[b,h,q,k]    = (s_Q s_K/√D) Σ_{r,s} A_Q[b,q,r,h] A_K[b,k,s,h] P[b,q,k,r,s]
α             = Softmax_k(L + mask)
O[b,h,q,e]    = s_V Σ_k Σ_u α[b,h,q,k] A_V[b,k,u,h] B_V[b,k,u,e]
```

### Parameters (Table 1, exact formulas)

Learned tensors per attention layer (all bias-free):

| Tensor | Shape | Count |
|---|---|---|
| `W^{aQ}` | `(R_Q·h) × d_model` | `R_Q·h·d_model` |
| `W^{bQ}` | `(R_Q·d_h) × d_model` | `R_Q·d_h·d_model` |
| `W^{aK}, W^{bK}` | `(R_K·h / R_K·d_h) × d_model` | `R_K(h+d_h)d_model` |
| `W^{aV}, W^{bV}` | `(R_V·h / R_V·d_h) × d_model` | `R_V(h+d_h)d_model` |
| `W^O` | `d_model × (h·d_h)` | `d_model·h·d_h` |

**Parameter-count formula** (Table 1):

```
params(tpa) = d_model · (R_Q + R_K + R_V)(h + d_h) + d_model · h · d_h
```

(in this repo's symbols: `H·(R_Q+R_K+R_V)(nH + hd) + H²·hd` with
`H = hidden_size`, `nH = num_heads`, `hd = H/nH`; no biases, no `num_kv_heads`
split — TPA uses `h` query heads and `h` factor streams). This is the formula
`full_tests/param_count_check.py` and `ft-param-estimator.js` must agree on.

### Knobs

| Key | Type | Default | Constraint | Notes |
|---|---|---|---|---|
| `model.attention.tpa.query_rank` (`R_Q`) | int ≥ 1 | 8 | positive | paper's T6-75.2M setting `R_Q=8`; large-scale uses lower |
| `model.attention.tpa.key_rank` (`R_K`) | int ≥ 1 | 2 | positive | paper: "rank 1 or 2" typical |
| `model.attention.tpa.value_rank` (`R_V`) | int ≥ 1 | 2 | positive | same |

No head/feature split knobs (unlike `kpa`): TPA factors are vectors, so
`h1 = num_heads, d1 = 1, h2 = 1, d2 = head_dim` implicitly.

**Assumption to verify at implementation time**: exact small-model T6 ranks
(Appendix G) — the extract confirmed `R_Q=8, R_K=2, R_V=2` at the 75.2M scale
and `R_K=R_V=1` at 1.5B; confirm the 150M/350M settings before freezing
defaults (adjust the spec `defaults`, the YAML defaults and the annex wording
together if different).

### State & cache

Four per-token latent streams `{A_K, B̃_K, A_V, B_V}` of `R_K·h + R_K·d_h +
R_V·h + R_V·d_h` floats. Compatibility class: **(ii) compressed/latent
attention → `LatentKVSlot`** (`src/model/cache/capabilities.py` L26), exactly
the `kpa` layout (`kpa` is its matrix generalization). Incremental decode:
append factors, re-expand K/V on demand (`expand_fn`), exact parity.

### Complexity (for `tab:all-mixers`)

- Training (materialized, our path): `Θ(B T h d (R_Q+R_K+R_V))` factor
  materialization + `2Θ(B h T² d_h)` attention → **`O(n² d)`** overall.
- Specialized/FlashTPA training: `Θ(BT²R_QR_K d_h) + Θ(BHT²R_QR_K) + Θ(BHT²R_V E)`.
- Inference decode per step vs cache `M`: `Θ(M R_QR_K d_h) + Θ(h M R_QR_K) + Θ(h M R_V E)`; KV cache `(R_K+R_V)(h+d_h)` per token (vs `2 h d_h`).

### Trainability

Fully trainable from scratch (T6 pretrains with plain next-token prediction);
both causal (decoder) and bidirectional (encoder) work — the factorization is
mask-agnostic. **Not** training-free: no `TRAINING_FREE_LAYERS`/`dense_forward`
surrogate needed (hard rule 5 does not apply).

### Positional encoding

Consumes positional information (softmax attention over token pairs); RoPE
applies to the `d_h` axis. Repo convention (tucker/kpa precedent): apply the
shared PE to the *materialized* per-head Q/K via `apply_pe_to_qk`; the paper's
pre-rotated `B̃_K` optimization is algebraically equivalent (Theorem 3.1) and is
recorded in the docstring/annex as a decode-time optimization, not implemented
in v1. → `needs_pos_encoder=True`, `use_pe.tpa_use_pe` default true.

### Mode constraints

Encoder + decoder both valid (causal mask in decoder mode, `kpa.py` L360–366
pattern). Text tasks (mlm/sbert/causal_lm/classification). No vision/audio
restriction (sequence mixer), though examples target text encoder MLM.

### BibTeX data

Zhang, Y., Liu, Y., Yuan, H., Qin, Z., Yuan, Y., Gu, Q., & Yao, A. C.-C.
(2025). "Tensor Product Attention Is All You Need". arXiv:2501.06425
(NeurIPS 2025, Spotlight). Key: `zhang_tpa_2025`.

---

## 3. Decisions (Phase 3)

| Decision | Value | Evidence |
|---|---|---|
| Registry name / class | `tpa` / `TPAAttention` | paper short name; no `_attn` suffix |
| Family dir / paper label | `src/model/attention/latent/` / **Latent** | README L173 "Latent (12)" lists `kpa`, `tucker`, `mla`…; architecture §7.4 |
| Closest peer | `latent/kpa.py` (KPA, arXiv:2610.12657) | kpa docstring L31–32: vector special case recovers TPA; mirror constructor, `_materialize`, `make_cache_slot` (axes `{a_k,b_k,a_v,b_v}`), `cached_latent_attention` branch, `apply_pe_to_qk`, causal mask, `module_softmax`, BitLinear |
| Knobs | `query_rank=8, key_rank=2, value_rank=2` in shared `schema/_model/_attention_latent.yaml#/properties/tpa` | Table 1 + T6 configs (Appendix G to re-verify) |
| `needs_pos_encoder` | `True` | Theorem 3.1; softmax attention; tucker/kpa precedent |
| KV-cache decision | **(1) simple decode form**: `cache=CacheSpec(supported=True, slot_type="latent_kv", notes=…)`, `cache_implemented=True`, incremental branch in the module + `"tpa"` in `IMPLEMENTED_MIXERS` → parity suite auto-covers | §3.3 factorized cache; exact expansion; `capabilities.py` F1 latent list |
| `qk_norm_supported` / `qk_clip_supported` | `True` / `True` | operate on materialized Q/K, same as kpa |
| Training-free | No | fully trainable |
| Example | `configs/examples/latent_tpa_adamw.yaml` (user-approved; **no** full `configs/tpa.yaml`) | mirrors `latent_kpa_adamw.yaml` |
| Census bumps | enum **53 → 54**; README **Latent (12) → (13)**; `tests/test_schema_attention_layers.py` L178 `assertEqual(len(CACHE_SPECS), 53)` → `54`; architecture §7.4 "53 enum values in 9 families" → "54 enum values in 9 families" | actual repo numbers read 2026-11-12 |

---

## 4. File-by-file plan table

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/attention/latent/tpa.py` | New module, `kpa.py` skeleton. Docstring cites arXiv:2501.06425, states eq. 3.1 in `.. math::`, notes (a) the pre-rotated `B̃_K` RoPE optimization (Theorem 3.1) and (b) the relation to `kpa` (TPA = vector/CP special case). `from __future__ import annotations`, Google docstrings, CPU-safe. `__init__(self, config, pos_encoder=None, params=None)`: `params = mixer_params(config, "tpa", params)`; `query_rank=int(params.value("query_rank", 8))`, `key_rank=…2`, `value_rank=…2`, each `>0` else `ValueError`; `hidden_size % num_heads == 0` check. Projections (BitLinear iff `config.bitnet.enabled`, `bias=False`): `w_aq: H→R_Q·num_heads`, `w_bq: H→R_Q·head_dim`, and same for K/V. `make_cache_slot`: `LatentKVSlot(meta, axes={"a_k","b_k","a_v","b_v": "token"}, **cache.codec_slot_kwargs())`. `forward(x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)` (S1/S2 per §7.13): cached branch — compute q, `_expand(c_all)` materializes k/v from the four streams, `cached_latent_attention(slot, q, {"a_k","b_k","a_v","b_v"}, scale=…, expand_fn=_expand, pe=pe, pe_type, use_pe, x_for_pe=x, logical_layer_idx, dropout, causal=True, use_fixed_softmax=self.use_fixed_softmax)`; uncached branch — `apply_pe_to_qk`, `scores = q @ kᵀ · d_h^{-1/2}`, `apply_pe_to_scores`, decoder causal `triu` mask (kpa L360–366), `module_softmax(self, scores, dim=-1)`, dropout, `out_proj`. `_materialize(a_flat, b_flat, rank)`: `einsum("btrh,btrd->bthd", a.view(B,T,R,h), b.view(B,T,R,hd)) / R` → `(B,h,T,hd)`. |
| 2 | `src/model/attention/latent/__init__.py` + `src/model/attention/__init__.py` | Add `from .tpa import TPAttention` / `TPAttention` to `__all__` next to the `KPAAttention` entries. |
| 3 | *(none)* | `hybrid_layer.py` untouched — dispatch reads the registry; TPA is not training-free so `TRAINING_FREE_LAYERS` (derived, ~L80) is unaffected. |
| 4 | `src/registry/builtin/attention.py` | After the `kpa` spec (~L597–623) insert the single declaration: `MixerSpec(name="tpa", builder="src.model.attention.latent.tpa:TPAttention", description="Tensor Product Attention — CP-style sums of contextual vector-factor outer products for Q/K/V; factorized KV cache; RoPE-compatible (paper: arxiv:2501.06425)", description_es="Atención de Producto Tensorial — sumas CP de productos externos de factores vectoriales contextuales para Q/K/V; caché KV factorizada; compatible con RoPE (paper: arxiv:2501.06425)", references=("2501.06425",), params_schema="schema/_model/_attention_latent.yaml#/properties/tpa", defaults={"query_rank": 8, "key_rank": 2, "value_rank": 2}, cache=CacheSpec(supported=True, slot_type="latent_kv", notes="TPA (arXiv:2501.06425); per-token factor cache {A^K, B^K(rot), A^V, B^V} of (R_K+R_V)(h+d_h) floats; kpa-style latent slot, exact prefill/decode parity."), cache_implemented=True, qk_norm_supported=True, qk_clip_supported=True, needs_pos_encoder=True)`. Enum, `model.attention.tpa`, `use_pe.tpa_use_pe`, pickers all derive. |
| 5 | *(none)* | Flattener removed in 1.2; no rename entry exists. |
| 6 | `src/schema/_model/_dims.yaml` | One 4-column row in the EN bilingual table after `kpa` (~L262): `` | `tpa` | Tensor Product Attention — CP sums of contextual vector-factor outer products, factorized KV cache (paper: arxiv:2501.06425) | ✅ | ✅ | `` and the ES counterpart (~L326): `` | `tpa` | Atención de Producto Tensorial — sumas CP de productos externos de factores vectoriales contextuales, caché KV factorizada (paper: arxiv:2501.06425) | ✅ | ✅ | ``. Enum list untouched (registry-injected). |
| 7 | `src/schema/_model/_attention_latent.yaml` | Add `tpa:` block after the `kpa` block (same style as `kpa` at ~L261+): `type: object`, `additionalProperties: false`, `title: "Tensor Product Attention"`, `title_es: "Atención de Producto Tensorial"`, bilingual `description`/`description_es` stating eq. `Q_t = (1/R_Q) Σ_r a_r ⊗ b_r`, cache `(R_K+R_V)(h+d_h)` floats/token, that `kpa` with `h1=num_heads, d1=1, h2=1, d2=head_dim` recovers it, and `(paper: arxiv:2501.06425)`; three integer properties `query_rank` (default 8), `key_rank` (default 2), `value_rank` (default 2), each `minimum: 1`, bilingual titles/descriptions with `**References:** (paper: arxiv:2501.06425)` and `examples: [8]/[2]/[2]`. |
| 8 | *(none)* | `_model.yaml` is regenerated by the assembler. |
| 9 | *(none)* | `_positional_encoding.yaml` gets `use_pe.tpa_use_pe` injected from `needs_pos_encoder`. |
| 10 | `src/model/cache/capabilities.py` | Add `"tpa",  # F1 (vector-factor streams, expand + RoPE per step)` to `IMPLEMENTED_MIXERS` in the F1 latent block next to `"kpa"` (~L69). `CACHE_SPECS` derives from the registry — no other edit. |
| 11 | `configs/examples/latent_tpa_adamw.yaml` | Byte-level copy of `configs/examples/latent_kpa_adamw.yaml` with: `layer_pattern: [tpa]`, `attention.tpa: {query_rank: 8, key_rank: 2, value_rank: 2}` replacing the `kpa` block (keep the unrelated `retnet.heads: 8` sub-block as the kpa example does), `csv_log_path: training_metrics_es_arch_tpa_adamw.csv`. Must pass `load_training_config`. |
| 12 | *(skipped)* | User chose example-only; no `configs/tpa.yaml`. |
| 13 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (~L272, after `kpa`'s comment block): `# Tensor Product Attention (arXiv:2501.06425): CP vector-factor Q/K/V` / `# with a per-token latent factor cache; trains densely in both modes.` / `"tpa",`. Auto-generates `attn_single_tpa` + all `attn_<a>__tpa` / `attn_tpa__<b>` pairs. |
| 14 | `tests/test_schema_attention_layers.py` | Add `"tpa"` to the hand-expected enum set and bump L178 `self.assertEqual(len(CACHE_SPECS), 53)` → `54` (docstring L163 "…(53)" → "(54)"). |
| 15 | `tests/test_latent_attention.py` | New `TestTPAAttention` class mirroring the KPA tests: `test_forward_encoder_shape`, `test_forward_decoder_causal`, `test_ranks_honored` (build with non-default ranks, assert param shapes), `test_no_nan_inf`, `test_bitnet_linear_swap`, `test_cache_parity` (small prefill+decode vs full forward via the manager, if the KPA class has one — mirror it). |
| 16 | `tests/test_bitnet_coverage.py` | Add `"tpa"` to the mixer list next to `kpa`. |
| 17 | *(none)* | `tests/test_kv_cache_mixers.py` parametrizes from `IMPLEMENTED_MIXERS` — covered by row 10. |
| 18 | `docs/paper/figures/mixers/tpa.mmd` | House-style mermaid (copy header/init from `kpa.mmd`): `flowchart TD`; input `x_t [B,T,H]` → Linear → `A_Q [B,T,R_Q,h]` / `B_Q [B,T,R_Q,d_h]` (and K/V analogues); RoPE node on `B_Q`/`B_K` (dotted, "row-wise"); `Σ (1/R) a_r⊗b_r` materialization nodes → `Q,K,V [B,h,T,d_h]` → Scaled Dot-Product Attention → Concat → Linear. Dotted edge for the cached `{A_K,B̃_K,A_V,B_V}` streams annotated `(R_K+R_V)(h+d_h)/token`. |
| 19 | `docs/paper/figures/mixer-taxonomy.mmd` | Add the `tpa` node to the Latent family chain (next to `kpa`) and bump the family count label (12→13). |
| 20 | `docs/paper/figures/latent-map.mmd` | Add the `tpa` node linked to `kpa`/`tucker` (vector-factor CP corner). |
| 21 | `docs/bibliography/attention_types.bib` | `@misc{zhang_tpa_2025, …}` with the seven authors, title "Tensor Product Attention Is All You Need", year 2025, `eprint = {2501.06425}`, `archivePrefix = {arXiv}`, `note = {NeurIPS 2025 Spotlight}` (style follows `klein_tucker_2026` at L333). |
| 22 | `docs/paper/appendices/annex-3-latent-attention.tex` | New `\subsection{Tensor Product Attention (TPA)}` after the KPA subsection (KPA starts L458; take the next free number — GMA and KPA are the last ones, verify at edit time): `\fitfigure{figures/mixers/tpa.pdf}` + `\label{fig:mixer-tpa}` + `\label{sec:tpa}`; `\subsubsection{Mathematical Foundation}` with eqs. 3.1–3.6 verbatim (factorization, latent maps, `Q_t = (1/R_Q)A_Q^⊤B_Q`, RoPE Theorem 3.1, cache `(R_K+R_V)(h+d_h)`); `\subsubsection{Algorithmic Pseudocode}` (materialized training form + FlashTPA factor-space form, Algorithm 1 of Appendix A); `\subsubsection{Key Characteristics}` with the strength/limitation table (strengths: order-of-magnitude KV reduction, quality ≥ GQA/MLA, native RoPE; limitations: rank hyperparameters, materialized path costs the extra `Θ(BThd(R_Q+R_K+R_V))`, FlashTPA needs a custom kernel); `\citep{zhang_tpa_2026→zhang_tpa_2025}`; explicitly state TPA is the vector/CP special case of KPA (`\citep{nikpour_kpa_2026}`) and the activation-space counterpart of Tucker weight factorization (`\citep{klein_tucker_2026}`); extend the annex's closing unified-principle list if present. |
| 23 | `docs/paper/sections/04-architecture-taxonomy.tex` | Latent family paragraph: one sentence for TPA (CP vector factors, `(R_K+R_V)(h+d_h)` cache, NeurIPS 2025) + family count 12→13. Grep the old count and update every occurrence. |
| 24 | `docs/paper/appendices/annex-summary-tables.tex` | `tab:all-mixers` row after `kpa` (L91): `tpa & Latent & $\mathcal{O}(n^2 d)$ & $\mathcal{O}((R_K{+}R_V)(h{+}d_h))$ per token & CP vector-factor Q/K/V; factorized KV cache; RoPE-compatible; vector special case of KPA & \citep{zhang_tpa_2025} \\`; bump the Latent family header count and the intro census total. |
| 25 | `docs/paper/appendices/annex-schema.tex` | After the `model.attention.kpa.*` rows (~L98–106): `\multicolumn{3}{l}{\textit{model.attention.tpa.*}} \\` + `attention.tpa.query_rank & int $\ge 1$ & CP rank $R_Q$ of the query factors. Default: 8. \\`, `attention.tpa.key_rank & int $\ge 1$ & CP rank $R_K$ of the key factors (cached). Default: 2. \\`, `attention.tpa.value_rank & int $\ge 1$ & CP rank $R_V$ of the value factors (cached). Default: 2. \\`. |
| 26 | `docs/paper-es/appendices/annex-3-*.tex` | Faithful Spanish translation of #22 (same structure; figures via `../paper/figures/mixers/tpa.pdf`). Locate by grepping `kpa` in `docs/paper-es/appendices/`. |
| 27 | `docs/paper-es/sections/01-*.tex`, `04-*.tex`, `annex-summary-tables.tex`, `annex-schema.tex` | Spanish counterparts of #23–#25 + the global census sentence in section 01 (abstract-level mixer count). |
| 28 | `README.md`, `docs/architecture.md`, `configs/README.md` | README "Mixer Categories": **Latent (12) → (13)**, add `tpa` to the list + descriptor (L173); also L291 "(50 architectures / 9 families)" → check the current number on the branch and bump consistently (grep `50 architectures`, `53`, `12)`). architecture §7.4: "53 enum values" → "54 enum values", add the `| tpa | Latent | tpa.* | yes |` row after `kpa`; §7.13 cache list: add `tpa` to the latent-KV group. `configs/README.md`: add `tpa` to the latent family list + knob pointer. |
| 29 | `docs/source/api/model/attention/latent/tpa.rst` + `latent/index.rst` | `tpa.rst` mirroring `kpa.rst`: title `Tensor Product Attention`, `.. automodule:: src.model.attention.latent.tpa` with `:members:`, `:undoc-members:`, `:show-inheritance:`; add `tpa` to the `latent/index.rst` toctree. |
| 30 | `docs/schema.assembled.yaml` | `uv run --extra cpu --extra train python -m src.registry.snapshot --out docs/schema.assembled.yaml` (implementation time). |
| 31 | `docs/paper/appendices/annex-15-kv-cache.tex` + `docs/paper-es/appendices/annex-23-cache-kv.tex` | Extend the latent-cache discussion (EN ~L180): "TPA caches the four vector-factor streams $\{A^K, B^K, A^V, B^V\}$ per token at $(R_K+R_V)(h+d_h)$ floats — the vector special case of KPA's Kronecker streams \citep{zhang_tpa_2025}." + ES counterpart. |
| 32 | `docs/paper/sections/01-introduction.tex` (+ES), `10-*.tex`, `11-*.tex` | Global mixer census bump (grep the current total, e.g. "53 mixers"/"53 architectures", → 54) in EN + ES; check 10/11 for recurring census mentions. |

**Hard rules restated**: the schema is the contract (`additionalProperties: false`); the registry is the single source of truth (D13/D14) — the spec is the *only* hand declaration; the runtime config is nested only (D15); `hidden_size % num_heads == 0`, `num_kv_heads` divides `num_heads`, `len(layer_pattern) == num_layers`; inference never fails because of the cache (D5 — though TPA is not training-free); the papers are the theory source of truth (D11) — annex equations must match arXiv:2501.06425 and `paper-es` must translate them faithfully.

---

## 5. Ordered execution checklist

1. ☐ Row 1–2: module + exports.
2. ☐ Row 4: `MixerSpec` in `src/registry/builtin/attention.py`.
3. ☐ Rows 6–7: `_dims.yaml` bilingual rows + `_attention_latent.yaml` knob block.
4. ☐ Row 10: `IMPLEMENTED_MIXERS` (cache parity branch is inside row 1's module).
5. ☐ Row 11: example YAML; `uv run --extra cpu --extra train python -c "from src.training.config_loader import load_training_config; load_training_config('configs/examples/latent_tpa_adamw.yaml')"` sanity check.
6. ☐ Rows 14–16: tests.
7. ☐ Targeted gates:
```bash
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_latent_attention.py \
  tests/test_bitnet_coverage.py tests/test_kv_cache_mixers.py \
  tests/test_registry_core.py tests/test_registry_binding.py \
  tests/test_schema_assembly.py \
  -v --tb=short -p no:warnings
```
8. ☐ Full suite:
```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```
9. ☐ e2e enumeration + **targeted real-data training smoke**:
```bash
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F tpa
# real-data training (default --realonall; first run downloads the Borges corpus/tokenizer)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only tpa
# minimal fallback: solo combo only
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_tpa
# offline synthetic fallback
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only tpa --no-realonall
# sizing: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
```
   Expected: every selected combo PASS with finite loss on real corpus windows; NaN/Inf, schema rejection or timeout is a blocker. Row 13 (`ATTENTIONS`) precedes this step.
10. ☐ Rows 18–20: figures, then `( cd docs/paper/figures && ./build-figures.sh both )` (chromium available at `/usr/bin/chromium`).
11. ☐ Rows 21–28, 31–32: docs EN/ES.
12. ☐ Row 29 + `bash docs/build_docs.sh`.
13. ☐ Param ground truth + JS estimator (after row 11):
```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01
```
14. ☐ Row 30: snapshot regeneration.
15. ☐ PDF rebuild (`pdflatex`/`bibtex` available):
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
   Verify: no `Fatal error`, no `Citation .* undefined`, PDFs newer than the edited `.tex`/`.bbl`, `\citep{zhang_tpa_2025}` resolves, `git status --short docs/paper docs/paper-es` shows only intended files.
16. ☐ Website sync (§6), then commit/push (§7).

Preconditions/fallbacks: `pdflatex`, `bibtex`, `chromium`, `node` all present on this machine (verified read-only). If the HF Hub is unreachable for `--realonall`, record the `--no-realonall` PASS and re-run the real-data pass later.

---

## 6. Website sync (planned, not executed)

Per `.opencode/command/sync-website.md`:

1. Refresh the mirror; copy: regenerated `docs/schema.assembled.yaml`, `src/schema/**`, `src/registry/**`, `configs/examples/latent_tpa_adamw.yaml`, `docs/paper/**` + `docs/paper-es/**` (tex/bbl/pdf + **new** `figures/mixers/tpa.svg`, `latent-map.svg`, `mixer-taxonomy.svg`), `docs/bibliography/**`, `docs/architecture.md`.
2. **Site-only edits**:
   - `ft-diagram.js`: `CATS["tpa"]` (Latent class) + `SUBSTRUCT["tpa"]` tensor-flow template (or string alias to the `kpa` template with the Kronecker splits collapsed to `h`/`d_h`); no new `STYLES` class (existing Latent family).
   - `ft-param-estimator.js`: `mixerParams()` case `tpa`: `dModel * (rQ + rK + rV) * (nH + hd) + dModel * nH * hd` — knob-aware via the `tpa.query_rank/key_rank/value_rank` fields.
   - `index.html`: grep `kpa`; extend every hardcoded mixer list found (census counts included).
3. Paper-tab HTML rebuild (`flatten_tex.py --web` + `pandoc` + `postprocess_html.py`, en + es) + top-level `paper/en.pdf` / `es.pdf`.
4. Validation Gate: `param_count_check.py` → `0 failures` (intentional `param_truth.json` diff), estimator `--tol 0.01` → `ALL PASS`, JSON round-trip, CLI parse, `schema-refs`/`example-yaml`/`cli-flags` probes PASS.
5. Mirror stays uncommitted until §7.

## 7. Commit plan (planned, not executed)

1. Main repo — `feat(attention): add Tensor Product Attention mixer (2501.06425)`; body covering module, registry spec, bilingual schema rows + knob fragment, latent-slot cache decision with parity, example, tests, e2e `ATTENTIONS` + real-data smoke result, figures (mmd + pdf/svg), docs EN/ES (annex-3, summary tables, schema annex, kv-cache annex, bibliography, sections 01/04/10/11, README/architecture/configs README, Sphinx rst), regenerated `docs/schema.assembled.yaml`, rebuilt PDFs. Stage exactly:
   `src/model/attention/latent/tpa.py src/model/attention/latent/__init__.py src/model/attention/__init__.py src/registry/builtin/attention.py src/model/cache/capabilities.py src/schema/_model/_dims.yaml src/schema/_model/_attention_latent.yaml configs/examples/latent_tpa_adamw.yaml full_tests/run_e2e.py tests/test_schema_attention_layers.py tests/test_latent_attention.py tests/test_bitnet_coverage.py docs/paper/figures/mixers/tpa.mmd docs/paper/figures/mixers/tpa.pdf docs/paper/figures/mixers/tpa.svg docs/paper/figures/mixer-taxonomy.* docs/paper/figures/latent-map.* docs/bibliography/attention_types.bib docs/paper/appendices/annex-3-latent-attention.tex docs/paper/appendices/annex-summary-tables.tex docs/paper/appendices/annex-schema.tex docs/paper/appendices/annex-15-kv-cache.tex docs/paper/sections/ docs/paper-es/**(counterparts) docs/paper/paper.pdf docs/paper/paper.bbl docs/paper-es/paper-es.pdf docs/paper-es/paper-es.bbl README.md docs/architecture.md configs/README.md docs/source/api/model/attention/latent/ docs/schema.assembled.yaml full_tests/param_truth.json` — never `*.aux *.log *.out *.toc *.blg`. Then `git push origin HEAD:main`.
2. Mirror — `git -C erickfmm.github.io …` commit `docs(site): sync Tensor Product Attention mixer (2501.06425)`, then push `origin HEAD:main`.

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted pytest | §5 step 7 | all green |
| Full pytest | §5 step 8 | all green (incl. auto-parametrized `test_kv_cache_mixers.py::…[tpa]` parity) |
| e2e enumeration | `--category attn --list-combos \| grep -F tpa` | `attn_single_tpa` + every `attn_<a>__tpa` / `attn_tpa__<b>` pair |
| e2e real-data smoke | `--category attn --only tpa` | all selected combos PASS, finite loss (Borges corpus) |
| e2e solo smoke | `--only attn_single_tpa` | PASS |
| e2e offline fallback | `--only tpa --no-realonall` | PASS |
| Figures | `build-figures.sh both` | `tpa.pdf`/`tpa.svg`, updated `latent-map.*`, `mixer-taxonomy.*` |
| Param count | `param_count_check.py` | `0 failures`; TPA rows match `H(R_Q+R_K+R_V)(nH+hd)+H·nH·hd` |
| JS estimator | `param_estimate_check.mjs … --tol 0.01` | `ALL PASS` |
| Sphinx | `bash docs/build_docs.sh` | builds, `latent/tpa` page present |
| PDF EN/ES | §5 step 15 | no fatal/undefined-citation errors, new annex subsection + resolved `\citep{zhang_tpa_2025}` |
| KV parity | via full pytest | prefill+decode == full forward within tolerance |
| Mirror probes | §6 step 4 | all PASS |

## 9. Risks, assumptions and open questions

- **Rank defaults**: `R_Q=8, R_K=2, R_V=2` confirmed only for the 75.2M T6 config; verify Appendix G's small-model ranks before freezing (they only change defaults + docs, not the mechanism).
- **PE deviation**: repo applies RoPE to materialized Q/K (`tucker`/`kpa` precedent), the paper pre-rotates `B_K`. Algebraically equivalent (Theorem 3.1); must be stated in the annex. If bit-exactness with the reference implementation ever matters, pre-rotation is a follow-up.
- **Line drift**: all anchors read 2026-11-12 at `HEAD=f0f6516`; re-grep before editing.
- Naming collision: `tpa` not present anywhere in the registry/bibliography (checked); bib key `zhang_tpa_2025` unique.
- `num_kv_heads`: TPA ignores it (h factor streams); document that setting `num_kv_heads < num_heads` has no effect for `tpa` (as for `kpa`).
- FlashTPA specialized kernel deliberately not implemented (materialized path only) — record in Key Characteristics; complexity row uses `O(n² d)`.
- BitNet: BitLinear swap covered by `test_bitnet_coverage.py`; outer-product einsum is CPU-safe fp32.
- HF Hub reachability for the real-data smoke (fallback documented).
- Task/mode: no rule-6 restriction (text encoder/decoder both fine; not audio/vision-specific).

## 10. Out of scope

This command wrote only `.opencode/plan/add-mixer-2501.06425.md`. No code, schema, preset, test, figure, doc, PDF, mirror or git change was made; no pytest/e2e/LaTeX/uv command was executed.
