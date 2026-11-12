# Plan: add KPA mixer (arXiv:2610.12657)

**Paper**: Nikpour, Rajabzadeh, Dialameh, Partovi Nia, Omidi — *Kronecker Product
Attention: Memory-Efficient KV Compression via Structured Factorization*
(NeurIPS 2026 SLM-Agents Workshop, arXiv:2610.12657v1).

KPA attacks the autoregressive KV-cache bottleneck: instead of caching dense
per-token key/value matrices `K_t, V_t ∈ R^{h×d_h}`, it factorizes each as a
normalized sum of Kronecker products of two compact, token-dependent matrix
factors (`A ∈ R^{h1×d1}` head-partition, `B ∈ R^{h2×d2}` within-head), caching
only the factors — `(R_K+R_V)(h1·d1 + h2·d2)` floats per token vs
`2·h1·h2·d1·d2` for MHA (6.4×–10× compression). KPA strictly generalizes Tensor
Product Attention (rank-1 CP = degenerate Kronecker split) and introduces
Kronecker-Separable RoPE, which rotates only the `B` factor while preserving the
relative-position property — no MLA-style decoupled RoPE branch. It matches or
beats TPA/MHA/MQA/GQA/MLA at 125M–1.5B on FineWeb-Edu. It belongs in this repo
as a **Latent-family** mixer: fully trainable from scratch, softmax-attention
shaped, with a per-token factor cache that maps exactly onto the existing
`LatentKVSlot` machinery used by `tucker`/`mla`.

---

## 2. Paper extract

### Forward equations (verbatim, §2.1–2.5)

Kronecker product and mixed-product identity:

```
A ⊗ B = [a_11 B ... a_1n1 B; ...; a_m1 1 B ... a_m1 n1 B]           (Def. 2.1)
(M ⊗ N)(P ⊗ Q) = (MP) ⊗ (NQ)                                        (1)
```

Let `h = h1·h2`, `d_h = d1·d2`. For token `t` with hidden state `x_t ∈ R^{d_model}`:

```
Q_t = (1/R_Q) Σ_{r=1..R_Q} A^Q_r(x_t) ⊗ B^Q_r(x_t) ∈ R^{h×d_h}      (2)
K_t = (1/R_K) Σ_{r=1..R_K} A^K_r(x_t) ⊗ B^K_r(x_t) ∈ R^{h×d_h}      (3)
V_t = (1/R_V) Σ_{r=1..R_V} A^V_r(x_t) ⊗ B^V_r(x_t) ∈ R^{h×d_h}      (4)
A^K_r(x_t) = W^{A^K}_r x_t,   B^K_r(x_t) = W^{B^K}_r x_t            (5)
```

with `W^{A^K}_r ∈ R^{(h1 d1)×d_model}`, `W^{B^K}_r ∈ R^{(h2 d2)×d_model}`
(reshaped after multiplication); analogous maps for Q and V. In practice the
rank index is merged into aggregate weights `W^{A^K} ∈ R^{(R_K h1 d1)×d_model}`,
`W^{B^K} ∈ R^{(R_K h2 d2)×d_model}`, reshaped to `R_K × h1 × d1` and
`R_K × h2 × d2` — one batched projection per token per factor kind.

Degenerate split `h1=h, d1=1, h2=1, d2=d_h` recovers TPA exactly (rank-1 outer
products). `rank(A_r ⊗ B_r) = rank(A_r)·rank(B_r)` → strictly richer at matched
parameter budget.

**Factored score** (Prop. 2.2; heads indexed by pairs `(i1, i2)`, `i1 ∈ [h1]`,
`i2 ∈ [h2]`; scaled by `1/√d_h` before softmax in practice):

```
score_(i1,i2)(t,s) = (1/(R_Q R_K)) Σ_{r=1..R_Q} Σ_{r'=1..R_K}
    [ (A^Q_r)_{i1}^⊤ (A^K_{r'})_{i1} ]  ·  [ (B^Q_r)_{i2}^⊤ (B^K_{r'})_{i2} ]   (6)
```

(head-level × feature-level inner product; proof = mixed-product identity on
`(a⊗b)^⊤(c⊗d) = (a^⊤c)(b^⊤d)`). Output = standard
`softmax(scores)·V_t` with V from (4); scores can be computed without
materializing K (analogous to TPA), but the practical nanoGPT implementation
materializes the sums.

**Kronecker-Separable RoPE** (Def. 2.3 + Thm 2.4). `R^(2)_θ(t) ∈ R^{d2×d2}` is a
RoPE rotation with frequencies calibrated to `d2`:
`θ_j = 10000^{−2j/d2}, j = 0..d2/2−1`. The operator on `R^{d_h}` is
`R_KPA(t) = I_{d1} ⊗ R^(2)_θ(t)`. Then:

```
(I_{d1} ⊗ R^(2)_θ(t))((A^Q_r)_{i1} ⊗ (B^Q_r)_{i2}) = (A^Q_r)_{i1} ⊗ R^(2)_θ(t)(B^Q_r)_{i2}   (7)
(I_{d1} ⊗ R^(2)_θ(s))((A^K_{r'})_{i1} ⊗ (B^K_{r'})_{i2}) = (A^K_{r'})_{i1} ⊗ R^(2)_θ(s)(B^K_{r'})_{i2}   (8)
score_(i1,i2)(t,s) = (1/(R_Q R_K)) Σ_{r,r'} [(A^Q_r)^⊤_{i1}(A^K_{r'})_{i1}]
                                 · [(B^Q_r)^⊤_{i2} R^(2)_θ(s−t) (B^K_{r'})_{i2}]                (9)
```

using `R^(2)_θ(t)^⊤ R^(2)_θ(s) = R^(2)_θ(s−t)`. Consequences: only the `B^K`
factors are pre-rotated before caching; `A^K` is position-independent and cached
as-is; no decoupled RoPE branch.

**KV cache** (§2.5 / Appendix B): per token cache
`{A^K_r, B^K_r}_{r=1..R_K} ∪ {A^V_r, B^V_r}_{r=1..R_V}` (B^K pre-rotated). Cost
`(R_K+R_V)(h1 d1 + h2 d2)` vs `2 h1 h2 d1 d2` for MHA:

```
C_KPA = 2 h1 h2 d1 d2 / ((R_K+R_V)(h1 d1 + h2 d2))                  (10)
```

Compression grows with factor asymmetry.

### Algorithm (implementation forms)

*Parallel/training* (what our module implements — mirrors TPA-in-practice):
1. `A^Q = W^{A^Q} x → (B, n, R_Q, h1, d1)`; same for `B^Q (R_Q,h2,d2)`, `A^K/B^K (R_K,·)`, `A^V/B^V (R_V,·)`.
2. Materialize per-head Q/K/V: for head `(i1,i2)`,
   `Q[:, :, i1, i2, :] = (1/R_Q) Σ_r A^Q[:,:,r,i1,:] ⊗ B^Q[:,:,r,i2,:]`
   (einsum: `(B,n,R,h1,1,d1) * (B,n,R,1,h2,d2) → (B,n,R,h1,h2,d1,d2)`, sum over R, reshape to `(B, n, h1·h2, d1·d2)`).
3. Apply PE to Q,K (repo `pos_encoder`; see RoPE note below), scores
   `QK^⊤/√d_h`, softmax (`module_softmax`, honoring `use_fixed_softmax`),
   causal mask in decoder mode, `· V`, merge heads, `out_proj`, dropout.

*Recurrent/decode*: per step, project the new token's six factor sets, append
`{A^K, B^K_rope, A^V, B^V}` to the latent slot, reconstruct K/V rows for all
cached tokens (or use factored scores per (6)), one softmax row, output.

### Parameters (exact inventory; all bias-free)

| Tensor | Shape |
|---|---|
| `W^{A^Q}` | `(R_Q·h1·d1) × H` |
| `W^{B^Q}` | `(R_Q·h2·d2) × H` |
| `W^{A^K}` | `(R_K·h1·d1) × H` |
| `W^{B^K}` | `(R_K·h2·d2) × H` |
| `W^{A^V}` | `(R_V·h1·d1) × H` |
| `W^{B^V}` | `(R_V·h2·d2) × H` |
| `out_proj` | `H × H` (`nH·hd = H`) |

**Parameter-count formula** (with `H = hidden_size`, `nH = num_heads`,
`hd = H/nH`, `s = h1·d1 + h2·d2`):

```
P_kpa = H · (R_Q + R_K + R_V) · (h1·d1 + h2·d2) + H²
```

`ft-param-estimator.js` and `full_tests/param_count_check.py` must agree on this
(no biases; BitLinear has the same count).

### Knobs (spec `params_schema` leaves, `model.attention.kpa.*`)

| Key | Type | Default | Constraints |
|---|---|---|---|
| `h1` | int\|null | `null` (auto) | `h1·h2 == num_heads`; auto = most asymmetric split with larger `h1` (paper's best: `8×2`, `10×2`) |
| `h2` | int\|null | `null` (`num_heads // h1`) | `h1·h2 == num_heads` |
| `d1` | int | `8` (paper §App. C) | `d1·d2 == head_dim` |
| `d2` | int\|null | `null` (`head_dim // d1`) | `d1·d2 == head_dim` |
| `query_rank` | int | `6` (R_Q, §3.1) | ≥1 |
| `key_rank` | int | `2` (R_K, §3.1) | ≥1 |
| `value_rank` | int | `2` (R_V, §3.1) | ≥1 |

Auto-split rule (deterministic, testable): largest divisor `h1` of `num_heads`
such that `h1 ≤ num_heads // 2` when a non-trivial split exists, else
`h1 = num_heads, h2 = 1` (degenerate TPA-like split); `d2 = head_dim // d1`
must divide evenly or fall back to `d1 = 1, d2 = head_dim` with a warning.
Constraints validated in `__init__` with `ValueError` (peer style).

### State & cache

Per-token state = 6 factor sets; cached = 4 (`A^K, B^K(rotated), A^V, B^V`),
totaling `(R_K+R_V)·s` floats/token. Maps to compatibility class **latent-KV**
(`slot_type="latent_kv"`): per-token axes, exact prefill+decode parity like
`tucker` (`src/model/cache/capabilities.py` classes; tucker precedent). No ring
buffers, no fixed state.

### Complexity

- Training: projections `O(n·H·(R_Q+R_K+R_V)·s)` + materialization
  `O(n·(R_Q+R_K+R_V)·h·d_h)` + standard attention `O(n²·nH·hd)` — Big-O
  `O(n² d)` train like every Latent peer (row for `tab:all-mixers`:
  train `O(n² d)`, inference/memory `O((R_K+R_V)·s · n)` per token).
- Decode per step: `O(H·(R_Q+R_K+R_V)·s)` projection + `O(t·nH·hd)` attention.

### Trainability / PE / modes

- Trainable from scratch (§3.1: nanoGPT, SwiGLU, RoPE, FineWeb-Edu ~50B tokens,
  125M–1.5B). **Not** training-free → no `TRAINING_FREE_LAYERS`, no
  `dense_forward` surrogate.
- Consumes positional information (RoPE on Q/K) → `needs_pos_encoder=True`,
  `use_pe_for(config, "kpa", True)`.
- Paper is causal-decoder-focused, but the math is standard softmax attention —
  encoder (bidirectional) mode is a trivial superset (peer behavior). Both modes
  supported; no task restriction (rule 6 satisfied: no forced mode).

### BibTeX

```bibtex
@inproceedings{nikpour_kpa_2026,
  title     = {Kronecker Product Attention: Memory-Efficient KV Compression via Structured Factorization},
  author    = {Nikpour, Bahareh and Rajabzadeh, Hossein and Dialameh, Maryam and Partovi Nia, Vahid and Omidi, Parsa},
  booktitle = {NeurIPS 2026 Workshop on Small Language Models for Agentic Systems (SLM-Agents)},
  year      = {2026},
  eprint    = {2610.12657},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

## 3. Decisions

- **Name**: registry `kpa`, class `KPAAttention`, module
  `src/model/attention/latent/kpa.py`. No collision (grep of
  `src/registry/builtin/attention.py` shows no `kpa`; latent peers: `mla, gqla,
  mlra, tucker, iha, gta, mtla, cca, ccgqa, gma`).
- **Family**: `latent/` directory, paper family label **Latent** (no new family
  → no new STYLES class, no new family map).
- **Peer**: `latent/tucker.py` (constructor `(config, pos_encoder=None,
  params=None)`, `mixer_params(config, "kpa", params)`, `use_pe_for(config,
  "kpa", True)`, `BitLinear` switch on `config.bitnet.enabled`,
  `LatentKVSlot` + `cached_latent_attention`, `module_softmax`, decoder causal
  mask, forward call shape S1/S2 per architecture §7.13).
- **Knobs**: 7 leaves in the shared `src/schema/_model/_attention_latent.yaml`
  (tucker precedent: shared family file, not a new `_attention_kpa.yaml`);
  spec `params_schema="schema/_model/_attention_latent.yaml#/properties/kpa"`,
  `defaults={"h1": None, "h2": None, "d1": 8, "d2": None, "query_rank": 6,
  "key_rank": 2, "value_rank": 2}`.
- **`needs_pos_encoder`: `True`** — assembler auto-injects
  `use_pe.kpa_use_pe`; no `_positional_encoding.yaml` edit.
- **KV-cache: decision 1 (user-approved)** —
  `cache=CacheSpec(supported=True, slot_type="latent_kv", notes="KPA
  (arXiv:2610.12657); per-token Kronecker factor cache {A^K, B^K(rotated), A^V,
  B^V} of (R_K+R_V)(h1·d1+h2·d2) floats; tucker-style latent slot, exact
  parity.")`, `cache_implemented=True`, add `"kpa"` to `IMPLEMENTED_MIXERS`
  (`src/model/cache/capabilities.py`); `tests/test_kv_cache_mixers.py`
  auto-parametrizes.
- **Not training-free**; `qk_norm_supported=True`, `qk_clip_supported=True`
  (standard per-head Q/K after materialization).
- **Example**: `configs/examples/latent_kpa_adamw.yaml` (tiny encoder MLM).
  **No full preset** (user declined; D-C3 answer "No, example only").
- **Census bumps** (read from repo, do not trust other numbers):
  - `README.md` line 28: "51 mixer architectures — 50 mixer classes plus the
    `mamba` linear passthrough; … 51 `layer_pattern` values" → **52 / 51 / 52**.
  - `README.md` line 125: "**51** across 9 families" → **52**.
  - `README.md` line 173: "**Latent** (10)" → **(11)** + append `kpa` to list.
  - `docs/architecture.md` §7.4 (~line 390): "50 architectures / 9 families"
    census + latent row → **51**; §7.13 cache lists gain `kpa`.
  - `tests/test_schema_attention_layers.py` T12 docstring/assert "(50)" → 51.
  - `docs/paper/appendices/annex-summary-tables.tex` line 84
    `\textbf{Latent (10)}` → **(11)** + intro census; ES counterpart.
  - `mixer-taxonomy.mmd` latent-branch count label +1.
- **RoPE handling** (open point, see §9): default = apply the repo
  `pos_encoder` to the materialized per-head Q/K (exactly what `tucker` does);
  the paper's exact Kronecker-Separable schedule (`I_{d1} ⊗ R^{(2)}_θ` with
  frequencies over `d2`) is equivalent only when the base is recalibrated to
  `d2`. Annex must state the paper form; module docstring notes the
  calibration nuance.

---

## 4. File-by-file plan table

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/attention/latent/kpa.py` | New module. Docstring cites arXiv:2610.12657 with equations (2)–(10) mapped to code lines. `from __future__ import annotations`; Google-style docstrings; CPU-safe torch only. `class KPAAttention(nn.Module)` with `__init__(self, config, pos_encoder=None, params=None)`: `params = mixer_params(config, "kpa", params)`; read `h1/h2/d1/d2/query_rank/key_rank/value_rank` via `params.value(...)` with the auto-split fallback; validate `h1*h2 == num_heads`, `d1*d2 == head_dim`, `hidden_size % num_heads == 0` (`ValueError`); `proj_cls = BitLinear if config.bitnet.enabled else nn.Linear`; six bias-free projections `w_aq (H→R_Q·h1·d1)`, `w_bq (H→R_Q·h2·d2)`, `w_ak`, `w_bk`, `w_av`, `w_bv`, plus `out_proj (H→H)`; `self.scale = head_dim ** -0.5`; `self.use_pe = use_pe_for(config, "kpa", True)`; `self.use_fixed_softmax` consumed via `numerics.module_softmax` (attached by `hybrid_layer.py` ~line 176 — no edit). `forward(x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)`: factor projections → reshape → einsum materialization of Q/K/V (equations 2–4) → repo `pos_encoder` on q/k when `self.use_pe` → scores/`module_softmax`/causal-mask-in-decoder/`@ V` → merge → `out_proj` + dropout. `make_cache_slot` returning `LatentKVSlot(axes={"k_lat": "token", "v_lat": "token"}, ...)` storing the packed factor tensors; cached branch reconstructs K/V from all cached factors (prefill+decode parity). `dense_forward` NOT needed (trainable). |
| 2 | `src/model/attention/latent/__init__.py` + `src/model/attention/__init__.py` | Import `KPAAttention` + `__all__` entry, matching the `TuckerAttention` lines already there. |
| 3 | *(none)* | `src/model/hybrid_layer.py` needs **no** edit: dispatch reads the registry; `TRAINING_FREE_LAYERS` derives from the spec flag (kpa is not training-free anyway). |
| 4 | `src/registry/builtin/attention.py` | Insert `MixerSpec(name="kpa", builder="src.model.attention.latent.kpa:KPAAttention", description="Kronecker Product Attention — sums of Kronecker products of contextual matrix factors; strictly generalizes TPA; Kronecker-Separable RoPE (paper: arxiv:2610.12657)", description_es="Atención de Producto de Kronecker — sumas de productos de Kronecker de factores matriciales contextuales; generaliza estrictamente a TPA; RoPE Kronecker-separable (paper: arxiv:2610.12657)", references=("2610.12657",), params_schema="schema/_model/_attention_latent.yaml#/properties/kpa", defaults={"h1": None, "h2": None, "d1": 8, "d2": None, "query_rank": 6, "key_rank": 2, "value_rank": 2}, cache=CacheSpec(supported=True, slot_type="latent_kv", notes="KPA (arXiv:2610.12657); per-token Kronecker factor cache {A^K, B^K(rotated), A^V, B^V}; tucker-style latent slot, exact parity."), cache_implemented=True, qk_norm_supported=True, qk_clip_supported=True, needs_pos_encoder=True)` in the latent cluster (alphabetical position among the specs, near `tucker` at line ~966). **Single declaration** — enum, `model.attention.kpa` fragment and `use_pe.kpa_use_pe` all derive from it. |
| 5 | *(none)* | No flattener/`from_flat`/rename entry exists (D15). |
| 6 | `src/schema/_model/_dims.yaml` | One row in **both** bilingual tables (EN ~line 260 after the `tucker` row, ES ~line 322), 4 columns: `` `kpa` | Kronecker Product Attention — Kronecker-factorised K/V with Kronecker-Separable RoPE (paper: arxiv:2610.12657) | ✅ | ✅ `` (ES translation accordingly). The enum list itself is registry-injected — do not hand-edit it. |
| 7 | `src/schema/_model/_attention_latent.yaml` | New top-level `kpa:` property under `properties:` (~after the `tucker:` block ending ~line 268): `additionalProperties: false`, bilingual `title/title_es`, block descriptions (EN+ES) explaining equations (2)–(4), the factor cache cost `(R_K+R_V)(h1·d1+h2·d2)`/token and compression (10), "Only applies when `kpa` appears in `layer_pattern`", `**References:** (paper: arxiv:2610.12657)`; then 7 leaf properties (`h1`, `h2`, `d1`, `d2`, `query_rank`, `key_rank`, `value_rank`) each with type (`integer` / `["integer","null"]` for the nullables), bilingual titles/descriptions with defaults and constraints, mirroring the `tucker` rank leaves (~lines 202–230) as template. |
| 8 | *(none)* | `_model.yaml` and `_positional_encoding.yaml` are fully generated surfaces — the assembler rebuilds `model.attention.properties` from the spec fragment and injects `use_pe.kpa_use_pe`. |
| 9 | `src/model/cache/capabilities.py` | Add `"kpa"` to the hand-maintained `IMPLEMENTED_MIXERS` frozenset. `CACHE_SPECS` derives from the registry — no other edit. The incremental branch itself lives in the module (row 1). |
| 10 | `configs/examples/latent_kpa_adamw.yaml` | Tiny encoder MLM example (es_arch shape, `load_training_config`-clean): `model_class: frankenstein_text_encoder`, `dims: {vocab_size: 100, hidden_size: 64, num_layers: 2, num_heads: 4, layer_pattern: [kpa, kpa], mode: encoder}`, `attention: {kpa: {query_rank: 6, key_rank: 2, value_rank: 2}}`, rms_norm, gelu FFN 128, `positional_encoding: rope`, adamw group-prefixed params (copy the pattern from `configs/examples/pe_mixed_use_pe.yaml`), `training: {task: mlm, seed: 42, num_epochs: 1, batch_size: 2, max_length: 64, mlm_probability: 0.15, max_samples: 1000, cache_dir: ./temp_data/latent_kpa_cache, ...}`. Note `hidden_size=64, num_heads=4 → hd=16`, so the example pins `d1: 4, d2: 4` explicitly (default `d1=8` would not divide 16 — auto-fallback handles it, but be explicit in the example). |
| 11 | *(none)* | Full preset declined by user. |
| 12 | `full_tests/run_e2e.py` | Append `"kpa"` to `ATTENTIONS` (line ~273, in the latent cluster after `"tucker", "iha", "gta", "mtla", "cca", "ccgqa",`) with a one-line comment `# kpa: Kronecker Product Attention (arxiv:2610.12657), latent factor cache`. Auto-generates `attn_single_kpa` + all `attn_<a>__kpa` / `attn_kpa__<b>` pairs — no builder edits. |
| 13 | `tests/test_schema_attention_layers.py` | Add `"kpa"` to the expected enum set (line ~51 cluster) and `"kpa": {"h1", "h2", "d1", "d2", "query_rank", "key_rank", "value_rank"}` to `expected_per_mixer` (~line 97); bump the T12 count in `test_cache_specs_match_layer_pattern_enum` (docstring "(50)" → "(51)", and the numeric assertion if separate). |
| 14 | `tests/test_latent_attention.py` | New `TestKPAAttention` class: `test_builds_and_forward_encoder`, `test_forward_decoder_causal`, `test_output_shape`, `test_knobs_honored` (explicit `h1/h2/d1/d2`, ranks), `test_auto_split_default`, `test_invalid_split_raises` (`h1*h2 != num_heads` → ValueError), `test_no_nan_inf`, `test_bitnet_enabled_forward`, `test_param_count_formula` (assert `H·(R_Q+R_K+R_V)·(h1·d1+h2·d2) + H²` on a tiny config). |
| 15 | `tests/test_bitnet_coverage.py` | Add `kpa` to the mixer list (grep `tucker` for the list location). |
| 16 | *(none)* | `tests/test_kv_cache_mixers.py` parametrizes itself from `IMPLEMENTED_MIXERS` — parity coverage is automatic once row 9 lands. |
| 17 | `docs/paper/figures/mixers/kpa.mmd` | Full mermaid diagram in house style (copy the `cca.mmd`/`tucker` init header + `flowchart TD`): input `x (B,n,H)` → six projections `W^{A·}/W^{B·}` with shape labels `(R·h1·d1)×H` / `(R·h2·d2)×H` → factor tensors `A^· (R,h1,d1)`, `B^· (R,h2,d2)` → Kronecker-sum node `Q/K/V (B,n,h1·h2,d1·d2)` (⊗ with Σ over R, dotted normalization 1/R) → RoPE on Q/K (rotate-B-only note) → scores/softmax → ⊗V → out_proj → `y (B,n,H)`; side branch "KV cache: {A^K, B^Kᴿ, A^V, B^V} — (R_K+R_V)(h1·d1+h2·d2)/token". |
| 18 | `docs/paper/figures/mixer-taxonomy.mmd` | Extend the Latent family chain with a `kpa` node; bump the family count label (Latent 10 → 11). |
| 19 | `docs/paper/figures/latent-map.mmd` | Add the `kpa` node (family map exists for latent). |
| 20 | `docs/bibliography/attention_types.bib` | `nikpour_kpa_2026` entry (§2 above). |
| 21 | `docs/paper/appendices/annex-3-latent-attention.tex` | `\subsection{<next>. Kronecker Product Attention (kpa)}` numbered after the annex's last subsection (grep `tucker` in this file for placement): `\fitfigure{figures/mixers/kpa.pdf}` + `\label{fig:kpa}`; Mathematical Foundation with equations (1)–(10) verbatim from §2 of this plan (Kronecker def, factorized Q/K/V (2)–(5), factored score (6), Kronecker-Separable RoPE Def 2.3 + Thm 2.4 (7)–(9), cache cost and compression (10)); an algorithm float (parallel materialized form + decode factor-cache form); Empirical Performance (FineWeb-Edu ~50B tokens, 125M–1.5B, nine zero-shot benchmarks, Tables 1–4 numbers: 353M best 52.90 avg with h=(8×2) at 6.4×, 773M best 55.12 with h=(10×2) at 6.7×, 1.5B 10× compression beats TPA on 8/9) + strength/limitation `\citep{nikpour_kpa_2026}`; extend the annex's closing unified-principle list if present. |
| 22 | `docs/paper/sections/04-architecture-taxonomy.tex` | Latent family paragraph: one sentence on KPA + bump the family count (grep `Latent` / `10` in the file; quote the old-count line in the PR body). |
| 23 | `docs/paper/appendices/annex-summary-tables.tex` | `tab:all-mixers` row: `kpa & Latent & $\mathcal{O}(n^2 d)$ & $\mathcal{O}((R_K{+}R_V)(h_1 d_1{+}h_2 d_2) \cdot n)$ & Sum of Kronecker products of contextual factors; strictly generalizes TPA; Kronecker-Separable RoPE & \citep{nikpour_kpa_2026} \\`; header line 84 `\textbf{Latent (10)}` → `(11)`; intro census total +1. |
| 24 | `docs/paper/appendices/annex-schema.tex` | Mixer row for `model.attention.kpa` (7 knobs, types, defaults) matching the fragment. |
| 25 | `docs/paper-es/appendices/annex-3-latent-attention.tex` | Faithful Spanish translation of #21 (figures referenced from `../paper/figures/`). |
| 26 | `docs/paper-es/sections/01-*.tex`, `04-*.tex`, `annex-summary-tables.tex`, `annex-schema.tex` | Spanish counterparts of #22–#24 + the global census sentence in section 01. |
| 27 | `README.md`, `docs/architecture.md`, `configs/README.md` | Census pass: README lines 28, 125, 173 (+ `kpa` in the Latent list); architecture §7.4 census/row + §7.13 cache list gains `kpa`; configs/README latent family list. Grep the old counts (`51`, `50 mixer`, `Latent (10)`) everywhere before editing. |
| 28 | `docs/source/api/model/attention/latent/kpa.rst` + `latent/index.rst` | `automodule:: src.model.attention.latent.kpa` page + toctree entry (mirror `tucker`'s rst in the existing `latent/` dir). |
| 29 | `docs/schema.assembled.yaml` | Regenerate: `uv run --extra cpu --extra train python -m src.registry.snapshot --out docs/schema.assembled.yaml`. |
| 30 | `docs/paper/appendices/annex-23-kv-cache.tex` + `docs/paper-es/appendices/annex-23-cache-kv.tex` | Add `kpa` to the incremental-decode list (latent-KV class, tucker grouping; note the pre-rotated `B^K` cache invariant). *(EN kv-cache annex file is `annex-23-kv-cache.tex` in this repo; the plan-mode prompt's "annex-15" numbering is stale — locate by grepping `tucker`.)* |
| 31 | `docs/paper/sections/01-introduction.tex` (+ES), `10-*.tex`, `11-*.tex` | Global mixer-census bump wherever the total appears (grep the old total; FPA precedent). |

Hard rules restated: the schema is the contract (`additionalProperties: false`,
D1/D14); the `MixerSpec` is the single declaration (D13) — enum, fragment,
`use_pe` key derive from it; nested runtime config only (D15); invariants
`hidden_size % num_heads == 0`, `num_kv_heads | num_heads`,
`len(layer_pattern) == num_layers`; inference never fails because of the cache
(D5) — the uncached path must remain exact; papers are the theory source of
truth (D11) — annex equations verbatim from arXiv:2610.12657, `paper-es`
translates them faithfully; kpa is trainable so hard rule 5 does not apply.

---

## 5. Ordered execution checklist

Implementation order: rows 1–2 → 4 → 6–7 → 9–10 → 12 → tests 13–15(+16 auto) →
e2e smoke → figures 17–19 → docs 20–31 → Sphinx → PDFs → website → git.

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_latent_attention.py \
  tests/test_bitnet_coverage.py tests/test_kv_cache_mixers.py \
  tests/test_registry_core.py tests/test_registry_binding.py \
  tests/test_schema_assembly.py \
  -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

```bash
# e2e — enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F kpa

# e2e TARGETED SMOKE — ONLY kpa: actual training on REAL data
# (--realonall is the default; first invocation downloads the Borges
# tokenizer/corpus from the HF Hub; '--only kpa' matches attn_single_kpa
# plus every pair attn_<a>__kpa / attn_kpa__<b>)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only kpa

# minimal real-data run if the pair sweep is too heavy
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_kpa

# offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only kpa --no-realonall

# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke (real training box): swap the extra for --extra cu118 --extra train.
```

Expected: every selected combo PASS with finite training loss over real corpus
windows; run dir keeps `config.yaml` + metrics. NaN/Inf loss, schema rejection
or timeout = blocker.

```bash
# figures (chromium available at /usr/bin/chromium)
( cd docs/paper/figures && ./build-figures.sh both )

# param ground truth + JS estimator (after the example lands)
uv run --extra cpu --extra train python full_tests/param_count_check.py
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01

# Sphinx API docs
bash docs/build_docs.sh
```

```bash
# PDF rebuild, from the repository root — only at implementation time
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

Preconditions verified during planning: `pdflatex`, `bibtex`, `chromium`,
`node` all present. Fallbacks: if the HF Hub is unreachable, record the
`--no-realonall` PASS and re-run the real-data pass later; if mermaid rendering
fails, commit the `.mmd` and render before the website sync; if TeX breaks,
leave sources updated and rebuild on a TeX machine before release. PDF checks:
no `Fatal error`, no `Citation .* undefined`, `paper.pdf`/`paper-es.pdf` newer
than the edited `.tex`/`.bbl`, `\citep{nikpour_kpa_2026}` resolves,
`git status --short docs/paper docs/paper-es` shows only intended files.

---

## 6. Website sync (planned, not executed)

Per `.opencode/command/sync-website.md`:

1. Bootstrap/refresh the `erickfmm.github.io` mirror; copy per its file map:
   regenerated `docs/schema.assembled.yaml`, `src/schema/**`,
   `src/registry/**`, `configs/*.yaml` + `configs/README.md`,
   `configs/examples/**`, `docs/paper/**` + `docs/paper-es/**` (tex, bbl, pdf,
   and the **new `kpa.svg`** figure — sources/PDFs stay in the main repo),
   `docs/bibliography/**`, `docs/architecture.md`.
2. Site-only edits (mirror is the source of truth):
   - `ft-diagram.js`: add `CATS["kpa"] = "latent"` (or the exact category-key
     convention used for `tucker`/`mla`) and a `SUBSTRUCT["kpa"]`
     tensor-flow template mirroring `tucker`'s but with the six factor
     projections and the ⊗-sum core; no new `STYLES` class (existing Latent
     family).
   - `ft-param-estimator.js`: add the `mixerParams()` `kpa` case implementing
     exactly `H·(R_Q+R_K+R_V)·(h1·d1+h2·d2) + H²` (bias-free; knob-aware with
     the same auto-split defaults h1/h2/d2 as the spec).
   - `index.html`: grep `tucker`; extend every hardcoded mixer list found with
     `kpa`.
3. Paper tab HTML rebuild (`flatten_tex.py --web` + `pandoc` +
   `postprocess_html.py`, en + es) and top-level `paper/en.pdf` / `es.pdf`.
4. Validation Gate: `param_count_check.py` → `0 failures` (intentional
   `param_truth.json` diff), estimator `--tol 0.01` → `ALL PASS`, JSON
   round-trip script, CLI parse script, state-script probes (`schema-refs`,
   `example-yaml`, `cli-flags`, …) all PASS.
5. Mirror stays uncommitted until the commit phase.

---

## 7. Commit plan (planned, not executed)

Conventions: English, conventional tag, structured body; never stage
`*.aux *.log *.out *.toc *.blg`, caches or mermaid temp files;
`param_truth.json` changes are intentional.

1. Main repo — `feat(attention): add KPA mixer (arXiv:2610.12657)`; body
   covering module, registry spec, schema (bilingual `_dims.yaml` rows +
   `_attention_latent.yaml` knob block), latent_kv cache decision +
   `IMPLEMENTED_MIXERS`, example preset, tests (schema/latent/bitnet + auto
   kv-parity), e2e `ATTENTIONS` + real-data smoke result, figures (`kpa.mmd`
   + rendered pdf/svg + taxonomy/latent-map bumps), docs (annex-3 EN/ES,
   summary tables, schema annex, kv-cache annex, bibliography, sections
   01/04, README/architecture/configs-README census, Sphinx rst), regenerated
   `docs/schema.assembled.yaml`, rebuilt PDFs. Stage:
   `src/model/attention/latent/kpa.py src/model/attention/latent/__init__.py
   src/model/attention/__init__.py src/registry/builtin/attention.py
   src/schema/_model/_dims.yaml src/schema/_model/_attention_latent.yaml
   src/model/cache/capabilities.py configs/examples/latent_kpa_adamw.yaml
   full_tests/run_e2e.py full_tests/param_truth.json
   tests/test_schema_attention_layers.py tests/test_latent_attention.py
   tests/test_bitnet_coverage.py docs/paper/figures/mixers/kpa.*
   docs/paper/figures/mixer-taxonomy.* docs/paper/figures/latent-map.*
   docs/bibliography/attention_types.bib docs/paper/appendices/annex-3-latent-attention.tex
   docs/paper/appendices/annex-summary-tables.tex docs/paper/appendices/annex-schema.tex
   docs/paper/appendices/annex-23-kv-cache.tex docs/paper/sections/
   docs/paper-es/ docs/source/api/model/attention/latent/
   docs/schema.assembled.yaml README.md docs/architecture.md
   configs/README.md docs/paper/paper.pdf docs/paper/paper.bbl
   docs/paper-es/paper-es.pdf docs/paper-es/paper-es.bbl`
   (prune non-existent/unchanged paths at commit time). Then
   `git push origin HEAD:main`.
2. Mirror — `git -C erickfmm.github.io …` commit
   `docs(site): sync KPA mixer (arXiv:2610.12657)`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted pytest | the §5 targeted command | all green |
| Full pytest | full-suite command | all green (84+ files) |
| Schema assembly | `test_schema_assembly.py` | `kpa` in enum, fragment, `use_pe.kpa_use_pe` auto-injected |
| e2e enumeration | `--category attn --list-combos \| grep -F kpa` | `attn_single_kpa` + all pairs listed |
| e2e real-data smoke | `--category attn --only kpa` | all selected combos PASS, finite loss on Borges corpus |
| e2e minimal | `--only attn_single_kpa` | PASS |
| e2e offline | `--only kpa --no-realonall` | PASS (synthetic) |
| KV parity | `tests/test_kv_cache_mixers.py` | kpa params: full-forward == prefill+decode |
| Figures | `build-figures.sh both` | `kpa.pdf`/`kpa.svg` produced; taxonomy/latent-map updated |
| Param count | `full_tests/param_count_check.py` | `0 failures`; truth matches formula |
| Estimator | `param_estimate_check.mjs --tol 0.01` | `ALL PASS` |
| Sphinx | `bash docs/build_docs.sh` | kpa page built, no warnings about missing module |
| PDFs | §5 pdflatex sequence | no Fatal error, no undefined citations, PDFs newer than sources |

---

## 9. Risks, assumptions and open questions

- **RoPE calibration**: default path applies the repo `pos_encoder` to the
  materialized per-head Q/K (`tucker` precedent). Standard RoPE over `hd` equals
  the paper's `I_{d1} ⊗ R^{(2)}_θ` only if frequencies are recalibrated to
  `d2` (`θ_j = 10000^{−2j/d2}`). Implementation choice: rotate only the `B`
  factor before materialization when `pos_encoder` is a RoPE module, else use
  the shared encoder as-is; document in the docstring and annex. Flag for the
  implementer; if ambiguous, ask the user.
- **Line drift**: all anchors (tucker spec ~line 966, `_dims.yaml` ~260/~322,
  `ATTENTIONS` ~267–273, T12 ~152) were read at planning time; re-grep before
  editing.
- **Auto-split determinism**: the `h1/h2` auto rule (most asymmetric split with
  larger `h1`) must be identical in the module, the estimator JS and
  `param_truth.json` or the 1% gate fails.
- **Small configs**: default `d1=8` requires `head_dim % 8 == 0`; tiny test
  configs (hd=16) must pin `d1` explicitly or rely on the documented fallback.
- **BitNet**: BitLinear on the six factor projections follows the family
  pattern; `test_bitnet_coverage` will verify.
- **HF Hub reachability** for the real-data smoke (fallback `--no-realonall`).
- **Naming**: `kpa` free in registry, docs, figures, mirror lists (checked).
- **KV-cache annex numbering**: prompt said EN `annex-15-kv-cache`; the repo has
  `annex-23-kv-cache.tex` (EN) / `annex-23-cache-kv.tex` (ES) — locate by
  grepping `tucker`.
- Paper variants (h=(2×8), (4×4), …) are one knob change each — only the
  flagship asymmetric configuration is encoded in defaults; others recorded
  here as non-goals.

## 10. Out of scope

This command created only this plan file. No code, schema, preset, test,
figure, doc, PDF, mirror or git change was made.
