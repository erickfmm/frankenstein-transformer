# Plan: add ProbSparse mixer (arXiv:2209.00260)

**Paper**: Xianchao Wu, *"Deep Sparse Conformer for Speech Recognition"*, InterSpeech 2022
(arXiv:2209.00260). The paper makes a Conformer encoder **sparser and deeper**:
(1) it replaces the MHSA module with **ProbSparse self-attention** (query-sparsity
top-k selection borrowed from Informer, arXiv:2012.07436), dropping attention
cost from O(L²) to O(L log L); and (2) it scales residual branches with DeepNorm
α factors (DeepNet, arXiv:2203.00555) to train 50–100-layer encoders. On
CSJ-500h Japanese ASR, DSC-12en beats Conformer-Large (5.52/4.03/4.50 CER vs
6.97/4.65/5.29) at the same 135M params while training 20% faster and decoding
1.23× faster. **Scope of this plan**: the ProbSparse attention mechanism only
(the sequence mixer). The DeepNorm residual scaling is a *residual/norm*
strategy, not a mixer — out of scope, flagged below.

---

## 2. Paper extract

### 2.1 Forward equations (verbatim)

ProbSparse replaces the query tensor Q with a same-shape tensor Q̄ containing
only the top-u (= O(log L)) **dominant queries** under a sparsity measure:

Attention kernel form (paper Eq. 5–7):

```latex
A(Q, K, V) = \sum_j \left[ \frac{k(Q, k_j)}{\sum_l k(Q, k_l)} \right] v_j
           = \mathbb{E}_{p(k_j|Q)}[v_j]
           \approx \mathrm{Softmax}\!\left( \frac{\bar{Q} K^\top}{\sqrt{d}} \right) V
```

with exponential kernel `k(q_i, k_j) = exp(q_i k_j^T / sqrt(d))`.

Query-sparsity "max-mean" measure (paper Eq. 8, KL-motivated — dominant
dot-product pairs push the query's attention distribution away from uniform):

```latex
\bar{M}(q_i, K) = \max_j \left\{ \frac{q_i k_j^\top}{\sqrt{d}} \right\}
                - \frac{1}{L} \sum_{j=1}^{L} \frac{q_i k_j^\top}{\sqrt{d}}
```

Multi-head sparse attention (paper Eq. 9–10):

```latex
\mathrm{MHSA\text{-}Sparse}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W_O
\quad \mathrm{head}_i = A(X W_Q^i,\; X W_K^i,\; X W_V^i)
```

Score with relative position encoding (Algorithm 1, line 19):

```latex
S = \left( \bar{Q}_{U_1} K^\top + \bar{Q}_{U_2} P^\top \right) / \sqrt{d_q},
\qquad Q_{U_1} = (Q + U_1)^{\top},\; Q_{U_2} = (Q + U_2)^{\top}
```

where `P = X_P W_P` is the projected relative-position tensor of the keys,
`X_P ∈ R^{b×L_K×d}`, and `U_1, U_2 ∈ R^{h×d_q}` are learnable per-head bias
vectors (Transformer-XL style, arXiv:1901.02860).

**Non-selected queries** (Algorithm 1 line 21 "only change a part of V"): only
the top-L′_Q query rows of the output are overwritten by `Softmax(Mask(S))·V`;
the remaining rows are filled with the head-mean of V (Informer reference
implementation convention `context[V.mean(dim=-2)]` — **assumption to verify**
at implementation time against the Informer repo; recorded in §9).

**Out-of-scope equations** (residual/norm family, NOT this mixer — requires
explicit user approval as a separate change): DeepNorm
`DeepNorm(x) = LayerNorm(x × α + f(x))` with
`α_{1..4} = 0.81·(N⁴M)^{1/16}` (paper Eq. 11–15).

### 2.2 Algorithm (paper Algorithm 1, verbatim structure)

```
Given: X, X_P, Y, c1, c2, mask, p_dropout        # Y = X for self-attention
Trainable: W_{Q,K,V,P,O}, U_{1,2}
1  Q, K, V, P = Y·W_Q, X·W_K, X·W_V, X_P·W_P
2  reshape to (b, h, L, d_head); Q_U1 = (Q+U1), Q_U2 = (Q+U2)
3  L'_K = c1 · d · ln(L_K)            # key subsample size
4  K_part = Sample(K, L'_K, dim=-2)   # random key subset for the max-mean probe
5  L'_Q = c2 · d · ln(L_Q)            # number of dominant queries kept
6  M̄ = max_j(Q_U1 K_part^T) − mean_j(Q_U1 K_part^T)     # Eq. 8 (omits √d)
7  M̄_top = M̄.topk(L'_Q)[1]            # indices of dominant queries
8  Q̄_U1 = Q_U1[M̄_top, :];  Q̄_U2 = Q_U2[M̄_top, :]
9  S = (Q̄_U1 K^T + Q̄_U2 P^T) / √d_q   # full keys, reduced queries
10 S = Dropout(Softmax(Mask(S, mask)), p_dropout)
11 V_out[M̄_top, :] = S·V; V_out[rest] = mean(V)         # per head
12 return reshape(V_out) · W_O
```

No recurrent/decode form exists in the paper (see §2.6).

### 2.3 Parameter inventory + count formula

Learned tensors per layer (paper §3.1, Algorithm 1 line 2):

| Tensor | Shape (paper) | Repo shape (`H=hidden_size`, `nH=num_heads`, `hd=H/nH`) |
|---|---|---|
| `W_Q` | `d × d_k` per head | `H × H` |
| `W_K` | `d × d_k` per head | `H × H` |
| `W_V` | `d × d_v` per head | `H × H` |
| `W_O` | `h·d_v × d` | `H × H` |
| `W_P` (relative-pos projection) | `d × d` | `H × H` |
| `U_1`, `U_2` (per-head bias vecs) | `(h, d_q)` each | `nH × hd` each = `H` each |
| biases | — | per house convention (mirror `swa.py`/`sparsek.py`) |

**Exact attention-parameter formula** (bias handling must match the peer
projection style — verify against `sparsek.py` at implementation time; formula
below assumes the peer's convention exactly):

```
params(probsparse) = 5·H²  (W_Q, W_K, W_V, W_O, W_P)  +  2·H  (U_1, U_2)
                     + biases if and only if the peer projections carry them
```

`ft-param-estimator.js` `mixerParams()` case and
`full_tests/param_count_check.py` (which regenerates `param_truth.json` from
the real model) must agree on this formula within the ≤1% gate.

### 2.4 Knobs (→ spec `params_schema` + `defaults`)

| Schema key | Leaf | Type | Default | Constraint | Meaning |
|---|---|---|---|---|---|
| `model.attention.probsparse.c1` | `c1` | number | 5.0 | ≥ 0.1 | key-sampling factor: `L'_K = c1 · H · ln(L_K)` |
| `model.attention.probsparse.c2` | `c2` | number | 5.0 | ≥ 0.1 | query-top-k factor: `L'_Q = c2 · H · ln(L_Q)` |

Both clamp to the available sequence length (`L' = min(L', L)`); the paper uses
`c1 = c2 = 5.0`. No other knobs. (Paper's `p_dropout` maps to the existing
`attention dropout` plumbing, not a mixer knob.)

### 2.5 State & cache

No recurrent state, no buffers: the only "state" is standard K/V. Cache
compatibility **class (i)** (`KVSlot`) — sparsity is **query-side**, so at
decode time (exactly 1 new query per step) the top-u selection is degenerate
(every query is dominant) and attention over the cached KV prefix is **exact**,
identical to dense-over-cache. Per-step decode cost: `O(n_ctx · hd · nH)`.

### 2.6 Complexity

- Training (parallel form): `O(L_Q · L_K · d · nH / u)` ≈ **O(L log L)** time
  and memory (paper abstract + §3.1; the max-mean probe adds
  `O(L_Q · log L_K · d)` via key subsampling).
- Inference (decode): `O(n_ctx)` per step over the KV cache.

### 2.7 Trainability

**Trainable from scratch** — ProbSparse is active during training (the CSJ
models are trained end-to-end with it). **Not** `training_free`: no
`dense_forward` surrogate, no `TRAINING_FREE_LAYERS` membership, valid in
training presets, hard rule 5 does not apply.

### 2.8 Positional encoding

The mixer **consumes** relative positional information (P, U₁, U₂ terms) →
`needs_pos_encoder=True` → `use_pe_for(config, "probsparse", True)`; the
assembler auto-injects `use_pe.probsparse_use_pe` (default true). Per the
user's decision the module implements the **faithful paper PE** (W_P + U₁/U₂ +
the `Q̄_U2 P^T` score term) *in addition to* routing through the repo PE
machinery (`apply_pe_to_qk`) where the peer does.

### 2.9 Mode constraints

Paper formulates the encoder (bidirectional, ASR). Decoder/causal form is not
in the paper, but query sparsity is orthogonal to key masking → implement
decoder mode with the peer's causal-mask pattern (`sparsek.py:229-233`
`masked_fill(k_pos > q_pos, -inf)` before softmax, adapted to full-key scores).
Flagged as a repo extension, not paper-verified (§9). Vision/audio compatible
(plain per-token mixer; the paper itself is an audio architecture).

### 2.10 BibTeX data

- Primary: Wu, Xianchao. "Deep Sparse Conformer for Speech Recognition."
  InterSpeech 2022. arXiv:2209.00260.
- Secondary (origin of ProbSparse): Zhou, Haoyi et al. "Informer: Beyond
  Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021.
  arXiv:2012.07436.
- (Context only, cited in the annex prose: DeepNet arXiv:2203.00555;
  Transformer-XL arXiv:1901.02860.)

---

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `probsparse` | Paper §3.1 "ProbSparse self-attention"; snake_case, no `_attn` suffix (rule from `9b10768`) |
| Class / builder | `ProbSparseAttention` @ `src.model.attention.sparse.probsparse` | Sparse-family naming (`SparseKAttention`, `SlidingWindowAttention`) |
| Family | `sparse/` directory; paper family label **Sparse** | Query-sparsity top-k; closest to `sparsek` (selection axis: queries vs keys) |
| Peer | `src/model/attention/sparse/sparsek.py` (skeleton, knobs, causal mask) + `swa.py:164-260` (cache branch, `kv_decode_append`, `make_cache_slot`) | See anchors in §4 |
| Knobs | `c1`, `c2` (§2.4) in `schema/_model/_attention_sparse.yaml#/properties/probsparse` | Paper §3.1 hyper-parameters |
| `needs_pos_encoder` | `True` | §2.8 |
| KV cache | **Decision 1**: `cache=CacheSpec(supported=True, slot_type="kv", notes=…class (i), query-side sparsity degenerate at decode → exact dense-over-cache…)` + `cache_implemented=True` + incremental branch + `IMPLEMENTED_MIXERS` membership (parity suite auto-covers) | User-approved; `swa`/`lsa` precedent; class (i) in `capabilities.py:23-31` |
| training_free | `False` | §2.7 |
| qk_norm / qk_clip | `True` / `True` | Standard Q/K projections; `attach_qk_norm` + `record_qk_max_logits` like `sparsek.py:182,234` |
| Relative PE | **Faithful paper PE**: W_P + U₁/U₂ + `Q̄_U2 P^T` term (user-approved), on top of `apply_pe_to_qk` routing | §2.1, §2.8 |
| Example | `configs/examples/sparse_probsparse_adamw.yaml` (encoder MLM, `es_arch_swa_adamw.yaml` shape) | AGENTS.md mixer row |
| Full preset | **Yes** — `configs/probsparse.yaml` modeled on `configs/lsa.yaml`, DSC-12en shape: d=512, nH=8, 12 encoder layers, c1=c2=5.0 | User-approved |
| Census bumps | **55 → 56 mixers** (`grep -c MixerSpec builtin/attention.py` = 55; `len(CACHE_SPECS)` gate `test_schema_attention_layers.py:182` = 55 → 56); **Sparse 11 → 12** (`README.md:170`, `mixer-taxonomy.mmd:9`, `annex-summary-tables.tex:47` EN/ES, `docs/architecture.md` §7.4); ⚠️ README headline drift: `README.md:28` says 53, `:125` says 52 — reconcile all to 56 | Read from repo, not hardcoded |
| e2e | Append `"probsparse"` to `ATTENTIONS` (trainable → included; not training-free) | `run_e2e.py:267-322` |

---

## 4. File-by-file plan table

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/attention/sparse/probsparse.py` | New module. Docstring cites arXiv:2209.00260 + 2012.07436 with the §2.1 equations. `from __future__ import annotations`, Google docstrings, CPU-safe. `def __init__(self, config, pos_encoder=None, params=None)`; `params = mixer_params(config, "probsparse", params)` (helper at `params.py:17-43`); knobs `self.c1 = float(params.value("c1", 5.0))`, `self.c2` idem (clamp ≥0.1); `self.use_pe = bool(use_pe_for(config, "probsparse", True))`; `attach_qk_norm(self, config, self.head_dim)`; parameters `wq/wk/wv/wo/wp` (H×H each, bias convention copied from `sparsek.py` projections) + `u1/u2` (`nn.Parameter(torch.zeros(1, nH, 1, hd))`). `forward(self, x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)` in `(batch, seq, hidden)`; Q/K/V → `(bsz, nH, seq, hd)`; `apply_pe_to_qk(...)`; max-mean probe with key subsample `L'_K = min(int(c1·H·ln(L_K)), L_K)` (fixed generator seeded by `(logical_layer_idx, step)` for cache parity — see §9); `L'_Q = min(int(c2·H·ln(L_Q)), L_Q)`; `topk(L'_Q)` query indices; scores `S = (Q̄_U1 Kᵀ + Q̄_U2 Pᵀ)·scale` with `P = X_P·W_P` from the relative-PE tensor supplied by the PE plumbing (pos_encoder path); decoder causal mask (peer `sparsek.py:229-233` pattern adapted to full-key scores); `record_qk_max_logits(self, scores)` then `attn = module_softmax(self, scores, dim=-1)` (fixed-numerics honor); dropout; scatter top rows into a `V.mean(-2, keepdim)`-filled context (Informer convention, §2.1); `Wo`. Incremental branch mirroring `swa.py:223-260`: `slot = cache if getattr(cache, "kind", "") == "kv" else None`; `K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)`; rebuild P rows from `kv_positions`; single-query scores (selection degenerate — skip topk); exact dense-over-cache. `make_cache_slot` returning a `KVSlot` (pattern `swa.py:164-174`). |
| 2 | `src/model/attention/sparse/__init__.py` + `src/model/attention/__init__.py` | Add `from .probsparse import ProbSparseAttention` + `__all__` entry (peer lines `sparse/__init__.py:9,18`; `attention/__init__.py:13,80`). |
| 3 | *(no edit)* | `src/model/hybrid_layer.py` — dispatch reads the registry (`hybrid_layer.py:161-171`); `TRAINING_FREE_LAYERS` derives from the spec (`:79-82`); `use_fixed_softmax` attach at `:176`. Nothing to change. |
| 4 | `src/registry/builtin/attention.py` | One `MixerSpec` (single declaration; enum, `model.attention` fragment and `use_pe` key all derive from it). Insert alphabetically near the sparse block (~`:935-1018`): `name="probsparse"`, `builder="src.model.attention.sparse.probsparse:ProbSparseAttention"`, `description="ProbSparse: query-sparsity top-k attention via max-mean KL measure (paper: arxiv:2209.00260)"`, `description_es="ProbSparse: atención top-k por dispersidad de consultas con medida max-mean KL (paper: arxiv:2209.00260)"`, `references=("2209.00260", "2012.07436")`, `params_schema="schema/_model/_attention_sparse.yaml#/properties/probsparse"`, `defaults={"c1": 5.0, "c2": 5.0}`, `cache=CacheSpec(supported=True, slot_type="kv", notes="ProbSparse (arXiv:2209.00260, Informer arXiv:2012.07436); class (i) — sparsity is query-side, so single-query decode is degenerate and dense-over-cached-KV is exact.")`, `cache_implemented=True`, `qk_norm_supported=True`, `qk_clip_supported=True` (`training_free` defaults False, `needs_pos_encoder` defaults True). |
| 5 | *(removed)* | No config flattener exists (D15); no rename entry. |
| 6 | `src/schema/_model/_dims.yaml` | One row in **both** bilingual tables (4 columns): EN after the `sparsek` row (`:246`): `` | `probsparse` | ProbSparse: query-sparsity top-k attention via max-mean KL measure (paper: arxiv:2209.00260) | ✅ | ✅ | ``; ES after `:312`: `` | `probsparse` | ProbSparse: atención top-k por dispersidad de consultas con medida max-mean KL (paper: arxiv:2209.00260) | ✅ | ✅ | ``. Enum list untouched (registry-injected, `schema_assembly.py:75-84`). |
| 7 | `src/schema/_model/_attention_sparse.yaml` | New `probsparse:` block under `properties` (after `sparda`, `:141`), modeled on `msa` (`:4-32`): `type: object`, `additionalProperties: false`, bilingual folded descriptions citing arXiv:2209.00260 (+2012.07436), `c1` / `c2` leaves: `type: number`, bilingual `title`/`title_es` ("ProbSparse Key-Sampling Factor c₁" / "Factor de muestreo de claves c₁ de ProbSparse", etc.), descriptions with the `L'_K = c1·d·ln L_K` / `L'_Q = c2·d·ln L_Q` formulas, `examples: [5.0]`, `default: 5.0` matching the spec `defaults`. |
| 8 | *(no edit)* | `src/schema/_model.yaml` — assembler rebuilds `model.attention.properties` from spec fragments (`schema_assembly.py:89-104`). (Optional: extend the knob-less prose list at `_model.yaml:53,63` is **not** needed — probsparse has knobs.) |
| 9 | *(no edit)* | `src/schema/_model/_positional_encoding.yaml` — assembler injects `use_pe.probsparse_use_pe` from `needs_pos_encoder` (`schema_assembly.py:106-131`). |
| 10 | `src/model/cache/capabilities.py` | Add `"probsparse"` to `IMPLEMENTED_MIXERS` (`:47-92`, sparse family block `:74-81`) with comment `# F1 (query-side sparsity degenerate at decode; exact dense-over-cache)`. `CACHE_SPECS` derives from the spec — no other edit. |
| 11 | `configs/examples/sparse_probsparse_adamw.yaml` | Tiny encoder-MLM example, `es_arch_swa_adamw.yaml` shape: `model_class: frankenstein_text_encoder`, `model.dims`: `hidden_size: 96, num_layers: 2, num_heads: 4, layer_pattern: [probsparse]`, `model.attention.probsparse: {c1: 5.0, c2: 5.0}`, `model.positional_encoding_parameters.use_pe: {probsparse_use_pe: true}`, AdamW prefixed keys (`adamw-attention_lr: 3.0e-4`, etc.), `training.task: mlm` with the toy text dataset used by sibling examples. Must pass `load_training_config` + assembled-schema validation. |
| 12 | `configs/probsparse.yaml` | Full preset (user-approved), modeled on `configs/lsa.yaml`: DSC-12en shape — `hidden_size: 512`, `num_layers: 12`, `num_heads: 8`, `layer_pattern: [probsparse]`, `c1: 5.0, c2: 5.0`, `ffn.hidden_size: 2048`, rms_norm, rope PE (paper uses relative PE — rope is the repo-native carrier for `use_pe`), mlm task, adamw. Joins the smoke-tested preset set (`test_schema_consistency.py`, `test_yaml_examples.py` auto-glob). |
| 13 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (`:267-322`) with the peer comment style: `# ProbSparse / Deep Sparse Conformer (arXiv:2209.00260): query-sparsity` / `# top-k via max-mean KL measure; trainable natively; exact KV-cache decode.` / `"probsparse",`. Auto-generates `attn_single_probsparse` + all `attn_<a>__probsparse` / `attn_probsparse__<b>` pairs (`:864-886`). No builder edits. |
| 14 | `tests/test_schema_attention_layers.py` | Add `"probsparse"` to the `expected` set (`:18-75`); bump `self.assertEqual(len(CACHE_SPECS), 55)` → `56` at `:182`. |
| 15 | `tests/test_sparse_attention.py` | New `ProbSparseAttentionTests(unittest.TestCase)` after `SparseKAttentionTests` (`:321-335`): `test_output_shape`, `test_decoder_mode` (causal mask), `test_gradient_flows` (incl. `u1`/`u2`/`wp` grads), `test_knobs_honored` (`c1`/`c2` extremes: `c2` large → all queries kept), `test_no_nans`, `test_bitnet_on` (BitNet default), `test_incremental_parity` (prefill+decode vs full forward via `make_cache_slot`, mirroring the parity suite's contract). |
| 16 | `tests/test_bitnet_coverage.py` | Add `'probsparse'` to the inline `layer_pattern` list (`:23`). |
| 17 | *(no hand edit)* | `tests/test_kv_cache_mixers.py` parametrizes itself from `IMPLEMENTED_MIXERS` — covered by row 10. |
| 18 | `docs/paper/figures/mixers/probsparse.mmd` | House-style mermaid (`sparsek.mmd` template, 16 lines): nodes `x[B,n,H]` → `qkv[Wqkv]` → `sample[K_part, c1·d·ln L_K]` → `m̄[max-mean measure]` → `topk[L'_Q = c2·d·ln L_Q queries]` → `s[S = (Q̄_U1 Kᵀ + Q̄_U2 Pᵀ)/√d]` → `softmax` → `o[W_o·(S·V), mean-V for non-top rows]`; dotted edge for the `P = X_P·W_P` relative-PE path. Same `%%{init: …}%%` header. |
| 19 | `docs/paper/figures/mixer-taxonomy.mmd` | Sparse chain `:9`: `s["Sparse (11)"]` → `"Sparse (12)"`, append `--> s12["probsparse"]` after `s11["lsa"]`. |
| 20 | `docs/paper/figures/sparse-map.mmd` | Add a `probsparse` node (query-side top-k sibling of `sparsek`). |
| 21 | `docs/bibliography/attention_types.bib` | `@misc{wu_deep_sparse_conformer_2022, title={Deep Sparse Conformer for Speech Recognition}, author={Wu, Xianchao}, year={2022}, url={https://arxiv.org/abs/2209.00260}, doi={10.48550/ARXIV.2209.00260}, publisher={{arXiv}}}` (template: `lou_sparsek_2024` at `:159-167`); plus the Informer entry `zhou_informer_2021` (AAAI 2021, arXiv:2012.07436) if absent. |
| 22 | `docs/paper/appendices/annex-4-sparse-attention.tex` | New `\subsection{ProbSparse Attention: Query-Sparsity \texorpdfstring{Top-$k$}{Top-k} Selection}` inserted after the last mixer subsection, **before** `\subsection{Design Space and Selection Criteria}` (`:769`): `\fitfigure{figures/mixers/probsparse.pdf}` + `\label{fig:probsparse}`; "Mathematical Foundation" with Eqs. 5–10 + the max-mean measure + Algorithm 1 as an `algorithm` float (faithful to arXiv:2209.00260, D11); "Empirical Performance and Trade-offs" (CSJ-500h table: 5.52/4.03/4.50 vs Conformer 6.97/4.65/5.29 at 135M; 20% faster training, 1.23× decode) + strengths/limitations itemization (query-side vs key-side sparsity; random key probe; non-top rows = mean-V); `\citep{wu_deep_sparse_conformer_2022, zhou_informer_2021}`; extend the closing design-space list. |
| 23 | `docs/paper/sections/04-architecture-taxonomy.tex` | Sparse paragraph `:77` "The registry includes **eleven sparse blocks**" → "**twelve** sparse blocks … and ProbSparse \citep{wu_deep_sparse_conformer_2022}"; census `:7` "fifty-five sequence mixers" → "fifty-six"; figure caption `:23` likewise. |
| 24 | `docs/paper/appendices/annex-summary-tables.tex` | Intro census `:4` "**55 sequence mixers across nine families**" → 56; family header `:47` `\textbf{Sparse (11)}` → `(12)`; new `tab:all-mixers` row after `:53`: `probsparse & Sparse & $\mathcal{O}(L \log L)$ & $\mathcal{O}(n_{ctx})$ & Query-sparsity top-$k$ via max-mean KL measure; exact KV-cache decode & \citep{wu_deep_sparse_conformer_2022} \\`. |
| 25 | `docs/paper/appendices/annex-schema.tex` | Row after `:288`: `probsparse & ProbSparse: query-sparsity top-k attention & \checkmark & \checkmark \\`. |
| 26 | `docs/paper-es/appendices/annex-4-sparse-attention.tex` | Faithful Spanish translation of #22 (figures via `../paper/figures/`), same numbering/anchor (`:769`). |
| 27 | `docs/paper-es/sections/01-introduccion.tex`, `04-taxonomia-arquitectura.tex`, `10-discusion.tex`, `docs/paper-es/appendices/annex-summary-tables.tex`, `annex-schema.tex` | ES counterparts of #23–#25 (+ ES `01:14` "cincuenta y cinco" → "cincuenta y seis"; ES `10-discusion.tex:16` census; ES summary-tables intro `:5`, header `:47` "Disperso (11)" → "(12)", row after `:53`; ES schema row after `:287`). |
| 28 | `README.md`, `docs/architecture.md`, `configs/README.md` | Census pass: `README.md:170` sparse row add `` `probsparse` `` and `(11)`→`(12)`; `:180` "**55** mixer values" → 56; reconcile stale `:28` (53) and `:125` (52) to 56; `docs/architecture.md:392` census "55 enum values" → 56 + new table row `| probsparse | Sparse | probsparse.* | yes |` near `:411-413` + §7.13 implemented list (`:777`) add `` `probsparse` ``; `configs/README.md:299` sparse list add `probsparse` (arXiv:2209.00260). |
| 29 | `docs/source/api/model/attention/sparse/probsparse.rst` + `sparse/index.rst` | `automodule:: src.model.attention.sparse.probsparse :members: :undoc-members: :show-inheritance:` page; toctree entry after `sparsek`. |
| 30 | `docs/schema.assembled.yaml` | Regenerate: `python -m src.registry.snapshot --out docs/schema.assembled.yaml`. |
| 31 | `docs/paper/appendices/annex-23-kv-cache.tex` + `docs/paper-es/appendices/annex-23-cache-kv.tex` | Add `probsparse` to the incremental-decode (implemented) list — NOT the `tab:kv-fallbacks` table (`:235`); class (i) KV slot, decode is exact. |
| 32 | `docs/paper/sections/01-introduction.tex` (+ ES), `09/10-*.tex` | Global census `01:14` "fifty-five sequence mixers" → fifty-six (EN) and ES `01:14`; check `09-empirical-evaluation.tex`/`10-discussion.tex` for count mentions (ES `10:16` known). |

**Hard rules restated**: schema is the contract (`additionalProperties: false`);
the registry is the single source of truth (D13), validated schema = core YAML +
registry fragments (D14), runtime config is nested-only (D15);
`hidden_size % num_heads == 0`, `num_kv_heads | num_heads`,
`len(layer_pattern) == num_layers`; inference never fails because of the cache
(D5); probsparse is NOT training-free (rule 5 n/a); the papers are the theory
source of truth (D11) — the annex formulation must match arXiv:2209.00260
exactly and `paper-es` must translate it faithfully.

---

## 5. Ordered execution checklist

1. ☐ Rows 1–2: module + exports (`probsparse.py`, two `__init__.py`)
2. ☐ Row 4: `MixerSpec` (the single declaration)
3. ☐ Rows 6–7: bilingual `_dims.yaml` rows + knob fragment
4. ☐ Row 10: `IMPLEMENTED_MIXERS` + incremental branch (already in row 1)
5. ☐ Rows 11–12: example + full preset
6. ☐ Row 13: e2e `ATTENTIONS`
7. ☐ Rows 14–16: tests
8. ☐ Row 30: regenerate snapshot
9. ☐ Targeted gates:

```bash
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_sparse_attention.py \
  tests/test_bitnet_coverage.py tests/test_registry_core.py \
  tests/test_registry_binding.py tests/test_schema_assembly.py \
  tests/test_kv_cache_mixers.py \
  -v --tb=short -p no:warnings
```

10. ☐ Full suite:

```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

11. ☐ e2e enumeration + **targeted real-data training** (default `--realonall`
trains on the real Borges corpus; first run downloads tokenizer/corpus from the
HF Hub):

```bash
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F probsparse

uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only probsparse

# heavy-sweep fallback: solo combo only
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_probsparse

# offline synthetic fallback (no HF download)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only probsparse --no-realonall

# sizing: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU box: swap the extra for --extra cu118 --extra train
```

Expected: every selected combo (`attn_single_probsparse` + all pairs) reports
**PASS with finite training loss** on real corpus windows; the run directory
keeps `config.yaml` + metrics. NaN/Inf loss, schema rejection or timeout is a
**blocker**, not a warning.

12. ☐ Rows 18–20: figures, then render (chromium available at `/usr/bin/chromium`):

```bash
( cd docs/paper/figures && ./build-figures.sh both )
```

13. ☐ Param ground truth + JS estimator (after row 11/12 land):

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py   # "0 failures"
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01
```

14. ☐ Rows 21–28, 31–32: docs EN/ES (annex, summary tables, schema annex,
kv-cache annex, sections 01/04/10, bibliography, census pass).
15. ☐ Row 29 + Sphinx:

```bash
bash docs/build_docs.sh
```

16. ☐ PDFs (pdflatex/bibtex available):

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

PDF checks: no `Fatal error`, no `Citation .* undefined`, `paper.pdf` /
`paper-es.pdf` newer than the edited `.tex`/`.bbl`, `\citep{wu_deep_sparse_conformer_2022}` resolves, `git status --short docs/paper docs/paper-es` shows only intended files.
Fallbacks: no TeX → leave sources, rebuild before next release; no Chrome →
commit `.mmd`, render before website sync; HF Hub unreachable → record
`--no-realonall` PASS, re-run real-data later.

---

## 6. Website sync (planned, not executed)

Per `.opencode/command/sync-website.md`:

1. Bootstrap/refresh the `erickfmm.github.io` mirror; copy per the file map
   (`:31`): regenerated `docs/schema.assembled.yaml` → `schema.yaml`;
   `src/schema/**`, `src/registry/**`; `configs/*.yaml` (+ `probsparse.yaml`)
   + `configs/README.md`; `configs/examples/**` (+ the new example);
   `docs/paper/**` + `docs/paper-es/**` (tex/bbl/pdf + the **new
   `probsparse.svg`** — sources/PDFs stay in the main repo);
   `docs/bibliography/**`; `docs/architecture.md`.
2. Site-only edits (mirror is their source of truth):
   - `ft-diagram.js`: `CATS["probsparse"] = "Sparse"`; `SUBSTRUCT["probsparse"]`
     tensor-flow template modeled on `SUBSTRUCT["sparsek"]` with the extra
     sample→max-mean→topk pipeline and the `P·W_P` dotted branch; bump any
     Sparse-family count in `STYLES` labels. No new STYLES class (family
     exists).
   - `ft-param-estimator.js`: `mixerParams()` case with the exact formula
     `5·H² + 2·H (+ biases per repo convention)` — must match §2.3.
   - `index.html`: grep a peer (`sparsek`, `swa`); extend every hardcoded mixer
     list and the Sparse family count found.
3. Paper tab HTML rebuild: `flatten_tex.py --web` + `pandoc` +
   `postprocess_html.py` (en + es); top-level `paper/en.pdf` / `es.pdf` copies.
4. Validation Gate (`:159`): `param_count_check.py` → "0 failures"
   (`param_truth.json` diff is intentional); estimator `--tol 0.01` → ALL PASS;
   JSON round-trip script; CLI parse script; state probes
   (`schema-refs`, `example-yaml`, `cli-flags`, …) all PASS.
5. Mirror stays uncommitted until the commit phase.

---

## 7. Commit plan (planned, not executed)

Conventions: English, conventional tag, structured body; never stage
`*.aux *.log *.out *.toc *.blg`, caches, mermaid temp files; `param_truth.json`
changes intentional.

1. **Main repo** — `feat(attention): add ProbSparse mixer (arXiv:2209.00260)`
   - Body: module, registry spec, bilingual table rows + knob fragment,
     KV-cache decision (class (i), exact decode), example + full preset,
     tests, e2e `ATTENTIONS` + real-data smoke result, figures (mmd + rendered
     pdf/svg), docs (annex-4 EN/ES, summary tables, schema annex, kv-cache
     annex, bibliography, sections 01/04/10, README/architecture/configs
     README + census reconciliation, Sphinx rst), regenerated
     `docs/schema.assembled.yaml`, rebuilt PDFs.
   - Stage: `src/model/attention/sparse/probsparse.py`,
     `src/model/attention/sparse/__init__.py`,
     `src/model/attention/__init__.py`,
     `src/registry/builtin/attention.py`,
     `src/model/cache/capabilities.py`,
     `src/schema/_model/_dims.yaml`,
     `src/schema/_model/_attention_sparse.yaml`,
     `configs/examples/sparse_probsparse_adamw.yaml`,
     `configs/probsparse.yaml`,
     `full_tests/run_e2e.py`, `full_tests/param_truth.json`,
     `tests/test_schema_attention_layers.py`,
     `tests/test_sparse_attention.py`,
     `tests/test_bitnet_coverage.py`,
     `docs/paper/figures/mixers/probsparse.{mmd,pdf,svg}`,
     `docs/paper/figures/mixer-taxonomy.{mmd,pdf,svg}`,
     `docs/paper/figures/sparse-map.{mmd,pdf,svg}`,
     `docs/bibliography/attention_types.bib`,
     `docs/paper/appendices/annex-4-sparse-attention.tex`,
     `docs/paper/appendices/annex-summary-tables.tex`,
     `docs/paper/appendices/annex-schema.tex`,
     `docs/paper/appendices/annex-23-kv-cache.tex`,
     `docs/paper/sections/01-introduction.tex 04-architecture-taxonomy.tex 10-discussion.tex`,
     `docs/paper/paper.{tex,bbl,pdf}`, `docs/paper-es/` counterparts,
     `README.md`, `docs/architecture.md`, `configs/README.md`,
     `docs/source/api/model/attention/sparse/probsparse.rst`,
     `docs/source/api/model/attention/sparse/index.rst`,
     `docs/schema.assembled.yaml`.
   - Then `git push origin HEAD:main`.
2. **Mirror** — `git -C erickfmm.github.io add …` (file-map copies +
   `ft-diagram.js`, `ft-param-estimator.js`, `index.html`, paper HTML/PDF,
   new svg); commit `docs(site): sync ProbSparse mixer (arXiv:2209.00260)`;
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Schema/bilingual/registry gates | targeted pytest block (§5.9) | all pass; `len(CACHE_SPECS) == 56` |
| Full suite | §5.10 | all pass, no collection errors |
| KV parity | `pytest tests/test_kv_cache_mixers.py -k probsparse` | prefill+decode ≡ full forward (auto-parametrized) |
| e2e enumeration | `run_e2e.py --category attn --list-combos \| grep -F probsparse` | `attn_single_probsparse` + every pair id |
| e2e real-data training | `run_e2e.py --category attn --only probsparse` | all selected combos PASS, finite loss, real corpus |
| e2e solo fallback | `--only attn_single_probsparse` | PASS |
| e2e offline fallback | `--only probsparse --no-realonall` | PASS on synthetic |
| Figures | `build-figures.sh both` | `probsparse.svg/pdf`, taxonomy + sparse-map re-rendered |
| Param truth | `param_count_check.py` | "… configs counted, 0 failures" (new entries for both YAMLs) |
| JS estimator | `param_estimate_check.mjs … --tol 0.01` | ALL PASS |
| Sphinx | `bash docs/build_docs.sh` | builds, new page present |
| PDF EN/ES | §5.16 | no Fatal error, no undefined citations, PDFs newer than sources |

---

## 9. Risks, assumptions and open questions

- **Non-top query rows**: paper Algorithm 1 line 21 does not state what
  non-selected query rows output; the Informer reference implementation fills
  them with the head-mean of V. **Assumption** — verify against the Informer
  repo during implementation; if it uses zeros instead, switch and note it in
  the annex.
- **Key-sampling determinism vs cache parity**: the max-mean probe randomly
  samples keys; with a KV cache the probe is skipped at decode (single query),
  so parity holds regardless — but training/eval RNG must not change between
  the full-forward and parity-test paths (seed by `logical_layer_idx`).
- **Relative-PE tensor `X_P` source**: the repo PE plumbing supplies PE to
  Q/K via `apply_pe_to_qk`; the paper's `P = X_P·W_P` needs a per-key
  relative-position tensor. Plan: derive `X_P` from the PE embeddings the
  pos_encoder exposes (or `kv_positions` in decode). **Assumption to verify**
  — if no clean hook exists, fall back to `use_pe`-gated `apply_pe_to_qk` only
  and gate the U/W_P path behind the knob default `false` (requires user
  sign-off before deviating from the "faithful paper PE" decision).
- **Bias convention** in the parameter formula — mirror the peer exactly and
  confirm with `param_count_check.py`.
- **Decoder/causal form is a repo extension** (paper is encoder-only ASR);
  causal masking is mask-orthogonal, but the annex must present the causal
  variant as this repo's extension, not paper-verified.
- **Census drift**: README headlines are stale (28: "53", 125: "52" vs 55);
  reconciling to 56 is intentional — note it in the commit body to avoid
  confusing reviewers.
- Line anchors were read at plan time (2026-11-12); re-check for drift before
  editing. Tool availability confirmed: pdflatex, bibtex, chromium present;
  HF Hub reachability unknown until the smoke runs.
- **Out of scope (needs explicit approval)**: DeepNorm α-residuals
  (residual/norm family change), the paper's bidirectional l2r/r2l decoder,
  CSJ/ASR training presets.

## 10. Out of scope

No code, schema, preset, test, figure, doc, PDF, mirror or git change was made
by this command. This document is the only file written.
