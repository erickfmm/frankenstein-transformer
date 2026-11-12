# Plan: add Multi-Token Attention (MTA) mixer (arXiv:2504.00927)

**Paper:** *Multi-Token Attention*, Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar (FAIR at Meta), COLM 2025, arXiv:2504.00927v2.

MTA addresses the "single token attention" bottleneck of standard soft attention: each
attention weight conditions on the similarity of only one query and one key vector, so
locating a context region identified by *multiple* tokens (e.g. a sentence mentioning
both "Alice" and "rabbit") must be compressed into a single query vector. MTA keeps the
full quadratic attention skeleton but applies tiny learned convolutions over the
attention score maps — a **key-query convolution** (per head, kernel `c_q × c_k`, causal)
and a **head-mixing convolution** (non-overlapping groups of `c_h` heads) — both before
and after the softmax, followed by a **gated group normalization** before the final
concat. At 880M params / 105B SlimPajama tokens it beats Transformer, Diff Transformer
and Talking-Heads baselines on perplexity and long-context retrieval (multi-needle,
BabiLong) while adding only ~0.001% parameters. It is a trainable-from-scratch dense
softmax-attention variant — exactly the family this repo catalogues — with a clean causal
formulation and a KV-cache-compatible decode path (classic KV + a `c_q − 1` query tail).

---

## 2. Paper extract

### 2.1 Forward equations (verbatim, EN paper)

Standard multi-head attention (paper Eq. 1):

```
K = H W_k,  V = H W_v,  Q = H W_q                     (W_q, W_k, W_v ∈ R^{D×d}, M heads, d = D/M)
Â = Q Kᵀ / √d,   A = Softmax(Mask_{−∞}(Â))           (softmax over keys; Mask sets (i,j) to −∞ for i < j)
out = (A V) concat over heads · W_o,  W_o ∈ R^{D×D}
```

**Key-query convolution** (paper Eq. 3–5; Eq. 5 is the default "double causal mask" form):

```
A = Softmax( Conv2d_θ( Â ) )                                        (Eq. 3, exact)
a_ij = Softmax( Σ_{i′=0}^{c_q−1} Σ_{j′=−⌊c_k/2⌋}^{⌈c_k/2⌉−1}
                1_{i≥j−j′} θ_{i′,j′} q_{i−i′} k_{j−j′}ᵀ / √d )      (Eq. 4; only past queries, indicator kills future keys)
A = Softmax( Mask_{−∞}( Conv2d_θ( Mask_0( Â ) ) ) )                 (Eq. 5, DEFAULT — simpler, masks slightly more)
```

- `Conv2d_θ` spans the **query and key length dimensions**; batch and head are
  independent; **each head has its own θ** (`c_q × c_k` weights, no bias).
- Zero-padding before the conv so every position has a valid window.
- Post-softmax variant (paper Eq. 6, additive interaction instead of multiplicative):
  `A = Mask_0( Conv2d_θ( Softmax( Mask_{−∞}(Â) ) ) )`. Default implementation uses
  **both** pre- and post-softmax (one convolution each).

**Head-mixing convolution** (paper Eq. 7): for kernel size `c_h`, the M heads are
divided into `M/c_h` non-overlapping groups; within each group a per-group kernel
`w ∈ R^{c_h×c_h}` mixes heads (identical to a fully-connected layer over the group):

```
A¹_new = w₁₁ A¹ + w₁₂ A²,   A²_new = w₂₁ A¹ + w₂₂ A²                (Eq. 7, c_h = 2 example; can be pre- or post-softmax)
```

**Combination** (§3.3): when both mixings are pre-softmax they fuse into a single 3-D
conv (`c_q × c_k` over the (q,k) plane, third dim = head groups of `c_h`); default MTA
applies key-query conv **pre-softmax** and head conv **post-softmax** (plus the
post-softmax key-query conv — "we apply the MTA convolution both pre-softmax and
post-softmax" by default, §4).

**Gated group normalization** (§3.3, following Diff Transformer's Ye et al. 2024, with
layer-scaling replaced by sigmoid gating): before the final concat, per head the output
is group-normalized and scaled by `sigmoid(g_h)` scalar gates (one gate per head),
letting the model switch heads on/off across layers and push back against the residual
stream. Ablations: scalar gating 10.90 PPL vs depth scaling 10.92 vs no scaling 11.03 vs
layer-norm scaling 11.41 (Table 5) — scalar gating is the default.

### 2.2 Reference algorithm

Parallel / training form (per layer, per head group), the default configuration:

1. `Q, K, V = proj(H)`; `Â = QKᵀ/√d`; apply causal mask (0-fill) → `Conv2d_θ(c_q × c_k)`
   over (q, k) per head → causal mask (−∞) → softmax → **pre-softmax MTA output**.
2. Head conv (pre-softmax option) may instead fuse into step 1 as a 3-D conv over head
   groups; default keeps the head conv post-softmax (Eq. 7 applied to step-1 weights).
3. Apply the key-query conv **again post-softmax** (Eq. 6, 0-masked) on the weights.
4. Group norm + per-head sigmoid gate → `out = (A V) · W_o`.

Recurrent / decode form: standard KV cache; additionally the **last `c_q − 1` query
vectors** (and their per-head logits against the cached keys) must be retained so the
key-query conv window over the query axis can be rebuilt for the new step. Per-step cost:
`O(n·d)` for the score row plus `O(c_q · c_k · nH)` for the conv correction on the new
row and `O(c_h² · nH/c_h)` for the head mixing — negligible next to the KV matmul.

### 2.3 Parameter inventory and count formula

Learned tensors beyond standard MHA (per MTA layer):

| Tensor | Shape | Count |
|---|---|---|
| pre-softmax key-query kernels θ_pre | `[nH, c_q, c_k]` (no bias) | `nH·c_q·c_k` |
| post-softmax key-query kernels θ_post | `[nH, c_q, c_k]` (no bias) | `nH·c_q·c_k` |
| head-mixing kernels w (pre and post) | `[nH/c_h, c_h, c_h]` × 2 | `2·(nH/c_h)·c_h² = 2·nH·c_h` |
| group-norm gain + sigmoid gate | per head | `2·nH` |

**Exact extra-parameter formula** (with `H` = hidden, `nH` = heads, `nKv` = KV heads,
`hd = H/nH`; knobs at defaults, key-query conv applied on the layer):

```
extra_mta = nH · (2·c_q·c_k + 2·c_h) + 2·nH
total_mta_layer = 4·H·H (+biases per repo convention) + extra_mta
```

The paper's 880M run reports +0.001% parameters (Appendix Table 8). `nKv`, `fH` do not
enter the extra term (MTA is defined over query heads; GQA-style `nKv < nH` keeps the
standard GQA score path and MTA applies per query head — assumption to verify at
implementation time, §9).

### 2.4 Knob table

| Schema key | Leaf | Type | Default | Constraints | Notes |
|---|---|---|---|---|---|
| `model.attention.mta.cq` | `cq` | int | 6 | ≥1 | query-axis kernel size (paper default 6) |
| `model.attention.mta.ck` | `ck` | int | 11 | ≥1, odd | key-axis kernel size (paper default 11; odd so the window is centred) |
| `model.attention.mta.ch` | `ch` | int | 16 | ≥1, divides `num_heads` | head-mixing group/kernel size (paper default 16) |
| `model.attention.mta.pre_softmax` | `pre_softmax` | bool | true | — | apply key-query conv pre-softmax (Eq. 5) |
| `model.attention.mta.post_softmax` | `post_softmax` | bool | true | — | apply key-query conv post-softmax (Eq. 6) |
| `model.attention.mta.gated_group_norm` | `gated_group_norm` | bool | true | — | group norm + per-head sigmoid gate; false = plain attention scaling |

Knobs are read via `mixer_params(config, "mta", params)`; there is **no** flat dataclass
field and no rename entry. Kernel init is **identity** (paper §4.6: identity init — a
regular Transformer — beats zero/constant init; not a knob, hard-coded).

### 2.5 State & cache

- Static decode state: classic K/V cache **plus** a FIFO tail of the last `c_q − 1`
  query vectors per layer (query axis of the key-query conv). Compatibility classes
  **(i) + (v)** of `src/model/cache/capabilities.py` (KVSlot + `ConvTailSlot` /
  `TokenTailSlot` precedent, like the conv embedding tail).
- Head conv and group norm are per-step functions of the current score row — stateless.

### 2.6 Complexity

- Training: `O(n²·hd·nH)` standard + `O(n²·c_q·c_k·nH)` for the two convs over the
  score map → still `O(n²)` in sequence length; paper notes incompatibility with fused
  SDPA kernels (Appendix A) — the score map must be materialised.
- Inference (cached decode): `O(n·d)` per step + `O(c_q·c_k·nH)` conv correction.

### 2.7 Trainability / PE / modes

- Trainable from scratch (880M, 105B tokens); **not** training-free → not in
  `TRAINING_FREE_LAYERS`.
- Consumes positional information exactly like standard attention (RoPE-friendly in the
  paper) → `needs_pos_encoder=True`; assembler auto-injects `use_pe.mta_use_pe`.
- Both encoder and decoder; the causal form is the paper's default (decoder-only LM).
  No vision/audio restriction.

### 2.8 BibTeX

```bibtex
@inproceedings{golovneva_multi_token_2025,
  title     = {Multi-Token Attention},
  author    = {Golovneva, Olga and Wang, Tianlu and Weston, Jason and Sukhbaatar, Sainbayar},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2025},
  eprint    = {2504.00927},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `mta` | paper short name; snake_case; no `_attn` suffix |
| Class | `MultiTokenAttention` | `<Name>Attention` convention |
| Family | **Dense** (census 5 → 6), top-level module `src/model/attention/mta.py` | MTA is a full-quadratic softmax variant; baselines are Diff Transformer / Talking Heads (dense peers); annex-2 holds dense peers (`ca`) — user-confirmed |
| Peer | `src/model/attention/ca.py` + its spec (`src/registry/builtin/attention.py:32-52`) | top-level dense mixer with `params_schema` knobs, cache class (i), `cache_implemented=True` — mirrors constructor, `use_pe`, `numerics.module_softmax` dispatch, masking, S1/S2 call shapes |
| Knobs | 6 (table §2.4) in `schema/_model/_attention_mta.yaml` | paper defaults cq=6, ck=11, ch=16 |
| `needs_pos_encoder` | `True` | softmax-attention-like (§2.7) |
| KV-cache | **Full incremental** (user-confirmed): `cache=CacheSpec(supported=True, slot_type="kv", notes=…query tail c_q−1, classes (i)+(v)…)`, `cache_implemented=True`; add `"mta"` to `IMPLEMENTED_MIXERS` (`src/model/cache/capabilities.py:47`) and implement the incremental branch (KV + `ConvTailSlot`/`TokenTailSlot` query tail) → parity suite auto-covers it |
| `training_free` | `False` | trainable from scratch (§2.7) |
| `qk_norm_supported` / `qk_clip_supported` | `True` / `True` | forms per-head logits like `ca`/`standard` |
| Example | `configs/examples/dense_mta_adamw.yaml` only; **no** full preset (user-confirmed) | paper's 880M reference is beyond preset scale |
| Census bumps | mixers 50 → **51** (50 concrete + `mamba`); Dense **5 → 6**; e2e `ATTENTIONS` +1; `test_cache_specs_match_layer_pattern_enum` count 50 → 51; README "Mixer Categories" Dense row; `docs/architecture.md` §7.4 (~line 390) + census lines 25/82/253; `configs/README.md` | read from repo, not hardcoded |
| Paper variant scope | plan the default MTA (pre+post key-query conv, post-softmax head conv, scalar-gated group norm); alternatives (post-only, pre-only head conv, depth/layer-norm scaling) recorded as knobs/open questions | §4.6 ablations |

---

## 4. File-by-file plan

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/attention/mta.py` (new) | `MultiTokenAttention` docstring citing arXiv:2504.00927; `from __future__ import annotations`; Google-style docstrings; CPU-safe (no fused SDPA — score map materialised); builder-style `__init__(config, pos_encoder=None, params=None)` per `docs/architecture.md` §11.1 mirroring `ca.py`; Eq. 1 projections; Eq. 5 pre-softmax key-query conv (`Conv2d` over (q,k) per head, 0-mask then −∞ mask, identity init); Eq. 6 post-softmax conv; Eq. 7 head conv over `nH/ch` groups; gated group norm (per-head sigmoid gate); knobs via `mixer_params(config, "mta", params)`; `use_pe_for(config, "mta", True)`; honour `self.use_fixed_softmax` through `numerics.module_softmax`; encoder (no mask) and decoder (causal double-mask) paths; `forward` call shapes S1/S2 per §7.13; incremental branch: cached KV + `c_q−1` query tail rebuild of the conv window. |
| 2 | `src/model/attention/__init__.py` | Import `MultiTokenAttention` + `__all__` entry (top-level family — no subdirectory `__init__`). |
| 3 | — | `src/model/hybrid_layer.py` **no edit**: dispatch reads the registry; `TRAINING_FREE_LAYERS` derives from the spec flag (`training_free=False`). |
| 4 | `src/registry/builtin/attention.py` | Append `MixerSpec(name="mta", builder="src.model.attention.mta:MultiTokenAttention", description="Multi-Token Attention — causal key-query (cq×ck) and head-group (ch) convolutions over the attention score map, pre+post softmax, with gated group norm (paper: arxiv:2504.00927)", description_es="Atención Multi-Token — convoluciones causales consulta-clave (cq×ck) y por grupos de cabezas (ch) sobre el mapa de puntajes, pre+post softmax, con normalización de grupo con compuerta (paper: arxiv:2504.00927)", references=("2504.00927",), params_schema="schema/_model/_attention_mta.yaml#/properties/mta", defaults={"cq":6,"ck":11,"ch":16,"pre_softmax":True,"post_softmax":True,"gated_group_norm":True}, cache=CacheSpec(supported=True, slot_type="kv", notes="MTA (arXiv:2504.00927); classes (i)+(v) — classic KV plus a FIFO tail of the last cq−1 query vectors to rebuild the key-query conv window; head conv and gated group norm are per-step stateless."), cache_implemented=True, qk_norm_supported=True, qk_clip_supported=True)`. Single declaration: enum, `model.attention` fragment, `use_pe.mta_use_pe` all derive from it. |
| 5 | — | No flattener / flat rename entry exists (D15). |
| 6 | `src/schema/_model/_dims.yaml` | One 4-column row in **both** bilingual tables: `` `mta` | Multi-Token Attention — convolutions over attention scores (queries, keys, heads) with gated group norm (paper: arxiv:2504.00927) | …es… | `` — the enum list itself is registry-injected, do not touch. |
| 7 | `src/schema/_model/_attention_mta.yaml` (new) | Full bilingual knob block mirroring `_attention_ca.yaml` structure: `type: object`, `additionalProperties: false`, `mta` property with `title`/`title_es`, EN/ES `description` (mechanism, examples, references), `cq`/`ck`/`ch` (int, min 1, `ck` odd via `multipleOf` note in description), `pre_softmax`/`post_softmax`/`gated_group_norm` (bool), `defaults` matching the spec. |
| 8 | — | `src/schema/_model.yaml` no edit (assembler rebuilds `model.attention.properties`). |
| 9 | — | `_positional_encoding.yaml` no edit (assembler injects `use_pe.mta_use_pe`). |
| 10 | `src/model/cache/capabilities.py` | `CACHE_SPECS` already derives from the spec; real edits: add `"mta"` to `IMPLEMENTED_MIXERS` (frozenset, ~line 47, F1 dense section) + module docstring cache-class note if the convention requires it. |
| 11 | `configs/examples/dense_mta_adamw.yaml` (new) | Tiny encoder-MLM example modelled on `es_arch_fpa_adamw.yaml`: `model_class: frankenstein_encoder`, `layer_pattern: [mta]` (or `standard, mta, standard, mta`), small dims (`hidden_size: 96`, `num_heads: 4`), default knobs shown, AdamW keys, `training.task: mlm`; must pass `load_training_config` and schema validation. |
| 12 | — | No full preset (user decision). |
| 13 | `full_tests/run_e2e.py` | Append `"mta",` to `ATTENTIONS` (~lines 237–272) with a one-line comment `# Multi-Token Attention (arXiv:2504.00927): conv over attention score maps;` — auto-generates `attn_single_mta` + all `attn_<a>__mta` / `attn_mta__<b>` pairs. Trainable → belongs in the list. |
| 14 | `tests/test_schema_attention_layers.py` | Add `"mta"` to the hand-expected enum set; bump `test_cache_specs_match_layer_pattern_enum` count 50 → 51. |
| 15 | `tests/test_attention_modules.py` | `TestMultiTokenAttention`: `test_build_encoder_forward_shape`, `test_decoder_causal_forward`, `test_no_future_leakage` (shift-input check), `test_knobs_honored` (cq/ck/ch variants, ch divides nH), `test_identity_init_matches_standard` (zeroed extra params ≈ standard attention output), `test_no_nan_inf`, `test_bitnet_on`. |
| 16 | `tests/test_bitnet_coverage.py` | Add `"mta"` to the mixer list. |
| 17 | `tests/test_kv_cache_mixers.py` | **No hand edit** — parametrizes from `IMPLEMENTED_MIXERS`; full-forward vs prefill+decode parity comes free once row 10 lands. |
| 18 | `docs/paper/figures/mixers/mta.mmd` (new) | House-style mermaid (`%%{init}%%` header, `flowchart TD`): Q/K/V projections → scores `Â[B,nH,n,n]` → 0-mask → `Conv2d θ_pre (cq×ck)` → −∞ mask → softmax → head conv `w (ch×ch)` → post-softmax conv `θ_post` → gated group norm → `·V` → `W_o`; dotted edge for the cached query tail (`c_q−1`). |
| 19 | `docs/paper/figures/mixer-taxonomy.mmd` | Extend the Dense family chain with `mta`; bump its count label (Dense 5 → 6). |
| 20 | `docs/paper/figures/dense-map.mmd` | Add the `mta` node to the Dense family map (Dense has a map). |
| 21 | `docs/bibliography/attention_types.bib` | Add `golovneva_multi_token_2025` (§2.8). |
| 22 | `docs/paper/appendices/annex-2-recurrent-attention.tex` | `\subsection{<next>. Multi-Token Attention}` after the annex's last dense subsection (grep `ca`/`Continuum` to locate): `\fitfigure{figures/mixers/mta.pdf}` + label; Mathematical Foundation with Eqs. 3–7 verbatim; algorithm float (parallel form + cached decode with query tail); Empirical Performance (880M SlimPajama PPL 10.91 vs 11.25 Transformer; needle/BabiLong wins; +0.001% params) + strength/limitation table (limitation: incompatible with fused SDPA kernels, score map materialised); `\citep{golovneva_multi_token_2025}`; extend the annex's unified-principle list if present. |
| 23 | `docs/paper/sections/04-architecture-taxonomy.tex` | Dense family sentence (line ~17 area) gains MTA after the GQA/xsa items + count bump; quote old counts at implementation time. |
| 24 | `docs/paper/appendices/annex-summary-tables.tex` | `tab:all-mixers` row (`mta | Dense | O(n²·hd·nH + n²·cq·ck·nH) | KV + query tail | cq, ck, ch, …`) + Dense family header 5 → 6 + intro census 50 → 51. |
| 25 | `docs/paper/appendices/annex-schema.tex` | Mixer row: `mta` keys `cq/ck/ch/pre_softmax/post_softmax/gated_group_norm` with defaults and constraints. |
| 26 | `docs/paper-es/appendices/annex-<n>-dense>.tex` | Faithful Spanish translation of #22 (figures via `../paper/figures/`); locate by grepping the peer's escaped name in `docs/paper-es/appendices/`. |
| 27 | `docs/paper-es/sections/01-*.tex`, `04-*.tex`, ES `annex-summary-tables.tex`, `annex-schema.tex` | Spanish counterparts of #23–#25 + the section-01 global census 50 → 51. |
| 28 | `README.md`, `docs/architecture.md`, `configs/README.md` | README "Mixer Categories" Dense row (5 → 6, add `mta` + one-clause description, line ~168); `docs/architecture.md` census lines 25 / 82 / 253 and §7.4 table row + §7.13 cache list (KV + query-tail class); `configs/README.md` dense family list; grep the old numbers (`50`, `49 classes`, `Dense (5)`) everywhere and bump each occurrence. |
| 29 | `docs/source/api/model/attention/mta.rst` + `attention/index.rst` | `automodule:: src.model.attention.mta` page + toctree entry (top-level family dir). |
| 30 | `docs/schema.assembled.yaml` | Regenerate: `python -m src.registry.snapshot --out docs/schema.assembled.yaml` — must show `mta` in the enum, the `model.attention.mta` fragment and `use_pe.mta_use_pe`. |
| 31 | `docs/paper/appendices/annex-23-kv-cache.tex` + `docs/paper-es/` cache annex | Add `mta` to the incremental-decode list: class (i)+(v), KV + `c_q−1` query tail (verify the ES annex filename by grep at implementation time; EN is `annex-23-kv-cache.tex`). |
| 32 | `docs/paper/sections/01-introduction.tex` (+ES), `10-*.tex`, `11-*.tex` | Global census bump 50 → 51 wherever the total mixer count appears (FPA precedent; abstract sentence lives in section 01). |

**Hard rules restated:** schema is the contract (`additionalProperties: false`); the
registry is the single source of truth (D13/D14) — spec-first, no second hand-edited
list; runtime config is nested only (D15); `hidden_size % num_heads == 0`, `num_kv_heads`
divides `num_heads`, `len(layer_pattern) == num_layers`; inference never fails because of
the cache (D5) — non-cached paths fall back to full recompute; the papers are the theory
source of truth (D11) — annex equations must match arXiv:2504.00927v2 and `paper-es`
translates them faithfully.

---

## 5. Ordered execution checklist

1. [ ] Implement rows 1–2 (module + export).
2. [ ] Row 4 (spec) + row 7 (knob YAML) + row 6 (bilingual `_dims.yaml` rows).
3. [ ] Row 10 (IMPLEMENTED_MIXERS + incremental branch) — parity suite then covers `mta`.
4. [ ] Row 11 (example YAML).
5. [ ] Rows 13–17 (e2e list, tests, bitnet coverage).
6. [ ] Row 30 (snapshot regeneration) — before docs that quote it.
7. [ ] Targeted gates:

```bash
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_attention_modules.py tests/test_bitnet_coverage.py \
  tests/test_registry_core.py tests/test_registry_binding.py \
  tests/test_schema_assembly.py tests/test_kv_cache_mixers.py \
  -v --tb=short -p no:warnings
```

8. [ ] Full suite: `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`
9. [ ] e2e smoke with **real-data training**:

```bash
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F mta

uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only mta            # real Borges corpus (default --realonall)

uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_mta   # minimal solo run if the pair sweep is heavy

uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only mta --no-realonall  # offline fallback
# sizing: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
```

Expected: `--list-combos` shows `attn_single_mta` + every pair; each selected combo PASS
with finite training loss on real corpus windows; NaN/Inf, schema rejection or timeout is
a blocker.

10. [ ] Rows 18–20 figures; render: `( cd docs/paper/figures && ./build-figures.sh both )` (chromium available).
11. [ ] Rows 21–28, 31, 32 docs EN/ES.
12. [ ] Row 29 Sphinx: `bash docs/build_docs.sh`.
13. [ ] Param ground truth + estimator:

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py   # 0 failures
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01                                   # ALL PASS
```

14. [ ] PDF rebuild (pdflatex + bibtex available at `/usr/bin`):

```bash
cd docs/paper
pdflatex -interaction=nonstopmode paper.tex && bibtex paper \
  && pdflatex -interaction=nonstopmode paper.tex \
  && pdflatex -interaction=nonstopmode paper.tex
rm -f *.aux *.log *.out *.toc *.blg
cd ../paper-es   # repeat with paper-es.tex
```

Verify: no `Fatal error`, no `Citation .* undefined`, `golovneva_multi_token_2025`
resolves, PDFs newer than the edited `.tex`/`.bbl`, `git status --short docs/paper
docs/paper-es` shows only intended files.

15. [ ] Website sync (§6), then commit (§7).

Fallbacks: no TeX → leave sources updated, rebuild before next release; no Chrome →
commit `.mmd` and note the render; HF Hub unreachable → record the `--no-realonall`
PASS and re-run real-data later.

---

## 6. Website sync (planned, not executed)

Per `.opencode/command/sync-website.md`:

1. Bootstrap/refresh the `erickfmm.github.io` mirror; copy per its file map:
   regenerated `docs/schema.assembled.yaml`, `src/schema/**`, `src/registry/**`,
   `configs/*.yaml` + `configs/README.md` (now listing `dense_mta_adamw.yaml`),
   `configs/examples/**`, `docs/paper/**` + `docs/paper-es/**` (tex, bbl, pdf, and the
   **new `mixers/mta.svg`**, `mixer-taxonomy.svg`, `dense-map.svg` renders),
   `docs/bibliography/**`, `docs/architecture.md`.
2. Site-only edits (mirror is their source of truth):
   - `ft-diagram.js`: add `CATS["mta"]` (Dense) + a `SUBSTRUCT["mta"]` tensor-flow
     template mirroring the `.mmd` dataflow (scores → 0-mask → Conv2d → −∞ mask →
     softmax → head conv → post-conv → gated group norm → ·V → W_o, dotted query-tail
     edge); reuse the Dense `STYLES` class — no new class.
   - `ft-param-estimator.js`: add the `mixerParams()` case with the exact formula from
     §2.3: `extra = nH·(2·cq·ck + 2·ch) + 2·nH` on top of the standard
     attention projection count (bias- and knob-aware; `0` when only one softmax-side
     conv is enabled).
   - `index.html`: grep a peer (`ca`, `standard`); extend every hardcoded mixer list
     found with `mta`.
3. Paper tab HTML rebuild (`flatten_tex.py --web` + `pandoc` + `postprocess_html.py`,
   en + es) and `paper/en.pdf` / `paper/es.pdf` copies.
4. Validation Gate: `param_count_check.py` → `0 failures` (intentional
   `param_truth.json` diff), estimator `--tol 0.01` → `ALL PASS`, JSON round-trip,
   CLI parse, state-script probes (`schema-refs`, `example-yaml`, `cli-flags`, …) PASS.
5. Mirror stays uncommitted until the commit phase.

---

## 7. Commit plan (planned, not executed)

1. **Main repo** — `feat(attention): add Multi-Token Attention mixer (2504.00927)`:

   ```bash
   git add src/model/attention/mta.py src/model/attention/__init__.py \
     src/registry/builtin/attention.py src/model/cache/capabilities.py \
     src/schema/_model/_dims.yaml src/schema/_model/_attention_mta.yaml \
     configs/examples/dense_mta_adamw.yaml full_tests/run_e2e.py \
     tests/test_schema_attention_layers.py tests/test_attention_modules.py \
     tests/test_bitnet_coverage.py \
     docs/paper/figures/mixers/mta.mmd docs/paper/figures/mixers/mta.pdf \
     docs/paper/figures/mixers/mta.svg docs/paper/figures/mixer-taxonomy.mmd \
     docs/paper/figures/mixer-taxonomy.pdf docs/paper/figures/mixer-taxonomy.svg \
     docs/paper/figures/dense-map.mmd docs/paper/figures/dense-map.pdf \
     docs/paper/figures/dense-map.svg \
     docs/bibliography/attention_types.bib \
     docs/paper/appendices/annex-2-recurrent-attention.tex \
     docs/paper/appendices/annex-summary-tables.tex docs/paper/appendices/annex-schema.tex \
     docs/paper/appendices/annex-23-kv-cache.tex docs/paper/sections/01-introduction.tex \
     docs/paper/sections/04-architecture-taxonomy.tex \
     docs/paper/sections/10-*.tex docs/paper/sections/11-*.tex docs/paper/paper.pdf \
     docs/paper-es/appendices/ docs/paper-es/sections/ docs/paper-es/paper-es.pdf \
     README.md docs/architecture.md configs/README.md \
     docs/source/api/model/attention/mta.rst docs/source/api/model/attention/index.rst \
     docs/schema.assembled.yaml
   git commit  # conventional body: module, spec, schema rows+knobs, cache decision,
              # example, tests, e2e + real-data smoke result, figures, docs EN/ES,
              # snapshot, PDFs
   git push origin HEAD:main
   ```

   Never stage `*.aux *.log *.out *.toc *.blg`, caches, mermaid temp files;
   `full_tests/param_truth.json` diff (if any) is intentional and staged explicitly.

2. **Mirror** — `git -C erickfmm.github.io …` commit `docs(site): sync Multi-Token
   Attention mixer (2504.00927)`; stage the copied trees + `ft-diagram.js`,
   `ft-param-estimator.js`, `index.html`, rebuilt paper HTML/PDFs; then
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Schema gates | targeted pytest (§5.7) | all PASS; enum count 51 |
| Full suite | §5.8 | 0 failures |
| e2e enumeration | `--category attn --list-combos \| grep -F mta` | `attn_single_mta` + all pairs |
| e2e real-data training | `--category attn --only mta` | all selected combos PASS, finite loss on real corpus |
| e2e solo | `--only attn_single_mta` | PASS |
| e2e offline | `--only mta --no-realonall` | PASS (fallback evidence) |
| KV parity | `tests/test_kv_cache_mixers.py` | `mta` case PASS (full vs prefill+decode) |
| Figures | `build-figures.sh both` | `mta.svg/pdf`, taxonomy/dense-map re-rendered |
| PDFs | §5.14 | no fatal/undefined-citation errors; new `\citep` resolves |
| Sphinx | `bash docs/build_docs.sh` | builds with `mta.rst` |
| Param count | `param_count_check.py` | `0 failures` |
| Estimator | `param_estimate_check.mjs … --tol 0.01` | `ALL PASS` (≤1%) |
| Mirror probes | sync-website Validation Gate | all PASS |
| Snapshot | regenerated `docs/schema.assembled.yaml` | contains `mta` enum, fragment, `use_pe.mta_use_pe` |

---

## 9. Risks, assumptions and open questions

- **Line drift**: all anchors (spec tuple ~line 32, `ATTENTIONS` ~237–272,
  `IMPLEMENTED_MIXERS` ~47, README ~164–178, architecture 25/82/253/390) were read at
  planning time — re-grep before editing.
- **GQA interaction**: the paper defines MTA over full MHA; with `num_kv_heads <
  num_heads` the score path follows the repo's GQA convention and MTA applies per query
  head — assumption to verify against `ca.py`/`gqa` handling.
- **Conv tail mechanics**: confirm whether `ConvTailSlot`/`TokenTailSlot` stores raw
  queries or per-step score rows; the minimal state is the last `c_q−1` queries (recompute
  their score rows against cached keys). If the slot API makes raw-query storage awkward,
  store score-row tails (larger memory, simpler parity).
- **QK-norm interaction**: `qk_norm_supported=True` assumed; verify the norm applies
  before the conv (same place standard attention applies it).
- **Naming collisions**: grep confirms no `mta`/`MultiTokenAttention` exists in
  `src/`, `configs/`, `docs/`, or the estimator.
- **`ch` divides `num_heads`** must be validated at config level (`if/then` or
  runtime check in the builder with a clear error).
- **Per-layer key-query conv cadence**: the paper applies the key-query conv every 4th
  layer for efficiency (head conv on all layers). In this repo `layer_pattern` already
  mixes per layer — the example alternates `standard`/`mta` instead of adding a cadence
  knob; recorded as an open question (a `qk_conv_every` knob is a possible follow-up).
- **BitNet**: weight-quantized projections unaffected (conv kernels are tiny
  non-projection params — verify the BitNet linears do not try to quantize them).
- Tool availability confirmed: `pdflatex`, `bibtex`, `chromium`, `node` all present.
  HF Hub reachability for the real-data smoke is an implementation-time check.

## 10. Out of scope

No code, schema, preset, test, figure, doc, PDF, mirror or git change was made by this
command — this document is the only file written. The pre-existing untracked
`.opencode/plan/add-mixer-2603.09078.md` (another session's plan) was left untouched.
