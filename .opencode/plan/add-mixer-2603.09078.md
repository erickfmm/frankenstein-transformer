# Plan: add XSA (Exclusive Self Attention) mixer (arXiv:2603.09078)

**Paper**: Zhai, S. (2026). *Exclusive Self Attention*. arXiv:2603.09078 (Apple technical report).
XSA is a minimal, trainable modification of softmax attention that removes the
"attention similarity bias": the attention output tends to have high cosine
similarity with the token's own value vector, so SA wastes capacity
re-aggregating what the residual+FFN path already carries. XSA runs standard
causal SDPA and then **orthogonalizes the output against the token's own
(normalized) value vector**, forcing the attention layer to carry only
contextual, non-self information. Zero new parameters, two lines of code,
consistent LM gains up to 2.7B params, growing with sequence length. It is a
textbook Dense-family mixer for this registry — the closest structural
relatives already here are `standard` (identical projections/cache) and
`gated_softmax` (identical post-SDPA-modification pattern).

---

## 2. Paper extract (verbatim)

### Forward equations (Eq. 1–2 of the paper)

Standard SA (Eq. 1):

```latex
q_i = W_q x_i, \quad k_j = W_k x_j, \quad v_j = W_v x_j, \quad
a_{i,j} = \frac{\exp(q_i^T k_j)}{\sum_{j'=1}^{i} \exp(q_i^T k_{j'})}, \quad
y_i = \sum_{j=1}^{i} a_{i,j} v_j,
```

XSA (Eq. 2) — same first line, then one extra step:

```latex
z_i = y_i - (y_i^T v_i) \frac{v_i}{\|v_i\|_2^2}
```

i.e. `v̂_i = v_i / ‖v_i‖₂` and `z_i = y_i − (y_i · v̂_i) v̂_i` — remove the
projection of the attention output onto the *self* value direction. Output
projection `out = z W_o` (multi-head, per head dim `d_h = D/H`).

### Algorithm (Algorithm 1, parallel/training form — the only form)

```python
# x: (B,T,D); Wq, Wk, Wv, Wo: (D,D); H heads
Q = (x @ Wq).reshape(B,T,H,D//H).transpose(1,2)
K = (x @ Wk).reshape(B,T,H,D//H).transpose(1,2)
V = (x @ Wv).reshape(B,T,H,D//H).transpose(1,2)
Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)   # standard SDPA
Vn = F.normalize(V, dim=-1)                                    # XSA step
Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn                # XSA step
out = Z.transpose(1,2).reshape(B,T,D) @ Wo
```

No recurrent form is given; the exclusion is **row-local** (uses only the
current token's `v_i`), so any standard KV cache carries all the state needed.

### Parameter inventory and exact parameter-count formula

Learned tensors (all bias-free in the paper; `(D,D)` each):

| Name | Shape | Bias |
|---|---|---|
| `W_q` (`q_proj`) | `(H, H)` i.e. `hidden × hidden` | no |
| `W_k` (`k_proj`) | `(H, H)` | no |
| `W_v` (`v_proj`) | `(H, H)` | no |
| `W_o` (`out_proj`) | `(H, H)` | no |

**Parameter count = 4·H²** — identical to `standard` attention (XSA adds
*zero* parameters). No `num_kv_heads` path (paper is MHA-only); no head-geometry
knobs. `ft-param-estimator.js`'s `mixerParams` case for `xsa` must equal the
`standard` case exactly.

### Hyperparameters / knobs

**None.** The paper defines no knobs (uses `F.normalize`'s implicit eps). Like
`kda` before its fragment, "no knobs" is the outcome: no
`_attention_xsa.yaml`, no `params_schema`, no `defaults` on the spec, no
`attention.xsa` config subkey. (The module may internally use
`F.normalize(v, dim=-1)` default eps `1e-12` — a constant, not a knob.)

### State & cache

- No recurrent state, no extras. Per-step decode cost: standard KV attention
  `O(t·d_h)` + one `O(d_h)` Gram–Schmidt step per head (negligible).
- Compatibility class **(i) classic per-token KV** (`src/model/cache/capabilities.py`
  §classes, lines 23–30). The cached branch mirrors `gated_softmax`'s precedent
  (spec note: "the gate depends only on the current token — nothing extra to
  cache") — here: the exclusion depends only on the current token's `v_i`,
  which is always available at decode; parity is exact by construction.

### Complexity

- Training: `O(n²·d)` per head (softmax attention) + `O(n·d)` exclusion — paper
  Fig. 2 measures "minimal overhead" (bfloat16, B200).
- Inference (cached): `O(t·d)` per step + `O(d)` exclusion.

### Trainability

Fully trainable from scratch (LM pre-training 0.7B/1.4B/2.7B, FineWeb-100BT,
100B tokens). **Not** training-free → never in `TRAINING_FREE_LAYERS`.

### Positional encoding

Softmax-attention-like; the paper's reference stack uses **RoPE** on Q/K.
→ `needs_pos_encoder=True` (the flag's default), matching `standard`,
`gated_softmax`, `sigmoid`.

### Mode constraints

Paper is decoder/causal LM only in experiments, but the mechanism is
mode-agnostic (any attention mask); the module implements encoder
(bidirectional) + decoder (causal) like every peer. Valid in both modes; no
task restriction (rule 6 unaffected). Not grid/vision-specific; audio/ViT
models may use it like any dense mixer.

### BibTeX data

- Author: Shuangfei Zhai (Apple)
- Year: 2026
- Title: *Exclusive Self Attention*
- arXiv: 2603.09078 (v1, 10 Mar 2026)
- Key: `zhai_exclusive_2026`

```bibtex
@article{zhai_exclusive_2026,
  title   = {Exclusive Self Attention},
  author  = {Zhai, Shuangfei},
  journal = {arXiv preprint arXiv:2603.09078},
  year    = {2026}
}
```

---

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `xsa` | Paper acronym; user-confirmed |
| Class / module | `ExclusiveSelfAttention` in **top-level** `src/model/attention/xsa.py` | Dense family peers `standard.py`, `sigmoid.py`, `memory.py` are top-level; no gate → not `gated/` |
| Paper family label | **Dense** (chain `standard, sigmoid, ca, gqa` → 4 becomes 5) | `docs/architecture.md` §7.4 table rows 397–400; no new family → no new map/STYLES |
| Closest peers | `src/model/attention/standard.py` (full-recompute path, projections, PE, qk-norm) + `src/model/attention/gated/gated_softmax.py` (post-SDPA modification pattern + `cached_qk_attention` cached path, lines 159–182) | Both read verbatim above |
| Knobs | **none** (no fragment file, no spec `defaults`/`params_schema`) | Paper defines none |
| `needs_pos_encoder` | `True` (flag default — may simply be omitted; every Dense peer omits it meaning True) | Softmax attention; paper uses RoPE |
| KV cache | **Decision 1 (simple)**: `cache=CacheSpec(supported=True, slot_type="kv", notes=...)`, `cache_implemented=True`, incremental branch via `cached_qk_attention`, `"xsa"` added to `IMPLEMENTED_MIXERS` (F1 dense section) | Exclusion is row-local on `v_i`; class (i) |
| QK stabilization | `qk_norm_supported=True`, `qk_clip_supported=True` | Identical score path to `standard`/`gated_softmax` |
| Training-free | No | Fully trainable |
| Example preset | `configs/examples/es_arch_xsa_adamw.yaml` (encoder MLM, `es_arch_ca_adamw.yaml` shape minus the `ca`/`retnet` subkeys) | User-confirmed **example only** — no `configs/xsa.yaml` |
| e2e | Append `"xsa"` to `ATTENTIONS` (~line 300) with an arXiv comment | Trainable → belongs in the pair sweep |
| Census bumps | Enum 50 → **51**; Dense 4 → **5**; `len(CACHE_SPECS)` assert 50 → **51** (`tests/test_schema_attention_layers.py:167`); `README.md:28,145,180` "50 mixers" → 51 ("50 mixer classes plus mamba"); `docs/architecture.md:392` "50 enum values in 9 families" → 51; `mixer-taxonomy.mmd:5` `d["Dense (4)"]` → `d["Dense (5)"]` + `--> d4["xsa"]`; `annex-summary-tables.tex` intro + Dense family header; `.mmd` count 101 → 102 | Actual numbers read from the repo this session |

---

## 4. File-by-file plan table

Execution order top to bottom. Rows marked *(none)* are explicitly verified
non-edits.

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/attention/xsa.py` | New module. Docstring cites Zhai 2026, arXiv:2603.09078, with the equations. `from __future__ import annotations`; Google-style docstrings; `params = mixer_params(config, "xsa", params)` (line ~65 pattern from `standard.py`) even though there are no knobs (keeps the contract). Constructor copies `StandardAttention.__init__` (`standard.py:64–85`) verbatim: q/k/v/out `BitLinear if config.bitnet.enabled else nn.Linear` (bias=False), dropout, `mode`, `pe_type`, `use_pe = bool(use_pe_for(config, "xsa", True))`, `attach_qk_norm(self, config, self.head_dim)`. `forward(x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)`: **full path** = `standard.py:158–170` (apply_pe_to_qk → scores → record_qk_max_logits → causal mask in decoder mode → `module_softmax` → dropout), then keep `Y = attn_weights @ v` in `(B,H,n,hd)`, `vn = F.normalize(v, dim=-1)`, `Y = Y - (Y * vn).sum(-1, keepdim=True) * vn`, reshape, `out_proj`. **Cached path** = `gated_softmax.py:159–182` (`cached_qk_attention(...)` with `use_fixed_softmax=self.use_fixed_softmax`, `causal=True`) returning `attended` `(B,H,n,hd)`; orthogonalize against the current-step `v` exactly as above; `out_proj`. No `dense_forward` (not training-free). |
| 2 | `src/model/attention/__init__.py` | Import `ExclusiveSelfAttention` + `__all__` entry, mirroring the `StandardAttention`/`SigmoidAttention` lines (no family subpackage involved). |
| 3 | *(none)* `src/model/hybrid_layer.py` | No edit: dispatch reads the registry; `TRAINING_FREE_LAYERS` derives from `training_free` (absent → not training-free); `use_fixed_softmax` attached generically (~line 176). |
| 4 | `src/registry/builtin/attention.py` | One `MixerSpec` appended to `ATTENTION_SPECS` (alphabetical position: after `wr`/between `tucker` and `mamba` — place it where sorting keeps the tuple tidy, e.g. after `tucker`): `name="xsa"`, `builder="src.model.attention.xsa:ExclusiveSelfAttention"`, `description="Exclusive Self Attention — post-SDPA orthogonalization against the self value vector (paper: arxiv:2603.09078)"`, `description_es="Atención Exclusiva — ortogonalización post-SDPA frente al vector de valor propio (paper: arxiv:2603.09078)"`, `references=("2603.09078",)`, `cache=CacheSpec(supported=True, slot_type="kv", notes="XSA (arXiv:2603.09078); class (i) — the exclusion z_i = y_i − (y_i·v̂_i)v̂_i depends only on the current token's value, so a plain KV slot carries everything (gated_softmax precedent); exact parity.")`, `cache_implemented=True`, `qk_norm_supported=True`, `qk_clip_supported=True`. (No `params_schema`/`defaults`/`training_free`; omit `needs_pos_encoder` or set `True`.) Single declaration — enum, `use_pe.xsa_use_pe`, fragment absence all derive from it. |
| 5 | *(none)* | Config flattener is gone; no flat rename entry exists. |
| 6 | `src/schema/_model/_dims.yaml` | One 4-column row in **each** bilingual `layer_pattern` table: EN after the `ca` row (line 226): `` | `xsa` | Exclusive Self Attention — post-SDPA orthogonalization against the self value vector (paper: arxiv:2603.09078) | ✅ | ✅ | ``; ES after line 287: `` | `xsa` | Atención Exclusiva — ortogonalización post-SDPA frente al vector de valor propio (paper: arxiv:2603.09078) | ✅ | ✅ | ``. Enum list untouched (registry-injected). |
| 7 | *(none)* | No knobs → no `_attention_xsa.yaml`. |
| 8 | *(none)* `src/schema/_model.yaml` | Assembler rebuilds `model.attention.properties` from spec fragments. |
| 9 | *(none)* `src/schema/_model/_positional_encoding.yaml` | Assembler injects `use_pe.xsa_use_pe` from `needs_pos_encoder`. |
| 10 | `src/model/cache/capabilities.py` | Add `"xsa",  # F1 (post-SDPA exclusion over current-step v)` to `IMPLEMENTED_MIXERS` in the F1 dense/softmax section (~line 56, next to `gated_softmax`). `CACHE_SPECS` derives from the spec — no other edit. |
| 11 | `configs/examples/es_arch_xsa_adamw.yaml` | Copy `configs/examples/es_arch_ca_adamw.yaml` with: `layer_pattern: [xsa]`; delete the whole `attention:` block (no knobs); rename `csv_log_path: training_metrics_es_arch_xsa_adamw.yaml`. Everything else (dims 512/10 layers/8 heads, mlm task, adamw prefixed params, dataset block) unchanged. Must pass `load_training_config`. |
| 12 | *(none)* | User chose example only — no `configs/xsa.yaml`. |
| 13 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (~line 300, continuing the comment-per-mixer style): `# Exclusive Self Attention (arXiv:2603.09078): standard softmax attention + post-SDPA orthogonalization against the self value vector; trains natively in encoder and decoder modes; exact KV-cache incremental decoding.` then `"xsa",`. Auto-generates `attn_single_xsa` + all pairs — no builder edits. |
| 14 | `tests/test_schema_attention_layers.py` | Add `"xsa"` to the `expected` set (line ~18–70); bump `self.assertEqual(len(CACHE_SPECS), 50)` (line 167) → `51`. |
| 15 | `tests/test_attention_modules.py` (dense family test — confirm the file that holds `TestStandardAttention`/`TestSigmoidAttention` at implementation time) | `TestXSAAttention`: `test_forward_encoder_shape`, `test_forward_decoder_causal` (output at position i unchanged when future tokens change), `test_exclusion_orthogonalizes` (cos(y_i, v_i) ≈ 0 in the head-shaped pre-projection output via a direct call, or: with attention restricted to a single token, output ≈ 0), `test_bitnet_on` (BitLinear path), `test_no_nan_inf`, `test_cached_parity` smoke (covered fully by suite #17). |
| 16 | `tests/test_bitnet_coverage.py` | Add `"xsa"` to the mixer list. |
| 17 | *(auto)* `tests/test_kv_cache_mixers.py` | **No hand edit** — parametrizes from `IMPLEMENTED_MIXERS`; row 10 membership makes full-forward vs prefill+decode parity run automatically. |
| 18 | `docs/paper/figures/mixers/xsa.mmd` | House-style mermaid (copy `sigmoid.mmd` header/init): flowchart TD — `x["x (B,n,H)"]` → `q_proj/k_proj/v_proj` nodes → scores → softmax → SDPA output `Y` → `vn["v̂ = v/‖v‖"]` → `dot["(Y·v̂)"]` → `sub["Z = Y − (Y·v̂)v̂"]` (dotted edge from `v_proj` to `vn`/`dot`) → `out_proj` → `z`. Tensor-shaped labels, labelled edges. |
| 19 | `docs/paper/figures/mixer-taxonomy.mmd` | Line 5: `d["Dense (4)"]` → `d["Dense (5)"]`, chain `... --> d3["gqa"] --> d4["xsa"]`. |
| 20 | `docs/paper/figures/dense-map.mmd` | Add an `xsa` node wired next to `standard`/`sigmoid` (same diff style as the last addition). |
| 21 | `docs/bibliography/attention_types.bib` | Entry `zhai_exclusive_2026` (verbatim block from §2 above). |
| 22 | `docs/paper/appendices/annex-2-recurrent-attention.tex` | New `\subsubsection{Exclusive Self Attention}` inserted after *Continuum Attention* (line 88 block, ends before `\subsubsection{Retentive Networks...}` line 193) keeping the dense mixers grouped: `\fitfigure{figures/mixers/xsa.pdf}` + `\label{fig:xsa}`; **Mathematical Foundation** with Eq. 1–2 verbatim (§2 above) + the attention-similarity-bias motivation (cos⟨y_i, v_i⟩ grows with depth, Fig. 1) + implicit attention sink reading (§4 comparison); algorithm float (Algorithm 1 pseudocode); **Empirical Performance and Trade-offs** — FineWeb-100BT 0.7/1.4/1.4B/2.7B, +0.26/+1.03/+1.36 avg downstream points, gains grow with seq length (512→16k), LR-robust, attention-sink-robust, zero parameters, negligible overhead; strength/limitation itemization (strengths: parameter-free, minimal overhead, long-context; limitations: LM-only evidence, per-token normalization cost, no theory); `\citep{zhai_exclusive_2026}`; extend the annex's closing unified-principle list if one exists. |
| 23 | `docs/paper/sections/04-architecture-taxonomy.tex` | Dense family paragraph: add one sentence ("XSA strips the self-value direction from the softmax output (arXiv:2603.09078), a parameter-free exclusion that sharpens the SA/FFN division of labor") + bump the family count mention (grep `Dense` / the old count near the family prose). |
| 24 | `docs/paper/appendices/annex-summary-tables.tex` | `tab:all-mixers` row `xsa & Dense & $O(n^2 d)$ / $O(t d)$ + $O(d)$ exclusion & 0 extra params & — & \citep{zhai_exclusive_2026} \\` (match actual column schema); Dense family header count 4→5; intro census 50→51. |
| 25 | `docs/paper/appendices/annex-schema.tex` | Row after `ca` (line 246 pattern): `xsa & Exclusive Self Attention --- post-SDPA orthogonalization against the self value vector & \checkmark & \checkmark \\` (match actual columns). |
| 26 | `docs/paper-es/appendices/annex-2-recurrent-attention.tex` | Faithful Spanish `\subsubsection{Atención Exclusiva (XSA)}` mirroring row 22; `\fitfigure{../paper/figures/mixers/xsa.pdf}`; equations identical (math is not translated), prose translated; `\citep{zhai_exclusive_2026}`. |
| 27 | `docs/paper-es/sections/01-*.tex`, `04-*.tex`, `annex-summary-tables.tex`, `annex-schema.tex` | Spanish counterparts of rows 23–25 + the global census sentence in section 01 (grep `50` / `Densa`). |
| 28 | `README.md`, `docs/architecture.md`, `configs/README.md` | README line 28 `**50 mixer architectures** — 49 mixer classes...` → `**51 mixer architectures** — 50 mixer classes plus the mamba linear passthrough; the schema exposes all 51`; lines 145 & 180 `50` → `51`; Mixer Categories table Dense row +1. architecture.md §7.4 line 392 `50 enum values in 9 families` → `51 enum values in 9 families`; table row `` | `xsa` | Dense | — | yes | `` after `gqa` (line 400). configs/README.md Dense family list += `xsa`. Grep every remaining `50 mixers`/`Dense (4)` occurrence and fix. |
| 29 | `docs/source/api/model/attention/xsa.rst` + `docs/source/api/model/attention/index.rst` | Top-level `automodule:: src.model.attention.xsa` page (copy `sigmoid.rst` shape) + toctree entry in `index.rst` (no family subdir — dense lives at the top level). |
| 30 | `docs/schema.assembled.yaml` | Regenerate: `python -m src.registry.snapshot --out docs/schema.assembled.yaml` (adds `xsa` to the enum, `use_pe.xsa_use_pe`). |
| 31 | `docs/paper/appendices/annex-15-kv-cache.tex` + `docs/paper-es/appendices/annex-23-cache-kv.tex` | Add `xsa` to the incremental-decode (class i) list: note that the exclusion consumes only the current step's `v_i`, so the plain `KVSlot` decode path is exact. |
| 32 | `docs/paper/sections/01-introduction.tex` (+ ES), `10-*.tex`, `11-*.tex` | Global census bump wherever the total mixer count appears (grep `50` in those files; abstract-style sentence lives in 01). |

### Hard rules restated

Schema is the contract (`additionalProperties: false`); the registry is the
single source of truth (D13/D14) — row 4 is the only declaration; runtime
config is nested only (D15); `hidden_size % num_heads == 0`,
`num_kv_heads` divides `num_heads`, `len(layer_pattern) == num_layers`;
inference never fails because of the cache (D5); `xsa` is fully trainable so
rule 5 (training-free) does not apply; the paper is the theory source of
truth (D11) — annex formulations must match arXiv:2603.09078 and `paper-es`
must translate them faithfully.

---

## 5. Ordered execution checklist

- [ ] 1. Row 1–2: module + exports
- [ ] 2. Row 4: registry spec
- [ ] 3. Row 6: bilingual `_dims.yaml` rows
- [ ] 4. Row 10: `IMPLEMENTED_MIXERS`
- [ ] 5. Row 11: example YAML
- [ ] 6. Row 13: e2e `ATTENTIONS`
- [ ] 7. Rows 14–16: tests (17 auto-covers)
- [ ] 8. Row 30: regenerate snapshot
- [ ] 9. Targeted gates + full suite (below)
- [ ] 10. e2e smoke incl. real-data training (below)
- [ ] 11. Rows 18–20: figures + render
- [ ] 12. Rows 21–28, 31, 32: docs EN/ES
- [ ] 13. Row 29: Sphinx
- [ ] 14. PDFs EN/ES
- [ ] 15. Param truth + JS estimator (site files first, §6)
- [ ] 16. Website sync (§6)
- [ ] 17. Commits (§7)

### Commands (run at implementation time — NOT now)

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_modules.py \
  tests/test_bitnet_coverage.py tests/test_registry_core.py \
  tests/test_registry_binding.py tests/test_schema_assembly.py \
  tests/test_kv_cache_mixers.py \
  -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

```bash
# e2e — enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F xsa

# e2e TARGETED SMOKE — ONLY xsa: actual training on REAL data (Borges corpus;
# --realonall is the default; first run downloads tokenizer/corpus from HF Hub).
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only xsa

# minimal fallback if the pair sweep is too heavy
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_xsa

# offline synthetic fallback (no HF download)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only xsa --no-realonall

# sizing: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU: swap the extra for --extra cu118 --extra train.
```

Expected evidence: every selected combo (`attn_single_xsa` + all
`attn_<a>__xsa` / `attn_xsa__<b>` pairs) reports **PASS** with finite training
loss over real corpus windows; run dir keeps `config.yaml` + metrics. NaN/Inf
loss, schema rejection or timeout = blocker.

```bash
# figures (chromium available: /usr/bin/chromium)
( cd docs/paper/figures && ./build-figures.sh both )

# param ground truth + JS estimator (after the example lands)
uv run --extra cpu --extra train python full_tests/param_count_check.py
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01

# Sphinx
bash docs/build_docs.sh
```

```bash
# PDF rebuild (pdflatex + bibtex available)
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

**Preconditions/fallbacks**: `pdflatex`/`bibtex`/`chromium`/`node` all
confirmed present this session. If the HF Hub is unreachable, record the
`--no-realonall` PASS and re-run the real-data pass later. Post-PDF checks:
no `Fatal error`, no `Citation .* undefined`, `paper.pdf`/`paper-es.pdf`
newer than the edited `.tex`/`.bbl`, `zhai_exclusive_2026` resolves, and
`git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf`/figure files.

---

## 6. Website sync (planned, not executed)

Follow `.opencode/command/sync-website.md`:

1. Copy per the file map: regenerated `docs/schema.assembled.yaml` +
   `src/schema/**`, `src/registry/**`, `configs/examples/**` +
   `configs/README.md`, `docs/paper/**` + `docs/paper-es/**` (tex, bbl, pdf,
   and the **new `xsa.svg`** — sources/PDFs stay in the main repo),
   `docs/bibliography/**`, `docs/architecture.md`.
2. **Site-only edits** (mirror `erickfmm.github.io/frankenstein-transformer/`,
   fetch from `https://raw.githubusercontent.com/erickfmm/erickfmm.github.io/main/frankenstein-transformer/` if not cloned):
   - `ft-diagram.js`: `CATS["xsa"] = "dense"` (match however dense peers are
     keyed); `SUBSTRUCT["xsa"]` — tensor-flow template cloning `standard`'s
     with two extra nodes `v̂ = v/‖v‖₂` and `Z = Y − (Y·v̂)v̂` between the
     softmax aggregate and `W_o`. No new `STYLES` class (existing Dense family).
   - `ft-param-estimator.js`: `mixerParams()` case for `xsa` identical to
     `standard` (formula `4*H*H`, no knobs) — keep it a distinct case, not an
     alias, so future divergence is cheap.
   - `index.html`: grep a dense peer (`sigmoid`/`ca`); extend every hardcoded
     mixer list found with `xsa`.
3. Paper tab HTML rebuild (`flatten_tex.py --web` + `pandoc` +
   `postprocess_html.py`, en + es) and top-level `paper/en.pdf` / `es.pdf`
   copies.
4. Validation Gate: `param_count_check.py` → `0 failures` (regenerates
   `full_tests/param_truth.json` — intentional diff); estimator `--tol 0.01` →
   `ALL PASS`; JSON round-trip, CLI parse, and state-script probes
   (`schema-refs`, `example-yaml`, `cli-flags`, …) all PASS.
5. Mirror stays uncommitted until the commit phase.

---

## 7. Commit plan (planned, not executed)

`/make-a-commit` conventions (English, conventional tag, structured body;
never stage `*.aux *.log *.out *.toc *.blg`, caches or mermaid temp files;
`param_truth.json` diffs are intentional).

1. **Main repo** — `feat(attention): add XSA mixer (arXiv:2603.09078)`.
   Body: module + exports; registry `MixerSpec` (single declaration);
   bilingual `_dims.yaml` rows; `IMPLEMENTED_MIXERS` + exact KV decode branch;
   example `es_arch_xsa_adamw.yaml`; tests (enum set, `CACHE_SPECS` 51,
   family test class, bitnet coverage, auto KV parity); e2e `ATTENTIONS` +
   real-data smoke result; figures (`xsa.mmd`, taxonomy Dense (5),
   `dense-map.mmd`, rendered pdf/svg); docs EN/ES (annex-2 subsubsections,
   summary/schema/kv-cache annexes, sections 01/04/10/11, README /
   architecture §7.4 / configs README census 51); bibliography entry; Sphinx
   `xsa.rst`; regenerated `docs/schema.assembled.yaml`; rebuilt PDFs. Then
   `git push origin HEAD:main`.
   Stage: `src/model/attention/xsa.py src/model/attention/__init__.py
   src/registry/builtin/attention.py src/schema/_model/_dims.yaml
   src/model/cache/capabilities.py configs/examples/es_arch_xsa_adamw.yaml
   full_tests/run_e2e.py tests/test_schema_attention_layers.py
   tests/test_attention_modules.py tests/test_bitnet_coverage.py
   docs/paper/figures/mixers/xsa.{mmd,pdf,svg}
   docs/paper/figures/mixer-taxonomy.{mmd,pdf,svg}
   docs/paper/figures/dense-map.{mmd,pdf,svg}
   docs/bibliography/attention_types.bib
   docs/paper/appendices/annex-2-recurrent-attention.tex
   docs/paper/appendices/annex-summary-tables.tex
   docs/paper/appendices/annex-schema.tex
   docs/paper/appendices/annex-15-kv-cache.tex
   docs/paper/sections/ docs/paper-es/ README.md docs/architecture.md
   configs/README.md docs/source/api/model/attention/
   docs/schema.assembled.yaml docs/paper/paper.pdf docs/paper/paper.bbl
   docs/paper-es/paper-es.pdf docs/paper-es/paper-es.bbl full_tests/param_truth.json`
2. **Mirror** — `git -C erickfmm.github.io …` commit
   `docs(site): sync XSA mixer (arXiv:2603.09078)`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted pytest | the §5 targeted block | all green |
| Full pytest | `python -m pytest tests/ --continue-on-collection-errors …` | all green |
| Schema census | `test_cache_specs_match_layer_pattern_enum` | `len(CACHE_SPECS) == 51`, enum contains `xsa` |
| Bilingual tables | `test_layer_pattern_enum_matches_documentation` | EN + ES rows present |
| Example loads | `test_yaml_examples.py` | `es_arch_xsa_adamw.yaml` validates + builds |
| KV parity | `tests/test_kv_cache_mixers.py` (auto) | `xsa` full-forward ≡ prefill+decode |
| e2e enumeration | `--category attn --list-combos \| grep -F xsa` | `attn_single_xsa` + every pair id |
| e2e real-data training | `--category attn --only xsa` | all selected combos PASS, finite loss |
| e2e solo fallback | `--category attn --only attn_single_xsa` | PASS |
| e2e offline | `--only xsa --no-realonall` | PASS |
| Figures | `build-figures.sh both` | `xsa.{pdf,svg}` regenerated; taxonomy/dense-map updated |
| Param truth | `param_count_check.py` | `0 failures`; xsa = 4·H² |
| Estimator | `param_estimate_check.mjs … --tol 0.01` | `ALL PASS` |
| Sphinx | `bash docs/build_docs.sh` | `xsa.rst` page builds |
| PDF EN/ES | §5 pdflatex sequence | no fatal/undefined-citation errors; cite resolves |
| Mirror probes | sync-website Validation Gate | all PASS |

---

## 9. Risks, assumptions and open questions

- **Assumption — line drift**: all cited line numbers (registry tuple,
  `ATTENTIONS` ~267–300, `_dims.yaml` 226/287, `len(CACHE_SPECS)` 167,
  `mixer-taxonomy.mmd` 5, architecture 392/397–400, README 28/145/180) were
  verified this session; re-grep before editing.
- **Assumption — dense family test file**: confirm whether
  `tests/test_attention_modules.py` holds the `standard`/`sigmoid` test
  classes (row 15) and mirror that file's naming.
- **Annex numbering**: `annex-2` uses `\subsubsection` (not `\subsection`);
  insert XSA after *Continuum Attention* to keep dense peers grouped.
- **NaN edge**: `F.normalize` eps (1e-12) guards zero-norm value vectors in
  early training; fixed-softmax/fixed-cross-entropy swaps flow through
  `module_softmax` unchanged (no bespoke normalizer, unlike `ca`/`sigmoid` —
  annex-15-fixed-numerics needs no edit).
- **QK-clip/norm**: score path identical to `standard`; both flags True —
  verify with `tests/test_qk_norm.py` after wiring.
- **Encoder mode**: paper is causal-only; encoder (bidirectional) XSA is a
  straightforward extrapolation (the exclusion is mask-agnostic) — flag this
  in the annex as a repo extension beyond the paper's evaluation.
- **No open questions remain** (name and preset settled by the user; family,
  cache class and peer are unambiguous).

## 10. Out of scope

No code, schema, preset, test, figure, doc, PDF, mirror or git change was made
by this command. This document is the only file written.
