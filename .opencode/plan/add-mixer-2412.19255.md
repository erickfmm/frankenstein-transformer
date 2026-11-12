# Plan: add MFA — Multi-matrix Factorization Attention (arXiv:2412.19255)

**Paper**: Hu, Li, Zhang, Wang, Zhou, Zhang, Shum, Jiang — *Multi-matrix Factorization
Attention* (StepFun / Tsinghua / Fudan / Megvii), arXiv:2412.19255v2 [cs.LG], Jan 2025.

MFA attacks the KV-cache-capacity trade-off of MHA variants (MQA/GQA/MLA): the paper
models every MHA variant as a low-rank factorization of Fully Parameterized Bilinear
Attention (FPBA) and shows that the *number and dimension of heads* — not just the
compression rank — govern capacity. MFA factorizes the QK circuit as
$W_c = S_q Q_c S_k^{\top}$ with shared $S_q, S_k, S_v \in \mathbb{R}^{H\times C}$ and
head-specific $Q_c, O_c \in \mathbb{R}^{C\times C}$: many high-rank heads (rank $C > d$)
at parameter cost $\approx CH$ per extra head, with a **single shared K/V head**
($k_j = x_j S_k$, $v_j = x_j S_v \in \mathbb{R}^C$) so the KV cache per token is $2C$,
independent of head count. MFA-KR additionally re-parameterizes
$S_v = (I + \mathrm{diag}(\alpha) N) S_k$ with zero-initialized $\alpha$, reusing the key
cache as value (cache $= C$ per token). At 7B/1T tokens MFA beats MHA accuracy with
87.5% less cache; MFA-KR reaches 93.7% less. This is exactly this repo's
latent/low-rank family (peer: `mla`, `mlra`, `tpa`) and it trains from scratch in both
modes — no `TRAINING_FREE_LAYERS` involvement.

---

## 2. Paper extract (verbatim)

### Forward equations

Inference formulation (paper Eq. 6; the paper writes the head sum with $n$ but Table 1
and all configs use $m$ heads for MFA — $m$ is used below and in the plan):

$$O_i = \sum_{c=1}^{m}\Big(\sum_{j=1}^{i} \phi\big(\tfrac{x_i S_q Q_c (x_j S_k)^{\top}}{\sqrt{d}}\big)\, x_j S_v\Big) O_c^{\top}$$

Factorization formulation (paper Eq. 7):

$$O_i = \sum_{c=1}^{m}\Big(\sum_{j=1}^{i} \phi\big(\tfrac{x_i (S_q Q_c S_k^{\top}) x_j^{\top}}{\sqrt{d}}\big)\, x_j S_v O_c^{\top}\Big)$$

where $\phi$ is softmax, $S_q, S_k, S_v \in \mathbb{R}^{H\times C}$ are shared across
heads, $Q_c, O_c \in \mathbb{R}^{C\times C}$ are head-specific, $C$ is the low-rank
factorization dimension (per-head rank), $m$ the head count ($m > n$ typical), and
$d$ the softmax scale divisor (paper scales by $\sqrt{d}$ with $d$ the MHA head dim;
implementation uses $\sqrt{C}$, the per-head dim — record in the annex as the natural
reading; see Open questions).

MFA-KR re-parameterization (paper Eqs. 8–9):

$$S_v = S_k + \alpha \odot N S_k = (I + \mathrm{diag}(\alpha) N)\, W_K,\qquad N \in \mathbb{R}^{C\times C},\ \alpha \in \mathbb{R}^{C}$$

i.e. the value of token $j$ is computed from its cached key: $v_j = k_j (I + \mathrm{diag}(\alpha) N)$, $k_j = x_j S_k$. **$\alpha$ is initialized as the zero
vector** — $S_v = S_k$ at step 0 — which the authors found crucial for training
stability (Table 4: vanilla KR 6.55, +extra value proj 7.88 unstable, +residual 6.65,
+zero-init gating = MFA-KR 6.45 vs MFA 6.35).

Scale: $\phi(\cdot/\sqrt{d})$; RoPE is applied directly to the $C$-dim $q_c$ and $k$
vectors (single-key design, paper Table 6: no decoupled-RoPE machinery needed,
unlike MLA); ALiBi also validated (Table 5).

### Algorithm

*Parallel/training form* (per layer, standard SDPA path):

1. $K = X S_k \in \mathbb{R}^{T\times C}$, $V = X S_v \in \mathbb{R}^{T\times C}$ (single head each; $V = K\,(I+\mathrm{diag}(\alpha)N)$ when `key_reuse`).
2. For each head $c$: $Q^{(c)} = X S_q Q_c \in \mathbb{R}^{T\times C}$.
3. Apply PE (RoPE/ALiBi/none) to $Q^{(c)}$ and $K$.
4. $A^{(c)} = \mathrm{softmax}(Q^{(c)} K^{\top}/\sqrt{d} + M)$; $Y^{(c)} = A^{(c)} V \in \mathbb{R}^{T\times C}$.
5. $O = \sum_c Y^{(c)} O_c^{\top}$ (concat heads × per-head output projection to $H$).

*Recurrent/decode form*: append $k_t, v_t \in \mathbb{R}^C$ to the cache (with
`key_reuse`, cache **only** $k_t$ and recompute $v_t = k_t(I+\mathrm{diag}(\alpha)N)$);
per step compute $q_t^{(c)} = x_t S_q Q_c$, PE, then one-row softmax over the cached
keys — classic single-KV-head decode (compatibility class (i)).

### Parameter inventory (exact count formula)

| Tensor | Shape | Bias | Notes |
|---|---|---|---|
| $S_q$ (W_q_shared) | $H \times C$ | no | shared query down-projection |
| $S_k$ (W_k_shared) | $H \times C$ | no | shared key down-projection (the only K projection) |
| $S_v$ (W_v_shared) | $H \times C$ | no | only when **not** `key_reuse` |
| $Q_c$ (W_q_heads) | $m \times C \times C$ | no | head-specific query factors |
| $O_c$ (W_o_heads) | $m \times C \times H$ | no | per-head output projection (Table 1 accounts $mCH$ here) |
| $N$ | $C \times C$ | no | only when `key_reuse` |
| $\alpha$ | $C$ | — | only when `key_reuse`; **zero-init** |

Parameter-count formula, with $H$ = `hidden_size`, $m$ = `num_heads`,
$C$ = `mfa.factor_rank`:

- **MFA** (`key_reuse: false`): $P = 3HC + mC^2 + mCH$ (paper Table 1: $H(3C + mC) + mC^2$).
- **MFA-KR** (`key_reuse: true`): $P = 2HC + mC^2 + mCH + C^2 + C$ ($S_v$ replaced by $N, \alpha$).

These are the formulas `full_tests/param_count_check.py` and
`ft-param-estimator.js: mixerParams` must agree on (tol 1%).

### Knobs (→ spec `params_schema` fragment in `_attention_latent.yaml#/properties/mfa`)

| Schema key | Type | Default | Constraint | EN description | ES description |
|---|---|---|---|---|---|
| `model.attention.mfa.factor_rank` ($C$) | int | 128 | $\geq 1$ (typically $d_{\text{mha}} < C \le H$) | Low-rank factorization dimension; per-head rank and shared K/V width (paper: 256 at H=2048). | Dimensión de factorización de bajo rango; rango por cabeza y ancho de K/V compartido (paper: 256 con H=2048). |
| `model.attention.mfa.key_reuse` | bool | false | — | Enable MFA-KR: value re-parameterized from the cached key via $S_v=(I+\mathrm{diag}(\alpha)N)S_k$, $\alpha$ zero-initialized; halves the KV cache again. | Habilita MFA-KR: valor reparametrizado desde la clave en caché vía $S_v=(I+\mathrm{diag}(\alpha)N)S_k$, $\alpha$ inicializado en cero; reduce la caché KV a la mitad otra vez. |

Spec `defaults = {"factor_rank": 128, "key_reuse": False}`. The head count $m$ is the
global `model.dims.num_heads` (invariant `hidden_size % num_heads == 0` still enforced
structurally; MFA does not use `num_kv_heads`).

### State & cache

- Cached per token: $k \in \mathbb{R}^C$ and $v \in \mathbb{R}^C$ post-PE (MFA), or only
  $k \in \mathbb{R}^C$ (MFA-KR). One cache "head" of width $C$ — **compatibility class
  (i)**, classic KV with `H_kv = 1`, `D = C`; per-step decode cost $O(m \cdot T \cdot C)$.
- Matches `KVSlot` (`slot_type="kv"`): the module materializes the $m$ query heads from
  one shared K/V — the same slot shape `gqa`/`memory` use, just with a single head.

### Complexity

- Training: $O(T^2 m C)$ time, $O(T C)$ extra memory — same as MHA with $d := C$ but
  cache/KV width decoupled from head count.
- Inference decode: $O(m T C)$ per step; KV cache $2C$ floats/token (MFA) or $C$
  (MFA-KR), independent of $m$.

### Trainability / PE / modes

- Fully trainable from scratch (7B/1T-token runs); **not** training-free → no
  `TRAINING_FREE_LAYERS`, no `dense_forward` surrogate.
- Consumes positional information (RoPE natively; ALiBi validated) →
  `needs_pos_encoder=True` (the default).
- Paper is decoder/causal only, but the formulation is plain bidirectional-capable
  SDPA over shared K/V → supports encoder (MLM) and decoder (causal) modes like peers;
  no vision/audio restrictions beyond the family norm.

### BibTeX

```bibtex
@article{hu_mfa_2025,
  title   = {Multi-matrix Factorization Attention},
  author  = {Hu, Jingcheng and Li, Houyi and Zhang, Yinmin and Wang, Zili and Zhou, Shuigeng and Zhang, Xiangyu and Shum, Heung-Yeung and Jiang, Daxin},
  journal = {arXiv preprint arXiv:2412.19255},
  year    = {2025}
}
```

---

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `mfa` (no `_attn` suffix) | paper short name; `grep -i mfa src/` → no collision |
| Class | `MFAAttention` in `src/model/attention/latent/mfa.py` | latent family (low-rank shared projections), like `mla` |
| Family label | **Latent** (paper family "Latent/low-rank") | `annex-3-latent-attention.tex`, `latent-map.mmd`, `_attention_latent.yaml` all exist |
| Peer | `src/model/attention/latent/mla.py` + the TPA commit `7c58c6d` as the scope blueprint | shared $S_{q,k,v}$ projections, latent KV pattern |
| Knobs | `factor_rank` (128), `key_reuse` (false) — one mixer covers MFA and MFA-KR | **user-approved**; precedent `gmla.use_nope` |
| `needs_pos_encoder` | `True` (default) | RoPE/ALiBi native |
| KV-cache decision | **(1) simple decode form**: `cache=CacheSpec(supported=True, slot_type="kv", notes="MFA (arXiv:2412.19255); class (i) — single shared K/V head of width C post-PE; with key_reuse only K is cached and V = K(I+diag(α)N) is recomputed per step.")`, `cache_implemented=True`, incremental branch + `IMPLEMENTED_MIXERS` membership | class-(i) single-head KV, cheaper than `mla`'s per-step re-expansion |
| Training-free | No | fully trainable |
| Example | `configs/examples/latent_mfa_adamw.yaml` | `es_arch_*` shape |
| Full preset | `configs/mfa.yaml` (H=2048, L=24, m=18, `factor_rank: 256`, MoE stripped → dense decoder) | **user-approved** |
| Census bumps | mixer enum **54 → 55** (53 concrete + `mamba` → 54 concrete + `mamba`); latent family **13 → 14** modules | `README.md:25,82,180`, `docs/architecture.md:82`, `tests/test_schema_attention_layers.py:53,180` |
| e2e | append `"mfa"` to `ATTENTIONS` (`full_tests/run_e2e.py` ~line 245) → auto `attn_single_mfa` + all pairs | TPA precedent |

---

## 4. File-by-file plan

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/attention/latent/mfa.py` | New module. Docstring cites arXiv:2412.19255 + equations. `from __future__ import annotations`, Google docstrings, CPU-safe. `MFAAttention(config, pos_encoder=None, params=None)`; knobs via `mixer_params(config, "mfa", params)` (`factor_rank=C`, `key_reuse`); `use_pe_for(config, "mfa", True)`; honor `self.use_fixed_softmax` via `numerics.module_softmax` (peer `mla.py`); shared `W_q/W_k/W_v` `H×C`, `W_q_heads` `m×C×C`, `W_o_heads` `m×C×H`; `key_reuse`: no `W_v`, instead `N` (`C×C`) + `alpha` (`C`, `zeros_`); forward: K/V = single head, per-head Q, PE on Q/K, SDPA (`math`/`sdpa`) with causal mask in decoder mode, per-head output projection + reshape; `α` zero-init; cached decode branch writing/reading a single-head `KVSlot` (S1/S2 shapes per `docs/architecture.md` §7.13; with `key_reuse`, V recomputed from cached K each step). |
| 2 | `src/model/attention/latent/__init__.py` + `src/model/attention/__init__.py` | Import `MFAAttention`, `__all__` entry (mirror the `tpa` lines). |
| 3 | `src/model/hybrid_layer.py` | **No edit** — dispatch reads the registry; `mfa` is not training-free so `TRAINING_FREE_LAYERS` (derived, ~line 80) is untouched. |
| 4 | `src/registry/builtin/attention.py` | `MixerSpec(name="mfa", builder="src.model.attention.latent.mfa:MFAAttention", description="Multi-matrix Factorization Attention — shared low-rank S_q/S_k/S_v + per-head Q_c/O_c, single shared K/V head (+ MFA-KR key-reuse variant) (paper: arxiv:2412.19255)", description_es="Atención de Factorización Multimatriz — S_q/S_k/S_v compartidos de bajo rango + Q_c/O_c por cabeza, una sola cabeza K/V compartida (+ variante MFA-KR de reúso de clave) (paper: arxiv:2412.19255)", references=("2412.19255",), params_schema="schema/_model/_attention_latent.yaml#/properties/mfa", defaults={"factor_rank": 128, "key_reuse": False}, cache=CacheSpec(supported=True, slot_type="kv", notes=see §3), cache_implemented=True, qk_norm_supported=True, qk_clip_supported=True, needs_pos_encoder=True)` — inserted alphabetically near `mla` (~line 702). Single declaration; enum/fragment/`use_pe` key derive from it. |
| 5 | *(removed)* | No flat rename entry — the flattener is gone (D15). |
| 6 | `src/schema/_model/_dims.yaml` | One 4-column row in **both** bilingual `layer_pattern` tables: EN `\| `mfa` \| Multi-matrix Factorization Attention — shared low-rank QK factorization, single shared K/V head (paper: arxiv:2412.19255) \|` and the ES counterpart. The enum itself is registry-injected — do not hand-edit. |
| 7 | `src/schema/_model/_attention_latent.yaml` | Add `mfa:` knob block after the `tpa` block (~line 433): `factor_rank` (int, default 128, minimum 1, bilingual description w/ example + reference) and `key_reuse` (bool, default false, bilingual, notes α zero-init and cache halving). |
| 8–9 | *(removed)* | `_model.yaml` / `_positional_encoding.yaml`: no edits (assembler generates `model.attention.mfa` and `use_pe.mfa_use_pe`). |
| 10 | `src/model/cache/capabilities.py` | Add `"mfa",  # (i) shared single-head KV of width C; key_reuse caches K only` to `IMPLEMENTED_MIXERS` (~line 47, near `"tpa"` line 70 comment list); `CACHE_SPECS` derives from the spec. |
| 11 | `configs/examples/latent_mfa_adamw.yaml` | New tiny example (mirror `latent_tpa_adamw.yaml`): encoder MLM, small dims (e.g. H=256, L=4, heads=4, `factor_rank: 64`, `key_reuse: false`), AdamW, `load_training_config`-clean. |
| 12 | `configs/mfa.yaml` | Full preset (user-approved): decoder `causal_lm`, H=2048, L=24, heads=18, `factor_rank: 256` (paper Table 7), `key_reuse: false`, RMSNorm + SwiGLU + RoPE base 500000 as `es_arch_*` peers allow; MoE stripped to dense FFN. |
| 13 | `full_tests/run_e2e.py` | Append `"mfa",` to `ATTENTIONS` (~lines 237–272) with comment `# Multi-matrix Factorization Attention (arXiv:2412.19255): shared low-rank QK factorization, single shared K/V head; MFA-KR via mfa.key_reuse.` → auto-generates `attn_single_mfa` + every pair. |
| 14 | `tests/test_schema_attention_layers.py` | Add `"mfa"` to the expected enum set (~line 53) **and** bump `self.assertEqual(len(CACHE_SPECS), 54)` → `55` (~line 180); extend the knob map (~line 105) with `"mfa": {"factor_rank", "key_reuse"}`. |
| 15 | `tests/test_latent_attention.py` (family file — confirm by grepping `tpa`) | `TestMFAAttention`: build on tiny config; encoder + decoder forward; output shape `(B,T,H)`; `factor_rank`/`key_reuse` honored; `key_reuse=True` starts as exact MFA-with-shared-S_v equivalence at init (α=0 ⇒ v=k ⇒ same as S_v=S_k); no NaN/Inf; gradients flow to all tensors incl. `alpha`; exact parameter formula ($3HC + mC^2 + mCH$, and $2HC + mC^2 + mCH + C^2 + C$ under KR); BitNet-on forward. |
| 16 | `tests/test_bitnet_coverage.py` | Add `"mfa"` to the mixer list. |
| 17 | `tests/test_kv_cache_mixers.py` | **No hand edit** — parametrizes from `IMPLEMENTED_MIXERS`; full-forward vs prefill+decode parity (rope/nope/alibi) covers `mfa` automatically (both `key_reuse` settings — extend the parametrization knobs if the suite carries per-mixer knob dicts; follow the `tpa` pattern). |
| 18 | `docs/paper/figures/mixers/mfa.mmd` | House-style mermaid (init header, `flowchart TD`): `X[B,T,H] → S_q/W_q_shared[H,C] → W_q_heads[m,C,C] → Qc` and `X → S_k[H,C] → K[B,T,C]`, `X → S_v[H,C] (or N,α dotted "key reuse" edge from K) → V`, PE on Q/K, per-head SDPA box `A=φ(QKᵀ/√d)V`, `W_o_heads[m,C,H] → O[B,T,H]`; cache note `KV/token = 2C (C with key_reuse)`. |
| 19 | `docs/paper/figures/mixer-taxonomy.mmd` | Extend the Latent chain with `mfa` and bump the family count label (13 → 14). |
| 20 | `docs/paper/figures/latent-map.mmd` | Add the `mfa` node (mirror the `tpa` 9-line addition) + rebuild pdf/svg. |
| 21 | `docs/bibliography/attention_types.bib` | `hu_mfa_2025` entry (§2 above). |
| 22 | `docs/paper/appendices/annex-3-latent-attention.tex` | New `\subsection{Multi-matrix Factorization Attention (MFA)}` (49th; currently 48 `\subsection`s): figure `\fitfigure{figures/mixers/mfa.pdf}` + label; Mathematical Foundation with Eqs. 6–9 verbatim, the FPBA/Table-1 capacity framing (TER = mC, SLSD = C), param formulas; algorithm float (parallel + cached decode incl. key-reuse); Empirical Performance (7B/1T: 49.9 vs 49.0 avg acc, 24.6K vs 196.6K KV/token; MFA-KR 12.3K, Table 4 ablation); strength/limitation table (cache independent of head count / K and V share one C-dim bottleneck); `\citep{hu_mfa_2025}`; extend the annex's closing unified-principle list if present. |
| 23 | `docs/paper/sections/04-architecture-taxonomy.tex` | Latent family paragraph sentence + counts (grep the old count next to the latent list; TPA bumped it at commit 7c58c6d). |
| 24 | `docs/paper/appendices/annex-summary-tables.tex` | `tab:all-mixers` row: `mfa & Latent & $O(T^2 mC)$ / $O(mTC)$ & $2C$ ($C$ KR) & shared $S_{q,k,v}$, per-head $Q_c,O_c$ & \citep{hu_mfa_2025}`; bump the latent family header count + intro census. |
| 25 | `docs/paper/appendices/annex-schema.tex` | `mfa` mixer row: knobs `factor_rank` (128), `key_reuse` (false). |
| 26 | `docs/paper-es/appendices/annex-3-latent-attention.tex` | Faithful Spanish translation of #22 (figures via `../paper/figures/`). |
| 27 | `docs/paper-es/sections/01-introduccion.tex`, `04-taxonomia-arquitectura.tex`, `10-discusion.tex`, `annex-summary-tables.tex`, `annex-schema.tex` | Spanish counterparts of #23–#25 + global census bump in 01 (mirror the 2-line diffs of commit 7c58c6d; also `docs/paper/sections/01-introduction.tex`, `10-*.tex`, `11-*.tex` EN census). |
| 28 | `README.md`, `docs/architecture.md`, `configs/README.md` | Census pass: README:25 "54 mixer values"→55, :82 "54 mixers (53 classes…)"→55 (54 classes), :180, Mixer Categories latent row; architecture §7.4 census + row + §7.13 cache list (add `mfa`) + :82; configs/README latent list + new example/preset rows. Grep every occurrence of `54`/`53` counts first. |
| 29 | `docs/source/api/model/attention/latent/mfa.rst` + `latent/index.rst` | `automodule:: src.model.attention.latent.mfa` page + toctree entry (7-line `tpa.rst` pattern). |
| 30 | `docs/schema.assembled.yaml` | Regenerate: `uv run --extra cpu --extra train python -m src.registry.snapshot --out docs/schema.assembled.yaml`. |
| 31 | `docs/paper/appendices/annex-23-kv-cache.tex` + `docs/paper-es/appendices/annex-23-cache-kv.tex` | Add `mfa` to the incremental-decode list (class (i), single shared head; note the key-reuse K-only cache). |
| 32 | `full_tests/param_truth.json` | Regenerated by `param_count_check.py` (intentional diff; new configs join). |

**Hard rules restated**: schema is the contract (`additionalProperties: false`); the
registry is the single source of truth (D13) and the validated schema is core YAML +
registry fragments (D14); the runtime config is nested only (D15);
`hidden_size % num_heads == 0`, `num_kv_heads | num_heads`, `len(layer_pattern) == num_layers`;
inference never fails because of the cache (D5); papers are the theory source of truth
(D11) — annex formulations must match arXiv:2412.19255 and `paper-es` must translate
them faithfully. `mfa` is fully trainable → no rule-5 wording anywhere.

---

## 5. Ordered execution checklist (run at implementation time, NOT now)

1. [ ] Rows 1–2: module + exports
2. [ ] Row 4: `MixerSpec` (registry)
3. [ ] Rows 6–7: bilingual `_dims.yaml` rows + `_attention_latent.yaml` knob block
4. [ ] Rows 10, 13: `IMPLEMENTED_MIXERS` + e2e `ATTENTIONS`
5. [ ] Rows 11–12: example + preset YAMLs
6. [ ] Rows 14–17: tests
7. [ ] Targeted gates:

```bash
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_latent_attention.py \
  tests/test_bitnet_coverage.py tests/test_kv_cache_mixers.py \
  tests/test_registry_core.py tests/test_registry_binding.py \
  tests/test_schema_assembly.py \
  -v --tb=short -p no:warnings
```

8. [ ] Full suite: `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`
9. [ ] e2e smoke (real data, default `--realonall`; first run downloads the Borges corpus/tokenizer):

```bash
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --list-combos | grep -F mfa
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --only mfa
# heavy? solo only:
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --only attn_single_mfa
# offline fallback:
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --only mfa --no-realonall
# sizing: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800 ; GPU: --extra cu118 --extra train
```

   Expected: every selected combo (`attn_single_mfa` + ~54 pairs) PASS with finite
   loss on real corpus windows; NaN/Inf, schema rejection or timeout = blocker.
10. [ ] Rows 18–20: figures, then `( cd docs/paper/figures && ./build-figures.sh both )` (chromium ✓ at `/usr/bin/chromium`)
11. [ ] Rows 21–28, 31: docs EN/ES
12. [ ] Row 29: Sphinx `bash docs/build_docs.sh`
13. [ ] Param gates:

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py   # 0 failures
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01                                   # ALL PASS
```

14. [ ] Row 30: regenerate `docs/schema.assembled.yaml`
15. [ ] PDFs (`pdflatex`/`bibtex` ✓):

```bash
cd docs/paper && pdflatex -interaction=nonstopmode paper.tex && bibtex paper && \
pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex && \
rm -f *.aux *.log *.out *.toc *.blg
cd ../paper-es && pdflatex -interaction=nonstopmode paper-es.tex && bibtex paper-es && \
pdflatex -interaction=nonstopmode paper-es.tex && pdflatex -interaction=nonstopmode paper-es.tex && \
rm -f *.aux *.log *.out *.toc *.blg
```

16. [ ] Website sync (§6) → commits (§7)

**Verification after PDFs**: no `Fatal error`, no `Citation .* undefined`, PDFs newer
than edited `.tex`/`.bbl`, `hu_mfa_2025` resolves, `git status --short docs/paper docs/paper-es`
shows only intended files. **Fallbacks**: no TeX → leave sources, rebuild before the
next release (website sync copies prebuilt PDFs only); no Chrome → commit `.mmd`, render
before sync; HF Hub unreachable → record the `--no-realonall` PASS and re-run real-data
later.

---

## 6. Website sync (planned; per `.opencode/command/sync-website.md`)

1. Mirror copy: regenerated `docs/schema.assembled.yaml`, `src/schema/**`,
   `src/registry/**`, `configs/*.yaml` + `configs/README.md`, `configs/examples/**`,
   `docs/paper/**` + `docs/paper-es/**` (tex/bbl/pdf + new `mfa.svg`, `latent-map.svg`,
   `mixer-taxonomy.svg`), `docs/bibliography/**`, `docs/architecture.md`.
2. Site-only edits:
   - `ft-diagram.js`: `CATS["mfa"]` (Latent category) + `SUBSTRUCT["mfa"]` tensor-flow
     template (or string alias to `mla`) matching the shared-`S` dataflow with the
     dotted key-reuse gate; no new `STYLES` class (existing latent family).
   - `ft-param-estimator.js`: `mixerParams()` case with the exact formulas:
     `key_reuse ? 2*H*C + m*C*C + m*C*H + C*C + C : 3*H*C + m*C*C + m*C*H`.
   - `index.html`: grep `tpa` / `mla`; extend every hardcoded mixer list found (incl. counts 54→55).
3. Paper tab HTML rebuild (`flatten_tex.py --web` + `pandoc` + `postprocess_html.py`, en+es) + top-level `paper/en.pdf` / `es.pdf`.
4. Validation Gate: `param_count_check.py` → 0 failures; estimator `--tol 0.01` → ALL
   PASS; JSON round-trip; CLI parse; state-script probes (`schema-refs`, `example-yaml`,
   `cli-flags`, …) PASS.
5. Mirror stays uncommitted until §7.

## 7. Commit plan (planned)

1. **Main repo** — stage exactly:
   `src/model/attention/latent/mfa.py src/model/attention/latent/__init__.py src/model/attention/__init__.py src/registry/builtin/attention.py src/schema/_model/_dims.yaml src/schema/_model/_attention_latent.yaml src/model/cache/capabilities.py configs/examples/latent_mfa_adamw.yaml configs/mfa.yaml full_tests/run_e2e.py tests/test_schema_attention_layers.py tests/test_latent_attention.py tests/test_bitnet_coverage.py docs/paper/figures/mixers/mfa.{mmd,pdf,svg} docs/paper/figures/latent-map.{mmd,pdf,svg} docs/paper/figures/mixer-taxonomy.{mmd,pdf,svg} docs/bibliography/attention_types.bib docs/paper/appendices/annex-3-latent-attention.tex docs/paper/appendices/annex-23-kv-cache.tex docs/paper/appendices/annex-summary-tables.tex docs/paper/appendices/annex-schema.tex docs/paper/sections/{01-introduction,04-architecture-taxonomy,10-discusion}.tex docs/paper/sections/11-*.tex docs/paper-es/appendices/annex-3-latent-attention.tex docs/paper-es/appendices/annex-23-cache-kv.tex docs/paper-es/appendices/annex-{summary-tables,schema}.tex docs/paper-es/sections/{01-introduccion,04-taxonomia-arquitectura,10-discusion}.tex README.md docs/architecture.md configs/README.md docs/source/api/model/attention/latent/{mfa.rst,index.rst} docs/schema.assembled.yaml docs/paper/paper.pdf docs/paper-es/paper-es.pdf full_tests/param_truth.json .opencode/plan/add-mixer-2412.19255.md`
   Message: `feat(attention): add Multi-matrix Factorization Attention mixer (2412.19255)` — body mirroring the TPA commit structure (module, spec, schema, cache decision, example+preset, tests, e2e real-data result, figures, docs, snapshot, PDFs, verification numbers). Then `git push origin HEAD:main`.
   Never stage `*.aux *.log *.out *.toc *.blg`, caches or mermaid temp files.
2. **Mirror** — `git -C erickfmm.github.io …` commit `docs(site): sync MFA mixer (2412.19255)`, then push `origin HEAD:main`.

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted pytest | §5 step 7 | all green |
| Full pytest | §5 step 8 | 0 failures (TPA baseline: 2079 passed) |
| Schema census | `test_schema_attention_layers.py`, `test_schema_consistency.py` | enum 55, `CACHE_SPECS` len 55, defaults match |
| e2e enumeration | `--list-combos \| grep -F mfa` | `attn_single_mfa` + all `attn_<a>__mfa` / `attn_mfa__<b>` |
| **e2e real-data** | `--category attn --only mfa` | all selected combos PASS, finite loss (Borges) |
| e2e offline | `--only mfa --no-realonall` | PASS |
| KV parity | `test_kv_cache_mixers.py` | full vs prefill+decode parity (rope/nope/alibi) |
| Figures | `build-figures.sh both` | `mfa.pdf/svg`, updated maps |
| Param truth | `param_count_check.py` | `0 failures` (both `key_reuse` settings) |
| Estimator | `param_estimate_check.mjs … --tol 0.01` | ALL PASS |
| Sphinx | `bash docs/build_docs.sh` | `latent/mfa.html` builds; no new warnings |
| PDFs | §5 step 15 | clean compile, `hu_mfa_2025` resolved |
| Mirror probes | sync-website Validation Gate | all PASS |

## 9. Risks, assumptions and open questions

- **Assumptions to verify at implementation time**: line numbers drift (validate by
  grep, e.g. `grep -n "tpa" …`); the latent family test file name (`tests/test_latent_attention.py`);
  census strings in `test_registry_core.py` / `test_kv_cache_fallback.py` may also
  hardcode 54 (TPA commit bumped them — grep `54` across `tests/`).
- **Softmax scale**: the paper writes $\sqrt{d}$ (MHA head dim) while MFA heads are
  $C$-dim; implement $\sqrt{C}$ (per-head dim, consistent with every other mixer here)
  and note the reading in the annex. Flag for reviewer.
- **Eq. 8 ambiguity**: $\alpha \odot N S_k$ is dimensionally loose in the paper; Eq. 9
  ($v = k(I+\mathrm{diag}(\alpha)N)$) is the operative form and what will be
  implemented + documented.
- **`num_kv_heads`**: MFA ignores it (single shared head); ensure the schema/if-then
  rules don't force it (peers `mla`/`tpa` already coexist with it).
- **α gradient flow at init**: α=0 makes MFA-KR ≡ shared S_v at step 0 — use it as an
  equivalence test, but confirm the gate actually trains (non-zero after a few steps).
- **Preset `configs/mfa.yaml`**: the paper's reference is MoE + seq 16384; the preset
  is a dense, small-seq analogue — document the deviation in the preset header comment.
- Naming collision: none (`grep -i mfa src/` empty; `falcon*` unrelated).

## 10. Out of scope

No code, schema, preset, test, figure, doc, PDF, mirror or git change was made by this
command. This document is the only file written. Implementation is a separate,
explicitly requested step.
