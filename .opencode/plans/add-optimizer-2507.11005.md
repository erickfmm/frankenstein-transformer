# Plan: add AdaMuon optimizer (arXiv 2507.11005)

> Canonical location: `.opencode/plans/add-optimizer-2507.11005.md`.

## 1. Header

**Paper:** *AdaMuon: Adaptive Muon Optimizer* — Chongjie Si, Debing Zhang,
Wei Shen (Shanghai Jiao Tong University; Xiaohongshu Inc.), arXiv:2507.11005
(v3, published 2025-07-15).

**Summary.** Muon turns matrix-shaped momentum into spectrally normalized
(direction-only) updates, which stabilizes large-scale training but applies a
uniform update magnitude that ignores element-wise gradient variance. AdaMuon
adds coordinate-wise adaptivity to Muon through two coupled mechanisms:
(1) an **element-wise second-momentum estimator accumulated on the
orthogonalized update** `O_t` (not on the raw gradient or momentum), and
(2) a **sign-stabilized orthogonal update** — the momentum buffer is passed
through `sign(·)` before Newton–Schulz orthogonalization, which Theorem 1 of
the paper proves is the *unique* element-wise transform that is scale-free,
sign-consistent, odd, and bounded, making `O_t` stable enough for
variance-based normalization. A final **RMS-aligned rescaling**
`γ_t = 0.2·√(mn)/‖Ô_t‖_F` matches the update RMS to Adam's ≈0.2, so
Adam learning-rate schedules are reused with no retuning, and (Appendix B)
makes explicit bias correction unnecessary. Experiments on GPT-2
(125M–1.5B) and Qwen2.5 (1.5B, 7B) beat AdamW and Kimi-variant Muon,
with up to ~40% training-efficiency gains and consistent benchmark wins.

**Why it belongs here.** The repo already has an *Orthogonalized* optimizer
bucket (`muon`, `turbo_muon`, `af_muon` — `docs/architecture.md` §8.1, line
681). AdaMuon is a drop-in next member of that family: same Newton–Schulz
machinery as `src/model/optimizer/muon.py`, one extra state buffer, no new
mandatory hyperparameters (β is shared by both EMAs), and it is RMS-aligned
so the existing Adam-scale per-group LRs in presets work unchanged.

Decision gate: **PASS** — it is a real optimizer method; it does not duplicate
any entry in `factory.OPTIMIZER_REGISTRY` (verified: no `ada_muon`/`AdaMuon`
anywhere in `src/`, `configs/`, `tests/`, `docs/`, `full_tests/`, `README.md`);
the paper fully specifies every equation and default.

## 2. Paper extract

### 2.1 Update rule (verbatim, Algorithm 1 + Eqs. (1)–(9))

For each 2-D weight `W ∈ ℝ^{n×m}`, learning rate `η`, weight decay `λ`,
momentum `β`, Newton–Schulz steps `T`, small constant `ε`:

```
M₀ = 0,  V₀ = 0
for t = 1, 2, …:
  G_t = ∇_{W_t} L(W_t)                                    (gradient)
  M_t = β·M_{t−1} + G_t                                   (first momentum; Muon-style, no (1−β))
  O_t = NewtonSchulz( Sign(M_t), T )                      (sign-stabilized orthogonal direction)
  V_t = β·V_{t−1} + (1−β)·(O_t ⊙ O_t)                     (Eq. 5: second momentum on O_t)
  Ô_t = O_t ⊘ ( √V_t + ε·1 )                              (Eq. 6: variance-normalized direction)
  γ_t = 0.2·√(mn) / ‖Ô_t‖_F                               (Eq. 8: RMS-aligned rescaling)
  W_{t+1} = W_t − η·( γ_t·Ô_t + λ·W_t )                   (Eq. 9: coupled update + weight decay)
```

Newton–Schulz iteration (Eq. 2), applied to `X₀ = Sign(M_t)/‖Sign(M_t)‖_F`:

```
X_k = a·X_{k−1} + b·(X_{k−1}X_{k−1}ᵀ)·X_{k−1} + c·(X_{k−1}X_{k−1}ᵀ)²·X_{k−1}
a = 3.4445,  b = −4.7750,  c = 2.0315,  T = 5
```

### 2.2 Pseudocode / schedule interactions

- Algorithm 1 is the complete method (reproduced above). **No Nesterov
  option**, no warmup-specific behavior. Any external LR schedule applies as
  usual (that is the point of the RMS alignment).
- **No bias correction** on either moment (Appendix B): any constant factor
  `c` in `V_t` is exactly cancelled by `γ ∝ √c`, and any constant factor in
  `M_t` is removed by `polar(cM) = polar(M)` (Eq. 15).
- Scope: **2-D parameters only**. In the paper's experiments 1-D params
  (biases) are handled by an auxiliary Adam at the same LR; in this repo the
  Muon-family convention (`muon.py` line 104: `if p.grad is None or p.ndim < 2:
  continue`) simply skips them — mirror that.
- Theorem 1 (why `sign`): an element-wise `f` applied before `polar` must
  satisfy scale invariance, sign consistency, odd symmetry, bounded range ⟹
  `f(x) = c·sign(x)`, and `c` is immaterial under `polar`.
- Convergence (Appendix C, λ=0): with L-smoothness, unbiased stochastic
  gradients, and a directional-alignment assumption, diminishing steps
  `η_t = η₀/√t` give `min E‖∇L‖_F = O(log T / √T)` (Thm. 2); under the PL
  condition with constant step the iterates converge to an `O(η²)`
  neighborhood (Thm. 3). Lemmas: `0 ≤ V_{t,ij} ≤ 1` (since `O_t ⊙ O_t ∈
  [0,1]`), and `‖γ_t·Ô_t‖_F = 0.2·√(mn)` exactly (useful as a unit-test
  invariant).

### 2.3 Hyperparameters

| Name | Symbol | Type | Default (paper) | Range tried | Scope |
|---|---|---|---|---|---|
| learning rate | η | float | shared with Adam: 6e-4 / 1e-3 | 6e-4, 1e-3 | per-group (`lr_*`) |
| momentum (shared by M and V) | β | float | **0.95** | 0.8–0.99 (Table 3: flat) | global |
| weight decay | λ | float | 0.1 (experiments) | — | per-group (`wd_*`) |
| NS steps | T | int | **5** | — | global |
| variance epsilon | ε | float | **1e-8** (Eq. 6 denominator) | — | per-group (`eps_*`) |
| RMS target constant | c_rms | float | **0.2** (fixed; "any fixed positive constant works") | — | not configurable |

No additional hyperparameters are introduced (§3.1: "β is inherited directly
from Muon's momentum parameter"). Repo-specific extras: `ns_eps` (NS
normalization epsilon, mirrors `muon`'s `ns_eps=1e-7` for numerical safety).

### 2.4 State buffers and per-step cost

- **Buffers:** 2 per 2-D parameter — momentum `M` (`n×m`) + second moment `V`
  (`n×m`). One more than Muon (paper §6 limitation: "one additional
  second-moment buffer introduces more computation and memory").
- **Per-step cost:** identical Newton–Schulz chain to Muon — `O(k)` matmuls of
  the parameter shape (dominant) — plus `O(nm)` element-wise work (sign, V
  EMA, `√V` divide, Frobenius norm for γ_t) and two reductions.
- Summary-table wording: State `2 (m, v)`, per-step cost `O(n·k)`, key
  hyperparams `lr, momentum, eps, wd, NS steps k`.

### 2.5 Category

Existing bucket **Orthogonalized** (`docs/architecture.md` §8.1 line 681:
`muon`, `turbo_muon`, `af_muon`). No new category needed. Paper annex bucket:
"Second-Order, Geometric & Orthogonality" header in
`annex-summary-tables.tex` (line 146) and the "Geometry-oriented" row of the
closing longtable in `annex-1-optimizers.tex` (line 1167).

### 2.6 BibTeX data

Authors: Chongjie Si; Debing Zhang; Wei Shen. Year: 2025. Title: *AdaMuon:
Adaptive Muon Optimizer*. arXiv:2507.11005 (v3). DOI
10.48550/ARXIV.2507.11005. Entry key (repo convention
`<firstauthor>_<shortname>_<year>`, cf. `boissin_turbo-muon_2025`,
`lagzian_afmuon_2026`): **`si_adamuon_2025`**. Full draft in §4 row 11.

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `ada_muon` | snake_case short name from the paper; matches family naming `muon`/`turbo_muon`/`af_muon`; grep confirms the name is unused |
| Class / module | `AdaMuon` / `src/model/optimizer/ada_muon.py` | mirrors `Muon(Optimizer)` in `muon.py` |
| Closest peer | `src/model/optimizer/muon.py` | Mirror: module docstring w/ citation, `defaults` dict, `ndim < 2` skip (line 104), `momentum_buffer` update (line 113), bf16 NS with `a,b,c = 3.4445, −4.7750, 2.0315` (lines 123–135), tall-matrix transpose, decoupled `p.mul_(1 − lr·wd)` (line 121) which is algebraically identical to Eq. 9's `−ηλW`. Add: sign gate, `V` EMA, element-wise divide, `γ_t` scaling. Deviation: **no `nesterov`** suffix (paper's Alg. 1 has none) |
| Category | Orthogonalized (existing) | §8.1 line 681; annex summary block line 146 |
| Peer for factory branch | `muon`/`turbo_muon` branch (`factory.py` lines 262–273) | same `momentum`/`ns_steps`/`ns_eps` handling, minus nesterov |
| lr default in module | `1e-3` (Adam-scale; NOT muon's 0.02) | Paper §3.3/§4.1: RMS-aligned to Adam, "direct reuse of existing learning rate schedules"; experiments share Adam's 6e-4/1e-3. Factory per-group lr override still wins |
| `eps` source | per-group shared suffix `eps_*` (factory default **1e-8**) doubles as Eq. 6's ε | Factory injects `group["eps"]` = 1e-8 for all non-sgd optimizers (`factory.py` line 174) — exactly the paper's default; `betas_*` remain tolerated-but-unused, as for `muon` |
| `supports_shared_per_group_suffixes` | `true` | norm for every current entry in `by_optimizer` |
| Example preset | `configs/examples/es_arch_hybrid_ada_muon.yaml` (task `mlm`, encoder) | Clone of the family recipe `es_arch_hybrid_turbo_muon.yaml` (already smoke-tested architecture; conv stays disabled so every trained param is 2-D), but the optimizer block covers **all 10 canonical groups** (`embeddings, pos, attention, ffn, norms, head, mhc, router, residual, other` — values from `DEFAULT_GROUP_LR`/`DEFAULT_GROUP_WD`, `base.py` lines 77–102), unlike the turbo_muon example which only overrides the 4 original ones. Subsets are still schema-valid (`minProperties: 0`); the full block documents the current taxonomy |
| Census bump | docs say **23** (README lines 22/27/93; architecture lines 23/83/527/668; both annex-summary intro/line-100) while the enum/registry already hold **24** → after this change the truth is **25**; bump every "23" to **25** and the paper sections' "twenty-four/veinticuatro" to "twenty-five/veinticinco" | Pre-existing off-by-one drift (AF-Muon was added without a README/architecture/annex census bump) — fixed in the same pass; flagged, not silently |
| Figure | optional `docs/paper/figures/optimizers/ada-muon.mmd` + `.pdf` in the `muon.mmd` mermaid style | 22 optimizer figures exist; include only if it can be rendered to PDF in the same vector style, else omit the float |

### Suffix table

| Key | Type | Default | Coercion | Scope | EN description | ES description |
|---|---|---|---|---|---|---|
| `ada_muon-momentum` | float | 0.95 | `to_float` | global | β shared by the momentum buffer and the element-wise second-moment EMA | β compartido entre el búfer de momentum y la EMA de segundo momento por elemento |
| `ada_muon-ns_steps` | int | 5 | `to_int` | global | Newton–Schulz iteration count | Iteraciones de Newton–Schulz |
| `ada_muon-ns_eps` | float | 1e-7 | `to_float` | global | numerical-stability epsilon in the NS Frobenius normalization | épsilon de estabilidad numérica en la normalización Frobenius de NS |
| `ada_muon-eps_<group>` | float | 1e-8 | `to_float` (shared `eps_*`) | per-group | denominator ε of the variance-normalized update (Eq. 6) | ε del denominador de la actualización normalizada por varianza (Ec. 6) |
| `ada_muon-lr_<group>` / `wd_<group>` / `betas_<group>` | — | via factory | shared | per-group | standard shared suffixes, one per each of the 10 canonical groups (`embeddings, pos, attention, ffn, norms, head, mhc, router, residual, other`); `betas_*` accepted but unused (Muon-family behaviour) | sufijos compartidos estándar, uno por cada uno de los 10 grupos canónicos (`embeddings, pos, attention, ffn, norms, head, mhc, router, residual, other`); `betas_*` se aceptan pero no se usan |

`specific_global_suffixes: [momentum, ns_steps, ns_eps]` — deliberately **no
`nesterov`** (paper's algorithm has no Nesterov term).

## 4. File-by-file plan table

Execution order: 1–9 implementation, 10 tests, 11–18 docs.

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/ada_muon.py` (new) | `torch.optim.Optimizer` subclass citing the paper in the module docstring (`Si, Zhang, Shen 2025, arXiv:2507.11005`), `from __future__ import annotations`, Google-style docstrings, CPU-safe (no `.cuda()`), no imports beyond `math`+`torch`. Constructor `(params, lr=1e-3, momentum=0.95, ns_steps=5, ns_eps=1e-7, weight_decay=0.0)`; validate `0 ≤ lr`, `0 ≤ momentum < 1`, `0 ≤ ns_eps`; `defaults = dict(...)`. `step()` sketch (equations mapped): `buf.mul_(momentum).add_(grad)` → M_t (Alg. 1 l.4); `g = torch.sign(buf)` → Sign(M_t) (l.5); `x = g.bfloat16(); x = x / (x.norm() + ns_eps)` → X₀ (Eq. 2); transpose if tall, run `ns_steps` NS iterations with `a,b,c = 3.4445, −4.7750, 2.0315`, transpose back; cast `o = x.to(p.dtype)` **before** the EMA (FP32 precision, unlike `muon.py` which casts at the final add); lazily init `state["exp_avg_sq"] = torch.zeros_like(p)`; `v.mul_(momentum).addcmul_(o, o, value=1−momentum)` → V_t (Eq. 5); `o_hat = o / (v.sqrt() + group["eps"])` → Ô_t (Eq. 6); `gamma = 0.2 * math.sqrt(p.size(0) * p.size(1)) / o_hat.norm().clamp_min(group["ns_eps"])` → γ_t (Eq. 8; clamp guards the all-zero-grad edge case, which the paper does not hit); if `weight_decay > 0`: `p.mul_(1 − lr·wd)` → −ηλW (Eq. 9); finally `p.add_(o_hat, alpha=−lr·gamma)` → −ηγÔ (Eq. 9). Keep module constant `_RMS_TARGET = 0.2` documented as fixed by the paper |
| 2 | `src/model/optimizer/__init__.py` | Add `from .ada_muon import AdaMuon` after line 1 `from .adamw import AdamWOptimizer` (top of import list); add `"AdaMuon",` to `__all__` after `"AdamWOptimizer",` (line 34) |
| 3 | `src/model/optimizer/factory.py` | Import `from .ada_muon import AdaMuon` (after line 1, next to `.adamw`); `OPTIMIZER_REGISTRY`: add `"ada_muon": AdaMuon,` after `"af_muon": AFMuon,` (line 60) to keep the Muon family grouped; `OPTIMIZER_ALLOWED_KEYS`: add `"ada_muon": _COMMON_PER_GROUP_KEYS \| {"momentum", "ns_steps", "ns_eps"},` after the `"af_muon"` entry (line 117); new dispatch branch after the muon/turbo_muon branch (line 273): `if optimizer_name == "ada_muon": momentum = to_float(scoped.get("momentum"), 0.95); ns_steps = to_int(scoped.get("ns_steps"), 5); ns_eps = to_float(scoped.get("ns_eps"), 1e-7); return optimizer_cls(converted_groups, momentum=momentum, ns_steps=ns_steps, ns_eps=ns_eps)` — per-group `eps` flows in via `converted_groups` |
| 4 | `src/schema/_optimizer.yaml` | (a) enum: `- ada_muon` after `- af_muon` (line 109); (b) EN use-case table row after line 40: `` | `ada_muon` | Adaptive orthogonal updates | Element-wise second moment on orthogonalized directions + sign stabilization + RMS-aligned scaling (arXiv:2507.11005) | ``; ES row after line 76: `` | `ada_muon` | Actualizaciones ortogonales adaptativas | Segundo momento por elemento sobre direcciones ortogonalizadas + estabilización por signo + escalado alineado por RMS (arXiv:2507.11005) | ``; (c) EN suffix list after line 146: `` - `ada_muon`: `momentum`, `ns_steps`, `ns_eps` `` and ES after line 181 (same); (d) `by_optimizer` after the `af_muon` entry (line 272): `ada_muon:` / `prefix: "ada_muon-"` / `supports_shared_per_group_suffixes: true` / `specific_global_suffixes: [momentum, ns_steps, ns_eps]` |
| 5 | `src/schema/_training.yaml` | One `if/then` block after the `af_muon` block (ends line 1273), same shape as lines 1129–1142: `const: ada_muon` → `parameters: additionalProperties: false` + `patternProperties: { ^ada_muon-: { type: [number, boolean, array, string] } }`. Exactly one block — `test_optimizer_prefixes_match_conditional_rules` (`tests/test_schema_consistency.py` lines 76–90) requires `by_optimizer` prefixes == `^…-:` regexes |
| 6 | `src/schema/_sbert.yaml` | Extend the `optimizer_kwargs` descriptions (EN line 352, ES line 353): in the `momentum`, `ns_steps`, `ns_eps` lists change "for muon/turbo_muon/af_muon" → "for muon/turbo_muon/af_muon/ada_muon" (EN) and `` `muon`/`turbo_muon`/`af_muon` `` → `` `muon`/`turbo_muon`/`af_muon`/`ada_muon` `` (ES); **do not** add `ada_muon` to the `nesterov` lists; append "for ada_muon the shared per-group `eps` is the variance-normalization epsilon (default 1e-8)" / "para `ada_muon` el `eps` por grupo compartido es el épsilon de normalización por varianza (por defecto 1e-8)" |
| 7 | `configs/examples/es_arch_hybrid_ada_muon.yaml` (new) | Full clone of `configs/examples/es_arch_hybrid_turbo_muon.yaml` (81 lines: `frankenstein_text_encoder`, `layer_pattern: [retnet, titan_attn, mamba, ode]`, `embedding.conv.enabled: false`, `bitnet.enabled: false`, task `mlm`) with `optimizer_class: ada_muon` and a parameter block covering **all 10 canonical groups** — lr/wd/betas/eps × `embeddings, pos, attention, ffn, norms, head, mhc, router, residual, other` — at the house defaults (`DEFAULT_GROUP_LR`/`DEFAULT_GROUP_WD`); `ada_muon-momentum: 0.95`, `ada_muon-ns_steps: 5`, `ada_muon-ns_eps: 1e-7` (**no `nesterov` key**); `csv_log_path: training_metrics_es_arch_hybrid_ada_muon.csv`. Drafted in §5 checklist step 7 |
| 8 | `configs/README.md` | (a) Optimizer chooser list: `- ada_muon` after `- af_muon` (line 141); (b) suffix list: `- ada_muon: `momentum`, `ns_steps`, `ns_eps`` after line 185; (c) examples block (after line 205): `- ada_muon-ns_steps: 5` |
| 9 | `full_tests/run_e2e.py` | `OPTIMIZERS` list (line 113): insert `"ada_muon"` after `"turbo_muon"` (line 115) to keep the family grouped (note: `af_muon` is absent from this list today — leave that as-is); `_with_optimizer` (after the muon/turbo_muon branch, lines 237–239): `if optimizer_class == "ada_muon": params[f"{prefix}momentum"] = 0.9; params[f"{prefix}ns_steps"] = 5` |
| 10 | `tests/test_optimizer_factory.py` | (a) Add `"ada_muon"` to the `required` set in `test_registry_contains_all_required` (line 29); (b) new class `BuildOptimizerAdaMuonTests` (after `BuildOptimizerAFMuonTests`): `test_builds_without_error`, `test_step_runs` (4×4 param, backward, step), `test_defaults_match_paper` (momentum 0.95, ns_steps 5, ns_eps 1e-7), `test_step_skips_vector_params` (1-D param unchanged after step), `test_update_norm_matches_rms_alignment` (wd=0, one step ⟹ `‖Δp‖_F ≈ lr·0.2·√(mn)` — Lemma 2 invariant), `test_string_coercion_and_invalid_fallback` (`"ada_muon-momentum": "0.9"` → 0.9; `"ada_muon-ns_steps": "bad"` → 5), `test_per_group_lr_routes_new_blocks` (`ada_muon-lr_mhc` / `ada_muon-lr_router` override reaches `param_groups[0]["lr"]`, mirroring `test_per_group_lr_for_new_block_groups` at lines 400–404), `test_unknown_group_suffix_raises` (`ada_muon-lr_bogus`) |
| 11 | `docs/bibliography/optimizers.bib` | `@misc{si_adamuon_2025, …}` in the Zotero/arXiv house style of `boissin_turbo-muon_2025` (line 268): `title = {AdaMuon: Adaptive Muon Optimizer}`, `rights = {{arXiv}.org perpetual, non-exclusive license}`, `url = {https://arxiv.org/abs/2507.11005}`, `doi = {10.48550/ARXIV.2507.11005}`, `shorttitle = {AdaMuon}`, `abstract = {…verbatim from arXiv…}`, `publisher = {{arXiv}}`, `author = {Si, Chongjie and Zhang, Debing and Shen, Wei}`, `urldate = {<implementation date>}`, `date = {2025}`, `year = {2025}`, `note = {Version Number: 3}`, `keywords = {Machine Learning (cs.LG), …}` |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | `\paragraph{AdaMuon (Adaptive Muon).}` at the end of subsection "Second-Order, Geometric, and Orthogonality Optimizers" (line 803), after the AF-Muon paragraph's closing longtable (line 1170). Structure mirroring the AF-Muon paragraph (lines 1097–1154): optional figure float `\fitfigure{\includegraphics{figures/optimizers/ada-muon.pdf}}` + `\label{fig:opt-ada-muon}` (omit if no PDF); intro sentence; `align` block with Eqs. (5)(6)(8)(9) exactly as §2.1 above + Theorem-1 sentence (sign is the unique admissible element-wise transform); `algorithm` float `alg:ada-muon` reproducing Algorithm 1; `\paragraph{Cons.}` (extra second-moment buffer + per-step EMA overhead, per paper §6; 2-D-only — 1-D params need an auxiliary optimizer in the paper's recipe); `\paragraph{Pros.}` (no new hyperparameters — β shared; RMS alignment lets Adam schedules/LRs be reused; bias correction provably unnecessary; GPT-2 + Qwen2.5 gains over AdamW/Muon); `\citep{si_adamuon_2025}` (+ `\citep{shen_convergence_2025}` for Muon). Update the closing "Geometry-oriented" longtable row (line 1167): `Muon, Turbo-Muon, AF-Muon, AdaMuon` |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | Row after AF-Muon (line 153) inside the "Second-Order, Geometric & Orthogonality" block (header line 146): `AdaMuon & Orthogonality + adaptive & 2 ($m$, $v$) & $\mathcal{O}(n \cdot k)$ & lr, momentum, eps, wd, NS steps $k$ & \citep{si_adamuon_2025} \\`. Census: line 4 intro "…, 23 optimizers, …" → "…, 25 optimizers, …"; line 100 "all 23 supported optimizers" → "all 25 supported optimizers" |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: "The system supports twenty-four optimizer families in seven categories" → "twenty-five"; "geometry-oriented methods (Muon, Turbo-Muon, AF-Muon)" → "(Muon, Turbo-Muon, AF-Muon, AdaMuon)" |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of #12: `\paragraph{AdaMuon (Muon Adaptativo).}` after the AF-Muon paragraph (line 1097+); figure path `../paper/figures/optimizers/ada-muon.pdf` (ES annex convention, line 1100 ES); same equations (formulations are theory — translate prose, keep LaTeX identical); `alg:ada-muon` with translated comments; Pros/Cons translated; `\citep{si_adamuon_2025}`. Update "Orientado a geometría" row (line 1167): `Muon, Turbo-Muon, AF-Muon, AdaMuon` |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | Row after AF-Muon (line 153): `AdaMuon & Ortogonalidad + adaptativo & 2 ($m$, $v$) & $\mathcal{O}(n \cdot k)$ & lr, momentum, eps, wd, pasos NS $k$ & \citep{si_adamuon_2025} \\`; line 4 "23 optimizadores" → "25 optimizadores"; line 100 "los 23 optimizadores soportados" → "los 25 optimizadores soportados" |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: "veinticuatro familias de optimizadores" → "veinticinco familias"; "métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon)" → "(Muon, Turbo-Muon, AF-Muon, AdaMuon)" |
| 18 | `docs/architecture.md`, `README.md` | Census pass (grep the old counts first — see §5 step 18): **architecture.md** line 23 `\| Optimizer families \| **23** \|` → `**25**`; line 83 "23 optimizers + registry + parameter routing" → "25 optimizers …"; line 527 "Same 23-family enum" → "Same 25-family enum"; line 668 `### 8.1 Optimizers (23)` → `(25)`; line 681 Orthogonalized row → `` | Orthogonalized | `muon`, `turbo_muon`, `af_muon`, `ada_muon` | Momentum orthogonalization; `ada_muon` adds an element-wise second moment on the orthogonalized direction with RMS-aligned Adam-scale updates; `af_muon` adds … | `` (keep the existing af_muon sentence). **README.md** stats table (~line 22) `\| **42** \| **23** \|` → `\| **42** \| **25** \|`; mermaid line 27 "23 optimizer families" → "25 optimizer families"; line 93 "Optimizer families \| 23 across 6 categories" → "25 across 6 categories" |

**Hard rules restated.** The schema is the contract (`additionalProperties:
false`); optimizer keys are prefixed `<name>-<group>_<param>` /
`<name>-<param>`; never read an undocumented key (`factory.ensure_no_unknown_parameters`
enforces it). The papers are the theory source of truth (AGENTS.md D11): the
annex formulations must match arXiv:2507.11005 verbatim, and `paper-es` must
translate them faithfully (equations unchanged, prose translated).

## 5. Ordered execution checklist

Implementation-time only — **none of these commands were run by the planning
session**.

- [ ] 1. `src/model/optimizer/ada_muon.py` (file #1). Include the module
      docstring citation and the Eq.-to-line mapping from §4 row 1.
- [ ] 2. `src/model/optimizer/__init__.py` (#2).
- [ ] 3. `src/model/optimizer/factory.py` (#3).
- [ ] 4. `src/schema/_optimizer.yaml` (#4).
- [ ] 5. `src/schema/_training.yaml` (#5).
- [ ] 6. `src/schema/_sbert.yaml` (#6).
- [ ] 7. `configs/examples/es_arch_hybrid_ada_muon.yaml` (#7). Optimizer block:

```yaml
  optimizer:
    optimizer_class: ada_muon
    parameters:
      ada_muon-lr_embeddings: 1e-6
      ada_muon-lr_pos: 2e-6
      ada_muon-lr_attention: 3e-6
      ada_muon-lr_ffn: 2e-6
      ada_muon-lr_norms: 5e-6
      ada_muon-lr_head: 2e-6
      ada_muon-lr_mhc: 3e-6
      ada_muon-lr_router: 2e-6
      ada_muon-lr_residual: 2e-6
      ada_muon-lr_other: 2e-6
      ada_muon-wd_embeddings: 0.01
      ada_muon-wd_pos: 0.01
      ada_muon-wd_attention: 0.01
      ada_muon-wd_ffn: 0.01
      ada_muon-wd_norms: 0.001
      ada_muon-wd_head: 0.01
      ada_muon-wd_mhc: 0.01
      ada_muon-wd_router: 0.01
      ada_muon-wd_residual: 0.01
      ada_muon-wd_other: 0.01
      ada_muon-betas_embeddings: [0.9, 0.95]
      ada_muon-betas_pos: [0.9, 0.95]
      ada_muon-betas_attention: [0.9, 0.95]
      ada_muon-betas_ffn: [0.9, 0.95]
      ada_muon-betas_norms: [0.9, 0.95]
      ada_muon-betas_head: [0.9, 0.95]
      ada_muon-betas_mhc: [0.9, 0.95]
      ada_muon-betas_router: [0.9, 0.95]
      ada_muon-betas_residual: [0.9, 0.95]
      ada_muon-betas_other: [0.9, 0.95]
      ada_muon-eps_embeddings: 1e-8
      ada_muon-eps_pos: 1e-8
      ada_muon-eps_attention: 1e-8
      ada_muon-eps_ffn: 1e-8
      ada_muon-eps_norms: 1e-8
      ada_muon-eps_head: 1e-8
      ada_muon-eps_mhc: 1e-8
      ada_muon-eps_router: 1e-8
      ada_muon-eps_residual: 1e-8
      ada_muon-eps_other: 1e-8
      ada_muon-momentum: 0.95
      ada_muon-ns_steps: 5
      ada_muon-ns_eps: 1e-7
```

      (lr/wd values are the house defaults from `DEFAULT_GROUP_LR` /
      `DEFAULT_GROUP_WD` in `src/model/optimizer/base.py` lines 77–102, and
      the block covers **all 10 canonical groups** — unlike
      `es_arch_hybrid_turbo_muon.yaml`, which only overrides the 4 original
      ones. `betas_*` are accepted-but-unused for this family, kept to
      document the shared suffixes. Everything else is identical to
      `es_arch_hybrid_turbo_muon.yaml`, including
      `embedding.conv.enabled: false` — conv kernels would not be 2-D and
      the Muon family skips/fails on them; `csv_log_path` renamed).
- [ ] 8. `configs/README.md` (#8).
- [ ] 9. `full_tests/run_e2e.py` (#9).
- [ ] 10. `tests/test_optimizer_factory.py` (#10).
- [ ] 11. `docs/bibliography/optimizers.bib` (#11).
- [ ] 12–18. Docs: annex EN (#12), summary EN (#13), 05 EN (#14), annex ES
      (#15), summary ES (#16), 05 ES (#17), census pass (#18). Before #18,
      inventory the old counts:

```bash
grep -n "23" README.md docs/architecture.md | grep -i optim
grep -n "twenty-four\|veinticuatro" docs/paper/sections/05-optimizer-families.tex docs/paper-es/sections/05-familias-optimizadores.tex
grep -n "23 optimiz" docs/paper*/appendices/annex-summary-tables.tex
```

- [ ] 19. (Optional) figure `docs/paper/figures/optimizers/ada-muon.mmd` in
      the `muon.mmd` style:
      `G → M=βM+G → sign(M) → X₀=sign(M)/‖·‖F → NS×k → O → V=βV+(1−β)O⊙O →
      Ô=O/(√V+ε) → γ=0.2√(mn)/‖Ô‖F → W←W−η(γÔ+λW)`; render to
      `ada-muon.pdf` with the same tool used for the existing 22 figures;
      if a same-style vector PDF cannot be produced, omit the figure float
      from #12/#15.
- [ ] 20. Targeted gates:

```bash
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings
```

- [ ] 21. Full suite:

```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

- [ ] 22. Fast e2e smoke (raise `--limit` if the new optimizer is not
      reached; `ada_muon` sits mid-list):

```bash
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3
```

- [ ] 23. PDF rebuild, from the repository root:

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

      Preconditions: `command -v pdflatex bibtex` — both present on this
      machine (`/usr/bin/pdflatex`, `/usr/bin/bibtex`, verified read-only).
      Fallback if unavailable: leave the `.tex`/`.bib` sources updated and
      note that the rebuild must happen on a TeX machine before the next
      release (the website sync copies prebuilt PDFs only).
      Post-compile verification: no `Fatal error` and no
      `Citation .* undefined` in the logs; `paper.pdf` / `paper-es.pdf`
      newer than the edited `.tex`/`.bbl`; the new `\citep{si_adamuon_2025}`
      resolves in both PDFs; `git status --short docs/paper docs/paper-es`
      shows only intended `.tex`/`.bbl`/`.pdf` (+#19 figure) changes.
- [ ] 24. Website sync (§6).
- [ ] 25. Commits (§7).

## 6. Website sync (planned, not executed)

Follow `.opencode/command/sync-website.md` (mirror:
`erickfmm.github.io/frankenstein-transformer/`, separate git repo):

1. `bash .opencode/command/sync-website-state.sh` — read the `[drift]` report.
2. Copy (plain `cp`, no rsync) main → mirror:
   `src/schema.yaml` → `schema.yaml`; `src/schema/**` → `schema/**`
   (includes the edited `_optimizer.yaml`, `_training.yaml`, `_sbert.yaml`);
   `configs/examples/es_arch_hybrid_ada_muon.yaml` →
   `examples/examples/…`; `configs/README.md` → `examples/README.md`;
   `docs/paper/{paper.tex,paper.bbl,paper.pdf,sections/*.tex,appendices/*.tex}`
   → `paper/en/…`; same for `docs/paper-es/**` → `paper/es/…`;
   `docs/paper/figures/optimizers/ada-muon.{mmd,pdf}` (if produced) →
   `paper/en/figures/optimizers/`; `docs/bibliography/optimizers.bib` →
   `paper/bibliography/`; `docs/architecture.md` →
   `paper/specs/architecture.md`; `README.md` is main-repo only (not
   mirrored).
3. PDFs: already rebuilt in §5 step 23 — copy
   `paper/en/paper.pdf → paper/en.pdf` and `paper/es/paper-es.pdf →
   paper/es.pdf` (probes `paper-en-copy`/`paper-es-copy`).
4. Paper-tab HTML (pandoc is installed):

```bash
cd erickfmm.github.io/frankenstein-transformer/paper
python3 flatten_tex.py en/paper.tex    en/paper.bbl    /tmp/paper-en-flat.tex
python3 flatten_tex.py es/paper-es.tex es/paper-es.bbl /tmp/paper-es-flat.tex
pandoc /tmp/paper-en-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=en -o paper-en.html
pandoc /tmp/paper-es-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=es -o paper-es.html
```

5. Re-run the state script; fix any `[probes]` FAIL per the command doc
   (expected: none — `schema-refs`, `example-yaml`, `cli-flags` should stay
   PASS; the new example YAML must load via `load_training_config`).
6. Validation Gate:

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py      # "… configs counted, 0 failures"; regenerates full_tests/param_truth.json
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01                                     # exit 0, ALL PASS
```

   plus the state-script probes (`schema-refs`, `example-yaml`, `cli-flags`,
   estimator) all PASS. `param_truth.json` changes are intentional.
7. Leave both repos **uncommitted** until the commit phase (§7).

## 7. Commit plan (planned, not executed)

`/make-a-commit` conventions: English, conventional tag, structured body,
push to `main`. Never stage `*.aux *.log *.out *.toc *.blg` or caches.

1. **Main repo** — `feat(optimizer): add AdaMuon (arXiv 2507.11005)`
   Body: implementation notes (sign-stabilized NS + second moment on O_t +
   RMS-aligned γ_t, no nesterov by design, per-group eps doubles as variance
   ε, zero-norm clamp); new suffixes `momentum`/`ns_steps`/`ns_eps`;
   schema/registry changes (enum, if/then, by_optimizer, SBERT description);
   docs (annexes EN/ES, summary tables + census 23→25 fixing the pre-existing
   off-by-one, family sections twenty-four→twenty-five, bibliography,
   rebuilt PDFs); example `es_arch_hybrid_ada_muon.yaml`; e2e list; tests.

   Stage:
   - `src/model/optimizer/ada_muon.py`, `src/model/optimizer/__init__.py`,
     `src/model/optimizer/factory.py`
   - `src/schema/_optimizer.yaml`, `src/schema/_training.yaml`,
     `src/schema/_sbert.yaml`
   - `configs/examples/es_arch_hybrid_ada_muon.yaml`, `configs/README.md`
   - `full_tests/run_e2e.py` (+ `full_tests/param_truth.json` if the
     Validation Gate regenerated it — intentional change)
   - `tests/test_optimizer_factory.py`
   - `docs/bibliography/optimizers.bib`
   - `docs/paper/appendices/annex-1-optimizers.tex`,
     `docs/paper/appendices/annex-summary-tables.tex`,
     `docs/paper/sections/05-optimizer-families.tex`,
     `docs/paper/paper.bbl`, `docs/paper/paper.pdf`,
     `docs/paper/figures/optimizers/ada-muon.mmd` + `.pdf` (if produced)
   - `docs/paper-es/appendices/annex-1-optimizadores.tex`,
     `docs/paper-es/appendices/annex-summary-tables.tex`,
     `docs/paper-es/sections/05-familias-optimizadores.tex`,
     `docs/paper-es/paper-es.bbl`, `docs/paper-es/paper-es.pdf`
   - `docs/architecture.md`, `README.md`

   Then `git push origin HEAD:main`.

2. **Mirror** — `git -C erickfmm.github.io …` commit
   `docs(site): sync AdaMuon optimizer (arXiv 2507.11005)` staging the
   mirror-side copies from §6 (schema files, example YAML + README, paper
   en/es tex/bbl/pdf + figure, bibliography, architecture spec, regenerated
   `paper-en.html`/`paper-es.html`/`en.pdf`/`es.pdf`, `param_truth`-driven
   files if any); then `git -C erickfmm.github.io push origin HEAD:main`.

## 8. Verification matrix

| Gate | Command | Expected result |
|---|---|---|
| Factory/base/schema/yaml gates | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings` | All pass; new `BuildOptimizerAdaMuonTests` green; `test_optimizer_prefixes_match_conditional_rules` green (one `ada_muon-` rule ↔ one `by_optimizer` entry); `test_yaml_examples` auto-globs the new example |
| Full suite | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | All pass (CPU-only; the new optimizer has no CUDA-specific code) |
| e2e smoke | `uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3` | Runs including `ada_muon` without error (raise `--limit` if it stops before the family) |
| PDF EN | §5 step 23 block | `paper.pdf` rebuilt; no `Fatal error`; no `Citation si_adamuon_2025 undefined`; newer than `.tex`/`.bbl` |
| PDF ES | same | `paper-es.pdf` rebuilt; same checks |
| PDF diff hygiene | `git status --short docs/paper docs/paper-es` | Only intended `.tex`/`.bbl`/`.pdf` (+ optional figure) |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `… configs counted, 0 failures` (regenerates `param_truth.json`) |
| Estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | Exit 0, `ALL PASS` (optimizer choice does not change param counts) |
| State probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags` (and estimator) PASS |
| Example loads | `uv run --extra cpu --extra train python -c "from src.training.config_loader import load_training_config; load_training_config('configs/examples/es_arch_hybrid_ada_muon.yaml')"` | No exception |

## 9. Risks, assumptions and open questions

**Assumptions to verify at implementation time**

- Census drift is real: `_optimizer.yaml` enum and `OPTIMIZER_REGISTRY` hold
  **24** entries today while README/architecture/annexes say **23** and
  `05-optimizer-families.tex` says twenty-four/veinticuatro. Re-grep before
  editing (#18 inventory commands); after the change everything must read
  **25** / twenty-five / veinticinco consistently. This fixes a pre-existing
  inconsistency (AF-Muon was evidently added without a full census bump) as a
  side effect of the mandated census pass — flagged here for visibility.
- `run_e2e.py`'s `OPTIMIZERS` list omits `af_muon` (23 entries vs 24 in the
  enum). Assumed deliberate (tied-embedding-specific optimizer); left
  untouched. If it was an oversight, adding it is a separate decision.
- Category-count statements are inconsistent across docs (README "6
  categories", architecture §8.1 table has 8 rows, paper says "seven
  categories"). Only family counts are bumped; category counts stay as each
  file states. Reconciling them is out of scope.
- The factory injects `betas`/`eps` into every non-sgd optimizer's groups
  (`factory.py` line 172–174). `ada_muon` reads `eps` (variance ε) and
  ignores `betas` — same tolerated-key behaviour as `muon`.
- **Parameter-group taxonomy = 10 blocks** (verified):
  `GROUP_NAMES` = `embeddings, pos, attention, ffn, norms, head, mhc,
  router, residual, other` (`base.py` lines 16–27);
  `_COMMON_PER_GROUP_KEYS` is generated from `GROUP_NAMES`
  (`factory.py` lines 83–91), and the schema's
  `shared_per_group_suffixes` lists all 40 suffixes (`_optimizer.yaml`
  lines 191–231; also `configs/README.md` line 163 and
  `architecture.md` §8.1 "Parameter groups"). `ada_muon` therefore
  supports every per-group suffix automatically — the new module never
  enumerates groups.
- bf16 is used inside NS only (mirrors `muon.py`); the V-EMA, `√V` divide and
  `γ_t` run on the parameter dtype (typically FP32) after casting `O_t` back.
  The paper does not specify precision; this matches the repo's Muon and is
  recorded as an implementation decision.

**Risks**

- Zero-gradient edge case: `sign(M)=0 ⟹ O=0 ⟹ ‖Ô‖_F=0 ⟹ γ_t=∞`. The paper
  never hits this (nonzero gradients); the implementation clamps the norm
  with `clamp_min(ns_eps)` and the resulting update is exactly 0. Covered by
  a test.
- Non-2-D parameters: `ndim < 2` is skipped (family convention); tensors with
  `ndim ≥ 3` (conv kernels) would break the NS matmuls exactly as they would
  under `muon` — the example keeps `embedding.conv.enabled: false`. Any
  vision/audio preset combining this optimizer with conv embeddings needs the
  same opt-out.
- Like `muon`/`turbo_muon`, 1-D parameters (norms, biases) receive **no**
  update under this optimizer (the paper uses an auxiliary Adam for them).
  Small norm/bias-heavy models will train those params never — the preset's
  per-group `lr_norms` keys exist but are inert for 1-D members of those
  groups. Same caveat as the existing family members; documented in Cons.
- Naming: `ada_muon` vs `adamw`/`adan` prefixes — `extract_prefixed_parameters`
  matches on exact `ada_muon-` prefix, so no collision with `adamw-` keys
  (verified: prefix extraction is exact-string, `base.py` lines 181–186).
- No task constraints: `training.task` rules (AGENTS.md hard rule 6/8) do not
  interact with optimizer choice; the example uses `mlm` + encoder, matching
  the family convention.

**Open questions**

- Expose `c_rms` (0.2) as a suffix? The paper fixes it ("any fixed positive
  constant works in the analysis"); plan keeps it a module constant. Revisit
  only if a future paper needs a different target RMS.
- The ablation variants in §4.3 (sign-only, second-moment-only) are not
  separate optimizers; only AdaMuon itself is planned.
- Should the optional mermaid figure be produced now or deferred? Plan marks
  it optional; omitting it does not break any gate.
- **Stale 4-block text in `AGENTS.md` line 58** (hard rule 7 still says
  "Groups: `embeddings`, `norms`, `attention`, `other`"). The code, schema,
  `configs/README.md` and `architecture.md` §8.1 all use the 10-block
  taxonomy, so AGENTS.md rule 7 is out of date. Recommend a separate
  one-line fix listing all 10 groups — **requires user approval** and is
  deliberately outside this plan's file set (it is repo documentation drift,
  not part of the AdaMuon feature). The `/add-optimizer` command file itself
  does not enumerate groups, so no command change is needed.
- Older Muon-family examples (`es_arch_hybrid_turbo_muon.yaml`,
  `es_arch_all_gated_muon.yaml`, `es_arch_sigmoid_derf_muon.yaml`) and the
  `run_e2e.py` `_with_optimizer` harness (lines 214–233) only set the 4
  original groups — schema-valid (subsets allowed) but dated. Expanding them
  to all 10 groups is cosmetic and out of scope; the new example shows the
  full taxonomy instead.

## 10. Out of scope

This command created **only** this plan file. No code, schema, preset, test,
doc, PDF, website-mirror file or git history was created, edited or deleted;
no `pytest`, e2e, LaTeX, param-check, `uv sync`, training, inference or
web-server command was run; nothing was staged, committed or pushed; the
mirror was not touched.
