# Plan: add Dion3 optimizer (arXiv:2608.11612)

> Canonical location: `.opencode/plan/add-optimizer-2608.11612.md`.
> Status: plan only — nothing outside the plan directory was written by the planning run.

## 1. Header — paper summary

**Dion3: Full-stack orthogonal updates** (Amsel, Zhang, Ahn, Naeimi, Feng, Chen, Dao,
Langford — NYU/Princeton/NVIDIA/Yale/Microsoft Research, Aug 13 2026, arXiv:2608.11612)
revises Muon to cut the cost of its cubic-time Newton-Schulz orthogonalization at every
level of the stack. Four contributions: (1) **Gram Newton-Schulz**, a mathematically
equivalent reformulation that iterates on the small symmetric Gram matrix `XXᵀ`
(≈55–68 % FLOP savings at typical aspect ratios); (2) CuteDSL symmetric-GEMM GPU kernels
(§5, hardware-specific); (3) the **Dion3 update rule**: orthogonalize only a *fraction*
`f ∈ (0,1]` of the momentum rows per step (top-`k = ⌈f·n⌉` rows by ℓ1 norm) with
**error feedback** that decays only the selected momentum rows; (4) megabatched
communication for distributed training (§7, out of scope). Dion3 matches or improves
Muon/NorMuon loss with up to 6× faster optimizer steps; `f = 1` reduces exactly to
Muon (Muon flavor) or NorMuon (NorMuon flavor with row-wise second moments + Frobenius
rescale). It belongs in this repo as a new member of the existing **Orthogonalized**
family bucket next to `muon`, `turbo_muon`, `af_muon`, `ada_muon` — the algorithmic core
(Gram NS + fractional selection + error feedback + optional NorMuon) is pure PyTorch,
CPU-safe, and needs no kernels; the paper's kernel/distributed layers are flagged
out of scope.

## 2. Paper extract

### 2.1 Update rule (verbatim from the paper)

**Algorithm 4 — Dion3 update rule (single weight matrix).** Input: weight
`W ∈ ℝ^{n×m}`, gradient `G`, momentum buffer `M`, row-wise 2nd-moment buffer
`v ∈ ℝ^n` (NorMuon only), fraction `f ∈ (0,1]`, momentum `µ`, decay `β₂`,
learning rate `η`:

1. `M ← M + G`  *(momentum accumulation; damping is deferred to line 9 — footnote 4:
   NS sees `M+G` rather than `µM+G`; since `M₀ = 0` these differ by a constant factor
   µ, removed by orthogonalization anyway)*
2. `S ← indices of the k = ⌈f·n⌉ rows of M with largest ℓ₁ norm`  *(Selection)*
3. `O ← polar(M[S, :])`  *(Orthogonalize the `k × m` submatrix via Gram Newton-Schulz)*
4. *(Optional NorMuon)* `v_i ← β₂ v_i + (1−β₂) · m · mean_j(O_ij²)` for `i ∈ S`
5. *(Optional NorMuon)* `Õ_ij ← O_ij / (√v_i + ε)`
6. *(Optional NorMuon)* `Õ ← Õ · ‖O‖_F / ‖Õ‖_F`
7. `W ← (1 − η·wd) · W`  *(decoupled weight decay, applied to **all** rows)*
8. `W_ij ← W_ij − η·Õ_ij` for `i ∈ S`  *(update **selected rows only**; `Õ = O` in the
   Muon flavor)*
9. `M_ij ← µ · M_ij` for `i ∈ S`  *(Error Feedback: decay only the selected rows)*

**Error feedback rationale:** decompose `M = M̃ + (M − M̃)` where `M̃` matches `M` on
selected rows and is zero elsewhere; instead of `M ← µM`, do `M ← µM̃ + (M − M̃)`. The
extra `(1−µ)(M − M̃)` term lets ignored rows build up in `M` so they are eventually
selected. At exact approximation (`f = 1`) this recovers standard momentum.

**Selection details:** `f ∈ (0,1]`, *recommend `f = 1/4` or `f = 1/8`*; `f = 1` ⇒ Muon.
Subselect rows or columns — when unsharded, **pick the smaller dimension**. Top-ℓ1
selection; random selection was only slightly worse in initial experiments.

**Algorithm 3 — Stabilized Gram Newton-Schulz** (the `polar(·)` of line 3; input
`X ∈ ℝ^{n×m}`, `n ≤ m`, coefficients `{(a_t, b_t, c_t)}_{t=1..5}` = Polar Express
per-iteration triples):

```
1:  X ← X / (‖X‖_F + ε)          ▷ normalize singular values to [0,1], ε = 1e-7
2:  X ← float16(X)               ▷ half precision (NOT bfloat16 — spurious eigenvalues)
3:  R₀ ← XXᵀ ;  Q₀ ← I
4:  for t = 1..5:
5:      if t = 3:                ▷ RESTART (stabilizes; best per condition-number sweep)
6:          X ← Q₂X ;  R₂ ← XXᵀ ;  Q₂ ← I
7:      end if
8:      Zₜ  ← bₜ·R_{t−1} + cₜ·R²_{t−1}
9:      Qₜ  ← Q_{t−1}·Zₜ + aₜ·Q_{t−1}
10:     (RZ)ₜ ← R_{t−1}·Zₜ + aₜ·R_{t−1}
11:     Rₜ  ← Zₜ·(RZ)ₜ + aₜ·(RZ)ₜ
12: end for
13: X ← Q₅X ;  return X
```

Mathematical basis (Theorem 2): with `pₜ(x) = x·hₜ(x²)` (any odd NS polynomial
`p(x)=ax+bx³+cx⁵` ⇔ `h(x)=a+bx+cx²`), the composed polynomial satisfies
`(p_T∘…∘p_1)(x) = q_T·x` where `r₀ = x², q₀ = 1, qₜ = q_{t−1}·hₜ(r_{t−1})`,
`rₜ = r_{t−1}·hₜ(r_{t−1})²`; lifted to matrices `Q_T → (XXᵀ)^{−1/2}`, so the output
equals standard Newton-Schulz. The restart at iteration 3 kills spurious negative
eigenvalues of the Gram matrix (observed down to `−4·10⁻⁴` in half precision); restarting
after iteration 2 gives the best condition-number bound (Appendix B.2.1, Figs. 18–19).

**Cost.** Gram NS: `(4T + 3α − 3)·n³` FLOPs with `α = m/n`, vs standard NS
`T(3α+1)n³` (symmetric GEMMs). Typical Muon shape (`T=5, α=4`): **−55 %** FLOPs
(−68 % vs plain GEMMs). Row selection shrinks `n → k = ⌈fn⌉` and raises `α` by `1/f`,
compounding the Gram advantage; selection itself costs `O(nm)` (ℓ1 norms + top-k).

**NorMuon flavor (Algorithm 4 lines 4–6, from Li et al., ICML 2026):** row-wise second
moment of the *orthogonalized* update, then per-row rescale, then a Frobenius-norm
rescale back to `‖O‖_F`. Normalization steps should run in `float32`.

**Finite precision / f=1 identity.** At `f = 1` Dion3 reduces to Muon/NorMuon up to:
row permutation by selection, momentum damping applied *after* forming the update,
NorMuon normalization in float32, and the (Triton) fused weight update — convergence
curves coincide (Appendix D.4: final losses 2.2141 vs 2.2146 at 1B).

### 2.2 Pseudocode / schedule interactions

- No warmup/schedule coupling beyond a standard LR schedule; the paper's runs use
  warmup–stable–cooldown (200-step warmup from 2e-3, cooldown final 25 %) — informational
  only, the repo's schedulers are unchanged.
- Paper recipe: matrix params get `η·√(d_out/d_in)`; embeddings/LM-head/norms/biases use
  AdamW (β₁=0.9, β₂=0.95); wd 0.01 on matrix params only. **Repo peers (`muon.py`,
  `ada_muon.py`) do not implement the √ scaling or the AdamW fallback** — keep
  repo-consistent, note in annex + risks.

### 2.3 Hyperparameters → suffix table

| Paper symbol | Meaning | Type | Default (repo) | Paper guidance | Coercion | Scope |
|---|---|---|---|---|---|---|
| `η` | learning rate | float | per-group (`DEFAULT_GROUP_LR`) | peak 0.01 at 1B (with √ scaling) | `to_float` via factory | per-group (`lr_*`) |
| `wd` | decoupled weight decay, all rows | float | per-group (`DEFAULT_GROUP_WD`, 0.01) | 0.01 on matrices | `to_float` via factory | per-group (`wd_*`) |
| `β₂` | NorMuon row-moment decay | float | per-group `betas[1]` = 0.95 | free input in Alg. 4 | `to_betas` via factory | per-group (`betas_*`) |
| `ε` (NorMuon) | row-normalization epsilon | float | per-group `eps` = 1e-8 | small | `to_float` via factory | per-group (`eps_*`) |
| `f` | row-selection fraction ∈ (0,1] | float | **0.25** | 1/4 or 1/8 recommended; 1 ⇒ Muon | `to_float` | **global** (`dion3-fraction`) |
| `µ` | momentum (error-feedback decay) | float | **0.95** | tuned 0.9 at 1B (Table 7); repo peer default 0.95 | `to_float` | **global** (`dion3-momentum`) |
| `T` | Gram NS iterations | int | **5** | fixed 5, restart after iteration 2 | `to_int` | **global** (`dion3-ns_steps`) |
| `ε` (NS) | Frobenius-normalization epsilon | float | **1e-7** | Alg. 3 line 1 | `to_float` | **global** (`dion3-ns_eps`) |
| flavor | NorMuon on/off | bool | **false** | two flavors shipped | `bool(...)` | **global** (`dion3-normuon`) |

### 2.4 State buffers / per-step cost / complexity

- **State:** 1 buffer `M` (same shape as `W`) + optional `v ∈ ℝ^n` (row-wise, only when
  `normuon: true`). 1D/scalar params are skipped entirely (peer behaviour of
  `muon.py` L104, `ada_muon`).
- **Per-step cost:** selection `O(nm)`; Gram NS on `k×m`:
  `(4T+3α−3)k³` FLOPs with `α = m/k` (≈ `O(k²(4k+3m))`); error feedback `O(km)`.
  Summary-table form: `O(k²(m+k))`, `k = ⌈fn⌉` — vs Muon's `O(n·k_NS)`-style entries.

### 2.5 Category

Existing bucket **Orthogonalized** (`docs/architecture.md` §8.1 L681:
"Orthogonalized | `muon`, `turbo_muon`, `af_muon`, `ada_muon`"); summary-table family
"Orthogonality" (EN L151–154, ES L151+); README "Geometry-Oriented" (L152). **No new
category.**

### 2.6 BibTeX data

Dion3: authors Noah Amsel, Jack Zhang, Kwangjun Ahn, Ali Naeimi, Austin Feng,
Berlin Chen, Tri Dao, John Langford; year 2026; title "Dion3: Full-stack orthogonal
updates"; arXiv:2608.11612 (cs.LG). Supporting citation for the NS coefficients:
Amsel, Persson, Musco, Gower, "The Polar Express: Optimal Matrix Sign Methods and
their Application to the Muon Algorithm", ICLR 2026
(openreview `yRtgZ1K8hO`). Related-but-not-planned: Dion (Ahn et al. 2025,
arXiv:2504.05295), NorMuon (Li et al., ICML 2026).

**Decision gate: PASSED.** The paper is an optimizer method; the registry
(`factory.py` L51–77) has no Dion/Dion3 entry (grep `dion` over `src/` → no matches);
algorithms and defaults are fully specified. The paper bundles exactly one *optimizer*
(the `dion` package exposes Muon/NorMuon as baselines — already covered by `muon`);
systems contributions (kernels §5, megabatching §7, Triton/CUDA-graph tricks) are out
of scope by design and flagged below.

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `dion3` (snake_case) | paper short name "Dion3"; peer pattern `turbo_muon` |
| Class name | `Dion3` | peer pattern `TurboMuon` |
| Closest peer | `src/model/optimizer/muon.py` | matrix-only orthogonalized family: mirror module docstring w/ arXiv cite, `from __future__ import annotations`, defaults dict, `@torch.no_grad() step()`, `ndim < 2` skip (L104), tall-matrix transpose (L127–130), decoupled wd `p.mul_(1−lr·wd)` (L120–121), final `.to(p.dtype)` (L140). Replace standard NS with Gram NS + restart; add selection, error feedback, optional NorMuon steps. |
| Category | existing Orthogonalized bucket (no new one) | §2.5 |
| Suffixes | `fraction`, `momentum`, `ns_steps`, `ns_eps`, `normuon` (all global) + shared per-group `lr_/wd_/betas_/eps_`; `betas[1]` doubles as NorMuon β₂ | §2.3; ada_muon precedent for reusing a shared knob instead of adding a suffix |
| `supports_shared_per_group_suffixes` | `true` | every peer is `true` (`_optimizer.yaml` L237–337) |
| Example preset | `configs/examples/es_arch_hybrid_dion3.yaml` (task `mlm`) | naming peer `es_arch_hybrid_ada_muon.yaml` / `es_arch_hybrid_turbo_muon.yaml`; MLM is the cheapest schema-valid task (no `image:`/`audio:` blocks) |
| Census bump | **25 → 26** (actual old count read from repo, verified in 8 places below) | §4 row 18 |

EN/ES description text drafted in §4 row 4.

## 4. File-by-file plan table

Execution order: 1→9 implementation, 10 tests, 11→18 docs+census.

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/dion3.py` (new) | `class Dion3(torch.optim.Optimizer)`. Module docstring cites "(Amsel et al. 2026, arXiv:2608.11612)" + one-paragraph description (peer: `muon.py` L1–12). `from __future__ import annotations`; Google-style docstrings; no `.cuda()`. Module constants: `_NS_COEFFS: tuple[tuple[float,float,float], ...]` — five per-iteration `(a_t,b_t,c_t)` Polar Express triples (source at implementation time from the `gram-newton-schulz` package / ICLR'26 paper; **documented fallback**: the constant Muon triple `(3.4445, −4.7750, 2.0315)` reused each iteration, mathematically valid NS — see Risks R1), `_RESTART_AFTER = 2`. `__init__(params, lr=0.02, fraction=0.25, momentum=0.95, ns_steps=5, ns_eps=1e-7, normuon=False, betas=(0.9,0.95), eps=1e-8, weight_decay=0.0)`; `ValueError` unless `0 < fraction <= 1`, `0 <= momentum < 1`, `ns_steps >= 1`, `ns_eps > 0`. `defaults = dict(...)` incl. `betas`/`eps` (factory injects them per group, `_optimizer_base` pattern). `step()` sketch (equation → line mapping): closure handling (peer L97–100) → per param: skip `p.grad is None or p.ndim < 2` (peer L104) → `transposed = p.size(0) > p.size(1)`; work on the smaller-as-rows view (peer L127–130) → state init `momentum_buffer` zeros_like (+ `row_second_moment` zeros(rows) when normuon) → **Alg4 L1**: `buf += grad` → **L2**: `k = ceil(fraction·rows)`; `S = torch.topk(buf.abs().sum(dim=1), k).indices` → **L3**: `O = _gram_newton_schulz(buf[S,:], coeffs, ns_steps, ns_eps)` (private method implementing Alg. 3: Frobenius-normalize, `.half()`, `R = X@X.T`, loop t=1..T with restart at t=`_RESTART_AFTER+1`, `Z/Q/RZ/R` updates, return `Q_T @ X` cast to param dtype; symmetric products written as `A@B` on CPU — the CuteDSL kernels are GPU-only and out of scope) → **L4–6 (normuon only, float32)**: `v[S] = β₂·v[S] + (1−β₂)·k·O.square().mean(dim=1)`; `Õ = O / (v[S].sqrt()[:,None] + eps)`; `Õ *= O.norm() / Õ.norm()` → **L7**: if `wd > 0`: `p.mul_(1 − lr·wd)` (all rows, peer L120–121) → **L8**: `p[S,:] −= lr·(Õ if normuon else O)` → **L9**: `buf[S,:] *= momentum` (error feedback). Un-transpose bookkeeping mirrors peer L137–140. |
| 2 | `src/model/optimizer/__init__.py` | Add `from .dion3 import Dion3` between L35 (`from .cautious_adamw import CautiousAdamW`) and L36 (`from .galore_adamw import GaLoreAdamW`) keeping alphabetical order; add `"Dion3",` to `__all__` after `"CautiousAdamW",` (L37) — file currently L1–54. |
| 3 | `src/model/optimizer/factory.py` | (a) `from .dion3 import Dion3` between L35/L36. (b) `"dion3": Dion3,` at the end of `OPTIMIZER_REGISTRY` after `"q_apollo": QApollo,` (L76). (c) `"dion3": _COMMON_PER_GROUP_KEYS \| {"fraction", "momentum", "ns_steps", "ns_eps", "normuon"},` after the `q_apollo` entry (L134). (d) Dispatch branch after the `apollo_mini` block (ends L369), before the final `return optimizer_cls(converted_groups)` (L371): `if optimizer_name == "dion3": fraction = to_float(scoped.get("fraction"), 0.25); momentum = to_float(scoped.get("momentum"), 0.95); ns_steps = to_int(scoped.get("ns_steps"), 5); ns_eps = to_float(scoped.get("ns_eps"), 1e-7); normuon = bool(scoped.get("normuon", False)); return optimizer_cls(converted_groups, fraction=fraction, momentum=momentum, ns_steps=ns_steps, ns_eps=ns_eps, normuon=normuon)`. (`betas`/`eps` already per-group via `_build_param_groups_for_optimizer`, L175–177, since `dion3 ∉ {"sgd_momentum"}`.) |
| 4 | `src/schema/_optimizer.yaml` | (a) `- dion3` appended to `optimizer_class.enum` after `- q_apollo` (L126). (b) Use-case tables: EN after L55 add ``| `dion3` | Faster orthogonal updates | Fractional Gram-Newton-Schulz Muon with error feedback (arXiv:2608.11612) |``; ES after L92 add ``| `dion3` | Ortogonalización acelerada | Muon fraccional con Gram-Newton-Schulz y retroalimentación de error (arXiv:2608.11612) |``. (c) Suffix bullet lists: EN after L155 add ``- `dion3`: `fraction`, `momentum`, `ns_steps`, `ns_eps`, `normuon` ``; ES after L191 the same. (d) `x_parameter_reference.by_optimizer` after the `q_apollo` block (L334–337): `dion3:` / `prefix: "dion3-"` / `supports_shared_per_group_suffixes: true` / `specific_global_suffixes: [fraction, momentum, ns_steps, ns_eps, normuon]`. |
| 5 | `src/schema/_training.yaml` | Append exactly **one** `if/then` block after the final `^q_apollo-` block (the `allOf` list ends with q_apollo, after L1524+): `- if:` / `properties:` / `optimizer:` / `properties:` / `optimizer_class:` / `const: dion3` / `then:` / `properties:` / `optimizer:` / `properties:` / `parameters:` / `additionalProperties: false` / `patternProperties:` / `^dion3-: { type: [number, boolean, array, string] }` — same shape as every peer block (L1202–1215 et seq.). Parity gate: `tests/test_schema_consistency.py::test_optimizer_prefixes_match_conditional_rules` (L76) requires exactly one block per `by_optimizer` prefix. |
| 6 | `src/schema/_sbert.yaml` | Extend `optimizer_kwargs` description (EN L352, ES L353) — new suffixes are introduced. EN: append `..., fraction/momentum/ns_steps/ns_eps/normuon (for dion3; betas[1] is the NorMuon row-moment decay when normuon is on)`. ES: `..., fraction/momentum/ns_steps/ns_eps/normuon (para dion3; betas[1] es el decaimiento del segundo momento por fila de NorMuon cuando normuon está activo)`. |
| 7 | `configs/examples/es_arch_hybrid_dion3.yaml` (new) | Clone of `es_arch_hybrid_ada_muon.yaml` (L1–117) with: `optimizer_class: dion3`; all 40 per-group keys re-prefixed `dion3-` (10× `lr_*`, 10× `wd_*`, 10× `betas_*` `[0.9, 0.95]`, 10× `eps_*` `1e-8`); globals `dion3-fraction: 0.25`, `dion3-momentum: 0.95`, `dion3-ns_steps: 5`, `dion3-ns_eps: 1.0e-7`, `dion3-normuon: false`; `csv_log_path: training_metrics_es_arch_hybrid_dion3.csv`. Model block unchanged (encoder, `layer_pattern: [retnet, titan_attn, mamba, ode]`, mlm). Must pass `load_training_config` (auto-globbed by `tests/test_yaml_examples.py`). |
| 8 | `configs/README.md` | (i) Optimizer list: `- dion3` after `- q_apollo` (L156). (ii) Suffix reference: `- \`dion3\`: \`fraction\`, \`momentum\`, \`ns_steps\`, \`ns_eps\`, \`normuon\`` after L201. (iii) Example keys: `- \`dion3-fraction: 0.25\`` after L209. |
| 9 | `full_tests/run_e2e.py` | (i) `"dion3"` appended to `OPTIMIZERS` (L238–243, after `"q_apollo"`). (ii) Documented specific globals in `_with_optimizer` (L597) after the ada_muon block (L626–628): `if optimizer_class == "dion3": params[f"{prefix}momentum"] = 0.9; params[f"{prefix}fraction"] = 0.25; params[f"{prefix}ns_steps"] = 5`. Note: appended last ⇒ `--category opt --limit 3` will **not** reach it — raise `--limit` to ≥ 25 (list grows 24→25 entries; `af_muon` is already absent, pre-existing, see R6). |
| 10 | `tests/test_optimizer_factory.py` | (i) Add `"dion3"` to the `required` set in `test_registry_contains_all_required` (L29). (ii) New class `BuildOptimizerDion3Tests` after `BuildOptimizerAdaMuonTests` (L468–542), methods: `test_builds_without_error`, `test_step_runs`, `test_step_runs_normuon_flavor` (`{"dion3-normuon": True}`), `test_defaults_match_paper` (fraction 0.25, momentum 0.95, ns_steps 5, ns_eps 1e-7, normuon False), `test_custom_fraction_suffix` (`{"dion3-fraction": 0.5}` → `defaults["fraction"] == 0.5`), `test_unselected_rows_unchanged` (f<1: rows outside S bitwise-identical), `test_error_feedback_decays_only_selected_rows` (state `momentum_buffer` row norms: selected ×µ, unselected untouched), `test_orthogonalized_rows_have_unit_rms` (updated selected rows ≈ `lr·O` with `‖O‖_F ≈ √k` ⇒ delta norm `≈ lr·√k`), `test_string_coercion_and_invalid_fallback` (`"0.5"→0.5`, `"bad"→0.25`; `ns_steps "4"→4`, `"bad"→5`), `test_unknown_group_suffix_raises` (`{"dion3-lr_bogus": 1.0}` → ValueError naming `lr_bogus`), `test_vector_params_skipped` (1D param unchanged). Pattern: Q-APOLLO class (L210–229) + ada_muon class (L468–542). |
| 11 | `docs/bibliography/optimizers.bib` | Append after `si_adamuon_2025` (L319–333), matching the Zotero-arXiv style of `boissin_turbo-muon_2025`/`zhu2025apollosgdlikememoryadamwlevel`: `@misc{amsel_dion3_2026, title={Dion3: Full-stack orthogonal updates}, author={Noah Amsel and Jack Zhang and Kwangjun Ahn and Ali Naeimi and Austin Feng and Berlin Chen and Tri Dao and John Langford}, year={2026}, eprint={2608.11612}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={https://arxiv.org/abs/2608.11612}}` and (supports the coefficients cite in row 12) `@inproceedings{amsel_polar_express_2026, title={The Polar Express: Optimal Matrix Sign Methods and their Application to the Muon Algorithm}, author={Noah Amsel and David Persson and Christopher Musco and Robert M. Gower}, booktitle={The Fourteenth International Conference on Learning Representations (ICLR)}, year={2026}, url={https://openreview.net/forum?id=yRtgZ1K8hO}}`. |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | Insert `\paragraph{Dion3 (Fractional Gram-Newton-Schulz Orthogonalization).}` between AdaMuon's `\paragraph{Pros.}` list (ends L1202) and the closing `\small` longtable (L1204), inside `\subsection{Second-Order, Geometric, and Orthogonality Optimizers}` (L803). **No figure** (no `figures/optimizers/dion3.pdf` is planned; AdaMuon L1156 is the no-figure precedent). Draft (to be kept verbatim-faithful to the arXiv paper — D11): prose ("Dion3 is a full-stack revision of Muon that attacks the cost of the orthogonalization step algorithmically…"), `align` block with equations (1) `M_t = M_{t−1} + G_t`, (2) `S_t = topk_i(‖M_{t,i,:}‖₁, k)`, `k = ⌈f n⌉`, (3) `O_t = GramNS(M_t[S_t,:])`, (4) `W_t = (1−ηλ)W_{t−1}; W_t[S_t,:] ← W_t[S_t,:] − η O_t`, (5) `M_t[S_t,:] ← µ M_t[S_t,:]` (error feedback); a `GramNS` paragraph with the iteration `Z_t = b_t R_{t−1} + c_t R_{t−1}²`, `Q_t = Q_{t−1}Z_t + a_t Q_{t−1}`, `R_t = Z_t(R_{t−1}Z_t + a_t R_{t−1}) + a_t(R_{t−1}Z_t + a_t R_{t−1})`, restart after iteration 2, float16, Frobenius normalization ε=10⁻⁷, cost `(4T+3α−3)k³` vs `T(3α+1)n³`, `\citep{amsel_polar_express_2026}`; a NorMuon-flavor sentence (`v_i ← β₂v_i + (1−β₂) k mean_j O_{ij}²`, `Õ = O⊘(√v+ε)`, `Õ ← Õ‖O‖_F/‖Õ‖_F`, float32); `\begin{algorithm}[H] \caption{Dion3 (Fractional Orthogonalization with Error Feedback)} \label{alg:dion3}` mirroring `alg:muon`/`alg:ada-muon` style (L1021–1039, L1170–1188); `\paragraph{Cons.}` (only a row fraction is updated per step — trajectory changes, benign per §8.1; ℓ1 selection + error feedback add bookkeeping; GPU speedups rely on symmetric kernels not shipped here; 2D-only like Muon) and `\paragraph{Pros.}` (up to 6× faster optimizer steps at matched loss; Gram NS mathematically equivalent, −55…−68 % FLOPs; `f=1` recovers Muon/NorMuon; single momentum buffer). Cite `\citep{amsel_dion3_2026}`. Also update the closing longtable row L1215: "Geometry-oriented & Muon, Turbo-Muon, AF-Muon, AdaMuon" → append ", Dion3" with clause "fractional Gram-Newton-Schulz orthogonalization with error feedback". |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | (i) L100: "catalogs all 25 supported optimizers" → "all 26". (ii) Row after AdaMuon (L154) inside the "Second-Order, Geometric & Orthogonality" block (L146–155): `Dion3 & Orthogonality (fractional) & 1 ($m$); + $v$ rows if NorMuon & $\mathcal{O}(k^2(m+k))$, $k=\lceil f n\rceil$ & lr, momentum, wd, fraction $f$, NS steps $T$, $\beta_2$ (NorMuon) & \citep{amsel_dion3_2026} \\`. |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | L4 (exact current text): "The system supports **twenty-five** optimizer families in seven categories: … geometry-oriented methods (Muon, Turbo-Muon, AF-Muon, AdaMuon); …" → "twenty-six" and "(Muon, Turbo-Muon, AF-Muon, AdaMuon, **Dion3**)". |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of #12, inserted after the ES AdaMuon block (ES AdaMuon `\paragraph` starts L1156; ES file mirrors EN structure line-for-line) before the ES closing longtable. Title: `\paragraph{Dion3 (Ortogonalización Fraccional con Gram-Newton-Schulz).}`. Equations identical to EN; translate prose 1:1 (papers are the theory source of truth — **paper-es must translate faithfully**). Figures live in `../paper/figures/optimizers/` — none referenced (no figure). Label `alg:dion3` (separate compilation, mirrors EN naming). Update the ES closing longtable geometry row to include Dion3. |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | (i) L4: "…45 mezcladores de secuencia en nueve familias, **25 optimizadores**, …" → "26 optimizadores". (ii) L100: "cataloga los 25 optimizadores soportados" → "los 26". (iii) Row after ES AdaMuon row (in the ES "Segundo Orden, Geométricos y Ortogonalidad" block): `Dion3 & Ortogonalidad fraccional & 1 ($m$); + $v$ por filas con NorMuon & $\mathcal{O}(k^2(m+k))$, $k=\lceil f n\rceil$ & lr, momentum, wd, fracción $f$, pasos NS $T$, $\beta_2$ (NorMuon) & \citep{amsel_dion3_2026} \\`. |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | L4 (exact current text): "El sistema soporta **veinticinco** familias de optimizadores en siete categorías: … métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon, AdaMuon); …" → "veintiséis familias" and "(Muon, Turbo-Muon, AF-Muon, AdaMuon, **Dion3**)". |
| 18 | `docs/architecture.md`, `README.md` (+ rows 8, 13, 16) — census pass | Grep the old count `25` (and "twenty-five"/"veinticinco") across the three files and update every hit: **architecture.md** L23 `` | Optimizer families | **25** | `` → `**26**`; L83 `` | `src/model/optimizer/**` | 25 optimizers + registry + parameter routing | `` → `26 optimizers`; L527 (§7.12 SBERT table) `Same 25-family enum` → `Same 26-family enum`; L668 `### 8.1 Optimizers (25)` → `(26)`; L681 Orthogonalized row → `muon, turbo_muon, af_muon, ada_muon, dion3` + character note "fractional Gram-Newton-Schulz with error feedback (arXiv:2608.11612)". **README.md** L27 mermaid `25 optimizer families` → `26`; L93 `| Optimizer families | 25 across 6 categories |` → `26 across 6 categories`; L152 Geometry-Oriented row add `dion3` (count 3→4) — note: that table is already missing `af_muon`/`ada_muon` (sums to 23, pre-existing drift, see R5 — do NOT fix without approval). |

**Hard rules restated** (bind every row above): the schema is the contract
(`additionalProperties: false` everywhere — D1); optimizer keys are prefixed
`dion3-<group>_<param>` / `dion3-<param>` and `factory.OPTIMIZER_ALLOWED_KEYS` rejects
anything else (D9); never read an undocumented key; the papers are the theory source of
truth (D11) so annex formulations must match arXiv:2608.11612 exactly and `paper-es`
must translate them faithfully; every preset/example is smoke-tested (D8).

## 5. Ordered execution checklist (implementation time — do NOT run now)

```bash
# 0. CPU-first environment (never a bare `uv run` — it swaps torch extras)
uv sync --extra cpu --extra train

# 1. Implementation (files 1–9 above), then targeted gates:
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings

# 2. Full suite:
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# 3. Fast e2e smoke (dion3 sits LAST in OPTIMIZERS — raise --limit to reach it):
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 25
```

Then docs (11–18), then PDFs:

```bash
# PDF rebuild, from the repository root — only at implementation time.
# Preconditions VERIFIED on this machine: /usr/bin/pdflatex and /usr/bin/bibtex exist.
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

**PDF fallback** (if TeX unavailable on the implementing machine): leave the `.tex`/`.bib`
sources updated, skip the `.bbl`/`.pdf` regeneration, and record that the rebuild must
happen on a TeX machine before the next release (the website sync copies prebuilt PDFs
only). **Post-compile verification:** logs contain no `Fatal error` and no
`Citation 'amsel_dion3_2026' ... undefined` (nor `amsel_polar_express_2026`);
`docs/paper/paper.pdf` and `docs/paper-es/paper-es.pdf` are newer than the edited
`.tex`/`.bbl`; `git status --short docs/paper docs/paper-es` shows only the intended
`.tex`/`.bbl`/`.pdf` changes and no `*.aux *.log *.out *.toc *.blg`.

## 6. Website sync (planned, not executed)

Follow `/sync-website` (`.opencode/command/sync-website.md`), driven by
`bash .opencode/command/sync-website-state.sh` (read-only report):

1. **Copy main → mirror** (`erickfmm.github.io/frankenstein-transformer/`, plain `cp`,
   no rsync): `src/schema.yaml` → `schema.yaml`; `src/schema/**` → `schema/**`
   (includes `_optimizer.yaml`, `_training.yaml`, `_sbert.yaml`);
   `configs/examples/es_arch_hybrid_dion3.yaml` → `examples/examples/…`;
   `configs/README.md` → `examples/README.md`;
   `docs/paper/{paper.tex,paper.bbl,paper.pdf}` + `sections/05-optimizer-families.tex` +
   `appendices/{annex-1-optimizers.tex,annex-summary-tables.tex}` → `paper/en/…`;
   `docs/paper-es/{paper-es.tex,paper-es.bbl,paper-es.pdf}` + ES sections/appendices →
   `paper/es/…`; `docs/bibliography/optimizers.bib` → `paper/bibliography/…`;
   `docs/architecture.md` → `paper/specs/architecture.md`.
   Never touch site-only files (`index.html`, `ft-param-estimator.js`, `ft-diagram.js`,
   `paper/flatten_tex.py`, generated HTML/PDFs are overwritten only via the steps below).
2. **Paper artifacts**: PDFs already rebuilt in §5 (TeX present) — otherwise copy the
   prebuilt ones; then top-level copies `cp paper/en/paper.pdf paper/en.pdf` and
   `cp paper/es/paper-es.pdf paper/es.pdf` (probes `paper-en-copy`/`paper-es-copy`).
3. **Paper-tab HTML**: `cd erickfmm.github.io/frankenstein-transformer/paper && python3
   flatten_tex.py en/paper.tex en/paper.bbl /tmp/paper-en-flat.tex && pandoc
   /tmp/paper-en-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3
   -V lang=en -o paper-en.html` and the ES twin (`es/paper-es.tex`, `es/paper-es.bbl`,
   `-V lang=es -o paper-es.html`).
4. **Validation Gate** (all must pass):
   `uv run --extra cpu --extra train python full_tests/param_count_check.py` →
   `… configs counted, 0 failures` (regenerates `full_tests/param_truth.json`;
   intentional diff);
   `node full_tests/param_estimate_check.mjs
   erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js
   full_tests/param_truth.json --tol 0.01` → exit 0 `ALL PASS`;
   re-run the state script — probes `schema-refs`, `example-yaml`, `cli-flags`,
   `estimator-drift`, `paper-en-copy`, `paper-es-copy` all PASS.
5. **Leave both repos uncommitted** until the commit phase.

## 7. Commit plan (planned, not executed — `/make-a-commit` conventions)

1. **Main repo** — `feat(optimizer): add Dion3 (arXiv:2608.11612)`.
   Body: implementation notes (Gram Newton-Schulz w/ restart-after-2, fractional ℓ1
   row selection, error feedback, optional NorMuon flavor); new suffixes
   (`fraction`, `momentum`, `ns_steps`, `ns_eps`, `normuon`); schema/registry changes
   (`_optimizer.yaml` enum + `by_optimizer`, `_training.yaml` if/then, `_sbert.yaml`
   descriptions); docs (annexes EN/ES, summary tables EN/ES, family sections EN/ES,
   bibliography, census 25→26, rebuilt PDFs); example `es_arch_hybrid_dion3.yaml`;
   tests (`BuildOptimizerDion3Tests`).
   Stage exactly:
   `src/model/optimizer/dion3.py src/model/optimizer/__init__.py src/model/optimizer/factory.py
   src/schema/_optimizer.yaml src/schema/_training.yaml src/schema/_sbert.yaml
   configs/examples/es_arch_hybrid_dion3.yaml configs/README.md full_tests/run_e2e.py
   tests/test_optimizer_factory.py docs/bibliography/optimizers.bib
   docs/paper/appendices/annex-1-optimizers.tex docs/paper/appendices/annex-summary-tables.tex
   docs/paper/sections/05-optimizer-families.tex docs/paper/paper.bbl docs/paper/paper.pdf
   docs/paper-es/appendices/annex-1-optimizadores.tex docs/paper-es/appendices/annex-summary-tables.tex
   docs/paper-es/sections/05-familias-optimizadores.tex docs/paper-es/paper-es.bbl
   docs/paper-es/paper-es.pdf docs/architecture.md README.md`
   (+ `full_tests/param_truth.json` if the website gate regenerated it — intentional).
   **Never stage** `*.aux *.log *.out *.toc *.blg`, `csvs.zip`, or caches.
   Then `git push origin HEAD:main`.
2. **Mirror** — `git -C erickfmm.github.io add …` the synced paths
   (`schema.yaml schema/_optimizer.yaml schema/_training.yaml schema/_sbert.yaml
   examples/examples/es_arch_hybrid_dion3.yaml examples/README.md
   paper/en/{paper.tex,paper.bbl,paper.pdf,sections/…,appendices/…}
   paper/es/… paper/bibliography/optimizers.bib paper/specs/architecture.md
   paper/en.pdf paper/es.pdf paper/paper-en.html paper/paper-es.html`;
   commit `docs(site): sync Dion3 optimizer (arXiv:2608.11612)`;
   `git -C erickfmm.github.io push origin HEAD:main`.

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Registry/allowed-keys/step | `pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py -v` | all pass incl. new `BuildOptimizerDion3Tests` |
| Schema parity | `pytest tests/test_schema_consistency.py -v` | `test_optimizer_prefixes_match_conditional_rules` green (exactly one `^dion3-` block ↔ `by_optimizer.dion3`), `test_all_presets_validate` green, SBERT `$ref` enum resolves |
| Example smoke | `pytest tests/test_yaml_examples.py -v` | `es_arch_hybrid_dion3.yaml` loads via `load_training_config` |
| Full suite | `pytest tests/ --continue-on-collection-errors` | 46 files green |
| e2e smoke | `full_tests/run_e2e.py --category opt --limit 25` | `opt_dion3` config trains a step, no NaN |
| PDF EN/ES | §5 sequence | no `Fatal error`, no `Citation … undefined`, PDFs newer than `.tex`/`.bbl`, clean `git status docs/paper*` |
| Param truth | `full_tests/param_count_check.py` | `configs counted, 0 failures` |
| Estimator | `param_estimate_check.mjs … --tol 0.01` | exit 0, `ALL PASS` |
| State probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, `estimator-drift`, `paper-en-copy`, `paper-es-copy` all PASS; drift list empty for the synced pairs |

## 9. Risks, assumptions, open questions

- **R1 (assumption to verify at implementation time).** The Polar Express per-iteration
  coefficients `(a_t,b_t,c_t)_{t=1..5}` are not printed in the paper text (they ship in
  the `gram-newton-schulz` package, github.com/Dao-AILab/gram-newton-schulz). Plan:
  source them there; if unavailable, use the constant Muon triple
  `(3.4445, −4.7750, 2.0315)` per iteration (`h(x)=a+bx+cx²`) — mathematically valid,
  slightly more iterations to converge — and say so in the module docstring.
- **R2.** `momentum` default 0.95 follows this repo's Muon-family peers; the paper's
  tuned value at 1B is µ=0.9 (Table 7; the e2e param map sets 0.9). Both documented.
- **R3.** NorMuon β₂ is mapped to the shared per-group `betas[1]` (default 0.95)
  instead of a new suffix (Algorithm 4 names β₂ as a free input; ada_muon precedent
  reuses a shared knob). Open question for the user if a dedicated `dion3-beta2`
  suffix is preferred.
- **R4.** Paper recipe details intentionally NOT ported (repo-peer consistency):
  `√(d_out/d_in)` LR scaling, AdamW fallback for non-matrix groups (repo Muon peers
  simply skip 1D params), MXFP8 weights, Triton fused update, CUDA graphs, CuteDSL
  symmetric kernels, megabatching/communication (§5, §7). Gram NS runs as plain
  `torch.matmul` on CPU — algorithmically identical, wall-clock benefits are
  hardware-specific.
- **R5 (pre-existing drift, needs approval — not fixed by this plan).** README's
  Optimizer Categories table (L145–152) already omits `af_muon` and `ada_muon`
  (sums 23 ≠ 25). The plan adds only `dion3` (Geometry-Oriented 3→4). Fixing the
  pre-existing omission is a separate user decision.
- **R6.** `full_tests/run_e2e.py::OPTIMIZERS` lacks `af_muon` (24 entries; pre-existing).
  After appending `dion3` it holds 25. `--limit 3` will not reach the new entry.
- **R7.** fp16 Gram NS on CPU is fine for tests/small tensors; the paper switched from
  bf16→fp16 for stability (B.2.2) — follow the paper (`half()`), cast back to the param
  dtype at the end (peer L140). The NorMuon normalization runs in float32.
- **R8.** Naming collision check: `grep -r dion src/` → no matches; `Dion3`/`dion3`
  free. Task/mixer compatibility: none — optimizers are orthogonal to
  `training.task` rules (hard rule 6/8) and to mixer choice; works for all 7 tasks and
  all model classes (matrix params updated; 1D params untouched, like `muon`).
- **R9.** Untracked `csvs.zip` exists in the worktree (pre-existing) — leave alone.
- **R10.** Other optimizers in the paper (Muon/NorMuon baselines; Dion 2504.05295 as
  the earlier low-rank variant) are out of scope: Muon is already in the registry,
  NorMuon's steps are exposed via the `normuon` flavor flag, and Dion proper is
  superseded by Dion3 per §6 ("improves on Dion in both speed and performance").
  Recording, not planning, them.

## 10. Out of scope

This command created **no** code, schema, preset, test, doc, PDF, mirror or git change.
Nothing outside `/home/erickfmm/.opencode/plan/` was written; no mutating command
(`pytest`, `uv sync`, `pdflatex`, `git add/commit/push`, sync) was executed. The single
pre-existing untracked file `csvs.zip` was reported and left untouched. Implementation
is a separate, explicitly requested step (switch agents / approve this plan first).
