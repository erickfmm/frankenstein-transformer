# Plan: add ROOT optimizer (arXiv 2511.20626)

> Canonical location: `.opencode/plans/add-optimizer-2511.20626.md`.
> Implemented in `feat(optimizer): add ROOT (arXiv 2511.20626)`.
> Source command: `/add-optimizer https://www.alphaxiv.org/abs/2511.20626`.
> Status: PLAN ONLY — nothing outside the plan directory was written by the
> planning run.

## 1. Header

**Paper:** *ROOT: Robust Orthogonalized Optimizer for Neural Network Training* —
Wei He*, Kai Han*, Hang Zhou, Hanting Chen, Zhicheng Liu, Xinghao Chen, Yunhe
Wang (Huawei Noah's Ark Lab; * equal contribution), arXiv:2511.20626v1
[cs.LG], 25 Nov 2025 (tech report). Code announced at
`https://github.com/huawei-noah/noah-research/tree/master/ROOT`.

**Summary.** Muon-style optimizers orthogonalize the momentum matrix with a
fixed-coefficient quintic Newton–Schulz (NS) iteration. The paper identifies
two robustness gaps: (1) **dimensional fragility** — the one-size-fits-all
coefficients $(a,b,c)=(3.4445,-4.7750,2.0315)$ leave a large orthogonalization
MSE precisely on square / near-square shapes (Table 1: MSE 0.0499 at
$2048\times2048$ vs 0.0003 at $2048\times16384$); (2) **outlier fragility** —
heavy-tailed gradient components distort the Frobenius normalization that NS
requires and are amplified by the polynomial iteration. ROOT (**R**obust
**O**rthogonalized **O**p**T**imizer) fixes both: it replaces the fixed
coefficients with *per-matrix-shape* calibrated coefficients $\{a^{(m,n)},
b^{(m,n)}, c^{(m,n)}\}$ ("AdaNewton", offline Mixed-1:3 calibration), and it
runs a proximal L1 **soft-threshold** decomposition of the momentum
($O_t=\mathcal{T}_\varepsilon[M_t]$, $B_t=M_t-O_t$) with a dynamic percentile
threshold $\varepsilon_t=\mathrm{Quantile}(|M_t|,p)$, $p=0.90$, so NS only ever
sees the robust base component $B_t$. On a 1B Llama-style model (FineWeb-Edu,
10B/100B tokens) ROOT beats Muon and AdamW on loss and zero-shot benchmarks
(avg 60.12 vs 59.59 vs 59.05) and transfers to ViT/CIFAR-10.

**Why it belongs here.** The repo already has an *Orthogonalized* bucket
(`muon`, `turbo_muon`, `af_muon`, `ada_muon`, `dion3` — `docs/architecture.md`
§8.1, line 729). ROOT is a drop-in next member of that family: same Newton–Schulz
machinery and hybrid AdamW-fallback recipe as `src/model/optimizer/muon.py`,
one momentum buffer (same state footprint as Muon), no closure requirement, no
task restriction, and no new mandatory per-group keys — only three new global
suffixes (`quantile`, `softthresh`, `ada_newton`) on top of the Muon knobs.

**Decision gate: PASS.** It is a genuine optimizer method; it does not
duplicate any entry in `factory.OPTIMIZER_REGISTRY` (28 entries, no `root`;
`grep -rn "\broot\b" src/schema/_optimizer.yaml src/model/optimizer/` finds
only prose uses); the paper fully specifies every equation, the coefficient
table, and the default percentile. No rejection report needed.

## 2. Paper extract

### 2.1 Update rule (verbatim equations)

Preliminaries / momentum (Eq. 1):

$$M_t = \mu M_{t-1} + \nabla_\theta \mathcal{L}_t(\theta_{t-1}), \qquad M_0 = 0$$

Robust decomposition (Eqs. 5–9). Model $M_t = B_t + O_t$ (base + outliers) and
solve $\min_{B,O} \lVert M_t - B - O\rVert_F^2 + \lambda\lVert O\rVert_1$ s.t.
$\lVert B\rVert \le \tau$. The proximal solution is the L1 soft-threshold:

$$\mathcal{T}_\varepsilon[x]_i = \operatorname{sign}(x_i)\cdot\max(|x_i|-\varepsilon,\,0)
= \begin{cases} x_i-\varepsilon & x_i>\varepsilon\\ 0 & |x_i|\le\varepsilon\\ x_i+\varepsilon & x_i<-\varepsilon\end{cases}$$

$$O_t = \mathcal{T}_\varepsilon(M_t)\ \text{(sparse outliers, discarded)},\qquad B_t = M_t - O_t\ \text{(robust base)}$$

applied element-wise to the matrix (Eq. 10). Dynamic threshold (§4.5 ablation):

$$\varepsilon_t = \operatorname{Quantile}(|M_t|,\,p), \qquad p\in\{0.85, 0.90, 0.95, 0.99\},\ \ p=0.90\ \text{optimal for LLM pre-training}$$

AdaNewton iteration (Eqs. 2–3). Quintic NS on the robust base, with
shape-specific coefficients:

$$X_0 = \frac{B_t}{\lVert B_t\rVert_F}$$
$$X_k = a^{(m,n)} X_{k-1} + b^{(m,n)} X_{k-1}(X_{k-1}^{\top}X_{k-1}) + c^{(m,n)} X_{k-1}(X_{k-1}^{\top}X_{k-1})^{2}, \quad k=1..K,\ K=5$$

Parameter update:

$$\theta_t = \theta_{t-1} - \eta\, B_t^{\text{orth}}, \qquad B_t^{\text{orth}} = X_K$$

**Weight decay / hybrid routing.** The paper follows the standard Muon recipe:
"All 2D weight matrices were optimized via Muon or ROOT, while 1D parameters
(biases, norms) and embeddings and class-tokens were optimized using AdamW"
(§4.6; same convention implied for the LLM runs). ViT fixed params: Muon
LR = 0.02, AdamW LR = 0.001, WD = 5×10⁻⁵. LLM runs use default Muon
hyperparameters with a 0.2 scaling factor (Moonlight / arXiv:2502.16982) to
align update RMS with AdamW.

**Repo-convention notes (implementation decisions, see §3):** the repo's
`muon.py` applies decoupled decay `p ← p(1 − lr·wd)` before the orthogonalized
update (`muon.py:155-156`) and applies **no** RMS/0.2 scale factor (the preset
LR absorbs it, `muon.py:175`). ROOT mirrors the repo convention exactly; the
paper's 0.2 factor is documented in the annex as an LR-calibration remark, not
implemented as a constant.

### 2.2 Algorithm (paper Algorithm 2)

```
Require: learning rate η, momentum μ, threshold ε (ε_t = Quantile(|M_t|, p))
1:  M_0 ← 0
2:  for t = 1, … do
3:      G_t ← ∇_θ L(θ_{t−1})                          # gradient
4:      M_t ← μ M_{t−1} + G_t                          # momentum accumulation
5:      O_t ← T_ε[M_t]                                 # outlier separation (soft-threshold)
6:      B_t ← M_t − O_t                                # clipped base components
7:      B_t^orth ← AdaNewton(B_t)                      # robust orthogonalization (per-shape a,b,c)
8:      θ_t ← θ_{t−1} − η B_t^orth                     # update
9:  end for
10: return θ_t
```

Schedule interactions (paper §4.1, not optimizer-internal): cosine LR decaying
to 10% of peak, 2 000-step warmup; peak 8e-4 (10B tokens) / 1.6e-3 (100B).
Nesterov is not in Algorithm 2 but the repo's Muon offers it; ROOT keeps the
same `nesterov` option (default true) for family consistency.

### 2.3 AdaNewton coefficient table (paper Table 1, Mixed-1:3 calibration)

Shapes are $(m, n)$ as printed; MSE is the orthogonalization error with the
adaptive coefficients (vs fixed 0.0499/0.0637/0.0761/0.0338/0.0362/0.0340/0.0425):

| m | n | a | b | c | MSE |
|---|---|---|---|---|---|
| 2048 | 2048 | 3.3334 | −4.2591 | 1.7791 | 0.0352 |
| 4096 | 4096 | 3.3732 | −4.6134 | 2.0576 | 0.0470 |
| 8192 | 8192 | 3.3886 | −4.9026 | 2.3105 | 0.0587 |
| 3072 | 2048 | 2.9091 | −3.8108 | 1.8600 | 0.0024 |
| 4096 | 2048 | 2.7739 | −3.4911 | 1.6992 | 0.0010 |
| 8192 | 2048 | 2.5925 | −3.0010 | 1.4138 | 0.0003 |
| 16384 | 2048 | 2.7045 | −3.2335 | 1.5336 | 0.0003 |

Fallback for unlisted shapes: Muon's fixed $(3.4445, -4.7750, 2.0315)$.

### 2.4 Hyperparameters

| Name | Type | Default | Range / notes | Scope |
|---|---|---|---|---|
| `lr` (η) | float | 0.02 (repo Muon default) | paper peak 8e-4–1.6e-3 with 0.2 RMS scaling; repo presets use per-group values | per-group (shared `lr_*`) |
| `momentum` (μ) | float | 0.95 | "default Muon hyperparameters" | global |
| `nesterov` | bool | true (repo Muon default; not in paper Alg. 2) | family consistency | global |
| `ns_steps` (K) | int | 5 | "strict budget of 5 iterations" (§4.2) | global |
| `ns_eps` | float | 1e-7 (repo Muon default) | numerical stability of $\lVert\cdot\rVert_F$ normalization | global |
| `quantile` (p) | float | 0.90 | ablated {0.85, 0.90, 0.95, 0.99}; 0.90 LLM-optimal, 0.85 best on ViT/CIFAR | global, **new** |
| `softthresh` | bool | true | enables outlier suppression; false ⇒ pure AdaNewton | global, **new** |
| `ada_newton` | bool | true | per-shape coefficients; false ⇒ fixed Muon coefficients (with `softthresh: true` reproduces the paper's "ROOT (SoftThresh)" ablation) | global, **new** |
| `weight_decay` (wd) | float | 0.0 / per-group | decoupled (repo convention); paper ViT used 5e-5 | per-group (shared `wd_*`) |
| `betas`, `eps` | tuple / float | (0.9, 0.95) / 1e-8 | AdamW fallback only | per-group (shared `betas_*`, `eps_*`) |

### 2.5 State buffers and per-step cost

- **2-D hidden matrices:** 1 buffer (`momentum_buffer`, same as Muon). The
  threshold $\varepsilon_t$ is recomputed per step — no extra state.
- **AdamW-fallback params** (`embeddings`/`pos`/`head` groups + every non-2-D
  tensor): `exp_avg` + `exp_avg_sq` (+ step counter), via
  `base.adamw_fallback_update`.
- **Per-step cost:** $\mathcal{O}(n\cdot k)$ matmul-dominated NS loop (identical
  to Muon) **plus** $\mathcal{O}(N\log N)$ for the quantile/sort of $|M_t|$
  ($N = mn$ elements of the matrix). Memory: 1 momentum matrix per parameter,
  i.e. same footprint as Muon.

### 2.6 Category

Existing bucket **Orthogonalized** (`docs/architecture.md` §8.1 line 729); in
the paper's family section wording, "geometry-oriented methods". No new
category. Character: *dimension-calibrated Newton–Schulz + soft-threshold
outlier suppression on the momentum before orthogonalization; hybrid AdamW
fallback as Muon*.

### 2.7 BibTeX data

Authors: Wei He, Kai Han, Hang Zhou, Hanting Chen, Zhicheng Liu, Xinghao
Chen, Yunhe Wang. Year: 2025. Title: *ROOT: Robust Orthogonalized Optimizer
for Neural Network Training*. arXiv:2511.20626 [cs.LG].

## 3. Decisions

1. **Names.** Registry `root`; class `ROOT` (acronym style like `EA4LLM`).
2. **Peer module.** `src/model/optimizer/muon.py` — mirror: module docstring
   citing the paper, `from __future__ import annotations`, Google-style
   docstrings, `defaults` dict, `ADAMW_FALLBACK_GROUPS` + `p.ndim != 2` routing
   (`muon.py:121-140`), momentum buffer + nesterov (`muon.py:144-153`),
   decoupled weight decay (`muon.py:155-156`), bfloat16 NS loop with the
   rows≤cols transpose (`muon.py:158-175`). Changed parts: (a) soft-threshold
   between momentum and normalization; (b) `(a,b,c)` from a shape lookup
   instead of the literal on `muon.py:158`.
3. **Category.** Orthogonalized (existing); summary-table row + §8.1 row +
   family-sentence list in both languages.
4. **Suffix table** (all global; shared per-group keys `lr_*`, `wd_*`,
   `betas_*`, `eps_*` also apply, `supports_shared_per_group_suffixes: true`):

| Key | Type | Default | Coercion | EN description | ES description |
|---|---|---|---|---|---|
| `root-momentum` | float | 0.95 | `to_float` | momentum coefficient for the gradient buffer | coeficiente de momentum del búfer de gradientes |
| `root-nesterov` | bool | true | `bool` | Nesterov-style momentum before orthogonalization | momentum estilo Nesterov antes de ortogonalizar |
| `root-ns_steps` | int | 5 | `to_int` | number of Newton–Schulz iterations | número de iteraciones Newton–Schulz |
| `root-ns_eps` | float | 1e-7 | `to_float` | epsilon for the Frobenius normalization | épsilon de la normalización de Frobenius |
| `root-quantile` | float | 0.90 | `to_float` | percentile `p` of \|M\| for the dynamic soft-threshold ε_t = Quantile(\|M_t\|, p) | percentil `p` de \|M\| para el umbral dinámico ε_t = Quantile(\|M_t\|, p) |
| `root-softthresh` | bool | true | `bool` | enable soft-threshold outlier suppression | activa la supresión de atípicos por soft-threshold |
| `root-ada_newton` | bool | true | `bool` | use per-shape calibrated NS coefficients (false = fixed Muon coefficients) | usa coeficientes NS calibrados por forma (false = coeficientes fijos de Muon) |

5. **`supports_shared_per_group_suffixes`: `true`** (hybrid Muon-style routing).
6. **Example preset.** `configs/examples/es_arch_hybrid_root.yaml` — task `mlm`,
   clone of `es_arch_hybrid_dion3.yaml` (same hybrid encoder: retnet / titan_attn
   / mamba / ode layers) with the optimizer block swapped to `root`. Smallest
   *valid* config in the family's established pattern: full per-group
   lr/wd/betas/eps matrix + the three new globals, auto-validated by
   `tests/test_yaml_examples.py`.
7. **Census bump 28 → 29.** Old number verified live at:
   `docs/architecture.md:83` (`28 optimizers + registry + parameter routing`),
   `docs/architecture.md:716` (`### 8.1 Optimizers (28)`),
   `README.md:27` (mermaid `Train<br/>28 optimizer families…`),
   `README.md:93` (`| Optimizer families | 28 across 7 categories |`),
   `docs/paper/sections/05-optimizer-families.tex:4` ("twenty-eight"),
   `docs/paper/appendices/annex-summary-tables.tex:100` ("all 28 supported
   optimizers"), `docs/paper-es/sections/05-familias-optimizadores.tex:4`
   ("veintiocho"), `docs/paper-es/appendices/annex-summary-tables.tex:100`
   ("los 28 optimizadores"). (`configs/README.md` has no census number.)
8. **RMS-scale decision.** No 0.2 factor implemented — mirrors repo `muon.py`
   (see §2.1 note); documented as a remark in the annex.
9. **Coefficient lookup key.** `(min(rows, cols), max(rows, cols))` of the
   working matrix (after the Muon transpose so rows ≤ cols); all seven Table-1
   shapes satisfy m ≥ n, so the canonical keys are
   `(2048,2048), (4096,4096), (8192,8192), (2048,3072), (2048,4096),
   (2048,8192), (2048,16384)`. Unlisted shapes fall back to Muon's fixed
   coefficients.

**Hard rules restated.** The schema is the contract
(`additionalProperties: false` — every YAML key must exist in
`src/schema/_optimizer.yaml` / the `_training.yaml` `if/then` blocks).
Optimizer keys are prefixed `root-<group>_<param>` / `root-<param>`; the
factory's `OPTIMIZER_ALLOWED_KEYS["root"]` is the whitelist and
`ensure_no_unknown_parameters` rejects everything else. Never read an
undocumented key. The papers are the theory source of truth (AGENTS.md D11):
the annex formulations must match arXiv:2511.20626 and the `paper-es` annex
must translate them faithfully.

## 4. File-by-file plan

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/root.py` (new) | `"""ROOT optimizer: robust orthogonalized updates. …"""` module docstring citing He et al. 2025, arXiv:2511.20626 + the hybrid recipe. `from __future__ import annotations`; `import torch`; `from torch.optim import Optimizer`; `from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update`. Module constants: `_MUON_COEFFICIENTS = (3.4445, -4.7750, 2.0315)` and `_ADA_NEWTON_COEFFICIENTS` dict with the seven §2.3 rows (comment: "Table 1 of arXiv:2511.20626, Mixed 1:3 calibration; keyed by shape-canonical (min, max)"). Helpers: `_shape_coefficients(rows, cols, ada_newton)` (returns per-shape triple or fixed fallback) and `_threshold_eps(t, quantile)` → `torch.quantile(t.abs().flatten(), quantile)` when `numel <= 16_777_216` else `t.abs().flatten().kthvalue(max(1, ceil(numel·q))).values` (torch.quantile element-count guard). Class `ROOT(Optimizer)` with `__init__(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, ns_eps=1e-7, quantile=0.90, softthresh=True, ada_newton=True, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)`; `defaults` dict; Google-style class/init/step docstrings. `@torch.no_grad() step(closure=None)`: identical loop head to `muon.py:121-140` (AdamW fallback for `ADAMW_FALLBACK_GROUPS` and `p.ndim != 2`); then `momentum_buffer` EMA (`muon.py:144-148`), nesterov blend (`muon.py:150-153`); **new** — if `group["softthresh"]`: `eps_t = _threshold_eps(g, float(group["quantile"]))` (Eq. ε_t), `o = torch.sign(g) * (g.abs() - eps_t).clamp(min=0)` (Eq. 7/10), `g = g - o` (Eq. 9, B_t); decoupled weight decay (`muon.py:155-156`); `a, b, c = _shape_coefficients(...)` **after** the transpose so the key is canonical; bfloat16 Frobenius normalization + quintic loop + transposed update exactly as `muon.py:159-175`. CPU-safe: no `.cuda()`, no autocast. |
| 2 | `src/model/optimizer/__init__.py` | Insert `from .root import ROOT` in the alphabetical slot between `from .radam import RAdamOptimizer` (line 22) and `from .schedulefree_adamw import ScheduleFreeAdamW` (line 23). Add `"ROOT",` to `__all__` between `"RAdamOptimizer",` (line 52) and `"SGDMomentum",` (line 53). |
| 3 | `src/model/optimizer/factory.py` | (a) Import `from .root import ROOT` between `.radam` (line 46) and `.schedulefree_adamw` (line 47). (b) `OPTIMIZER_REGISTRY`: append `"root": ROOT,` after `"eve": Eve,` (line 82) mirroring the enum's chronological tail. (c) `OPTIMIZER_ALLOWED_KEYS`: `"root": _COMMON_PER_GROUP_KEYS \| {"momentum", "nesterov", "ns_steps", "ns_eps", "quantile", "softthresh", "ada_newton"},` after the `"eve"` entry (lines 144-145). (d) Dispatch branch before the final `return optimizer_cls(converted_groups)` (line 435), after the `eve` branch: `if optimizer_name == "root": momentum = to_float(scoped.get("momentum"), 0.95); nesterov = bool(scoped.get("nesterov", True)); ns_steps = to_int(scoped.get("ns_steps"), 5); ns_eps = to_float(scoped.get("ns_eps"), 1e-7); quantile = to_float(scoped.get("quantile"), 0.90); softthresh = bool(scoped.get("softthresh", True)); ada_newton = bool(scoped.get("ada_newton", True)); return optimizer_cls(converted_groups, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, ns_eps=ns_eps, quantile=quantile, softthresh=softthresh, ada_newton=ada_newton)`. |
| 4 | `src/schema/_optimizer.yaml` | (a) `optimizer_class.enum`: append `- root` after `- ea4llm` (line 165). (b) EN use-case table (after the `ea4llm` row, line 62): `` | `root` | Robust orthogonal updates | Dimension-calibrated Newton-Schulz coefficients + soft-threshold outlier suppression on the momentum; AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2511.20626) | ``. (c) ES table (after line 115): `` | `root` | Ortogonalización robusta | Coeficientes Newton-Schulz calibrados por dimensión + supresión de atípicos por soft-threshold sobre el momentum; respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2511.20626) | ``. (d) EN references sentence (line 82): append ``; `root` (paper: arxiv:2511.20626)``; same in the ES twin (line 136). (e) EN "Optimizer-specific suffixes" bullet after the `dion3` bullet (line 195): "- `root`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`, `quantile`, `softthresh`, `ada_newton`. Same hybrid routing as `muon`: only hidden-layer 2-D matrices are orthogonalized, using per-shape calibrated Newton-Schulz coefficients when `ada_newton` is on; `quantile` is the percentile `p` defining the dynamic soft-threshold `ε_t = Quantile(|M_t|, p)` (default 0.90); `softthresh: false` disables outlier suppression; the `embeddings`, `pos` and `head` groups plus every non-2-D parameter use a decoupled AdamW fallback configured by the shared suffixes". ES bullet after line 236: "- `root`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`, `quantile`, `softthresh`, `ada_newton`. Mismo enrutamiento híbrido que `muon`: solo se ortogonalizan las matrices 2-D de capas ocultas, con coeficientes Newton-Schulz calibrados por forma cuando `ada_newton` está activo; `quantile` es el percentil `p` del umbral dinámico `ε_t = Quantile(|M_t|, p)` (por defecto 0.90); `softthresh: false` desactiva la supresión de atípicos; los grupos `embeddings`, `pos` y `head` más todo parámetro no 2-D usan un respaldo AdamW desacoplado configurado por los sufijos compartidos". (f) `x_parameter_reference.by_optimizer` — append after the `ea4llm` block (lines 395-398): `root:` / `prefix: "root-"` / `supports_shared_per_group_suffixes: true` / `specific_global_suffixes: [momentum, nesterov, ns_steps, ns_eps, quantile, softthresh, ada_newton]`. |
| 5 | `src/schema/_training.yaml` | Append the 29th `if/then` block at the end of the `allOf` list (after the `ea4llm` block), exactly one block (parity with `by_optimizer` enforced by `tests/test_schema_consistency.py::test_optimizer_prefixes_match_conditional_rules`): `- if:` `properties:` `optimizer:` `properties:` `optimizer_class:` `const: root` / `then:` `properties:` `optimizer:` `properties:` `parameters:` `additionalProperties: false` `patternProperties:` `^root-: { type: [number, boolean, array, string] }` — same shape as the `muon` block at lines 1288-1301. |
| 6 | `src/schema/_sbert.yaml` | `optimizer_kwargs.description` (line 400): extend "momentum (for SGD/muon/turbo_muon/af_muon/ada_muon/dion3)" → append `/root`; "nesterov (for muon/turbo_muon/af_muon)" → append `/root`; "ns_steps (…)" and "ns_eps (…)" lists → append `/root`; add "quantile/softthresh/ada_newton (for root)". Mirror every change in `description_es` (line 401) with the backticked Spanish style. |
| 7 | `configs/examples/es_arch_hybrid_root.yaml` (new) | Clone `configs/examples/es_arch_hybrid_dion3.yaml` verbatim (model block: retnet/titan_attn/mamba/ode layer_pattern, hidden 768, 12 layers; training: task mlm, seed 42, 4 epochs, the same dataset/cache/scheduler lines) and replace only the optimizer block: `optimizer_class: root`, all `dion3-` per-group keys re-prefixed `root-` (same lr/wd/betas/eps values), and the specific keys `root-momentum: 0.95`, `root-nesterov: true`, `root-ns_steps: 5`, `root-ns_eps: 1.0e-7`, `root-quantile: 0.90`, `root-softthresh: true`, `root-ada_newton: true`; `csv_log_path: training_metrics_es_arch_hybrid_root.csv`. Must pass `load_training_config` (auto-checked by `tests/test_yaml_examples.py`). |
| 8 | `configs/README.md` | (a) Class list (lines 132-160): add `- root` after `- ea4llm`. (b) Per-optimizer suffix list (after the `dion3` line 207): "- `root`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`, `quantile`, `softthresh`, `ada_newton` (same hybrid routing as `muon`; `quantile` = percentile of the dynamic soft-threshold, default 0.90)". (c) Example-keys block (near lines 213-218): add `- root-quantile: 0.90`. |
| 9 | `full_tests/run_e2e.py` | (a) `OPTIMIZERS` list (lines 246-251): append `"root"` after `"ea4llm"`. (b) Param map (after the `ea4llm` branch at lines 678-680): `if optimizer_class == "root": params[f"{prefix}momentum"] = 0.9; params[f"{prefix}ns_steps"] = 5; params[f"{prefix}quantile"] = 0.9`. |
| 10 | `tests/test_optimizer_factory.py` | (a) Add `"root"` to the `required` set in `test_registry_contains_all_required` (line 29). (b) New class `BuildOptimizerROOTTests(unittest.TestCase)` after `BuildOptimizerTurboMuonTests` (lines 197-235), mirroring the Muon test style: `test_builds_without_error` (`build_optimizer("root", _param_groups(), {})`), `test_step_runs` (4×4 param, `p.sum().backward()`, `opt.step()`), `test_default_betas_and_eps_present` (defaults `(0.9, 0.95)`, `1e-8`), `test_hidden_matrix_uses_momentum_state` (post-step `opt.state[p]` contains `momentum_buffer`), `test_embedding_group_uses_adamw_fallback` (group named `embeddings` → state has `exp_avg`), `test_specific_suffixes_parsed` (`{"root-quantile": 0.95, "root-softthresh": False, "root-ada_newton": False, "root-ns_steps": 3}` → reflected in `opt.defaults`/param_groups), `test_shape_coefficients_match_table1` (`_shape_coefficients(2048, 8192, True) == (2.5925, -3.0010, 1.4138)`; `_shape_coefficients(64, 64, False) == (3.4445, -4.7750, 2.0315)`; unlisted shape falls back). |
| 11 | `docs/bibliography/optimizers.bib` | Append after the last optimizer entry, matching the `liu_ea4llm_2025` Zotero-arXiv style: `@misc{he_root_2025, title={ROOT: Robust Orthogonalized Optimizer for Neural Network Training}, author={Wei He and Kai Han and Hang Zhou and Hanting Chen and Zhicheng Liu and Xinghao Chen and Yunhe Wang}, year={2025}, eprint={2511.20626}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={https://arxiv.org/abs/2511.20626},}`. |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | Insert a `\paragraph{ROOT (Robust Orthogonalized Optimizer).}` block at the end of `\subsection{Second-Order, Geometric, and Orthogonality Optimizers}` — after Dion3's `\paragraph{Pros.}` (ends line 1266) and before `\subsection{Evolutionary and Population-Based Hybrids}` (line 1268). Content: `figure[H]` with `\fitfigure{\includegraphics{figures/optimizers/root.pdf}}`, caption "Step flow of ROOT: momentum accumulation, soft-threshold outlier separation, and shape-calibrated Newton–Schulz orthogonalization.", `\label{fig:opt-root}`; intro sentence citing `~\citep{he_root_2025}` on the two fragilities; `align` block with Eq. (1) momentum, Eq. (7)+(10) soft-threshold $\mathcal{T}_\varepsilon$, Eq. (9) $B_t = M_t - O_t$, dynamic $\varepsilon_t=\mathrm{Quantile}(|M_t|,p)$, Eq. (3) quintic with superscripted $(m,n)$ coefficients, and the update $\theta_t = \theta_{t-1}-\eta B_t^{\text{orth}}$; the seven-row coefficient table from §2.3 as a `tabular` (shapes, $a,b,c$, MSE) labelled `tab:root-coefficients`; `algorithm[H]` float mirroring paper Algorithm 2 (Require: $\eta,\mu,p$; lines: gradient, momentum, $\varepsilon_t$, $O_t$, $B_t$, AdaNewton loop, update; `\Else` AdamW hybrid fallback branch like `alg:muon`); remark that shapes outside the calibrated table fall back to Muon's fixed coefficients and that the paper aligns RMS with AdamW via a 0.2 factor (absorbed by the LR here); `\paragraph{Cons.}` (quantile sort cost; calibrated table covers seven shapes; outliers are discarded, not shrunk) and `\paragraph{Pros.}` (dimension-invariant orthogonalization precision; outlier-robust, no retuning of the Muon recipe; same single momentum buffer as Muon). |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Row after the Dion3 row (line 155), inside the orthogonality category block: `ROOT & Orthogonality (robust) & 1 ($m$); + $v$ on the AdamW fallback & $\mathcal{O}(n \cdot k)$ + quantile sort & lr, momentum, wd, NS steps $k$, quantile $p$; $\beta$, eps for embeddings/pos/heads/vectors & \citep{he_root_2025} \\`. (b) Census sentence line 100: "all 28 supported optimizers" → "all 29 supported optimizers". |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: "The system supports twenty-eight optimizer families in eight categories" → "twenty-nine"; append ", ROOT" to "geometry-oriented methods (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3)". |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of #12, `\paragraph{ROOT (Optimizador Ortogonalizado Robusto).}` placed after Dion3's `\paragraph{Ventajas.}` (before the evolutionary subsection). Figure path uses the ES convention `\fitfigure{\includegraphics{../paper/figures/optimizers/root.pdf}}` (cf. line 11 `sgd-momentum.pdf`). Same equations (unchanged), Spanish prose, `\paragraph{Contras.}` / `\paragraph{Ventajas.}`, `\citep{he_root_2025}`. |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | (a) Row after the Dion3 row: `ROOT & Ortogonalidad (robusta) & 1 ($m$); + $v$ en el respaldo AdamW & $\mathcal{O}(n \cdot k)$ + ordenamiento por cuantil & lr, momentum, wd, pasos NS $k$, cuantil $p$; $\beta$, eps para embeddings/pos/heads/vectores & \citep{he_root_2025} \\`. (b) Census line ~100: "los 28 optimizadores" → "los 29 optimizadores". |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: "veintiocho familias de optimizadores" → "veintinueve familias de optimizadores"; append ", ROOT" to "métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3)". |
| 18 | `docs/architecture.md`, `README.md`, `configs/README.md` | Census pass (grep the old count at implementation time: `grep -rn "28 optimi\|Optimizer families | 28\|(28)" docs README.md`): `docs/architecture.md:83` `28 optimizers` → `29 optimizers`; `docs/architecture.md:716` heading `### 8.1 Optimizers (28)` → `(29)`; §8.1 Orthogonalized row (line 729) → add `root` to the class list and extend the character cell with "; `root` calibrates the NS coefficients per matrix shape and soft-thresholds momentum outliers before orthogonalization (single momentum buffer, dynamic percentile threshold)". `README.md:27` mermaid node `28 optimizer families` → `29 optimizer families`; `README.md:93` `| Optimizer families | 28 across 7 categories |` → `29 across 7 categories`. `configs/README.md` has no census number (verified) — only the class/suffix/example entries from row 8. |
| 19 | `docs/paper/figures/optimizers/root.mmd` (+ built `root.pdf`, `root.svg`) | New Mermaid step-flow diagram in the house style of `muon.mmd` (flowchart LR: Gradient → Momentum μ → Soft-threshold ε_t=Quantile(\|M\|,p) → B_t=M−O → Frobenius norm → NS iterations (per-shape a,b,c) ×5 → update −η·X). Build at implementation time from `docs/paper/figures`: `./build-figures.sh both` (script discovers every `.mmd`; needs Chrome/Chromium, auto-detected). If the figure cannot be produced, omit the `figure[H]` from #12/#15 and note it (the annex text stands alone). |

## 5. Ordered execution checklist

Happy path; run every command at **implementation time**, none during planning.

- [ ] 1. Implement `src/model/optimizer/root.py` (file #1).
- [ ] 2. Wire `__init__.py`, `factory.py` (#2, #3).
- [ ] 3. Schema: `_optimizer.yaml`, `_training.yaml`, `_sbert.yaml` (#4–#6).
- [ ] 4. Example `configs/examples/es_arch_hybrid_root.yaml` (#7).
- [ ] 5. `configs/README.md` + `full_tests/run_e2e.py` (#8, #9).
- [ ] 6. Tests (#10); then targeted gates:

```bash
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings
```

- [ ] 7. Full suite (must stay green):

```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

- [ ] 8. Bibliography + EN annex + EN summary table + EN family section (#11–#14).
- [ ] 9. ES annex + ES summary table + ES family section (#15–#17).
- [ ] 10. Census pass (#18) + figure (#19).
- [ ] 11. Fast e2e smoke (raise `--limit` if `root` is not reached; it is last in `OPTIMIZERS`):

```bash
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3
```

- [ ] 12. PDF rebuild (preconditions verified during planning: `/usr/bin/pdflatex`
      and `/usr/bin/bibtex` both exist). From the repository root:

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

      Fallback: if `pdflatex`/`bibtex` are unavailable on the implementing
      machine, leave the `.tex`/`.bib` sources updated and note that the rebuild
      must happen on a TeX machine before the next release (the website sync
      copies prebuilt PDFs only).
      Post-build verification: no `Fatal error` and no `Citation .* undefined`
      in the logs; `paper.pdf` / `paper-es.pdf` newer than the edited
      `.tex`/`.bbl`; the new `\citep{he_root_2025}` resolves;
      `git status --short docs/paper docs/paper-es` shows only intended
      `.tex`/`.bbl`/`.pdf` (plus `figures/optimizers/root.*`) changes.
- [ ] 13. Website sync (§6) — planned, not executed here.
- [ ] 14. Commit + push (§7).

## 6. Website sync (planned, not executed)

Follow `/sync-website` (`.opencode/command/sync-website.md`), driven by the
read-only state report `bash .opencode/command/sync-website-state.sh`:

1. **Copy main → mirror** (`erickfmm.github.io/frankenstein-transformer/`, plain
   `cp` + `mkdir -p`, no rsync): `src/schema.yaml` → `schema.yaml`;
   `src/schema/_optimizer.yaml`, `_training.yaml`, `_sbert.yaml` → `schema/**`;
   `configs/examples/es_arch_hybrid_root.yaml` → `examples/examples/**`;
   `configs/README.md` → `examples/README.md`; `docs/paper/{paper.tex,paper.bbl,paper.pdf}`
   + `appendices/annex-1-optimizers.tex`, `appendices/annex-summary-tables.tex`,
   `sections/05-optimizer-families.tex` → `paper/en/…`; the three ES twins →
   `paper/es/…`; `docs/bibliography/optimizers.bib` → `paper/bibliography/`;
   `docs/architecture.md` → `paper/specs/architecture.md`;
   `docs/paper/figures/optimizers/root.{mmd,pdf,svg}` → `paper/figures/optimizers/`.
2. **Paper artifacts**: PDFs were rebuilt in §5 (pdflatex=yes on this machine);
   ensure `paper/en/paper.pdf` / `paper/es/paper-es.pdf` in the mirror are the
   rebuilt ones, then top-level copies `cp paper/en/paper.pdf paper/en.pdf` and
   `cp paper/es/paper-es.pdf paper/es.pdf`. Delete every `*.aux *.log *.out
   *.toc *.blg` from the mirror. Rebuild the Paper-tab HTML (pandoc is
   installed): `python3 flatten_tex.py en/paper.tex en/paper.bbl
   /tmp/paper-en-flat.tex --web --manifest /tmp/paper-en-figures.json` (and the
   `es/` twin), `pandoc … --mathjax --toc -o paper-en.html` / `paper-es.html`,
   `python3 postprocess_html.py --manifest … paper-en.html` (and ES).
3. **Validation Gate** (in order, all must pass):
   - `uv run --extra cpu --extra train python full_tests/param_count_check.py`
     → prints `… configs counted, 0 failures` (regenerates
     `full_tests/param_truth.json`; also validates the new example preset) —
     changes to `param_truth.json` are intentional.
   - `node full_tests/param_estimate_check.mjs
     erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js
     full_tests/param_truth.json --tol 0.01` → exit 0 / `ALL PASS`
     (optimizer choice does not change the parameter-count estimator, so no
     `ft-param-estimator.js` edit is expected).
   - State-script probes `schema-refs`, `example-yaml`, `cli-flags`, estimator
     → all PASS (`bash .opencode/command/sync-website-state.sh`).
4. **Leave both repos uncommitted** until the commit phase.

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions (English, conventional tag, structured
body; never stage `*.aux *.log *.out *.toc *.blg` or caches;
`param_truth.json` changes are intentional).

1. **Main repo** — `feat(optimizer): add ROOT (arXiv 2511.20626)`
   - Body: implementation notes (per-shape AdaNewton coefficient table +
     fallback, quantile soft-threshold with kthvalue safeguard, Muon hybrid
     routing); new suffixes (`quantile`, `softthresh`, `ada_newton` plus the
     Muon knobs); schema/registry/`if/then` changes; example preset; e2e list;
     tests; docs (EN/ES annexes, summary tables, family sections, bibliography
     `he_root_2025`, census 28→29, rebuilt PDFs, figure).
   - Stage exactly:
     - `src/model/optimizer/root.py`, `src/model/optimizer/__init__.py`,
       `src/model/optimizer/factory.py`
     - `src/schema/_optimizer.yaml`, `src/schema/_training.yaml`,
       `src/schema/_sbert.yaml`
     - `configs/examples/es_arch_hybrid_root.yaml`, `configs/README.md`
     - `full_tests/run_e2e.py`
     - `tests/test_optimizer_factory.py`
     - `docs/bibliography/optimizers.bib`,
       `docs/paper/appendices/annex-1-optimizers.tex`,
       `docs/paper/appendices/annex-summary-tables.tex`,
       `docs/paper/sections/05-optimizer-families.tex`,
       `docs/paper/paper.tex` (only if the annex include list changes),
       `docs/paper/paper.bbl`, `docs/paper/paper.pdf`,
       `docs/paper/figures/optimizers/root.mmd`, `root.pdf`, `root.svg`,
       `docs/paper-es/appendices/annex-1-optimizadores.tex`,
       `docs/paper-es/appendices/annex-summary-tables.tex`,
       `docs/paper-es/sections/05-familias-optimizadores.tex`,
       `docs/paper-es/paper-es.tex` (if touched), `docs/paper-es/paper-es.bbl`,
       `docs/paper-es/paper-es.pdf`,
       `docs/architecture.md`, `README.md`
     - `full_tests/param_truth.json` (if regenerated before committing)
   - Then `git push origin HEAD:main`.
2. **Mirror** — `git -C erickfmm.github.io …` stage the synced site paths
   (`frankenstein-transformer/schema.yaml`, `schema/**`,
   `examples/**` incl. the new example + README, `paper/en/**`, `paper/es/**`,
   `paper/bibliography/optimizers.bib`, `paper/specs/architecture.md`,
   `paper/figures/optimizers/root.*`, `paper/paper-en.html`,
   `paper/paper-es.html`, `paper/en.pdf`, `paper/es.pdf`), commit
   `docs(site): sync ROOT optimizer (arXiv 2511.20626)`, then
   `git -C erickfmm.github.io push origin HEAD:main`.

## 8. Verification matrix

| Gate | Command (implementation time) | Expected result |
|---|---|---|
| Targeted optimizer gates | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings` | All pass; `BuildOptimizerROOTTests` green; prefix parity test sees exactly one `^root-` rule; every preset/example (incl. the new one) validates |
| Full suite | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | All 46 test files pass, no regressions |
| e2e smoke | `uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3` | `root` config builds + trains a step on CPU (raise `--limit` if the tail of `OPTIMIZERS` is not reached) |
| PDF EN | `cd docs/paper && pdflatex → bibtex → pdflatex ×2` | No `Fatal error`, no `Citation he_root_2025 undefined`, `paper.pdf` newer than edited `.tex`/`.bbl` |
| PDF ES | `cd docs/paper-es && pdflatex → bibtex → pdflatex ×2` | Same checks for `paper-es.pdf` |
| PDF cleanliness | `git status --short docs/paper docs/paper-es` | Only intended `.tex`/`.bbl`/`.pdf`/figure files; no `*.aux *.log *.out *.toc *.blg` |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `… configs counted, 0 failures` |
| Estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | Exit 0, `ALL PASS` |
| State probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, estimator probes PASS; drift section empty for synced files |

## 9. Risks, assumptions and open questions

- **`torch.quantile` element cap** (assumption to verify): some torch builds
  reject tensors above 2^24 ≈ 16.7 M elements (a 2048×16384 momentum matrix is
  33.5 M). The plan hard-codes the `kthvalue` fallback behind a
  `numel <= 16_777_216` guard; verify the exact limit on torch 2.6.0 CPU at
  implementation time and adjust the constant if needed.
- **Coefficient coverage.** Table 1 covers seven shapes; every other shape
  (including all tiny test tensors) falls back to Muon's fixed coefficients.
  This matches the paper's fallback spirit but is an implementation decision —
  documented in the module docstring, the annex remark, and covered by
  `test_shape_coefficients_match_table1`.
- **Official code cross-check.** The Huawei repo
  (`huawei-noah/noah-research/tree/master/ROOT`) is announced but was not
  consulted (planning stayed read-only on the paper). If it is public at
  implementation time, cross-check the coefficient table and the quantile
  computation; if it differs (e.g. per-layer learned coefficients), surface the
  diff to the user before finalizing.
- **Jointly-learned coefficients are out of scope.** §3.2.1 notes coefficients
  "can be optimized jointly with the model parameters during training"; the
  adopted strategy everywhere in the paper's experiments is offline calibration
  (Mixed 1:3). We implement the offline lookup only.
- **Nesterov is a repo extension** (paper Algorithm 2 has plain momentum);
  kept for family consistency with `muon`/`turbo_muon`/`af_muon`, default true.
- **RMS 0.2 factor not implemented** (repo `muon.py` has no scale factor);
  presets are expected to set per-group LRs as they already do for `muon`.
- **`p` default.** 0.90 is the LLM pre-training optimum; the CIFAR-10/ViT
  ablation favoured 0.85 — exposed via `root-quantile`, no second preset.
- **Naming collisions.** None: `root`/`ROOT` absent from the registry, schema
  enum, optimizer package, examples and tests (verified by grep; only prose
  uses of "root" exist, e.g. `af_muon.py:283`).
- **Task compatibility.** ROOT is unconstrained: no closure requirement, no
  `gradient_accumulation_steps: 1` rule, valid for `mlm`, `causal_lm`,
  `classification`, `segmentation`, `patch_prediction`, `text_classification`
  and SBERT (unlike `eve`/`ea4llm`, which are excluded from SBERT).
- **Figure dependency.** The annex figure needs Chromium for Mermaid; if
  unavailable, omit the figure and keep the text (explicitly allowed by the
  recipe).
- **Out of scope / flagged, not done:** no existing optimizer renamed, no
  family deleted, no Muon code path modified (ROOT only *mirrors* `muon.py`;
  if the implementer is tempted to factor a shared NS helper out of
  `muon.py`/`turbo_muon.py`/`root.py`, that is a refactor requiring separate
  user approval).

## 10. Out of scope

This command produced a plan only. **No code, schema, preset, test, doc, PDF,
website-mirror or git change was made**: nothing outside the plan directory
was created, edited or deleted; no `pytest`/e2e/param scripts, no
`pdflatex`/`bibtex`/`pandoc`/`flatten_tex.py`, no `uv sync`, no
training/inference/`web-server`, no mirror copy, and no
`git add`/`commit`/`push`/`checkout`/`stash`/`reset` were executed. The
pre-existing untracked `.opencode/plan/` directory in the repo (harness
plan-mode storage, present before this run) is untouched and left alone.
