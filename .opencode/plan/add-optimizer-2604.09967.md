# Plan: add Muon² optimizer (arXiv:2604.09967)

> **Location note.** This plan was produced by the plan-mode `add-optimizer` command.
> The runtime denied writes to the repo-local `.opencode/plan/` path, so the plan lives
> here (`~/.opencode/plan/`). When implementation starts, copy it to
> `<repo>/.opencode/plan/add-optimizer-2604.09967.md` (the repo's canonical plan
> directory per its AGENTS.md).

Paper: **Muon²: Boosting Muon via Adaptive Second-Moment Preconditioning** —
Ziyue Liu, Ruijie Zhang, Zhengyang Wang, Yequan Zhao, Yupeng Su, Zi Yang, Zheng Zhang
(UCSB / University at Albany SUNY). arXiv:2604.09967 (v2, 2026).

Muon² extends Muon by applying **Adam-style adaptive second-moment preconditioning to the
momentum matrix *before* the Newton–Schulz (NS) orthogonalization**. The paper's key insight
is that Muon's polar approximation struggles because the raw momentum matrix is
ill-conditioned (singular values spanning orders of magnitude, ~half in the NS "dead zone");
dividing the momentum by an element-wise second-moment EMA tightens its spectrum, so each NS
step aligns far better with the true orthogonalized update. Across GPT, LLaMA and MoE
pre-training up to 13B parameters, Muon² beats Muon at equal NS steps and matches it with
**40% fewer NS iterations** (N_s=3 vs 5), saving up to ¼ of training time at equal loss. A
memory-efficient variant **Muon²-F** replaces the dense second moment with Adafactor-style
row/column statistics at nearly unchanged memory. This repo already hosts the whole Muon
family (`muon`, `turbo_muon`, `af_muon`, `ada_muon`, `dion3`, `root`); Muon² is the
"precondition *before* orthogonalization" member and slots directly into the existing
Orthogonalized bucket, hybrid routing and shared-suffix machinery.

Decision gate: **PASS.** It is an optimizer method; it does not duplicate any entry in
`factory.OPTIMIZER_REGISTRY` (`ada_muon`, the closest relative, applies its second moment
*after* orthogonalization plus a sign transform and RMS rescale — a different update rule);
equations, algorithm and K defaults are fully specified in the paper (see §2; the only
unspecified numbers are β₁/β₂, handled as a documented assumption in §9).

---

## 2. Paper extract

### 2.1 Update rule (Algorithm 1 — Muon²), verbatim from the paper

For a 2-D weight `W_t ∈ ℝ^{n×m}`, gradient `G_t`, learning rate `η`, momentum coefficients
`β₁, β₂`, NS steps `K`, numerical constant `ε`:

```
M₀ ← 0, V₀ ← 0
for t = 1, 2, …:
  G_t  ← ∇_{W_t} L(W_t)                                                  (Eq. 1)
  M_t  ← β₁ M_{t−1} + (1−β₁) G_t                                         (Eq. 2)
  V_t  ← β₂ V_{t−1} + (1−β₂) (G_t ⊙ G_t)                                 (Eq. 3)
  M̃_t  ← M_t ⊘ (√V_t + ε·1)                                              (Eq. 4)
  O_t  ← NewtonSchulz(M̃_t, K)                                            (Eq. 5)
  W_{t+1} ← W_t − η √(m/n) · O_t                                         (Eq. 6)
```

LaTeX (as recovered from the arXiv HTML `alttext`, exact):

- `\mathbf{M}_{t}\leftarrow\beta_{1}\mathbf{M}_{t-1}+(1-\beta_{1})\mathbf{G}_{t}`
- `\mathbf{V}_{t}\leftarrow\beta_{2}\mathbf{V}_{t-1}+(1-\beta_{2})(\mathbf{G}_{t}\odot\mathbf{G}_{t})`
- `\widetilde{\mathbf{M}}_{t}\leftarrow\mathbf{M}_{t}\oslash(\sqrt{\mathbf{V}_{t}}+\epsilon\mathbf{1})`
- `\mathbf{O}_{t}\leftarrow\mathrm{Newton\text{--}Schulz}(\widetilde{\mathbf{M}}_{t},K)`
- `\mathbf{W}_{t+1}\leftarrow\mathbf{W}_{t}-\eta\sqrt{m/n}\mathbf{O}_{t}`
  (the `√(m/n)` learning-rate factor is credited to Bernstein 2025, "Deriving muon")

Newton–Schulz details (§3.3 of the paper): quintic polynomial
`φ(x) = a x + b x³ + c x⁵` with `(a,b,c) = (3.4445, −4.7750, 2.0315)` — identical to the
coefficients hard-coded in `src/model/optimizer/muon.py:158`. §3.4.1 states the NS input is
Frobenius-normalized "to reflect the actual input of the NS iteration" (as in Muon,
Jordan et al. 2024), i.e. `X₀ = M̃_t / (‖M̃_t‖_F + ε)`.

**No Nesterov term, no bias correction, and no weight decay appear in Algorithm 1** — the
momentum is an Adam-style EMA with the `(1−β)` scaling. Weight decay is only ever mentioned
in reference titles; the repo's decoupled convention for the Muon family is adopted (see §9
assumptions).

### 2.2 Algorithm 2 — Muon²-F (memory-efficient), verbatim

Same Require/Ensure; `r₀ ← 0 ∈ ℝⁿ`, `c₀ ← 0 ∈ ℝᵐ`; then per step:

- `M_t ← β₁ M_{t−1} + (1−β₁) G_t`
- `R_t ← G_t ⊙ G_t`
- `r_t ← β₂ r_{t−1} + (1−β₂) sum_col(R_t)` (row sums ⇒ `r_t ∈ ℝⁿ`)
- `c_t ← β₂ c_{t−1} + (1−β₂) sum_row(R_t)` (column sums ⇒ `c_t ∈ ℝᵐ`)
- `V̂_t ← (r_t c_tᵀ) / sum(r_t)` (Adafactor-style reconstruction)
- `M̃_t ← M_t ⊘ (√V̂_t + ε·1)`, then NS and update identical to Algorithm 1.

Muon²-F "remains close to the exact Muon² in both reduced NS iteration requirements and
model performance, while keeping memory overhead nearly unchanged from Muon" (Table 6:
LLaMA-7B 47.29 GB for Muon vs 47.53 GB for Muon²-F vs 60.66 GB for Muon²).

### 2.3 Hyperparameters

| Name | Type | Default in paper | Range / notes | Scope |
|---|---|---|---|---|
| `η` (lr) | float | swept 0.003–0.04 (GPT), 0.02–0.08 (LLaMA/MoE) | best ≈ 0.02 (GPT-Small/Base), 0.04–0.06 (LLaMA) | per-group (shared `lr_*`) |
| `β₁, β₂` | float pair | **not stated numerically** | momentum coefficients of Eqs. 2–3 | per-group (shared `betas_*`) |
| `K` (NS steps) | int | 5; **3 is the paper's headline setting** (matches Muon at 5) | 3 or 5 in all tables | global |
| `ε` (Eq. 4) | float | not stated (symbolic) | preconditioner epsilon | per-group (shared `eps_*`) |
| NS normalization eps | float | not stated | repo convention `ns_eps=1e-7` | global |

Warmup/schedule: nothing special — standard LR sweeps with the same pipelines (NanoGPT /
Nanotron) as the Muon baseline; Muon² needs no scheduler changes.

### 2.4 State buffers and per-step cost

- **Muon²**: 2 buffers per matrix param (`M`, `V`, each `n×m`) → +1 buffer over Muon;
  fallback params get AdamW's 2 buffers.
- **Muon²-F**: 1 buffer `M` + two vectors (`r ∈ ℝⁿ`, `c ∈ ℝᵐ`) ≈ Muon's memory.
- **Per-step cost**: `O(n·k)` like Muon (NS dominates) plus one element-wise
  multiply/sqrt pass; Muon² at `k=3` vs Muon at `k=5` ⇒ 40% fewer NS matmuls; the paper
  reports up to ¼ total training-time saving at equal loss (Table: GPT-Large 1044 → 1016
  GPU-hours).

### 2.5 Category

Existing bucket **Orthogonalized** (`docs/architecture.md` §8.1, line 740: `muon`,
`turbo_muon`, `af_muon`, `ada_muon`, `dion3`, `root`) — no new bucket needed. In the paper
annex summary table this bucket is "Second-Order, Geometric & Orthogonality"; in
`05-optimizer-families.tex` it is "geometry-oriented methods".

### 2.6 BibTeX data

Authors: Ziyue Liu, Ruijie Zhang, Zhengyang Wang, Yequan Zhao, Yupeng Su, Zi Yang,
Zheng Zhang. Year 2026. Title "Muon²: Boosting Muon via Adaptive Second-Moment
Preconditioning". arXiv:2604.09967 (cs.LG), DOI 10.48550/arXiv.2604.09967. Key:
`liu_muon2_2026` (matches `he_root_2025` / `si_adamuon_2025` / `amsel_dion3_2026` style).
Related in-repo citations: Muon = `shen_convergence_2025`; AdaMuon = `si_adamuon_2025`;
the `√(m/n)` factor cites "Bernstein 2025, Deriving muon" (mention in prose, no entry).

---

## 3. Decisions (Phase 3)

- **Registry name**: `muon2` (snake_case of the paper's short name Muon²).
- **Class name**: `Muon2`; file `src/model/optimizer/muon2.py`.
- **Closest peer**: `src/model/optimizer/muon.py` — mirror the hybrid routing
  (`ADAMW_FALLBACK_GROUPS` + `p.ndim != 2` → `adamw_fallback_update`), the transpose-wide
  bfloat16 NS loop and module/docstring style; borrow the second-moment buffer pattern from
  `ada_muon.py` (`exp_avg_sq` handling) and the factored row/col statistics from Adafactor
  (`src/model/optimizer/adafactor.py`).
- **Category**: existing Orthogonalized bucket everywhere (§2.5).
- **Suffix table** (all global; per-group β/ε come from the shared suffixes):

| Key | Type | Default | Coercion | Scope | Meaning |
|---|---|---|---|---|---|
| `muon2-ns_steps` | int | `5` | `to_int` | global | NS iterations `K` (paper: 3 already matches Muon at 5) |
| `muon2-ns_eps` | float | `1e-7` | `to_float` | global | Frobenius-normalization epsilon before NS |
| `muon2-factored` | bool | `false` | `bool(...)` | global | `true` ⇒ Algorithm 2 (Muon²-F row/col factored second moment) |

  `(β₁, β₂)` of Eqs. 2–3 are the shared per-group `betas_*` (default `(0.9, 0.95)` from
  `factory._build_param_groups_for_optimizer`), and Eq. 4's `ε` is the shared per-group
  `eps_*` (default `1e-8`) — the same values also drive the AdamW fallback, so one knob set
  configures both paths, exactly like `root`. Rationale: the paper defines β₁/β₂ but gives
  no numeric defaults, and the shared `betas` machinery already exists; adding redundant
  `momentum`/`beta2` globals would collide with the fallback's betas. No `nesterov` suffix —
  Algorithm 1 has no Nesterov term (unlike `muon`/`turbo_muon`/`root`).
- **`supports_shared_per_group_suffixes`**: `true` (norm for the registry).
- **Example preset**: `configs/examples/es_arch_hybrid_muon2.yaml` — MLM on the encoder
  with the same hybrid `layer_pattern` as `es_arch_hybrid_ada_muon.yaml` (the established
  shape for Muon-family examples: `es_arch_hybrid_{ada_muon,dion3,eve,root}.yaml`), with
  all shared per-group keys + the three `muon2-*` globals. Smallest valid config in the
  sense used by every peer example: one page, real dataset block copied from the peer,
  passes `load_training_config` and the schema strict gate.
- **Census bump**: **29 → 30** optimizers, read from the repo (do **not** trust any other
  number): `docs/architecture.md:727` `### 8.1 Optimizers (29)`, `docs/architecture.md:86`
  `29 optimizers + registry + parameter routing`, `README.md:27` (mermaid
  `Train<br/>29 optimizer families`) and `README.md:93` `| Optimizer families | 29 across 7
  categories |`, `docs/paper/appendices/annex-summary-tables.tex:100` "all 29 supported
  optimizers", `docs/paper-es/appendices/annex-summary-tables.tex:100` "los 29
  optimizadores soportados", `docs/paper/sections/05-optimizer-families.tex:4` "twenty-nine
  optimizer families", `docs/paper-es/sections/05-familias-optimizadores.tex:4`
  "veintinueve familias de optimizadores". Category counts ("7 categories", "eight
  categories", "ocho categorías") stay unchanged — no new bucket.

Hard rules restated: the schema is the contract (`additionalProperties: false`; every YAML
key must exist in `src/schema/**`); optimizer keys are prefixed `muon2-<group>_<param>` /
`muon2-<param>`; never read an undocumented key; the papers are the theory source of truth
(AGENTS.md D11), so the annex formulations must match arXiv:2604.09967 verbatim and
`paper-es` must translate them faithfully.

---

## 4. File-by-file plan

| # | File | Planned change (concrete drafts in §4.1–§4.19) |
|---|---|---|
| 1 | `src/model/optimizer/muon2.py` | New `Muon2(Optimizer)`; module docstring cites arXiv:2604.09967; hybrid routing as `muon.py`; Eqs. 2–6 mapped line by line; `factored` toggles Algorithm 2 |
| 2 | `src/model/optimizer/__init__.py` | `from .muon2 import Muon2` after line 19 (`from .muon import Muon`); `"Muon2"` in `__all__` after `"Muon"` (line 49) |
| 3 | `src/model/optimizer/factory.py` | Import; `OPTIMIZER_REGISTRY["muon2"] = Muon2` after `"root"` (line 84); `OPTIMIZER_ALLOWED_KEYS["muon2"] = _COMMON_PER_GROUP_KEYS \| {"ns_steps", "ns_eps", "factored"}` after the `root` entry (line 150); dispatch branch before the final `return optimizer_cls(converted_groups)` (line 458) |
| 4 | `src/schema/_optimizer.yaml` | (a) `- muon2` enum entry after `- root` (line 170); (b) one row in **both** use-case tables (EN after line 63, ES after line 118); (c) one bullet in EN+ES "Optimizer-specific suffixes" lists (after lines 203/245); (d) `x_parameter_reference.by_optimizer.muon2` after the `root` block (line 409); (e) append `` `muon2` (paper: arxiv:2604.09967)`` to both References paragraphs (lines 84 and 140) |
| 5 | `src/schema/_training.yaml` | One `if/then` block after the `root` block (line 1595): `const: muon2` → `patternProperties: {^muon2-: …}` |
| 6 | `src/schema/_sbert.yaml` | Extend `optimizer_kwargs` description EN (line 400) + ES (line 401): add `muon2` to the `ns_steps`/`ns_eps` lists and a `factored` entry |
| 7 | `configs/examples/es_arch_hybrid_muon2.yaml` | Full example YAML (draft §4.7) |
| 8 | `configs/README.md` | (a) `- muon2` in the chooser list after `- root` (line 162); (b) suffix bullet after line 213; (c) example key `- muon2-ns_steps: 3` after line 226 |
| 9 | `full_tests/run_e2e.py` | `"muon2"` appended to `OPTIMIZERS` (line 251); specific-globals block in `_with_optimizer` after the `root` block (line 685) |
| 10 | `tests/test_optimizer_factory.py` | `"muon2"` in `test_registry_contains_all_required` (line 30) + `BuildOptimizerMuon2Tests` class (draft §4.10) |
| 11 | `docs/bibliography/optimizers.bib` | `@misc{liu_muon2_2026, …}` after `he_root_2025` (line 381) |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | `\paragraph{Muon² (Second-Moment Preconditioned Orthogonalization).}` after ROOT's Pros (line 1347), before `\subsection{Evolutionary…}` (line 1349): figure + equations + algorithm float + Pros/Cons + `\citep{liu_muon2_2026}` (draft §4.12) |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | Row after ROOT (line 156) in "Second-Order, Geometric & Orthogonality"; census "29"→"30" (line 100) |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: "twenty-nine" → "thirty"; add "Muon²" to "geometry-oriented methods (…)" |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of #12 after ROOT's Pros (mirror position of EN line 1347); figure path `figures/optimizers/muon2.pdf` (same dir as EN) |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | Spanish row after ROOT (line 156); "los 29 optimizadores" → "los 30 optimizadores" (line 100) |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: "veintinueve familias" → "treinta familias"; add "Muon²" to "métodos orientados a geometría (…)" |
| 18 | `docs/architecture.md` + `README.md` + `configs/README.md` | Census pass: `architecture.md:86` (29→30 optimizers), `architecture.md:727` heading `(29)`→`(30)`, `architecture.md:740` Orthogonalized row gains `muon2` + character text; `README.md:27` + `README.md:93` (29→30); `configs/README.md` covered by row 8 |
| 19 | `docs/paper/figures/optimizers/muon2.{mmd,pdf,svg}` | *(optional)* new mermaid figure mirroring `muon.mmd` (draft §4.19); include only if it can be rendered in the same vector style — otherwise omit the `\begin{figure}` block from #12/#15 |

### 4.1 `src/model/optimizer/muon2.py` (new file)

`torch.optim.Optimizer` subclass; `from __future__ import annotations`; Google-style
docstrings; CPU-safe (no `.cuda()`, no device pinning); cite the paper in the module
docstring like `muon.py:1-17`. Constructor signature:

```python
class Muon2(Optimizer):
    def __init__(self, params, lr=0.02, betas=(0.9, 0.95), eps=1e-8,
                 ns_steps=5, ns_eps=1e-7, factored=False, weight_decay=0.0):
```

`defaults = dict(lr=lr, betas=betas, eps=eps, ns_steps=ns_steps, ns_eps=ns_eps,
factored=factored, weight_decay=weight_decay)`. Validate `lr ≥ 0`, `0 ≤ betas < 1`,
`eps ≥ 0`, `ns_steps ≥ 1` (mirror `ada_muon.py:88-95`).

`@torch.no_grad() step(closure=None)` — mirror `muon.py:105-176`:

1. Hybrid routing exactly as `muon.py:122,130`: `use_adamw = group name in
   ADAMW_FALLBACK_GROUPS`; `if use_adamw or p.ndim != 2: adamw_fallback_update(...,
   lr=group["lr"], betas=group["betas"], eps=group["eps"],
   weight_decay=group["weight_decay"]); continue`.
2. State init: `state["exp_avg"]` (M), and `state["exp_avg_sq"]` (V) when `factored=False`,
   else `state["row_stats"]` (r) + `state["col_stats"]` (c).
3. Eq. 2: `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)`.
4. Eq. 3 / Algorithm 2:
   - exact: `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)`, then
     `v_hat = exp_avg_sq`;
   - factored: `r = row_stats.mul_(beta2).add_(grad.pow(2).sum(dim=1), alpha=1-beta2)`,
     `c = col_stats.mul_(beta2).add_(grad.pow(2).sum(dim=0), alpha=1-beta2)`,
     `v_hat = r.outer(c).div_(r.sum().clamp_min(group["eps"]))` (compute in a temp; do not
     store the matrix; the clamp guards the t=1 zero-gradient case).
5. Eq. 4: `m_hat = exp_avg / (v_hat.sqrt() + group["eps"])` (`ε` = shared per-group eps).
6. Decoupled weight decay as `muon.py:155-156`: `p.mul_(1 - lr*wd)` when `wd > 0`.
7. Frobenius normalization + NS: copy `muon.py:159-173` verbatim (bfloat16,
   `x / (x.norm() + ns_eps)`, transpose-to-wide, quintic loop
   `a, b, c = 3.4445, -4.7750, 2.0315`, `ns_steps` iterations, transpose back).
8. Eq. 6: `p.add_(x.to(p.dtype), alpha=-group["lr"] * math.sqrt(m / n))` where
   `n, m = p.shape` read **before** the transpose trick (rows, cols of `W_t ∈ ℝ^{n×m}`) —
   the paper's `η√(m/n)` factor, which the repo's plain `muon` deliberately does **not**
   apply; here it is part of the paper's update rule and must be kept.

### 4.2 `src/model/optimizer/__init__.py`

After line 19 (`from .muon import Muon`) add `from .muon2 import Muon2`; in `__all__`
insert `"Muon2",` after `"Muon",` (line 49).

### 4.3 `src/model/optimizer/factory.py`

- Import block: `from .muon2 import Muon2` after line 43 (`from .muon import Muon`).
- `OPTIMIZER_REGISTRY`: `"muon2": Muon2,` after `"root": ROOT,` (line 84).
- `OPTIMIZER_ALLOWED_KEYS`:
  `"muon2": _COMMON_PER_GROUP_KEYS | {"ns_steps", "ns_eps", "factored"},` after the
  `root` entry (line 150).
- Dispatch in `build_optimizer`, before the final `return optimizer_cls(converted_groups)`
  (line 458):

```python
    if optimizer_name == "muon2":
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        factored = bool(scoped.get("factored", False))
        return optimizer_cls(
            converted_groups,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            factored=factored,
        )
```

  (`betas`/`eps` already flow through `_build_param_groups_for_optimizer`, which applies to
  every optimizer except `sgd_momentum` — factory.py:190-192.)

### 4.4 `src/schema/_optimizer.yaml`

(a) enum (line 170 `- root`): append `- muon2`.

(b) EN use-case table, after the `root` row (line 63):

```
      | `muon2` | Adaptive orthogonal updates | Adam-style second-moment preconditioning of the momentum before Newton-Schulz; 40% fewer NS iterations; optional factored moments (Muon²-F); AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2604.09967) |
```

ES table, after line 118:

```
      | `muon2` | Ortogonalización adaptativa | Precondicionamiento del momentum con segundo momento estilo Adam antes de Newton-Schulz; 40% menos iteraciones NS; momentos factorizados opcionales (Muon²-F); respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2604.09967) |
```

(c) EN "Optimizer-specific suffixes" bullet, after the `root` bullet (line 201):

```
      - `muon2`: `ns_steps`, `ns_eps`, `factored`. Adam-style second-moment preconditioning of the momentum before Newton-Schulz (β₁, β₂ from the shared `betas_*`; the Eq. 4 epsilon is the shared `eps_*`); the update scales by η√(m/n); `factored: true` switches to the memory-efficient row/column-factored second moment (Muon²-F). Same hybrid routing as `muon`: only hidden-layer 2-D matrices take the Muon² path; the `embeddings`, `pos` and `head` groups plus every non-2-D parameter use a decoupled AdamW fallback configured by the shared suffixes
```

ES bullet, after line 243:

```
      - `muon2`: `ns_steps`, `ns_eps`, `factored`. Precondicionamiento del momentum con segundo momento estilo Adam antes de Newton-Schulz (β₁, β₂ de los `betas_*` compartidos; el épsilon de la Ec. 4 es el `eps_*` compartido); la actualización escala por η√(m/n); `factored: true` activa el segundo momento factorizado por filas/columnas (Muon²-F). Mismo enrutamiento híbrido que `muon`: solo las matrices 2-D de capas ocultas toman la ruta Muon²; los grupos `embeddings`, `pos` y `head` más todo parámetro no 2-D usan un respaldo AdamW desacoplado configurado por los sufijos compartidos
```

(d) `x_parameter_reference.by_optimizer`, after the `root` block (line 409):

```yaml
        muon2:
          prefix: "muon2-"
          supports_shared_per_group_suffixes: true
          specific_global_suffixes: [ns_steps, ns_eps, factored]
```

(e) Both `**References:**` paragraphs (EN ends line 84, ES ends line 140): append
`` `muon2` (paper: arxiv:2604.09967)`` after `` `root` (paper: arxiv:2511.20626)``.

### 4.5 `src/schema/_training.yaml`

One block after the `root` block (line 1595), same shape as every peer
(`tests/test_schema_consistency.py::test_optimizer_prefixes_match_conditional_rules`
requires exactly one per optimizer):

```yaml
- if:
    properties:
      optimizer:
        properties:
          optimizer_class:
            const: muon2
  then:
    properties:
      optimizer:
        properties:
          parameters:
            additionalProperties: false
            patternProperties:
              ^muon2-: { type: [number, boolean, array, string] }
```

### 4.6 `src/schema/_sbert.yaml`

`optimizer_kwargs` description (lines 400 EN / 401 ES): in the `ns_steps` list change
"…for muon/turbo_muon/af_muon/ada_muon/dion3/root" → "…/root/muon2" and likewise in the
`ns_eps` list; then insert after the `ns_eps` clause:
EN: "`factored` (Adafactor-style factored second moment for `muon2`)";
ES: "`factored` (segundo momento factorizado estilo Adafactor para `muon2`)".

### 4.7 `configs/examples/es_arch_hybrid_muon2.yaml` (new file)

Clone of `es_arch_hybrid_ada_muon.yaml` with the prefix and optimizer block swapped:

```yaml
model_class: frankenstein_text_encoder
model:
  dims:
    vocab_size: 50000
    hidden_size: 768
    num_layers: 12
    num_loops: 1
    num_heads: 12
    layer_pattern:
    - retnet
    - titan_attn
    - mamba
    - ode
  norm:
    type: layer_norm
  embedding:
    conv:
      enabled: false
  bitnet:
    enabled: false
  ffn:
    hidden_size: 3072
  attention:
    ode:
      solver: rk4
      steps: 2
    retnet:
      heads: 12
training:
  task: mlm
  seed: 42
  num_epochs: 4
  batch_size: 2
  dataloader_workers: 2
  max_length: 256
  mlm_probability: 0.15
  max_samples: 6000000
  dataset_batch_size: 15000
  num_workers: 6
  cache_dir: ./temp_data/v2_dataset_cache
  local_parquet_dir: /home/erickfmm/.cache/huggingface/hub/datasets--erickfmm--red_pajama_es_hq_35/snapshots/bd7286c289a95dc3803c375bc36aaaeb138b1eab/train/
  prefer_local_cache: true
  stream_local_parquet: true
  join_temp_data_context_window: 0
  join_temp_data_min_remainder_tokens: 128
  gradient_accumulation_steps: 12
  optimizer:
    optimizer_class: muon2
    parameters:
      muon2-lr_embeddings: 1e-6
      muon2-lr_pos: 2e-6
      muon2-lr_attention: 3e-6
      muon2-lr_ffn: 2e-6
      muon2-lr_norms: 5e-6
      muon2-lr_head: 2e-6
      muon2-lr_mhc: 3e-6
      muon2-lr_router: 2e-6
      muon2-lr_residual: 2e-6
      muon2-lr_other: 2e-6
      muon2-wd_embeddings: 0.01
      muon2-wd_pos: 0.01
      muon2-wd_attention: 0.01
      muon2-wd_ffn: 0.01
      muon2-wd_norms: 0.001
      muon2-wd_head: 0.01
      muon2-wd_mhc: 0.01
      muon2-wd_router: 0.01
      muon2-wd_residual: 0.01
      muon2-wd_other: 0.01
      muon2-betas_embeddings: [0.9, 0.95]
      muon2-betas_pos: [0.9, 0.95]
      muon2-betas_attention: [0.9, 0.95]
      muon2-betas_ffn: [0.9, 0.95]
      muon2-betas_norms: [0.9, 0.95]
      muon2-betas_head: [0.9, 0.95]
      muon2-betas_mhc: [0.9, 0.95]
      muon2-betas_router: [0.9, 0.95]
      muon2-betas_residual: [0.9, 0.95]
      muon2-betas_other: [0.9, 0.95]
      muon2-eps_embeddings: 1e-8
      muon2-eps_pos: 1e-8
      muon2-eps_attention: 1e-8
      muon2-eps_ffn: 1e-8
      muon2-eps_norms: 1e-8
      muon2-eps_head: 1e-8
      muon2-eps_mhc: 1e-8
      muon2-eps_router: 1e-8
      muon2-eps_residual: 1e-8
      muon2-eps_other: 1e-8
      muon2-ns_steps: 5
      muon2-ns_eps: 1e-7
      muon2-factored: false
  scheduler_total_steps: 12000
  max_nan_retries: 4
  gradient_log_interval: 10
  csv_log_path: training_metrics_es_arch_hybrid_muon2.csv
  telemetry_log_interval: 5
```

### 4.8 `configs/README.md`

- Chooser list: `- muon2` after `- root` (line 162).
- Suffix bullet after line 213:

```
- `muon2`: `ns_steps`, `ns_eps`, `factored` (Adam-style second-moment preconditioning of the momentum before Newton-Schulz; β₁/β₂ come from the shared `betas_*`, the preconditioner epsilon from the shared `eps_*`; update scaled by η√(m/n); same hybrid routing as `muon`; `factored: true` switches to the memory-efficient Muon²-F variant)
```

- Example key after line 226: `` - `muon2-ns_steps: 3` `` (the paper's efficient setting).

### 4.9 `full_tests/run_e2e.py`

- `OPTIMIZERS` (line 246-252): change `"root",` → `"root", "muon2",`.
- `_with_optimizer`, after the `root` block (lines 682-685):

```python
    if optimizer_class == "muon2":
        params[f"{prefix}ns_steps"] = 5
        params[f"{prefix}ns_eps"] = 1e-7
```

### 4.10 `tests/test_optimizer_factory.py`

- Line 30: add `"muon2"` to the `required` set of `test_registry_contains_all_required`.
- New class after `BuildOptimizerROOTTests` (line 294), following its method pattern:

```python
@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerMuon2Tests(unittest.TestCase):
    """Muon2 defaults, paper invariants and hybrid routing (arXiv:2604.09967)."""

    def test_builds_without_error(self):
        opt = build_optimizer("muon2", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("muon2", groups, {})
        p.square().mean().backward()
        opt.step()
        self.assertFalse(torch.isnan(p).any())

    def test_step_runs_factored_flavor(self):
        p = nn.Parameter(torch.randn(4, 8))
        groups = [{"params": [p], "lr": 1e-2, "name": "attention"}]
        opt = build_optimizer("muon2", groups, {"muon2-factored": True})
        p.square().mean().backward()
        opt.step()
        self.assertIn("row_stats", opt.state[p])
        self.assertIn("col_stats", opt.state[p])
        self.assertFalse(torch.isnan(p).any())

    def test_defaults_match_paper(self):
        opt = build_optimizer("muon2", _param_groups(), {})
        self.assertEqual(opt.defaults["ns_steps"], 5)
        self.assertAlmostEqual(opt.defaults["ns_eps"], 1e-7)
        self.assertFalse(opt.defaults["factored"])
        self.assertEqual(opt.defaults["betas"], (0.9, 0.95))

    def test_hidden_matrix_uses_muon2_state(self):
        # Eqs. 2-3: exp_avg (M) and exp_avg_sq (V); no plain momentum_buffer.
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 0.02, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("muon2", groups, {})
        p.sum().backward()
        opt.step()
        state = opt.state[p]
        self.assertIn("exp_avg", state)
        self.assertIn("exp_avg_sq", state)
        self.assertNotIn("momentum_buffer", state)

    def test_embedding_group_uses_adamw_fallback(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "embeddings"}]
        opt = build_optimizer("muon2", groups, {})
        p.sum().backward()
        opt.step()
        self.assertIn("exp_avg", opt.state[p])
        self.assertIn("exp_avg_sq", opt.state[p])

    def test_update_scale_uses_sqrt_m_over_n(self):
        # Eq. 6: the first-step parameter delta equals a manual reimplementation
        # of Eqs. 2-6 (including the eta*sqrt(m/n) factor), wd=0.
        ...

    def test_specific_suffixes_parsed(self):
        opt = build_optimizer(
            "muon2", _param_groups(),
            {"muon2-ns_steps": "3", "muon2-ns_eps": "1e-6", "muon2-factored": True},
        )
        self.assertEqual(opt.defaults["ns_steps"], 3)
        self.assertAlmostEqual(opt.defaults["ns_eps"], 1e-6)
        self.assertTrue(opt.defaults["factored"])

        opt_bad = build_optimizer(
            "muon2", _param_groups(),
            {"muon2-ns_steps": "bad", "muon2-ns_eps": "bad"},
        )
        self.assertEqual(opt_bad.defaults["ns_steps"], 5)
        self.assertAlmostEqual(opt_bad.defaults["ns_eps"], 1e-7)

    def test_unknown_group_suffix_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("muon2", _param_groups(), {"muon2-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))

    def test_per_group_lr_routes(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "ffn"}]
        opt = build_optimizer("muon2", groups, {"muon2-lr_ffn": 0.02})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.02)
```

`tests/test_optimizer_base.py` needs no edits (it tests the shared helpers) — it stays a
gate only.

### 4.11 `docs/bibliography/optimizers.bib`

After `he_root_2025` (line 381):

```bibtex
@misc{liu_muon2_2026,
      title={Muon$^2$: Boosting Muon via Adaptive Second-Moment Preconditioning},
      author={Ziyue Liu and Ruijie Zhang and Zhengyang Wang and Yequan Zhao and Yupeng Su and Zi Yang and Zheng Zhang},
      year={2026},
      eprint={2604.09967},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.09967},
}
```

### 4.12 `docs/paper/appendices/annex-1-optimizers.tex`

Insert after ROOT's Pros block (line 1347), still inside the "Second-Order, Geometric, and
Orthogonality Optimizers" subsection (next is `\subsection{Evolutionary and Population-Based
Hybrids}` at line 1349). Figure block included **only if** the figure from §4.19 exists.

```latex
\paragraph{Muon² (Second-Moment Preconditioned Orthogonalization).}
\begin{figure}[H]
\centering
\fitfigure{\includegraphics{figures/optimizers/muon2.pdf}}
\caption{Step flow of Muon²: dual moment EMAs, element-wise second-moment preconditioning
of the momentum, and Newton--Schulz orthogonalization with the $\sqrt{m/n}$ update scale.}
\label{fig:opt-muon2}
\end{figure}
Muon$^2$~\citep{liu_muon2_2026} generalizes Muon by preconditioning the momentum with an
\textbf{Adam-style second-moment EMA before the Newton--Schulz orthogonalization}. The
paper shows the practical bottleneck of Muon's polar approximation is the ill-conditioned
momentum matrix (early-training spectra span $10^{-4}$--$1$ with roughly half the singular
values in the NS ``dead zone''); dividing by $\sqrt{V_t}$ tightens the spectrum, so every
NS step aligns better with the true orthogonalized update (cosine similarity $0.916$ at
$K{=}3$ vs Muon's $0.931$ at $K{=}5$, rising to $0.975$ at $K{=}5$). For a hidden-layer
weight matrix $W \in \mathbb{R}^{n \times m}$ with learning rate $\eta$:
\begin{align}
M_t &= \beta_1 M_{t-1} + (1-\beta_1) G_t && \text{(momentum EMA, Eq.~2)} \\
V_t &= \beta_2 V_{t-1} + (1-\beta_2)\, (G_t \odot G_t) && \text{(second-moment EMA, Eq.~3)} \\
\widetilde{M}_t &= M_t \oslash \big(\sqrt{V_t} + \epsilon \mathbf{1}\big) && \text{(preconditioning, Eq.~4)} \\
X_0 &= \frac{\widetilde{M}_t}{\|\widetilde{M}_t\|_F + \epsilon}, \qquad
X_{k+1} = aX_k + b\, X_k X_k^\top X_k + c\, X_k (X_k X_k^\top)^2 X_k && \text{(NS, Eq.~5)} \\
W_{t+1} &= W_t - \eta \sqrt{m/n}\, X_K && \text{(update, Eq.~6)}
\end{align}
with the same quintic coefficients $(a,b,c)=(3.4445,-4.7750,2.0315)$ as Muon and the
$\sqrt{m/n}$ learning-rate factor proposed by Bernstein (2025). The memory-efficient
variant \textbf{Muon$^2$-F} replaces $V_t$ with Adafactor-style row/column statistics:
\begin{align}
r_t &= \beta_2 r_{t-1} + (1-\beta_2)\,\operatorname{sum}_{\mathrm{col}}(G_t \odot G_t), \qquad
c_t = \beta_2 c_{t-1} + (1-\beta_2)\,\operatorname{sum}_{\mathrm{row}}(G_t \odot G_t), \\
\widehat{V}_t &= \frac{r_t\, c_t^\top}{\operatorname{sum}(r_t)},
\end{align}
keeping memory within $\sim$0.3\% of Muon at 7B scale (47.53 vs 47.29 GB) while retaining
most of Muon$^2$'s benefit. Across GPT, LLaMA and MoE pre-training up to 13B parameters,
Muon$^2$ outperforms Muon at equal NS iterations and matches it with $K{=}3$ instead of
$5$ (a 40\% reduction), saving up to a quarter of training time at equal loss.

\begin{algorithm}[H]
\caption{Muon² (Second-Moment Preconditioned Orthogonalization)}
\label{alg:muon2}
\begin{algorithmic}[1]
\Require Initial parameters $\theta$, learning rate $\eta$, $\beta_1, \beta_2$, NS iters $K$, preconditioner $\epsilon$
\Ensure Updated parameters $\theta_T$
\For{each parameter $P$}
    \If{$P$ is a hidden-layer 2D matrix $W \in \mathbb{R}^{n\times m}$}
        \State $G \gets \nabla f(W)$ \Comment{Gradient}
        \State $M \gets \beta_1 M + (1-\beta_1) G$ \Comment{Momentum EMA (Eq.~2)}
        \State $V \gets \beta_2 V + (1-\beta_2) (G \odot G)$ \Comment{Second-moment EMA (Eq.~3); factored variant keeps row/col sums only}
        \State $\widetilde{M} \gets M \oslash (\sqrt{V} + \epsilon\mathbf{1})$ \Comment{Precondition (Eq.~4)}
        \State $X_0 \gets \widetilde{M} / (\|\widetilde{M}\|_F + \epsilon)$ \Comment{Frobenius normalization}
        \For{$k = 0, 1, \ldots, K-1$}
            \State $A_k \gets X_k X_k^\top$ \Comment{Gramian}
            \State $X_{k+1} \gets 3.4445\,X_k - 4.7750\,A_k X_k + 2.0315\,A_k^2 X_k$ \Comment{Quintic NS step}
        \EndFor
        \State $W \gets W - \eta \sqrt{m/n}\, X_K$ \Comment{Scaled update (Eq.~6)}
    \Else
        \State $P \gets \operatorname{AdamW}(P, \nabla f(P); \eta, \beta_1, \beta_2, \epsilon, \lambda)$ \Comment{Hybrid fallback: embeddings, pos, heads, non-matrices}
    \EndIf
\EndFor
\Return $\theta_T$
\end{algorithmic}
\end{algorithm}

Following the hybrid recipe of Muon, only hidden-layer 2-D matrices take the Muon$^2$
path; embedding tables, learned positional encodings, output heads and every non-matrix
parameter are trained with decoupled AdamW using the group-level learning rate, weight
decay, $\beta_1, \beta_2$ and $\epsilon$ settings — the same $\beta_1, \beta_2, \epsilon$
also parameterize Eqs.~2--4, so one per-group knob set configures both paths.

\paragraph{Cons.}
\begin{itemize}[leftmargin=*]
\item One extra full-size state buffer over Muon (two moment EMAs per matrix), doubling
optimizer memory on the orthogonalized path unless the factored variant is enabled.
\item The $\sqrt{m/n}$ update scale differs from this repo's plain \texttt{muon} path, so
learning rates do not transfer one-to-one between the two registry entries.
\item $\beta_1, \beta_2$ defaults are not fixed by the paper; this implementation inherits
the shared per-group defaults $(0.9, 0.95)$.
\end{itemize}

\paragraph{Pros.}
\begin{itemize}[leftmargin=*]
\item Better orthogonalization at equal NS iterations and equal quality with 40\% fewer
iterations (3 instead of 5), cutting NS compute and communication.
\item Consistent gains over Muon across GPT, LLaMA and MoE pre-training up to 13B
parameters, with up to a quarter of training time saved at matched loss.
\item The \texttt{factored} toggle exposes Muon$^2$-F, which keeps memory within
$\sim$0.3\% of Muon while preserving most of the benefit.
\end{itemize}
```

### 4.13 `docs/paper/appendices/annex-summary-tables.tex`

Row after ROOT (line 156), plus census line 100 "all 29 supported optimizers" → "all 30
supported optimizers":

```latex
Muon² & Orthogonality + adaptive & 2 ($m$, $v$); factored: 1 ($m$) + row/col; + $v$ on the AdamW fallback & $\mathcal{O}(n \cdot k)$ & lr, $\beta_1$, $\beta_2$, eps, wd, NS steps $k$, factored & \citep{liu_muon2_2026} \\
```

### 4.14 `docs/paper/sections/05-optimizer-families.tex`

Line 4 (quoted verbatim today): "The system supports twenty-nine optimizer families in
eight categories: …geometry-oriented methods (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3,
ROOT)…" → "The system supports **thirty** optimizer families in eight categories:
…geometry-oriented methods (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT,
**Muon²**)…". No other sentence in the file mentions counts.

### 4.15 `docs/paper-es/appendices/annex-1-optimizadores.tex`

Faithful Spanish translation of §4.12, inserted after the ROOT paragraph's Pros (same
structural position as EN line 1347). Paragraph title:
`\paragraph{Muon² (Ortogonalización con Precondicionamiento de Segundo Momento).}`;
figure path `figures/optimizers/muon2.pdf` (figures are shared from
`docs/paper/figures/optimizers/`); algorithm caption "Muon² (Ortogonalización con
Precondicionamiento de Segundo Momento)"; equations kept in LaTeX exactly as in EN
(D11: `paper-es` translates prose, not math); "Pros./Contras" translated; hybrid-recipe
paragraph translated: "Siguiendo la receta híbrida de Muon, solo las matrices 2-D de
capas ocultas toman la ruta Muon²; las tablas de embeddings, los encoding posicionales
aprendidos, las cabezas de salida y todo parámetro no matricial se entrenan con AdamW
desacoplado usando el learning rate, weight decay, $\beta_1, \beta_2$ y $\epsilon$ del
grupo — los mismos $\beta_1, \beta_2, \epsilon$ que parametrizan las Ecs. 2–4."

### 4.16 `docs/paper-es/appendices/annex-summary-tables.tex`

Row after ROOT (line 156); census line 100 "los 29 optimizadores soportados" → "los 30
optimizadores soportados":

```latex
Muon² & Ortogonalidad + adaptativa & 2 ($m$, $v$); factorizado: 1 ($m$) + fila/col; + $v$ en el respaldo AdamW & $\mathcal{O}(n \cdot k)$ & lr, $\beta_1$, $\beta_2$, eps, wd, pasos NS $k$, factored & \citep{liu_muon2_2026} \\
```

### 4.17 `docs/paper-es/sections/05-familias-optimizadores.tex`

Line 4: "El sistema soporta veintinueve familias de optimizadores en ocho categorías:
…métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT)…" →
"El sistema soporta **treinta** familias … (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3,
ROOT, **Muon²**)…".

### 4.18 `docs/architecture.md` + `README.md`

- `docs/architecture.md:86`: `| src/model/optimizer/** | 29 optimizers + registry +
  parameter routing |` → `30 optimizers + …`.
- `docs/architecture.md:727`: `### 8.1 Optimizers (29)` → `### 8.1 Optimizers (30)`.
- `docs/architecture.md:740`: Orthogonalized family row — add `muon2` to the class list
  and extend the character cell: "…; `muon2` preconditions the momentum with an Adam-style
  second-moment EMA before orthogonalization (β₁/β₂ from the shared per-group `betas`,
  `η√(m/n)` update scale, `factored` toggles the Muon²-F row/col moments); …".
- `README.md:27` (mermaid): `Train<br/>29 optimizer families` → `Train<br/>30 optimizer
  families`.
- `README.md:93`: `| Optimizer families | 29 across 7 categories |` → `| Optimizer
  families | 30 across 7 categories |` (Muon² joins an existing category).

### 4.19 `docs/paper/figures/optimizers/muon2.{mmd,pdf,svg}` (optional)

New mermaid source mirroring `muon.mmd` (9 lines, same init header/theme):

```
%%{init: {"theme":"neutral","flowchart":{"nodeSpacing":22,"rankSpacing":26,"wrappingWidth":260}}}%%
flowchart TD
  G["G = ∇f(W)"]
  moments["M = β₁M + (1−β₁)G<br/>V = β₂V + (1−β₂)G⊙G"]
  pre["M̃ = M ⊘ (√V + ε)"]
  norm["X₀ = M̃/(‖M̃‖_F + ε)"]
  ns["repeat k = 0…K−1:<br/>A = XₖXₖᵀ<br/>B = 1.5A − 0.5A²<br/>Xₖ₊₁ = BXₖ"]
  orth["X_K ≈ orthogonal direction"]
  upd["W ← W − η√(m/n)·X_K"]

  G --> moments --> pre --> norm --> ns --> orth --> upd
```

Render to `.pdf` + `.svg` exactly as the sibling figures were produced (the repo keeps
`.mmd` sources; use the same mermaid tooling/versions). If the same vector style cannot be
reproduced at implementation time, **omit the figure and the `\begin{figure}` block** from
§4.12/§4.15 — the figure is optional per the recipe.

---

## 5. Ordered execution checklist (at implementation time — do NOT run now)

Implementation (files 1–9):

1. [ ] `src/model/optimizer/muon2.py` (§4.1)
2. [ ] `src/model/optimizer/__init__.py` (§4.2)
3. [ ] `src/model/optimizer/factory.py` (§4.3)
4. [ ] `src/schema/_optimizer.yaml` (§4.4)
5. [ ] `src/schema/_training.yaml` (§4.5)
6. [ ] `src/schema/_sbert.yaml` (§4.6)
7. [ ] `configs/examples/es_arch_hybrid_muon2.yaml` (§4.7)
8. [ ] `configs/README.md` (§4.8)
9. [ ] `full_tests/run_e2e.py` (§4.9)

Tests (file 10):

10. [ ] `tests/test_optimizer_factory.py` (§4.10), then targeted gates:

```bash
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings
```

11. [ ] Full suite:

```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

12. [ ] Fast e2e smoke (raise `--limit` if the new optimizer is not reached):

```bash
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3
```

Docs (files 11–18 + optional figure):

13. [ ] `docs/bibliography/optimizers.bib` (§4.11)
14. [ ] EN annex paragraph (§4.12) + summary row & census (§4.13) + family sentence (§4.14)
15. [ ] ES annex translation (§4.15) + summary row & census (§4.16) + family sentence (§4.17)
16. [ ] Census pass `architecture.md` + `README.md` (§4.18); optional figure (§4.19)

PDF rebuild (only at implementation time; `pdflatex`/`bibtex`/`pandoc` are all present at
`/usr/bin` on this machine — verified):

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

PDF verification: no `Fatal error` in the logs, no `Citation .* undefined` for
`liu_muon2_2026`, `paper.pdf` / `paper-es.pdf` newer than the edited `.tex`/`.bbl`,
`git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf` changes (never stage `*.aux *.log *.out *.toc *.blg`). Fallback if
TeX is unavailable: leave the sources updated and note that the rebuild must happen on a
TeX machine before the next release (the website sync copies prebuilt PDFs only).

Website sync + commit: see §6 and §7.

## 6. Website sync (planned, not executed)

Run `/sync-website` (`.opencode/command/sync-website.md`; state probe:
`.opencode/command/sync-website-state.sh`, mapping verified lines 106–133). Steps:

1. Copy into `erickfmm.github.io/frankenstein-transformer/`:
   - `src/schema.yaml` → `schema.yaml`; `src/schema/**` → `schema/**` (includes the edited
     `_optimizer.yaml`, `_training.yaml`, `_sbert.yaml`);
   - `configs/examples/**` → `examples/examples/**` (new
     `es_arch_hybrid_muon2.yaml`); `configs/README.md` → `examples/README.md`;
   - `docs/paper/**` → `paper/en/**` (`paper.tex`, `paper.bbl`, `paper.pdf`,
     `sections/*.tex`, `appendices/*.tex`; figures: only the new
     `figures/optimizers/muon2.svg` syncs — PDFs stay main-repo-only, state script line 129);
   - `docs/paper-es/**` → `paper/es/**` (`paper-es.tex`, `.bbl`, `.pdf`,
     `sections/*.tex`, `appendices/*.tex`);
   - `docs/bibliography/**` → `paper/bibliography/**` (updated `optimizers.bib`);
   - `docs/architecture.md` → `paper/specs/architecture.md`.
2. Rebuild the Paper-tab HTML with `flatten_tex.py` + `pandoc` (`--web` flavor, as the
   existing `paper/paper-en.html` / `paper/paper-es.html` were produced) so the new annex
   paragraph and SVG figure appear; refresh the top-level copies `paper/en.pdf` and
   `paper/es.pdf` from the freshly compiled PDFs (probes `paper-en-copy`/`paper-es-copy`
   compare them byte-for-byte).
3. Validation Gate:
   - `uv run --extra cpu --extra train python full_tests/param_count_check.py` →
     `0 failures` (regenerates `full_tests/param_truth.json`; a diff there is intentional);
   - `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` → PASS;
   - `bash .opencode/command/sync-website-state.sh` → probes `schema-refs`,
     `example-yaml`, `cli-flags` all PASS and `estimator-drift` PASS; drift list shows
     only the expected in-sync files.
4. The mirror stays uncommitted until the commit phase (§7).

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions: English, conventional tag, structured body; never
stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json` changes are intentional.

1. Main repo — `feat(optimizer): add Muon² (arXiv:2604.09967)`. Stage exactly:

```text
src/model/optimizer/muon2.py
src/model/optimizer/__init__.py
src/model/optimizer/factory.py
src/schema/_optimizer.yaml
src/schema/_training.yaml
src/schema/_sbert.yaml
configs/examples/es_arch_hybrid_muon2.yaml
configs/README.md
full_tests/run_e2e.py
tests/test_optimizer_factory.py
docs/bibliography/optimizers.bib
docs/paper/appendices/annex-1-optimizers.tex
docs/paper/appendices/annex-summary-tables.tex
docs/paper/sections/05-optimizer-families.tex
docs/paper/paper.bbl
docs/paper/paper.pdf
docs/paper-es/appendices/annex-1-optimizadores.tex
docs/paper-es/appendices/annex-summary-tables.tex
docs/paper-es/sections/05-familias-optimizadores.tex
docs/paper-es/paper-es.bbl
docs/paper-es/paper-es.pdf
docs/architecture.md
README.md
docs/paper/figures/optimizers/muon2.mmd        (only if §4.19 produced)
docs/paper/figures/optimizers/muon2.pdf        (only if §4.19 produced)
docs/paper/figures/optimizers/muon2.svg        (only if §4.19 produced)
```

   Body covers: implementation notes (Eqs. 2–6 mapping, `factored` = Muon²-F, hybrid
   routing, `η√(m/n)` scale), new suffixes (`ns_steps`, `ns_eps`, `factored`), schema
   (enum, both use-case tables, suffix lists, `by_optimizer`, `if/then`, SBERT kwargs),
   docs (EN/ES annexes, summary tables, family sections, bibliography, census 29→30,
   rebuilt PDFs), example preset, tests. Then `git push origin HEAD:main`.

2. Mirror — `git -C erickfmm.github.io …` commit
   `docs(site): sync Muon² optimizer (arXiv:2604.09967)` staging
   `frankenstein-transformer/schema.yaml`, `frankenstein-transformer/schema/_optimizer.yaml`,
   `frankenstein-transformer/schema/_training.yaml`, `frankenstein-transformer/schema/_sbert.yaml`,
   `frankenstein-transformer/examples/examples/es_arch_hybrid_muon2.yaml`,
   `frankenstein-transformer/examples/README.md`,
   `frankenstein-transformer/paper/en/**` (tex/bbl/pdf + annex/section tex + figure svg),
   `frankenstein-transformer/paper/es/**`,
   `frankenstein-transformer/paper/bibliography/optimizers.bib`,
   `frankenstein-transformer/paper/specs/architecture.md`,
   `frankenstein-transformer/paper/en.pdf`, `frankenstein-transformer/paper/es.pdf`,
   `frankenstein-transformer/paper/paper-en.html`, `frankenstein-transformer/paper/paper-es.html`;
   then `git -C erickfmm.github.io push origin HEAD:main`.

## 8. Verification matrix

| Gate | Command | Expected result |
|---|---|---|
| Targeted pytest | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings` | All pass; `BuildOptimizerMuon2Tests` collected; `test_optimizer_prefixes_match_conditional_rules` still sees exactly one `if/then` per prefix (now including `muon2`); `test_registry_contains_all_required` passes with `muon2` |
| Full suite | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | No new failures vs baseline |
| e2e smoke | `uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3` | `muon2` combo builds + trains 1 step on CPU; no `ValueError: Unknown parameters` |
| PDF EN/ES | `pdflatex`/`bibtex` sequence above | No `Fatal error`, no `Citation 'liu_muon2_2026' undefined`, PDFs newer than sources |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `0 failures` (param counts unaffected — optimizer adds no parameters) |
| Estimator | `node full_tests/param_estimate_check.mjs <estimator> full_tests/param_truth.json --tol 0.01` | PASS within 1% |
| State probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, `estimator-drift`, `paper-en-copy`, `paper-es-copy` all PASS |

## 9. Risks, assumptions and open questions

Assumptions to re-verify at implementation time:

1. **β₁/β₂ defaults**: the paper never states numeric values (only the LR sweeps). Plan:
   shared per-group `betas_*` default `(0.9, 0.95)` (factory default), driving both the
   Muon² EMAs and the AdamW fallback. If the authors release configs with different
   values, only the docs/examples change — not the contract.
2. **Weight decay**: absent from Algorithm 1; the repo convention for the Muon family
   (decoupled `W ← (1−ηλ)W` on the matrix path, AdamW-style decay in the fallback) is
   adopted, mirroring `muon.py:155-156`.
3. **Frobenius normalization before NS**: not drawn in Algorithm 1, but §3.4.1 says the NS
   input is Frobenius-normalized "to reflect the actual input of the NS iteration"
   (following Jordan et al. 2024). Plan normalizes `M̃_t` by `‖M̃_t‖_F + ns_eps`, matching
   the repo's Muon.
4. **ε mapping**: Eq. 4's `ε` → shared per-group `eps_*` (1e-8); the NS normalization
   epsilon → `ns_eps` (1e-7), following the `muon`/`ada_muon` precedent.
5. **`√(m/n)` orientation**: `W ∈ ℝ^{n×m}` (rows n, cols m) and the update multiplies by
   `√(m/n)`; the implementation must read `n, m = p.shape` before the transpose-to-wide
   trick so the scale is independent of the internal transposition.
6. **Figure tooling**: the `.mmd → .pdf/.svg` pipeline for existing figures is not
   scripted in the repo; reproduce with the same mermaid version or omit the figure.
7. **Naming collision**: `muon2` does not collide with any registry key, YAML prefix,
   group name or file (grep-verified against `OPTIMIZER_REGISTRY`,
   `_optimizer.yaml`, `_training.yaml`, `GROUP_NAMES`).

Risks:

- The `η√(m/n)` scale makes `muon2` LRs non-interchangeable with `muon` LRs (documented in
  Cons and in `configs/README.md`); e2e uses small LRs so no stability issue is expected.
- Factored path divides by `sum(r_t)`; clamp the denominator (`r.sum().clamp_min(eps)`)
  to avoid division by zero at t=1 with zero gradients.
- No task/mixer incompatibilities: `muon2` is task-agnostic and, unlike `eve`/`ea4llm`,
  needs neither `gradient_accumulation_steps: 1` nor closure-based stepping; SBERT
  compatibility follows the other Muon-family entries (allowed via `optimizer_kwargs`).

Open questions (defaults chosen, no user input required unless contested):

- **Muon²-F as a toggle vs a second registry entry** (`muon2_f`): the plan uses one entry
  with `factored` because the paper presents Muon²-F as a variant and this keeps the
  census at +1; a separate entry would additionally need its own enum/if-then/example rows.
- The paper also benchmarks PolarExpress coefficients `(2, −1.5, 0.5)` (appendix C) as an
  NS alternative — **out of scope**; this plan implements the paper's main algorithm with
  the classic quintic coefficients.

## 10. Out of scope

This command produced **only** this plan file. The plan-mode runtime denied the write to
the repo-local `.opencode/plan/add-optimizer-2604.09967.md`, so the plan was saved to the
runtime's plan directory (`~/.opencode/plan/`) instead; copy it into the repo when
implementation begins. No code, schema, preset, test, doc, PDF, figure, website-mirror or
git change was made; no build, test, training, LaTeX or sync command was executed.
