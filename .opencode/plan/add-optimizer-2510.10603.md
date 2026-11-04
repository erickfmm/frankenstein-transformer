# Plan: add EA4LLM optimizer (arXiv 2510.10603)

> Canonical location: `.opencode/plan/add-optimizer-2510.10603.md`.
> Status: PLAN ONLY — nothing outside the plan directory has been created, edited or deleted.
> Recipe: `docs/architecture.md` §11.2 + the **Optimizer** row of `AGENTS.md`, extended with
> paper reading, docs/PDF, e2e, website sync and git phases.

## 1. Header

**Paper:** *EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via
Evolutionary Algorithms* — WenTao Liu, Siyu Song, Hao Hao, Aimin Zhou (East China Normal
University / Shanghai Innovation Institute), ICML 2025 (PMLR 267), arXiv:2510.10603.

Gradient-based pretraining of LLMs costs 3–8× the parameter count in GPU memory
(parameters + gradients + optimizer states) and requires every operation to be
differentiable. EA4LLM replaces backpropagation with a **population-based evolution
strategies (ES)** estimator in the OpenAI-ES lineage (Salimans et al. 2017): each
iteration perturbs the parameters with N scaled Gaussian noises, scores each perturbed
point on the **token-level next-token log-likelihood fitness** (log-softmax + gather,
the causal-LM objective rewritten as an ES fitness), converts the N scores into
normalized weights via z-score normalization or rank shaping, and moves the parameters
along the noise-weighted average direction. The paper demonstrates the first stable
**full-parameter pretraining from scratch** with ES across Qwen3-style models from 0.5B
to 32B, with >4× lower GPU memory than Adam at ≥4B and **zero optimizer state** (no
moment buffers at all).

**Why it belongs in this repo:** it slots into the existing **Evolutionary** category
alongside `eve` (which is differential evolution *with* gradient rescue and 12× state);
EA4LLM is its polar opposite — purely zeroth-order, O(1) optimizer state — and it
reuses the closure-based trainer path built for `eve` unchanged
(`requires_closure`, `trainer.py:2053-2065`).

**Decision gate verdict: PROCEED.** It is a complete optimizer method (Algorithm 1 +
equations + defaults); it does not duplicate any entry in
`factory.OPTIMIZER_REGISTRY` (`src/model/optimizer/factory.py:53-81` — no ES method
exists; `eve` is steady-state DE with an Adam fallback, a different update rule,
different state (12× vs 0), and different mechanics).

## 2. Paper extract

### 2.1 Fitness: linking logits to ES fitness (§3.1)

Given tokenized sequence `x = (x_1, …, x_T)`, vocabulary `V`, logits
`z_t ∈ R^{|V|}` and binary padding mask `m_t ∈ {0,1}`:

```latex
p_t = \operatorname{softmax}(z_t)
R(x;\theta) = \frac{1}{\sum_t m_t}\sum_{t=1}^{T} m_t\,
              \log p_\theta\!\left(x_{t+1} \mid x_{\le t}\right)
F(\theta) = \frac{1}{|D|}\sum_{x \in D} R(x;\theta)
```

"the per-sample reward (fitness contribution) is the mean log-probability of the true
next token" (§3.1). Subsampled evaluation (§3.2): with `D_t ⊂ D` drawn uniformly
without replacement, `|D_t| = M`, all members of one iteration are scored on the *same*
subset ("the subset indices are broadcast across processes so all workers evaluate the
same `D_t`"), which is unbiased: `E[F_M(θ)] = F(θ)`.

> **Repo mapping:** the fitness is the negated training loss. The trainer's closure
> (`trainer.py:1808-1812`) returns the current-batch loss as a float, so
> `r_j = -closure(θ + σ·ε_j)`; the "subsample" is whatever minibatch the trainer drew —
> the optimizer must NOT touch data.

### 2.2 Perturbation, weights, update (§3.2, Algorithm 1)

```latex
\epsilon_j \sim \mathcal{N}(0, I), \quad j = 1,\dots,N; \qquad
\theta'_j = \theta + \sigma\,\epsilon_j; \qquad r_j = F(\theta'_j)
```

Weights — either **z-score normalization** (default) or **rank-based shaping**:

```latex
w_j = \frac{r_j - \bar r}{\operatorname{std}(r)}
\qquad\text{or}\qquad
w_j = \operatorname{rank}(r_j) - \frac{N-1}{2}
```

"where rank(·) assigns ranks 0, …, N−1 after sorting" (§3.2). The centered form
`rank − (N−1)/2` follows the paper's stated baseline (Salimans et al. 2017), whose
fitness-shaping centers the ranks; the exact centered expression sits in a figure the
arXiv conversion omitted — **assumption to verify at implementation time** (see §9).

Direction and update — standard scaling or NES-style `1/σ` scaling:

```latex
\hat g = \frac{1}{N}\sum_{j=1}^{N} w_j\,\epsilon_j
\quad\text{(or } \hat g = \frac{1}{N\sigma}\sum_{j=1}^{N} w_j\,\epsilon_j\text{)}
\qquad
\theta \leftarrow \theta + \alpha\,\hat g
```

**Antithetic sampling** (§3.2/§1: "combined with antithetic sampling, rank shaping,
grouped/layered controls…"): draw `ε` for `j = 1..N/2` and evaluate the ± pair
`θ ± σε`; z-score/rank shaping runs over all N rewards.

**Algorithm 1 (EA4LLM: Zeroth-Order Pretraining via ES)** — inputs: `θ_0`,
token-level fitness `R`, iterations `T`, population `N`, noise scale `σ`, learning
rate `α`, subset size `M`, normalization (z-score or rank):

```
for t = 1..T:
    sample uniform subset D_t, |D_t| = M; broadcast indices across processes
    draw ε_j ~ N(0, I), j = 1..N
    for j = 1..N:
        θ_tmp = θ + σ ε_j
        r_j = (1/M) Σ_{x∈D_t} R(x; θ_tmp)
    compute w_j from {r_j} by z-score normalization or rank shaping
    ĝ = (1/N) Σ_j w_j ε_j
    θ ← θ + α ĝ
return θ_T
```

### 2.3 Hyperparameters (paper §4.1 "Global Hyperparameters and Setup")

| Name | Symbol | Type | Default (paper) | Range / notes | Scope |
|---|---|---|---|---|---|
| population | `N` (also `P`) | int | **30** | ablated 20/30/40; must be ≥2 (even if antithetic) | global |
| sigma | `σ` | float | **1e-3** | noise scale; >0 | global |
| lr | `α` | float | **5e-4** | step size; maps to the shared per-group `lr_*` | per-group |
| antithetic | — | bool | **false** | "antithetic off" in the baseline toggles | global |
| rank_shaping | — | bool | **false** | "rank shaping off"; false → z-score normalization | global |
| nes_scaling | — | bool | **false** | "1/σ scaling off"; true → `1/(Nσ)` | global |
| seed | — | int/None | None (repo) | paper: "seeds are synchronized across processes" | global |
| weight decay | — | float | 0 (repo) | **paper defines none**; repo extension, see §3 | per-group (`wd_*`) |
| eval subset `M` | `M` | int | 5000 samples/iter | **not an optimizer knob** — the trainer's batch machinery | n/a (trainer) |

**Explicitly excluded** (Appendix A.1: "This design *removes* antithetic sampling
[as default], grouped/layered operators, and step-size multipliers, matching the actual
algorithm we employ"): grouped/layered multipliers, α/σ decay schedules
(`γ_α, γ_σ, S_α, S_σ` exist only as disabled toggles in the paper's runner and their
formula is not published). Do **not** invent suffixes for them.

### 2.4 State buffers and per-step cost

- **Optimizer state: none per parameter.** No moments, no population storage (unlike
  `eve`'s 12×). Persistent state = one step counter + a `torch.Generator` (CPU, plus
  lazily one per device — mirroring `eve.py:151-156,226-228`) + per-member integer
  seeds for the two-pass replay (O(N) *scalars*, transient).
- **Per-step cost:** `N` closure evaluations (forward + backward via the current
  closure; the backward is wasted but harmless — see §9) + `2N` noise generations
  (two-pass) + `O(n)` element-wise adds per member. No gradients are consumed.
- **Memory:** parameters + activations only; the paper's Figure 2 reports e.g.
  60.01 GB vs Adam's 137.61 GB at 8B and 137.61 vs 431.76 GB at 32B (>4× at ≥4B).

### 2.5 Category (docs/architecture.md §8.1)

Existing bucket **Evolutionary** (§8.1 last table row; annex subsection "Evolutionary
and Population-Based Hybrids"; summary-table header "Population-Based & Evolutionary").
No new category. Category count stays as each file states it (8 in the papers, 7 in
README — pre-existing inconsistency, out of scope).

### 2.6 BibTeX data

Authors: WenTao Liu, Siyu Song, Hao Hao, Aimin Zhou. Year: 2025. Title: "EA4LLM: A
Gradient-Free Approach to Large Language Model Optimization via Evolutionary
Algorithms". arXiv 2510.10603, cs.LG. Venue: ICML 2025 (PMLR 267).

## 3. Decisions

| Decision | Value |
|---|---|
| Registry name | `ea4llm` (snake_case of the paper's short name) |
| Class | `EA4LLM` (`src/model/optimizer/ea4llm.py`) |
| Closest peer | `src/model/optimizer/eve.py` — mirror: `requires_closure = True` class marker, `step(closure)` raising `RuntimeError` when the closure is missing (eve.py:447-453 message style), seeded `_cpu_gen`/`_dev_gen` generators (eve.py:151-156), `_evaluate(closure)` helper (eve.py:257-272), defaults dict + range validation, module docstring citing the paper, Google-style docstrings, CPU-safe (no `.cuda()`), `from __future__ import annotations`. **Not** mirrored: population slots, box, DE child, Adam fallback (EA4LLM has none). |
| Category | Evolutionary (existing) |
| `supports_shared_per_group_suffixes` | `true` (the norm) |
| Example preset | `configs/examples/es_arch_hybrid_ea4llm.yaml` — clone of the eve example (the only proven-valid closure-path example: encoder, `task: mlm`, `gradient_accumulation_steps: 1`), optimizer block swapped to `ea4llm-` keys. Named like `es_arch_hybrid_turbo_muon.yaml` / `es_arch_hybrid_eve.yaml`. |
| Census bump | **27 → 28** (actual old count read from the repo — see §4 row 19 for every occurrence) |

**Suffix table** (all `ea4llm-<suffix>` global keys; per-group shared keys unchanged):

| Key | Type | Default | Coercion (factory) | EN description (draft) | ES description (draft) |
|---|---|---|---|---|---|
| `ea4llm-population` | int | 30 | `to_int` | ES population size N (paper: 30) | Tamaño de población N (paper: 30) |
| `ea4llm-sigma` | float | 1e-3 | `to_float` | Gaussian noise scale σ | Escala de ruido gaussiano σ |
| `ea4llm-antithetic` | bool | false | `bool(...)` | Evaluate ±σε pairs (requires even population) | Evalúa pares ±σε (requiere población par) |
| `ea4llm-rank_shaping` | bool | false | `bool(...)` | Rank-based fitness shaping instead of z-score | Shaping por rangos en vez de z-score |
| `ea4llm-nes_scaling` | bool | false | `bool(...)` | NES-style 1/σ scaling of the ES direction | Escalado NES 1/σ de la dirección ES |
| `ea4llm-seed` | int | None | `to_int` (None passthrough) | RNG seed; None draws from the global torch RNG | Semilla RNG; None toma el RNG global |

Shared-suffix semantics for EA4LLM (to document): `lr_*` = per-group step size α
(default from the paper: 5e-4); `wd_*` = decoupled multiplicative decay
`θ ← θ(1 − α λ)` applied before the ES move — **a repo-consistency extension, the
paper defines no weight decay** (default 0.0); `betas_*`/`eps_*` are accepted for
schema parity but unused (ES keeps no moments; precedent: `sgd_momentum` accepts and
ignores them, `factory.py:100,182-184`).

**Design: O(1)-memory two-pass step.** Storing all N noise tensors would cost N×
parameters and forfeit the paper's memory claim. Instead: (1) draw one integer seed per
member from the persistent generator; (2) pass 1 — for each member, rebuild `ε_j` from
`seed_j` via a fresh `torch.Generator`, perturb `p.data += σ ε_j`, evaluate
`r_j = -closure()`, restore `p.data -= σ ε_j`; (3) compute weights; (4) pass 2 —
rebuild `ε_j` identically from `seed_j` and accumulate `g += w_j ε_j` into one transient
`zeros_like(p)` buffer per parameter, then apply
`p.data += (α_g / (N·σ if nes_scaling else N)) · g`. Non-finite rewards get weight 0.
With `antithetic`, draw `N/2` seeds and evaluate both signs; z-score/rank runs over all
N rewards.

## 4. File-by-file plan table

Hard rules restated: **the schema is the contract** (`additionalProperties: false`
everywhere); optimizer keys are prefixed `<name>-<param>` / `<name>-<group>_<param>`;
never read an undocumented key; **the papers are the theory source of truth (D11)** —
annex formulations must match arXiv 2510.10603 and `paper-es` must translate them
faithfully.

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/ea4llm.py` | New module. Docstring: `"""EA4LLM: gradient-free evolution strategies for LLM optimization.` citing Liu et al. 2025, arXiv:2510.10603, one-paragraph mechanism summary. `from __future__ import annotations`; imports `math, typing.Any/Dict/List/Optional, torch, Optimizer`. Class `EA4LLM(Optimizer)` with `requires_closure = True`. `__init__(self, params, lr=5e-4, population=30, sigma=1e-3, antithetic=False, rank_shaping=False, nes_scaling=False, weight_decay=0.0, seed=None)`; validation: `population >= 2`, `population % 2 == 0` when `antithetic`, `sigma > 0`, `lr >= 0`, `weight_decay >= 0` (ValueError, eve.py:126-135 style); `defaults = dict(lr, betas=(0.9,0.95), eps=1e-8, weight_decay, population, sigma, antithetic, rank_shaping, nes_scaling)`; seed handling mirroring eve.py:151-156 (`torch.Generator(device="cpu")`, lazy device generator). Helpers: `_randn_from_seed(seed, like)` (shape/dtype/device-matched standard normal from a fresh seeded generator), `_evaluate(closure)` (eve.py:257-272), `_weights(rewards, rank_shaping)` (z-score with zero-std guard → all-zero weights; centered ranks `rank − (N−1)/2`; non-finite → weight 0). `@torch.no_grad() step(closure=None)`: raise `RuntimeError` when closure is None (eve.py:447-453 message, adapted); collect params (skip empty); increment first-param state counter; per group apply decoupled decay `p.mul_(1 − lr_g·wd_g)` when `wd_g > 0`; two-pass seeded-replay step from §3; return mean reward as float. No `.cuda()`, no sparse-grad requirement (grads never read). |
| 2 | `src/model/optimizer/__init__.py` | Insert `from .ea4llm import EA4LLM` between `from .dion3 import Dion3` (line 11) and `from .eve import Eve` (line 12); add `"EA4LLM",` to `__all__` between `"Dion3",` (line 40) and `"Eve",` (line 41). |
| 3 | `src/model/optimizer/factory.py` | Import `from .ea4llm import EA4LLM` between `.dion3` (line 36) and `.eve` (line 37). Registry: `"ea4llm": EA4LLM,` after `"eve": Eve,` (line 80). Allowed keys: `"ea4llm": _COMMON_PER_GROUP_KEYS \| {"population", "sigma", "antithetic", "rank_shaping", "nes_scaling", "seed"},` after the `eve` entry (line 142). Dispatch after the eve block (line 411): `population = to_int(scoped.get("population"), 30)`, `sigma = to_float(scoped.get("sigma"), 1e-3)`, `antithetic = bool(scoped.get("antithetic", False))`, `rank_shaping = bool(scoped.get("rank_shaping", False))`, `nes_scaling = bool(scoped.get("nes_scaling", False))`, seed passthrough mirroring the eve `seed` handling (factory.py:400-401); `return optimizer_cls(converted_groups, population=…, sigma=…, antithetic=…, rank_shaping=…, nes_scaling=…, seed=…)`. |
| 4 | `src/training/config_loader.py` | Extend the three closure guards from `== "eve"` to membership in `{"eve", "ea4llm"}`: main task-sbert rejection (lines 656-662), `gradient_accumulation_steps == 1` requirement (lines 663-676), SBERT-block rejection (lines 682-690) — update messages to name both optimizers ("EvE/EA4LLM"). Reason: the trainer path in `trainer.py:2053-2065` and the SBERT loop restriction are shared by all `requires_closure` optimizers. |
| 5 | `src/schema/_optimizer.yaml` | (a) `optimizer_class.enum`: append `- ea4llm` after `- eve` (line 162). (b) EN use-case table, after the `eve` row (line 61): ``| `ea4llm` | Gradient-free training, memory-constrained regimes | Zeroth-order evolution strategies on the token-level log-likelihood fitness; Gaussian perturbations with z-score/rank shaping; no optimizer state, N forward evaluations per step (paper: arxiv:2510.10603) |``. (c) ES table after line 113: ``| `ea4llm` | Entrenamiento sin gradientes, regímenes con memoria limitada | Estrategias evolutivas de orden cero sobre la aptitud de log-verosimilitud por token; perturbaciones gaussianas con z-score/shaping por rangos; sin estado del optimizador, N evaluaciones forward por paso (paper: arxiv:2510.10603) |``. (d) EN references list (line 81): append `; `ea4llm` (paper: arxiv:2510.10603)`; same for ES (line 134). (e) EN "Optimizer-specific suffixes" list after the eve bullet (line 193): `- `ea4llm`: `population`, `sigma`, `antithetic`, `rank_shaping`, `nes_scaling`, `seed`. Gradient-free evolution strategies (zeroth-order): the shared `lr_*` is the per-group step size α, `wd_*` a decoupled decay extension (the paper defines none); `betas_*`/`eps_*` are accepted but unused. Requires `gradient_accumulation_steps: 1`; not available for SBERT; closure-based step supplied by the trainer`; mirror in ES after line 233 (`población`, `sigma`, `antitético`, `shaping por rangos`, `escalado NES`, `semilla`; `estrategias evolutivas sin gradientes (orden cero)`; `lr_*` es el tamaño de paso α por grupo; `wd_*` un decaimiento desacoplado (extensión, el paper no define ninguno); `betas_*`/`eps_*` se aceptan pero no se usan; requiere acumulación 1; no disponible para SBERT). (f) `x_parameter_reference.by_optimizer`, after the `eve` entry (lines 386-389): `ea4llm:` / `prefix: "ea4llm-"` / `supports_shared_per_group_suffixes: true` / `specific_global_suffixes: [population, sigma, antithetic, rank_shaping, nes_scaling, seed]`. |
| 6 | `src/schema/_training.yaml` | Append exactly one `if/then` block after the eve block (ends line 1567), same shape as every other optimizer (parity-checked by `tests/test_schema_consistency.py:76-90`): `- if:` / `properties:` / `optimizer:` / `properties:` / `optimizer_class:` / `const: ea4llm` — `then:` / `properties:` / `optimizer:` / `properties:` / `parameters:` / `additionalProperties: false` / `patternProperties:` / `^ea4llm-: { type: [number, boolean, array, string] }`. |
| 7 | `src/schema/_sbert.yaml` | **n/a** — EA4LLM is closure-based and rejected for SBERT by `config_loader` (row 4); closure-based optimizers are absent from the SBERT `optimizer_kwargs` description (`eve` is not listed there either). No new SBERT-usable suffixes are introduced. |
| 8 | `configs/examples/es_arch_hybrid_ea4llm.yaml` | Clone of `configs/examples/es_arch_hybrid_eve.yaml` with: `optimizer_class: ea4llm`; all `eve-` parameter prefixes → `ea4llm-`; drop the `betas_*`/`eps_*` block (accepted but unused — keeps the example honest) and the eve-specific `de_weight/de_jitter/init_sigma/adam_steps/use_pm/pm_eta`; add `ea4llm-population: 30`, `ea4llm-sigma: 1.0e-3`, `ea4llm-antithetic: false`, `ea4llm-rank_shaping: false`, `ea4llm-nes_scaling: false`; keep `gradient_accumulation_steps: 1` (line 46 — hard requirement); `csv_log_path: training_metrics_es_arch_hybrid_ea4llm.csv`. Keep the full `model:` block (dims 768/12/12, layer_pattern [retnet, titan_attn, mamba, ode], ffn 3072) and the ten `lr_*` + ten `wd_*` group keys verbatim from the eve example. Must pass `load_training_config` (auto-checked by `tests/test_yaml_examples.py`). |
| 9 | `configs/README.md` | (a) Add `- ea4llm` after `- eve` in the supported-values list (line 158). (b) Extend the closure note (line 162) to: "`eve` and `ea4llm` require `gradient_accumulation_steps: 1` and are not available for SBERT (closure-based step: the trainer re-evaluates the current batch on several candidate points inside `optimizer.step()`)." (c) Suffix reference, after the eve bullet (line 207): `- `ea4llm`: `population`, `sigma`, `antithetic`, `rank_shaping`, `nes_scaling`, `seed` (gradient-free evolution strategies; `lr_*` is the per-group step size α, `wd_*` a decoupled decay extension; `betas_*`/`eps_*` accepted but unused; N forward evaluations per step)`. (d) Example-keys list (around line 217): add `- `ea4llm-population: 30`` and `- `ea4llm-sigma: 1.0e-3``. |
| 10 | `full_tests/run_e2e.py` | Append `"ea4llm"` to `OPTIMIZERS` (line 250, after `"eve"`); add a param-map block after the eve block (line 677): `if optimizer_class == "ea4llm": params[f"{prefix}population"] = 30; params[f"{prefix}sigma"] = 1e-3` (documented specific globals; unsupported keys are dropped automatically by the `OPTIMIZER_ALLOWED_KEYS` filter at lines 682-697). |
| 11 | `tests/test_optimizer_factory.py` | (a) Add `"ea4llm"` to the `required` set in `test_registry_contains_all_required` (line 29). (b) New class `BuildOptimizerEA4LLMTests` after the EvE tests (mirroring its `_closure_for`/`_tiny_param` helpers, lines 796-809 — quadratic closure `((p − 0.25)**2).sum()`): `test_builds_without_error`, `test_step_requires_closure` (RuntimeError, message contains "closure"), `test_step_runs_with_closure` (8 steps, all finite, no NaN — ES descends the negated fitness = loss), `test_defaults_match_paper` (population 30, sigma 1e-3, antithetic/rank_shaping/nes_scaling False), `test_custom_suffixes_and_coercion` (`"ea4llm-population": "8"`, `"ea4llm-sigma": "0.01"` coerce; invalid strings fall back to defaults), `test_unknown_group_suffix_raises` (`ea4llm-lr_bogus`), `test_per_group_lr_routes` (`ea4llm-lr_attention: 0.02` reaches `param_groups[0]["lr"]`), `test_state_has_no_per_parameter_buffers` (after one step, `opt.state[p]` holds only scalars/counters — the paper's O(1)-state claim), `test_rank_shaping_flavor_runs`, `test_antithetic_flavor_runs`, `test_antithetic_odd_population_raises` (ValueError on population=7 + antithetic). |
| 12 | `docs/bibliography/optimizers.bib` | Append after `raj_eve_2026` (line 353), matching the Zotero arXiv style: `@misc{liu_ea4llm_2025,` / `title={EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms},` / `author={WenTao Liu and Siyu Song and Hao Hao and Aimin Zhou},` / `year={2025},` / `eprint={2510.10603},` / `archivePrefix={arXiv},` / `primaryClass={cs.LG},` / `url={https://arxiv.org/abs/2510.10603},}`. |
| 13 | `docs/paper/appendices/annex-1-optimizers.tex` | In `\subsection{Evolutionary and Population-Based Hybrids}` (line 1268), after the EvE Pros block (ends line 1352): `\paragraph{EA4LLM (Evolution Strategies for LLMs).}` Optional figure first — `figures/optimizers/ea4llm.pdf` via `ea4llm.mmd` in the same neutral-theme Mermaid style as `eve.mmd` (flowchart: draw seeds → pass 1 perturb+evaluate → weights → pass 2 replay+accumulate → apply); **omit the figure if the mermaid→PDF toolchain is unavailable** (the paragraph stands alone; existing figures are `.mmd`+`.pdf`+`.svg` triplets, 23 optimizers have figures). Body (faithful to §2 above): positioning sentence ("gradient-free alternate that removes backpropagation and optimizer state entirely, validated for full-parameter pretraining from 0.5B to 32B"); the fitness equations `p_t = softmax(z_t)`, `R(x;\theta) = (1/\sum_t m_t)\sum_t m_t \log p_\theta(x_{t+1}\mid x_{\le t})`, `F(\theta) = \frac{1}{|D|}\sum_x R(x;\theta)` and subsampled-eval unbiasedness; perturbation `r_j = F(\theta + \sigma\epsilon_j)`; weight equations (z-score and centered-rank); direction/update equations with the `1/\sigma` NES variant; antithetic pairs; defaults `N=30, \sigma=10^{-3}, \alpha=5\times10^{-4}`, toggles off (App. A.1 removes grouped multipliers and step-size schedules); implementation notes: closure-based step consuming the negated loss as fitness, per-group `\texttt{lr\_*}` as `\alpha_g`, decoupled `\texttt{wd\_*}` decay marked as a repo extension (the paper defines none), two-pass seeded replay giving O(1) optimizer state, `\texttt{gradient\_accumulation\_steps: 1}` and no SBERT/AMP (shared closure path with EvE); `\begin{algorithm}` float `alg:ea4llm` transcribing Algorithm 1 (closure-based form); `\paragraph{Cons.}` — N forward evaluations per step (population 30 ⇒ 30 fwd+bwd per step via the current closure), higher variance and worse sample efficiency than first-order, sensitive to `N/\sigma/\alpha/normalization`, toggles (antithetic, rank shaping, NES scaling) off in the paper's own large-scale runs, closure path means no clipping/AMP; `\paragraph{Pros.}` — zero optimizer state and no backward needed in principle (>4× GPU-memory reduction at ≥4B; 137.61 vs 431.76 GB at 32B), stable full-parameter pretraining 0.5B–32B (Figure 1), architecture-agnostic (non-differentiable ops allowed), unbiased subsampled fitness. Also extend the closing survey longtable row (line 1366): `Evolutionary hybrid & EvE, EA4LLM & ...` with the interpretation sentence extended ("EA4LLM removes gradients entirely — zeroth-order ES with token-level log-likelihood fitness and no optimizer state"). |
| 14 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Row under `\textbf{Population-Based \& Evolutionary}` (after the EvE row, line 168): `EA4LLM & Gradient-free evolution strategies & 0 (RNG + counters only) & $N$ fwd/bwd evals, no state & lr ($\alpha$), population $N$, $\sigma$, shaping, antithetic, $1/\sigma$ & \citep{liu_ea4llm_2025} \\`. (b) Census: line 4 "45 sequence mixers across nine families, 27 optimizers" → "… 28 optimizers"; line 100 "catalogs all 27 supported optimizers" → "all 28". |
| 15 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: "The system supports twenty-seven optimizer families in eight categories" → "twenty-eight"; append EA4LLM to the last family: "and evolutionary population-based hybrids (EvE)" → "and evolutionary population-based hybrids (EvE, EA4LLM)". |
| 16 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of row 13, placed in `\subsection{Híbridos Evolutivos y Poblacionales}` (line 1268) after the EvE Ventajas (line 1346+): `\paragraph{EA4LLM (Estrategias Evolutivas para LLMs).}` — same equations verbatim (math is language-neutral), Spanish prose mirroring the EvE paragraph register (see line 1277 for tone); figure path `../paper/figures/optimizers/ea4llm.pdf` (shared figures dir); `\paragraph{Contras.}` / `\paragraph{Ventajas.}`; extend the closing survey row's "Híbrido evolutivo & EvE" to include EA4LLM. |
| 17 | `docs/paper-es/appendices/annex-summary-tables.tex` | Row after EvE (line 168): `EA4LLM & Estrategias evolutivas sin gradientes & 0 (solo RNG y contadores) & $N$ evaluaciones fwd/bwd, sin estado & lr ($\alpha$), población $N$, $\sigma$, shaping, antitético, $1/\sigma$ & \citep{liu_ea4llm_2025} \\`. Census: line 4 "27 optimizadores" → "28 optimizadores"; line 100 "cataloga los 27 optimizadores" → "los 28". |
| 18 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: "veintisiete familias de optimizadores" → "veintiocho"; "y métodos evolutivos poblacionales (EvE)" → "(EvE, EA4LLM)". |
| 19 | `docs/architecture.md`, `README.md` | Census pass, every occurrence of the old count (27): `docs/architecture.md:83` (repo-map row "`src/model/optimizer/**` | 27 optimizers + registry…") → 28; `docs/architecture.md:716` heading `### 8.1 Optimizers (27)` → `(28)`; `docs/architecture.md:731` family-table row `\| Evolutionary \| eve \| …` → `\| Evolutionary \| eve, ea4llm \| …` (extend the character cell: "population-based, closure-based; `eve` = DE + Adam rescue with 12× state, `ea4llm` = gradient-free Gaussian ES with zero optimizer state, N forward evals per step"); `README.md:27` mermaid node "Train<br/>27 optimizer families" → 28; `README.md:93` "\| Optimizer families \| 27 across 7 categories \|" → "28 across 7 categories". (`configs/README.md` census covered in row 9; grep `27 optim\|27 supported\|twenty-seven\|veintisiete` at implementation time to catch any drift.) |

## 5. Ordered execution checklist (implementation time only — do NOT run now)

Implementation (rows 1-10) → tests (row 11) → docs (rows 12-19) → PDFs → website sync
→ commit/push.

```bash
# 0. baseline (must be clean apart from the plan file)
git status --short

# 1-10. implement rows 1-10 of the table above

# 11. targeted gates
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_run_e2e_configs.py -v --tb=short -p no:warnings
# expected: all pass; test_optimizer_prefixes_match_conditional_rules sees the new
# ^ea4llm- rule; the new example YAML validates through load_training_config.

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
# expected: no new failures.

# 12-19. docs rows, then rebuild both PDFs (pdflatex + bibtex ARE available:
# /usr/bin/pdflatex, /usr/bin/bibtex — verified read-only in planning)
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
cd ../..

# fast e2e smoke (raise --limit if ea4llm is not reached — it will be last in OPTIMIZERS)
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3
# expected: the ea4llm combo validates and trains without ValueError/RuntimeError.
```

**PDF preconditions and verification** (both `pdflatex` and `bibtex` exist on this
machine — `command -v pdflatex bibtex` → `/usr/bin/pdflatex`, `/usr/bin/bibtex`).
Fallback if unavailable: leave the `.tex`/`.bib` sources updated and note that the
rebuild must happen on a TeX machine before the next release — the website sync copies
prebuilt PDFs only. After compilation require: no `Fatal error` in the logs, no
`Citation liu_ea4llm_2025 undefined` (grep the `.blg`/`.log` before deleting them),
`paper.pdf` / `paper-es.pdf` newer than the edited `.tex`/`.bbl`, and
`git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf` changes (plus the optional figure triplet).

## 6. Website sync (planned, not executed)

Run `/sync-website` after the main-repo commit. Steps (per
`.opencode/command/sync-website.md`):

1. `bash .opencode/command/sync-website-state.sh` — read-only drift report.
2. Copy into `erickfmm.github.io/frankenstein-transformer/`: `src/schema.yaml` +
   `src/schema/**` (includes the edited `_optimizer.yaml`, `_training.yaml`);
   `configs/examples/**` (new `es_arch_hybrid_ea4llm.yaml`) + `configs/README.md` →
   `examples/README.md`; `docs/paper/**` (`paper.tex`, `sections/*.tex`,
   `appendices/*.tex`, `paper.bbl`, `paper.pdf`, `figures/optimizers/ea4llm.*` if
   produced) → `paper/en/…`; `docs/paper-es/**` → `paper/es/…`;
   `docs/bibliography/optimizers.bib` → `paper/bibliography/`; `docs/architecture.md` →
   `paper/specs/architecture.md`.
3. Rebuild the Paper-tab HTML: `flatten_tex.py` + `pandoc` (+ `postprocess_html.py`)
   per the sync command; refresh `paper/en.pdf` / `paper/es.pdf`.
4. **Validation Gate** (do not skip): `full_tests/param_count_check.py` → `0 failures`
   (regenerates `full_tests/param_truth.json` — intentional change if it shifts);
   `param_estimate_check.mjs … --tol 0.01` → pass (the estimator counts *model*
   parameters; a new optimizer changes nothing, so it should stay green); state-script
   probes `schema-refs`, `example-yaml`, `cli-flags`, estimator → all PASS.
5. The mirror stays uncommitted until the commit phase (below).

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions: English, conventional tag, structured body;
never stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json` changes are
intentional.

1. **Main repo** — `feat(optimizer): add EA4LLM (arXiv 2510.10603)`
   Body: implementation notes (closure-based zeroth-order ES, two-pass seeded replay,
   O(1) state); new suffixes `population/sigma/antithetic/rank_shaping/nes_scaling/seed`;
   schema/registry changes (enum, `by_optimizer`, `^ea4llm-` if/then rule, closure
   guards); docs (annexes EN/ES, summary tables, family sections, bibliography,
   census 27→28, rebuilt PDFs); example `es_arch_hybrid_ea4llm.yaml`; e2e list; tests.
   Stage exactly:
   - `src/model/optimizer/ea4llm.py`, `src/model/optimizer/__init__.py`,
     `src/model/optimizer/factory.py`
   - `src/training/config_loader.py`
   - `src/schema/_optimizer.yaml`, `src/schema/_training.yaml`
   - `configs/examples/es_arch_hybrid_ea4llm.yaml`, `configs/README.md`
   - `full_tests/run_e2e.py`
   - `tests/test_optimizer_factory.py`
   - `docs/bibliography/optimizers.bib`,
     `docs/paper/appendices/annex-1-optimizers.tex`,
     `docs/paper/appendices/annex-summary-tables.tex`,
     `docs/paper/sections/05-optimizer-families.tex`, `docs/paper/paper.bbl`,
     `docs/paper/paper.pdf`,
     `docs/paper/figures/optimizers/ea4llm.{mmd,pdf,svg}` (if produced)
   - `docs/paper-es/appendices/annex-1-optimizadores.tex`,
     `docs/paper-es/appendices/annex-summary-tables.tex`,
     `docs/paper-es/sections/05-familias-optimizadores.tex`,
     `docs/paper-es/paper-es.bbl`, `docs/paper-es/paper-es.pdf`
   - `docs/architecture.md`, `README.md`
   - `full_tests/param_truth.json` (only if the Validation Gate regenerated it)
   Then `git push origin HEAD:main`.
2. **Mirror** — `git -C erickfmm.github.io …` stage the synced paths from §6, commit
   `docs(site): sync EA4LLM optimizer (arXiv 2510.10603)`, then
   `git -C erickfmm.github.io push origin HEAD:main`.

## 8. Verification matrix

| Gate | Command (implementation time) | Expected result |
|---|---|---|
| Targeted pytest | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_run_e2e_configs.py -v --tb=short -p no:warnings` | All pass; `test_optimizer_prefixes_match_conditional_rules` green with the new rule; new EA4LLM tests green |
| Full pytest | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | No new failures vs baseline |
| e2e smoke | `uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3` | `ea4llm` combo validates and trains (closure path; no AMP, accumulation 1) |
| PDF checks | pdflatex/bibtex sequence above | No `Fatal error`, no `Citation liu_ea4llm_2025 undefined`, PDFs newer than sources, only intended files changed |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `0 failures` (model params unaffected; `param_truth.json` unchanged or intentionally regenerated) |
| Estimator | `node full_tests/param_estimate_check.mjs … --tol 0.01` | Pass within tolerance (estimator counts model parameters only) |
| State probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, estimator all PASS after sync |

## 9. Risks, assumptions and open questions

1. **Rank-shaping centering (verify).** The exact rank-shaping equation is in a figure
   the arXiv text conversion omitted ("rank(·) assigns ranks 0,…,N−1 after sorting",
   §3.2). The plan assumes centered ranks `w_j = rank(r_j) − (N−1)/2` per the paper's
   declared baseline (Salimans et al. 2017), because uncentered ranks would add a
   constant positive drift along the mean noise. At implementation time, confirm
   against the paper PDF's §3.2 figure; adjust `_weights()` if it differs.
2. **z-score denominator.** `std(r)` with ddof=0 vs ddof=1 is unspecified; either is a
   valid z-score — plan uses ddof=0 (population std). Immaterial to direction, scales
   α.
3. **Wasted backward.** The trainer closure (`trainer.py:1808-1812`) runs
   `loss.backward()`; ES discards gradients. Correct but ~2× the necessary evaluate
   cost. A `backward=False` closure variant is a natural follow-up but touches the
   trainer beyond this recipe — **out of scope**, noted here.
4. **Fitness direction.** Repo closures return a loss to *minimize*; the paper's
   fitness is *maximized*. Implementation must negate (`r_j = -closure()`). Getting
   this backwards ascends the loss — covered by `test_step_runs_with_closure`
   (quadratic loss must decrease over steps).
5. **NaN robustness.** A perturbed evaluation may return inf/NaN on odd batches;
   `_weights` zeroes non-finite rewards. If *all* rewards are non-finite, skip the
   update (return non-finite mean) rather than NaN the parameters.
6. **Naming collisions.** `ea4llm` and `EA4LLM` are free (registry, `__init__.__all__`,
   schema enum, examples dir, bib keys, annex labels `alg:ea4llm`/`fig:opt-ea4llm`).
   No collision.
7. **Task/mode compatibility.** `training.task` rules unchanged; EA4LLM works for any
   gradient-trained task *except* SBERT (closure unavailable) and requires
   `gradient_accumulation_steps: 1` and no AMP — all three enforced by config_loader
   (row 4) exactly as for eve.
8. **Pre-existing inconsistency (do not fix here):** README says "7 categories",
   the papers say "eight categories", §8.1 lists 9 table rows. Only the count 27→28 is
   in scope; the category-count drift is flagged for the user.
9. **Census drift:** the 27→28 list in row 19 was produced by grepping the repo at
   planning time; re-grep at implementation time in case unrelated edits landed.
10. **ICML status.** The paper header claims ICML 2025 (PMLR 267) while the arXiv ID
    (2510.10603) is October 2025; the bibliography entry uses year 2025 and cites
    arXiv only (matching `raj_eve_2026` style). No venue claim in the annex beyond
    "ICML 2025" as printed in the paper.
11. **Paper proposes one optimizer only.** EA4LLM is the single method; the ablated
    variants (evaluation budgets, population sizes, toggles) are configurations, not
    separate optimizers. Nothing to record under "other optimizers".

## 10. Out of scope

This command executed **no** code, schema, preset, test, doc, PDF, mirror or git
change. No command that mutates the repository was run (no pytest, no pdflatex, no
uv, no sync, no `git add/commit/push`). The only file written is this plan document.
Implementation is a separate, explicitly requested step (switch agents to execute).
