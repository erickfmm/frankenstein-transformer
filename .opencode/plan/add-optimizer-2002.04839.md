# Plan: add LaProp optimizer (arXiv:2002.04839)

**Paper**: Ziyin, L., Wang, Z. T., Ueda, M. (2021). *LaProp: Separating Momentum and
Adaptivity in Adam*. arXiv:2002.04839v3 [cs.LG]. LaProp identifies a coupling in
Adam-style optimizers — the accumulated (parameter-space) momentum is reweighted by the
current preconditioner through the factor $H_t^{-1}H_{t-1}$, so a single pathological
gradient (noise, OOD batch, explosion) can wipe the accumulated momentum and destabilize
training. LaProp decouples the two: it takes the EMA of the **preconditioned** gradient
(the momentum averages $g_t/\sqrt{\hat n_t + \epsilon}$ rather than $g_t$), leaving
parameter-space momentum additive, exactly like SGD. Cost, state and hyperparameter
count are identical to Adam; Adam settings carry over unchanged; $\nu \to 0$ recovers
signed-gradient methods. This makes it a natural member of the repo's
"Classical & Adaptive Baselines" bucket and a drop-in alternative to `adamw`.

---

## 1. Paper extract (verbatim source of truth)

### Update rule (§2.3, Eq. 7 + Algorithm 1)

Generalized adaptive framework (Eq. 2): $\theta_{t+1} = \theta_t - \lambda_t G_t \nabla\ell(\theta_t) + \mu_t K_t(\theta_t - \theta_{t-1})$.
Adam has $K_t = H_t^{-1}H_{t-1}$ (coupled). **LaProp sets $K_t = I$** (Eq. 7):

$$\theta_{t+1} = \theta_t - \lambda_t H_t^{-1}\nabla\ell(\theta_t) + \mu_t(\theta_t - \theta_{t-1})$$

**Algorithm 1 (LaProp)** — quoted line by line:
> Input: $\theta_1 \in \mathbb{R}^d$, learning rate $\{\lambda_t\}_{t=1}^T$, decay
> parameters $0 \le \mu, \nu < 1$, and $\epsilon \ll 1$, bias correction factors
> $0 < c_n, c_m < 1$. Set $m_0 = n_0 = 0$.
> **for** $t = 1$ to $T$ **do**
> &nbsp;&nbsp;$g_t = \nabla_\theta \ell(\theta_t)$
> &nbsp;&nbsp;$n_t = \nu n_{t-1} + (1-\nu) g_t^2$
> &nbsp;&nbsp;$m_t = \mu m_{t-1} + (1-\mu)\, \dfrac{g_t}{\sqrt{n_t/c_n} + \epsilon}$
> &nbsp;&nbsp;$\theta_{t+1} = \theta_t - \lambda_t\, m_t / c_m$
> **end for**

with (from Eq. 4) $c_m := \frac{1}{1-\mu^t}$ and $c_n := \frac{1}{1-\nu^t}$, i.e.
$m_t/c_m = m_t(1-\mu^t)$ and $n_t/c_n = \hat n_t = n_t(1-\nu^t)$ — the standard bias
corrections. **The key structural difference from Adam**: the *second* moment $n_t$ is
an EMA of $g_t^2$ (same as Adam), but the *first* moment $m_t$ is an EMA of the
already-normalized gradient $g_t/(\sqrt{\hat n_t}+\epsilon)$; the update is then just
$\theta \leftarrow \theta - \lambda\,\hat m_t$ with $\hat m_t = m_t/(1-\mu^t)$.

### Limits / relations (§3, §4)
- $\nu \to 0$: recovers signed stochastic gradient (SSG / sign-GD family).
- $\mu = 0$: RMSProp with bias correction (the paper notes LaProp "can be regarded as
  RMSProp with momentum" plus the $c_m, c_n$ corrections).
- Effect of a pathological gradient on the momentum is bounded by
  $\frac{1-\mu}{\sqrt{1-\nu}}$ (Proposition 1) — the stability claim.
- Regret $O(\sqrt T)$ under the same assumptions as Adam (§4); the bound differs from
  Adam/AMSGrad by a key factor.
- No special warmup/schedule interaction beyond standard LR schedules; the repo's
  existing schedulers apply unchanged.

### Hyperparameters
| Name | Symbol | Type | Default (paper/Adam carry-over) | Scope |
|---|---|---|---|---|
| lr | $\lambda$ | float | 1e-3 | per-group (`lr_<group>`) |
| betas | $(\mu, \nu)$ | (float, float) | (0.9, 0.999) — "exactly the same number of hyperparameters as Adam; the hyperparameter settings of Adam can be readily carried over" | per-group (`betas_<group>`) |
| eps | $\epsilon$ | float | 1e-8 ("$\epsilon \ll 1$") | per-group (`eps_<group>`) |
| weight_decay | — | float | 0 (paper does not modify Adam's wd handling; official LaProp code uses coupled L2 into the gradient) | per-group (`wd_<group>`) |

All four are **shared per-group suffixes** already routed by the factory for every
adaptive optimizer — LaProp needs **zero optimizer-specific suffixes**.

### State buffers & cost
- **2 buffers/parameter**: $m_t$ (`exp_avg`), $n_t$ (`exp_avg_sq`) — identical memory to Adam.
- **Per-step cost**: $\mathcal{O}(n)$ — one EMA update, one sqrt/div, one mul-add. Same as Adam.

### Category
Existing bucket **Classical & Adaptive Baselines / Adaptive**
(`docs/architecture.md` §8.1 family table; `annex-9-optimizers_adaptive.tex`;
summary-table group "Classical & Adaptive Baselines"). No new family.

### BibTeX data
Ziyin, Liu; Wang, Zhikang T.; Ueda, Masahito. "LaProp: Separating Momentum and
Adaptivity in Adam". arXiv:2002.04839v3 [cs.LG], 13 Jun 2021.

---

## 2. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `laprop` | Paper short name, snake_case, no collision (`grep -ri laprop src/ docs/ configs/ tests/ full_tests/` → empty) |
| Class / module | `LaProp(torch.optim.Optimizer)` in `src/model/optimizer/laprop.py` | Genuine `step()` needed (update order differs from Adam: precondition *then* momentum); cannot subclass AdamW like `adan.py` |
| Closest peer | `src/model/optimizer/anon.py` for the step() skeleton (state init, coupled wd, bias corrections) + `adan.py` for the constructor surface | `anon.py:150-205`, `adan.py:49-50` |
| `param_bindings` | **empty** — shared `lr_*`, `wd_*`, `betas_*`, `eps_*` only | `factory.py:41-67` auto-routes them for every adaptive (non-`sgd`) optimizer; same as `adamw`/`adan`/`ademamix` specs |
| `group_policy` | default (adaptive) | LaProp consumes per-group betas/eps |
| `supports_shared_per_group_suffixes` | `true` (spec default; not overridden) | |
| Capability flags | none (`closure_based=False`, `sbert_allowed=True`, `requires_grad_accum_one=False` — defaults) | It is a plain Adam-shaped optimizer; no closure, no grad-accumulation constraints |
| `custom_build` | none | Constructor takes the standard `(groups, lr, betas, eps, weight_decay)`-shaped kwargs; groups carry per-group values |
| Example preset | `configs/examples/es_arch_ca_laprop.yaml` (MLM encoder, copy of `es_arch_ca_adamw.yaml` with `optimizer_class: laprop` and `laprop-*` shared keys) | Smallest valid delta from an existing, schema-validated example |
| Census bump | **38 → 39** everywhere; also fix stale "36" at `docs/architecture.md:732` | Registry `OPTIMIZER_SPECS` has 38 entries; `run_e2e.py:344-349` lists 38; README:36 "38 optimizer families", README:126 "38 across 8 categories"; `docs/architecture.md:886` "### 8.1 Optimizers (38)"; `annex-summary-tables.tex:4` & `:119` "all 38" (EN and ES twins at `paper-es/...:4`) |
| Figures | **omit** (`docs/paper/figures/optimizers/laprop.*`) | LaProp is a two-buffer Adam-relative method with no distinctive dataflow diagram among the peer set that lack one; avoids inventing a new vector-style figure. Revisit later if wanted |

### File-by-file plan table (printed for veto before finalizing)

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/optimizer/laprop.py` | New module. Docstring citing arXiv:2002.04839; `from __future__ import annotations`. `class LaProp(Optimizer)` with `__init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`, validation (0≤lr, 0≤eps, 0≤β_i<1), `defaults` dict. `@torch.no_grad() step(closure=None)` mirroring `anon.py:127-205`: per-group loop, skip `p.grad is None`, reject sparse grads; state init (`step`, `exp_avg`, `exp_avg_sq`); coupled weight decay `grad.add_(p.data, alpha=group["weight_decay"])` when nonzero; then **Algorithm 1 order**: `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)`; `bias_correction2 = 1 - beta2**t`; `precond = grad / (exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + eps)` (ε added after bias correction, outside the sqrt — matches Algorithm 1); `exp_avg.mul_(beta1).add_(precond, alpha=1-beta1)`; `bias_correction1 = 1 - beta1**t`; `p.data.add_(exp_avg, alpha=-group["lr"]/bias_correction1)`. Google-style docstrings, CPU-safe. |
| 2 | `src/model/optimizer/__init__.py` | Add `from .laprop import LaProp` (after the `from .adan import Adan` block at line 5, alphabetical) and `"LaProp"` to `__all__` (after `"Adan"`, line 47). |
| 3 | `src/registry/builtin/optimizer.py` | Insert one `OptimizerSpec` into `OPTIMIZER_SPECS` alphabetically (between `lamb` at line 240 and `lion` at 246): `name="laprop"`, `builder="src.model.optimizer.laprop:LaProp"`, `description="Decoupled momentum + adaptivity — Adam-shaped but the momentum averages the *preconditioned* gradient (RMSProp-with-momentum view of Adam); parameter-space momentum stays additive, more stable on noisy/unstable tasks; same state and hyperparameters as Adam (paper: arxiv:2002.04839)"`, `description_es` faithful translation, `references=("2002.04839",)`, **no** `param_bindings`, **no** flags. This single declaration generates the enum, `by_optimizer` entry and `^laprop-` if/then rule via `schema_assembly._inject_optimizers` — `factory.py` is untouched. |
| 4 | `src/schema/_optimizer.yaml` | Documentation only: (a) one 3-column use-case row in **both** EN and ES tables (around the `namo_d` row seen at lines ~138-140): `` | `laprop` | Decoupled momentum + adaptivity — ... | `` and the Spanish twin; (b) append `` `laprop` (paper: arxiv:2002.04839)`` to the **Referencias** sentence (lines ~161-170, both languages); (c) LaProp gets **no** bullet in "Optimizer-specific suffixes" — it stays covered by the existing "Others: no specific suffixes (only shared ones)" line (~205). |
| 5 | — | `src/schema/_training.yaml`, `_sbert.yaml`: **no edit** (assembler injects enum/`if/then`). |
| 6 | `configs/examples/es_arch_ca_laprop.yaml` | Full copy of `configs/examples/es_arch_ca_adamw.yaml` with `optimizer_class: laprop` and every `adamw-` key renamed `laprop-` (`laprop-lr_embeddings`, `laprop-lr_norms`, `laprop-lr_attention`, `laprop-lr_other`, `laprop-wd_*`, `laprop-betas_*`, `laprop-eps_*` — all shared suffixes, schema-valid). |
| 7 | `configs/README.md` | Add `laprop` to the optimizer list, add one line in the per-optimizer suffix section: "`laprop`: no specific suffixes — only the shared `lr_*`/`wd_*`/`betas_*`/`eps_*` (β defaults (0.9, 0.999), Adam settings carry over)", and list the new example keys. Quote surrounding format when editing. |
| 8 | `full_tests/run_e2e.py` | Append `"laprop"` to the `OPTIMIZERS` list (line 344-349, second row end). No `_with_optimizer` block — defaults are safe. |
| 9 | `tests/test_optimizer_factory.py` | (a) add `"laprop"` to the `required` set in `test_registry_contains_all_required` (line 31); (b) new `class BuildOptimizerLaPropTests(unittest.TestCase)` after the Sophia class (~1473): `test_builds_without_error` (`build_optimizer("laprop", _param_groups(), {})` → instance), `test_step_runs` (tiny tensor, backward, `opt.step()` no raise), `test_shared_suffix_parsed` (`{"laprop-betas_attention": [0.9, 0.999], "laprop-eps": 1e-8}` accepted; assert group values), `test_unknown_key_raises` (`{"laprop-gamma": 1.0}` → ValueError). |
| 10 | `docs/bibliography/optimizers.bib` | Append: `@misc{ziyin_laprop_2021, title={{LaProp}: Separating Momentum and Adaptivity in {Adam}}, author={Ziyin, Liu and Wang, Zhikang T. and Ueda, Masahito}, year={2021}, eprint={2002.04839}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={https://arxiv.org/abs/2002.04839}}` (matching the Zotero/arXiv style of neighbors like `tieleman_hinton2012`). |
| 11 | `docs/paper/appendices/annex-9-optimizers_adaptive.tex` | `\paragraph{LaProp.}` after the other adaptive entries: motivation (Adam's $K_t = H_t^{-1}H_{t-1}$ coupling, Eq. 6 of the paper), equations from §1 above in LaTeX (`n_t`, `m_t` over preconditioned gradient, bias-corrected update), a small `algorithm` float mirroring Algorithm 1, Pros (Adam-identical cost/state/hyperparameters; more stable under noisy gradients; $\nu\to 0$ → sign-GD interpolation), Cons (no consistent win on every task; gains most visible in noisy/unstable regimes), `\citep{ziyin_laprop_2021}`. No figure (decision above). |
| 12 | `docs/paper/appendices/annex-summary-tables.tex` | Row under **Classical & Adaptive Baselines** (after the RAdam row at ~line 145): `LaProp & Adaptive & 2 ($m$, $v$) & $\mathcal{O}(n)$ & lr, $\beta_1$, $\beta_2$, eps, wd & \citep{ziyin_laprop_2021} \\`. Update the two census sentences: line 4 "38 optimizers" → "39 optimizers"; line 119 "all 38 supported optimizers" → "all 39 supported optimizers". |
| 13 | `docs/paper/sections/05-optimizer-families.tex` | Add LaProp to the adaptive-baselines family sentence + bump the family/total counts found by grepping "38". |
| 14 | `docs/paper-es/appendices/annex-9-optimizadores_adaptive.tex` | Faithful Spanish translation of #11 (same equations, translated prose; figures reference `../paper/figures/optimizers/` — none here). |
| 15 | `docs/paper-es/appendices/annex-summary-tables.tex` | Spanish row (same math) + census "38 optimizadores" → "39 optimizadores" (lines 4 and the ES "todos los 38" sentence). |
| 16 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Spanish family sentence + counts. |
| 17 | `docs/architecture.md` | §8.1 heading line 886 `(38)` → `(39)`; family table row count; **fix stale line 732** "Same 36-family enum" → "Same 39-family enum"; add `laprop` to the §8.1 list. |
| 18 | `README.md` | Line 36 "38 optimizer families" → "39 optimizer families"; line 126 "Optimizer families | 38 across 8 categories" → "39 across 8 categories". |
| 19 | `docs/schema.assembled.yaml` | Regenerate: `uv run --extra cpu --extra train python -m src.registry.snapshot --out docs/schema.assembled.yaml` (implementation time). |

**Hard rules restated**: schema is the contract (`additionalProperties: false`); keys
are `<laprop>-<group>_<param>` / `<laprop>-<param>`; the enum, `by_optimizer` and
`if/then` prefix rules are **generated from the spec** (`_inject_optimizers`) and never
hand-edited; never read an undocumented key; the registry is the single source of truth
(D13/D14); the paper is the theory source of truth (D11) — annex formulations must
match arXiv:2002.04839 and `paper-es` must translate them faithfully.

---

## 3. Ordered execution checklist (at implementation time — do NOT run now)

Implementation:
- [ ] 1. `src/model/optimizer/laprop.py` (file #1)
- [ ] 2. `src/model/optimizer/__init__.py` (#2)
- [ ] 3. `src/registry/builtin/optimizer.py` spec (#3)
- [ ] 4. `src/schema/_optimizer.yaml` prose (#4)
- [ ] 5. `configs/examples/es_arch_ca_laprop.yaml` (#6)
- [ ] 6. `configs/README.md` (#7), `full_tests/run_e2e.py` (#8)
- [ ] 7. Regenerate snapshot (#19)

Tests:
- [ ] 8. `tests/test_optimizer_factory.py` (#9)

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_registry_core.py tests/test_registry_binding.py tests/test_schema_assembly.py -v --tb=short -p no:warnings
# expected: all pass; test_schema_consistency auto-covers the new spec's defaults,
# docs rows and prefix rule (gate D17)

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# e2e enumeration + smoke
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --list-combos | grep -F laprop
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --only laprop
# (--realonall default trains on the Borges corpus; --no-realonall offline fallback; --limit N caps)
```

Docs:
- [ ] 9. Bibliography (#10), EN annex (#11), summary tables (#12), family section (#13), ES twins (#14-16), architecture/README census (#17-18)

PDFs (only at implementation time; `pdflatex`/`bibtex` confirmed available at `/usr/bin`):

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

Post-compile verification: no `Fatal error`, no `Citation .* undefined`, `paper.pdf` /
`paper-es.pdf` newer than the edited `.tex`/`.bbl`, `ziyin_laprop_2021` resolves,
`git status --short docs/paper docs/paper-es` shows only intended files. If TeX is
unavailable, leave sources updated and note the rebuild must happen before the next
release (website sync copies prebuilt PDFs only).

Website sync (planned): regenerate `docs/schema.assembled.yaml` **first**, then copy
into `erickfmm.github.io/frankenstein-transformer/`: `docs/schema.assembled.yaml`,
`src/schema/**`, `src/registry/**`, `configs/examples/**` + `configs/README.md`,
`docs/paper/**` (tex, bbl, pdf, `*.svg` figures only), `docs/paper-es/**`,
`docs/bibliography/**`, `docs/architecture.md`. Rebuild the Paper-tab HTML
(`flatten_tex.py --web` + `pandoc` + `postprocess_html.py`), refresh
`paper/en.pdf` / `paper/es.pdf`. Validation Gate: `python full_tests/param_count_check.py`
→ `0 failures`; `node full_tests/param_estimate_check.mjs … --tol 0.01`; state-script
probes (`schema-refs`, `example-yaml`, `cli-flags`, `paper-figures-en/es`, estimator)
all PASS. Mirror stays uncommitted until the commit phase.

Commits (planned):
1. Main repo — `feat(optimizer): add LaProp (2002.04839)`. Stage:
   `src/model/optimizer/laprop.py src/model/optimizer/__init__.py src/registry/builtin/optimizer.py src/schema/_optimizer.yaml configs/examples/es_arch_ca_laprop.yaml configs/README.md full_tests/run_e2e.py tests/test_optimizer_factory.py docs/bibliography/optimizers.bib docs/paper/appendices/annex-9-optimizers_adaptive.tex docs/paper/appendices/annex-summary-tables.tex docs/paper/sections/05-optimizer-families.tex docs/paper-es/appendices/annex-9-optimizadores_adaptive.tex docs/paper-es/appendices/annex-summary-tables.tex docs/paper-es/sections/05-familias-optimizadores.tex docs/architecture.md README.md docs/schema.assembled.yaml docs/paper/paper.pdf docs/paper/paper.bbl docs/paper-es/paper-es.pdf docs/paper-es/paper-es.bbl`
   (+ any `.tex`/`.bbl` the rebuild touches). Body: implementation notes, spec, docs,
   tests, example, snapshot. Then `git push origin HEAD:main`.
2. Mirror — `git -C erickfmm.github.io add frankenstein-transformer` (only the synced
   paths) → `docs(site): sync LaProp optimizer (2002.04839)` → push.
Never stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json` changes are
intentional.

---

## 4. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Factory tests | `pytest tests/test_optimizer_factory.py` | New `BuildOptimizerLaPropTests` pass; `test_registry_contains_all_required` green with `laprop` added |
| Base/registry gates | `pytest tests/test_optimizer_base.py tests/test_registry_core.py tests/test_registry_binding.py tests/test_schema_assembly.py` | Green; new spec validates, enum/prefix rule generated |
| Schema consistency | `pytest tests/test_schema_consistency.py` | Green (bilingual rows, defaults ↔ dataclass, prefixes ↔ `if/then`) |
| Example YAMLs | `pytest tests/test_yaml_examples.py` | `es_arch_ca_laprop.yaml` validates against assembled schema |
| Full suite | `pytest tests/ --continue-on-collection-errors` | All green |
| e2e enumerate | `run_e2e.py --category opt --list-combos \| grep -F laprop` | ≥1 combo listed |
| e2e smoke | `run_e2e.py --category opt --only laprop` | Trains, loss finite, exit 0 |
| PDF EN/ES | pdflatex+bibtex sequence above | No fatal/undefined-citation; PDFs newer than sources |
| Param count | `python full_tests/param_count_check.py` | `0 failures` |
| Estimator | `node full_tests/param_estimate_check.mjs … --tol 0.01` | PASS |
| State probes | `.opencode/command/sync-website-state.sh` probes | All PASS |
| git hygiene | `git status --short` | Only intended files staged/committed |

---

## 5. Risks, assumptions, open questions

- **Assumption (verify)**: `betas` default (0.9, 0.999) vs factory's implicit
  (0.9, 0.95). The factory always passes per-group `betas` explicitly with default
  `(0.9, 0.95)` (`factory.py:172-176`), so the constructor default is only
  documentation. Keep the class default at the paper's (0.9, 0.999) and document that
  Adam settings carry over; preset authors can set `laprop-betas_<group>: [0.9, 0.999]`.
- **Assumption (verify)**: ε placement. Algorithm 1 shows `g_t / (sqrt(n_t/c_n) + ε)` —
  ε **outside** the sqrt, added to the bias-corrected denominator. Implement exactly
  that (differs from `anon.py`, which adds ε inside the pow — that is a different
  paper's convention).
- **Weight decay**: paper doesn't redefine it; official LaProp repo couples L2 into the
  gradient. Plan mirrors that (coupled). If the maintainer prefers decoupled (AdamW
  style), it is a one-line change — flagged as an open question.
- **Census drift found**: `docs/architecture.md:732` says "36-family enum" while the
  registry has 38 — the plan fixes it to 39 as part of the census pass (in-scope
  consistency fix, not unrelated drift).
- No mixer/task incompatibilities: `laprop` is task-agnostic (MLM/SBERT/causal_lm/
  vision/audio all fine), no `closure_based`, no grad-accumulation restriction.
- The paper proposes only one optimizer (LaProp); no multi-optimizer ambiguity.

## 6. Out of scope

This command created **only** this plan file. No code, schema, preset, test, doc, PDF,
mirror or git change was made; no build/test/train command was executed.
