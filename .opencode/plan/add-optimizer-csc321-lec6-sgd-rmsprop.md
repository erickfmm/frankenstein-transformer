# Plan: add `sgd` (vanilla mini-batch SGD) and `rmsprop` optimizers (CSC321 Lecture 6)

> Source requested by the user: **https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf**
> (Hinton, Srivastava, Swersky — *Neural Networks for Machine Learning*, CSC321,
> University of Toronto, 2012). There is **no arXiv ID** for this source, so the
> plan slug is `csc321-lec6-sgd-rmsprop` instead of `<arxiv-id>`. Everything the
> template requires from "the paper" is extracted verbatim from the slides below.

## 1. Header

The source is the canonical CSC321 Lecture 6 slide deck ("Overview of
mini-batch gradient descent" / "A bag of tricks" / "rmsprop: Divide the
gradient by a running average of its recent magnitude"). Lecture 6a motivates
and defines **mini-batch SGD**: with redundant datasets, update the weights
from the gradient of a *mini-batch* instead of the full batch; the deck gives
the learning-rate heuristics (reduce on oscillation/divergence, increase when
descending slowly, turn down near the end) but no momentum — momentum is
presented separately (and is already implemented here as `sgd_momentum`).
Lecture 6e defines **RMSProp** as the mini-batch-safe version of `rprop`:
keep a moving average of the *squared* gradient per weight and divide the
gradient by its square root, so the divisor is nearly identical for adjacent
mini-batches ("Tijmen Tieleman, unpublished"). Both are the historical
baselines this repo is missing: plain SGD is the zero-state reference point,
and RMSProp is the ancestor of the entire adaptive family (AdaGrad → RMSProp →
Adam). They belong in this repo as registry components exactly like the other
36 optimizers: declarative `OptimizerSpec` → generated schema enum/prefix
rules → example YAML → tests → bilingual annexes.

## 2. Paper extract (verbatim from the slides)

### SGD (Lecture 6a — "Overview of mini-batch gradient descent")

Slides text (pdftotext, punctuation normalized):

> "Stochastic gradient descent — If the dataset is highly redundant, the
> gradient on the first half is almost identical to the gradient on the second
> half. So instead of computing the full gradient, update the weights using
> the gradient on the first half and then get a gradient for the new weights
> on the second half. The extreme version of this approach updates weights
> after each case. It's called 'online'."
>
> "Mini-batches are usually better than online. — Less computation is used
> updating the weights. — Computing the gradient for many cases
> simultaneously uses matrix-matrix multiplies which are very efficient,
> especially on GPUs. Mini-batches need to be balanced for classes."
>
> "A basic mini-batch gradient descent algorithm — Guess an initial learning
> rate. If the error keeps getting worse or oscillates wildly, reduce the
> learning rate. If the error is falling fairly consistently but slowly,
> increase the learning rate. […] Towards the end of mini-batch learning it
> nearly always helps to turn down the learning rate."

**Update rule** (the trivial one the deck assumes throughout; consistent with
the error-surface slides in 6a):

$$\theta_{t+1} = \theta_t - \eta\, g_t, \qquad g_t = \frac{1}{|B|}\sum_{i \in B} \nabla_\theta E_i(\theta_t)$$

- $\eta$: learning rate — the **only** hyperparameter. No momentum (that is a
  separate slide/section and already shipped as `sgd_momentum`).
- **State buffers: 0.** Per-step cost $\mathcal{O}(n)$ beyond the gradient.
- No warmup/schedule interaction defined in the deck beyond the manual
  LR-tuning heuristics above (the repo's schedulers already cover this).
- Weight decay: not part of the lecture's update; the implementation keeps
  torch-SGD semantics (`weight_decay` = L2 added to the gradient) via the
  shared `wd_*` suffixes, default 0.0 — documented as an implementation
  extension, not paper content.

### RMSProp (Lecture 6e — "rmsprop: Divide the gradient by a running average of its recent magnitude")

Slides text (verbatim, ligatures expanded):

> "rprop is equivalent to using the gradient but also dividing by the size of
> the gradient. — The problem with mini-batch rprop is that we divide by a
> different number for each mini-batch. So why not force the number we divide
> by to be very similar for adjacent mini-batches?
>
> **rmsprop: Keep a moving average of the squared gradient for each weight**
>
> **MeanSquare(w, t) = 0.9 MeanSquare(w, t−1) + 0.1 (∂E/∂w(t))²**
>
> Dividing the gradient by √MeanSquare(w, t) makes the learning work much
> better (Tijmen Tieleman, unpublished)."

**Update rule** exactly as printed:

$$\mathrm{MS}(w, t) = 0.9\,\mathrm{MS}(w, t-1) + 0.1 \left(\frac{\partial E}{\partial w}(t)\right)^{2}$$
$$w(t+1) = w(t) - \eta \, \frac{g_t}{\sqrt{\mathrm{MS}(w, t)}}$$

Implementation notes derived from the slides (to be documented in the annex):

- The 0.9/0.1 split is the EMA form torch implements as
  $v_t = \alpha v_{t-1} + (1-\alpha) g_t^2$ with $\alpha = 0.9$ — **not**
  torch's constructor default `alpha=0.99`. The slides' 0.9 is the default we
  register (`rmsprop-alpha: 0.9`).
- $v_0 = 0$ initialization (slides define no other start; first-step divisor
  becomes $\sqrt{0.1}\,|g_1|$).
- **No bias correction** (that arrives with Adam), **no first moment / no
  momentum** in the base method ("Combining rmsprop with standard momentum —
  Momentum does not help as much as it normally does. Needs more
  investigation." — extensions are explicitly left open and **out of scope**).
- $\varepsilon$: not in the slides; added as the standard numerical guard in
  the denominator ($\sqrt{\mathrm{MS}} + \varepsilon$, torch placement). Must
  be documented as an implementation guard, not paper content.
- **State buffers: 1** (`square_avg`, same shape as the parameter).
  Per-step cost $\mathcal{O}(n)$ beyond the gradient.
- Hyperparameters: $\eta$ (lr), decay $\alpha = 0.9$ (slides), $\varepsilon$
  (implementation guard). Both global; per-group lr/wd/eps arrive through the
  repo's shared suffix mechanism.

### Category

Both are classical baselines. In `docs/architecture.md` §8.1: `sgd` joins the
**SGD** family row (with `sgd_momentum`); `rmsprop` joins the **Adaptive**
family row (it is the founding adaptive-learning-rate method). In the paper
narrative both go to the "classical baselines" list of
`05-optimizer-families.tex` and the "Classical & Adaptive Baselines" block of
`tab:all-optimizers`. No new family bucket is needed (stays 8 families).

### BibTeX data (already in the repo — verified)

- `hinton_csc321_2012` — `docs/bibliography/optimizers.bib:492`
  (Hinton, Srivastava, Swersky, 2012, CSC321 course slides, the exact URL).
- `tieleman_hinton2012` — `docs/bibliography/optimizers.bib:516`
  (Tieleman & Hinton, 2012, "RMSProp: Divide the Gradient by a Running
  Average of Its Recent Magnitude", Lecture 6e).
- Secondary lineage cites already present: `robbins_monro_1951`,
  `rumelhart1986learning`, `bottou_2010`, `ruder_2016`, `duchi_adagrad_2011`,
  `sutskever2013importance`, `ibm_sgd_2025`, `ghojogh_ghodsi_2024`.

## 3. Decisions

| Decision | Choice | Evidence |
|---|---|---|
| Registry names | `sgd`, `rmsprop` (snake_case) | `registry.names(ComponentFamily.OPTIMIZER)` has neither (36 names, `src/registry/builtin/optimizer.py:39-436`); no collision with `sgd_momentum` |
| Class names | `SGDOptimizer` (matches `AdamWOptimizer`/`RAdamOptimizer` style), `RMSProp` (paper capitalization, avoids clashing with `torch.optim.RMSprop` import) | `src/model/optimizer/__init__.py` |
| Closest peer | `src/model/optimizer/sgd_momentum.py` — a 45-line `torch.optim.SGD` subclass that fixes `dampening=0` and forwards everything else. Both new modules mirror that pattern: subclass the torch built-in, pin the paper's shape, cite the deck in the docstring | `sgd_momentum.py:16-45` |
| `sgd` spec shape | No `param_bindings` (like `adamw`), `group_policy="sgd"` (only `lr_*`/`wd_*` routed — exactly like `sgd_momentum`, `builtin/optimizer.py:375-385`), `supports_shared_per_group_suffixes=True` (default) so `betas_*`/`eps_*` remain accepted-but-unused, `sbert_allowed=True`, `closure_based=False` | factory `_build_param_groups_for_optimizer` `src/model/optimizer/factory.py:155` |
| `rmsprop` spec shape | `param_bindings={"alpha": ParamBinding(as_float, 0.9)}`; default `group_policy="adaptive"` so the shared `eps_<group>` suffix is routed into each group dict (torch RMSprop reads `group["eps"]`); `betas_*` accepted-unused (harmless extra key in the group dict, precedent: every adaptive-policy optimizer); `supports_shared_per_group_suffixes=True`; `sbert_allowed=True`; `closure_based=False`. No `custom_build` — the constructor is not special | factory `factory.py:171-180`, `_build_adafactor` is the only `custom_build` precedent (`builtin/optimizer.py:24-36`) |
| torch reuse | Subclass `torch.optim.SGD` (momentum=0.0, dampening=0.0, nesterov=False) and `torch.optim.RMSprop` (momentum=0.0, centered=False, alpha default 0.9). torch's RMSprop step is algebraically the slides' equation with $\alpha=0.9$; the $\varepsilon$ placement ($\sqrt{v}+\varepsilon$) is the only deviation and is documented | torch 2.7.0 stable API; `SGDMomentum` precedent |
| lr constructor default | `1e-3` for both (repo peer convention, `sgd_momentum.py:37`); the factory always injects per-group `lr` anyway (`factory.py:160-163`) | note: torch RMSprop's own default is 1e-2 — divergence documented |
| Category | §8.1: `sgd` → SGD row; `rmsprop` → Adaptive row. Paper: both in "classical baselines" / "Classical & Adaptive Baselines". Families stay 8 | `docs/architecture.md:875-885`, `annex-summary-tables.tex:131` |
| Example presets | `configs/examples/es_arch_hybrid_sgd.yaml` and `configs/examples/es_arch_hybrid_rmsprop.yaml`, mirroring `es_arch_hybrid_muon2.yaml` (the smallest recent per-optimizer example: mlm task, hybrid `layer_pattern`, one optimizer block) | naming convention `es_arch_<topic>_<optimizer>.yaml` |
| Census bump | **36 → 38** optimizers, 8 families (verified today: `OPTIMIZER_SPECS` count = 36; `README.md:26` `**36**`; `INSTALL.md:203`; `docs/architecture.md:26` and `:875` `(36)`; EN `annex-summary-tables.tex:113` "all 36"; ES `annex-summary-tables.tex:4` "36 optimizadores" and `:113` "los 36"; EN `05-optimizer-families.tex:4` "thirty-six"; ES `05-familias-optimizadores.tex:4` "treinta y seis") | grep evidence above |

### Suffix table (generated from the spec, documented in `_optimizer.yaml` prose)

| Key | Type | Default | Coercion | Scope | EN description | ES description |
|---|---|---|---|---|---|---|
| `sgd-lr_<group>` | float | group default | (shared) | per-group | Learning rate per group (only hyperparameter of the method) | Tasa de aprendizaje por grupo (único hiperparámetro del método) |
| `sgd-wd_<group>` | float | 0.0 | (shared) | per-group | L2 weight decay folded into the gradient (implementation extension; the slides define none) | Weight decay L2 plegado en el gradiente (extensión de implementación; las diapositivas no definen ninguno) |
| `sgd-betas_*`, `sgd-eps_*` | — | — | (shared) | accepted-unused | Accepted by the shared-suffix contract; not used (no moments) | Aceptados por el contrato de sufijos compartidos; no se usan (sin momentos) |
| `rmsprop-lr_<group>` | float | group default | (shared) | per-group | Learning rate per group | Tasa de aprendizaje por grupo |
| `rmsprop-alpha` | float | **0.9** | `as_float` | global | EMA decay of the squared-gradient running average; the slides' exact 0.9/0.1 split (torch default is 0.99) | Decaimiento EMA del promedio móvil del gradiente al cuadrado; la división 0.9/0.1 exacta de las diapositivas (el valor por defecto de torch es 0.99) |
| `rmsprop-eps_<group>` | float | 1e-8 | (shared) | per-group | Denominator epsilon (numerical guard; not in the slides) | Epsilon del denominador (guarda numérica; no está en las diapositivas) |
| `rmsprop-wd_<group>` | float | 0.0 | (shared) | per-group | L2-coupled weight decay added to the gradient (torch semantics; the slides define none) | Weight decay L2 acoplado añadido al gradiente (semántica de torch; las diapositivas no definen ninguno) |
| `rmsprop-betas_<group>` | tuple | — | (shared) | accepted-unused | Accepted by the shared-suffix contract; unused (no first moment) | Aceptados por el contrato de sufijos compartidos; sin uso (sin primer momento) |

`supports_shared_per_group_suffixes = True` for both (the norm; maps straight
onto `OptimizerSpec.supports_shared_per_group_suffixes`, default).

## 4. File-by-file plan (execution order)

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/sgd.py` *(new)* | `"""Vanilla mini-batch SGD (Hinton, Srivastava & Swersky, 2012).` module docstring citing `https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf` (Lecture 6a) + Robbins–Monro lineage. `from __future__ import annotations`; `from torch.optim import SGD`. `class SGDOptimizer(SGD):` with Google-style class/method docstrings (Args: params, lr=1e-3, weight_decay=0.0; note "State buffers: none — the zero-state baseline"). `__init__` calls `super().__init__(params, lr=lr, momentum=0.0, dampening=0.0, weight_decay=weight_decay, nesterov=False)` — identical shape to `SGDMomentum.__init__` (`sgd_momentum.py:37-45`) with momentum pinned off. CPU-safe (no `.cuda()`). |
| 2 | `src/model/optimizer/rmsprop.py` *(new)* | Module docstring quoting the lecture: `MeanSquare(w, t) = 0.9 MeanSquare(w, t−1) + 0.1 (∂E/∂w(t))²` (Lecture 6e, Tieleman & Hinton 2012, `tieleman_hinton2012`). `from torch.optim import RMSprop`. `class RMSProp(RMSprop):` — `__init__(self, params, lr=1e-3, alpha=0.9, eps=1e-8, weight_decay=0.0)` validates `0.0 < alpha < 1.0`, `0.0 <= eps`, `0.0 <= lr` (ValueError like `anon.py:89-96`) and calls `super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0.0, centered=False)`. Docstring documents: 1 state buffer (`square_avg`), no bias correction, momentum/centered are deliberate non-goals per the slides, ε is an implementation guard. No `step()` override needed (torch's step *is* the slides' equation at α=0.9). |
| 3 | `src/model/optimizer/__init__.py` | Insert `from .rmsprop import RMSProp` between `from .radam import RAdamOptimizer` (line 28) and `from .root import ROOT` (line 29); insert `from .sgd import SGDOptimizer` immediately before `from .sgd_momentum import SGDMomentum` (line 31). Add `"RMSProp"` and `"SGDOptimizer"` to `__all__` in the same alphabetical slots (`"ROOT"` → `"RMSProp"` → `"SGDOptimizer"` → `"SGDMomentum"`). |
| 4 | `src/registry/builtin/optimizer.py` | Two specs, alphabetical like the rest of the tuple: **`rmsprop`** inserted between the `radam` spec (ends line 352) and the `root` spec (starts line 353): `OptimizerSpec(name="rmsprop", builder="src.model.optimizer.rmsprop:RMSProp", description="Classical adaptive baseline — Divides the gradient by a running average of its recent squared magnitude (slides: MeanSquare = 0.9·MS + 0.1·g²); one state buffer, no bias correction (source: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)", description_es="Línea base adaptativa clásica — Divide el gradiente por un promedio móvil de su magnitud cuadrada reciente (diapositivas: MeanSquare = 0.9·MS + 0.1·g²); un búfer de estado, sin corrección de sesgo (fuente: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)", param_bindings={"alpha": ParamBinding(as_float, 0.9)})` — default `group_policy="adaptive"` routes the shared `eps_*`. **`sgd`** inserted immediately before the `sgd_momentum` spec (line 375): `OptimizerSpec(name="sgd", builder="src.model.optimizer.sgd:SGDOptimizer", description="Pure mini-batch SGD baseline — w ← w − η·∇E; no momentum, zero optimizer state (source: CSC321 Lecture 6a, https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)", description_es="Línea base SGD de mini-lotes puro — w ← w − η·∇E; sin momentum, sin estado del optimizador (fuente: CSC321 Lecture 6a, https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)", group_policy="sgd")` — no `param_bindings`. `references=()` for both (not arXiv; the URL lives in the description, same as `sgd_momentum`'s DOI does). **This single declaration generates the enum, the `by_optimizer` entry and the `^sgd-`/`^rmsprop-` `if/then` prefix rules** (`src/registry/schema_assembly.py:_inject_optimizers`, line 139). No `factory.py` edit exists — `OPTIMIZER_REGISTRY`/`OPTIMIZER_ALLOWED_KEYS` are derived read-only views (`factory.py:129-135`) and `build_optimizer` has no per-optimizer branches (`factory.py:190-228`). |
| 5 | `src/schema/_optimizer.yaml` | Pure documentation (validation is registry-derived): (a) EN use-case table — add after the `adam_pm` row (line 36): ``| `sgd` | Pure SGD baseline | Mini-batch update, no momentum, zero optimizer state (source: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) |`` and ``| `rmsprop` | Classical adaptive baseline | Divides the gradient by a running average of its recent squared magnitude; one state buffer, no bias correction (source: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) |``; (b) same two rows in the ES table after line 100 (`adam_pm` row), translated; (c) EN "Optimizer-specific suffixes" list — add a bullet after the `anon` bullet (line 185): `- rmsprop: alpha, plus the shared lr_*/wd_*/eps_*. alpha is the EMA decay of the squared-gradient running average — the slides' exact 0.9/0.1 split (default 0.9; torch's own default is 0.99); eps_* is the denominator guard (not in the slides); wd_* is L2-coupled into the gradient (torch semantics); betas_* accepted but unused` (`sgd` needs no bullet — it falls under the existing "Others: no specific suffixes (only shared ones)" line 199); (d) same bullet in the ES list after line 234; (e) append `` `sgd`/`rmsprop` (paper: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)`` to the EN `**References:**` run (lines 79-93) and to the ES one (lines 144-158). Without (a)+(b) the assembler appends 2-column fallback rows from the spec descriptions — acceptable but uglier; with them the 3-column style is preserved. |
| 6 | `configs/examples/es_arch_hybrid_sgd.yaml` *(new)* | Copy `es_arch_hybrid_muon2.yaml` (model block verbatim: `frankenstein_text_encoder`, 768/12, `layer_pattern: [retnet, titan, mamba, ode]`, mlm training block) and replace the optimizer block: `optimizer_class: sgd`, parameters = `sgd-lr_{embeddings,pos,attention,ffn,norms,head,mhc,router,residual,other}` at 10× the muon2 values (plain SGD wants ~1e-5…1e-4 here; e.g. embeddings 1e-5, pos 2e-5, attention 3e-5, ffn 2e-5, norms 5e-5, head 2e-5, mhc 3e-5, router 2e-5, residual 2e-5, other 2e-5), `sgd-wd_*` 0.01 (0.001 norms), no `betas_*`/`eps_*` (accepted-unused; keep the file minimal), `csv_log_path: training_metrics_es_arch_hybrid_sgd.yaml` → `…_sgd.csv`. Must pass `load_training_config` (auto-globbed by `tests/test_yaml_examples.py`). |
| 7 | `configs/examples/es_arch_hybrid_rmsprop.yaml` *(new)* | Same skeleton; `optimizer_class: rmsprop`; parameters = `rmsprop-lr_*` at the muon2 scale (1e-6…5e-6 — RMSProp normalizes the step, so the adaptive scale applies), `rmsprop-wd_*` 0.01/0.001, `rmsprop-eps_*: 1e-8`, plus `rmsprop-alpha: 0.9` (explicit, documents the slides' split), `csv_log_path: training_metrics_es_arch_hybrid_rmsprop.csv`. |
| 8 | `configs/README.md` | (a) "Optimizer Chooser" (line 139): add `- sgd` and `- rmsprop` bullets near `- sgd_momentum` (line 142) with one-line guidance (sgd: zero-state reference baseline, needs the highest LR of the catalog; rmsprop: 2012 classical adaptive baseline, the Adam ancestor); (b) "Optimizer Parameters Reference" (line 183): add `- rmsprop: alpha` styled like `- sgd_momentum: momentum, nesterov` (line 200) with the same one-line explanation as the schema bullet; `sgd` needs no row (shared suffixes only); (c) example list if one exists in that file — verify at implementation time. |
| 9 | `full_tests/run_e2e.py` | Append `"sgd", "rmsprop"` to the `OPTIMIZERS` list (lines 289-295, keeping it inside the same bracket, e.g. after `"sgd_momentum"` for the SGD sibling and in rough alphabetical position for `rmsprop`). **No `_with_optimizer` block needed**: both use only safe defaults (no closure, no grad-accum constraint, no RNG), and the generated keys are auto-filtered by `OPTIMIZER_ALLOWED_KEYS` (`run_e2e.py:661-680` docstring). |
| 10 | `tests/test_optimizer_factory.py` | (a) Add `"sgd"`, `"rmsprop"` to the `required` set in `test_registry_contains_all_required` (line 31). (b) New `BuildOptimizerPlainSGDTests` class mirroring `BuildOptimizerSGDTests` (lines 76-90): `test_builds_without_error`, `test_step_runs` (backward + step on a 4×4 param), `test_zero_state_buffers` (`self.assertEqual(len(opt.state), 0)` after a step), `test_momentum_is_off` (`opt.param_groups[0]["momentum"] == 0.0`), `test_momentum_suffix_rejected` (`build_optimizer("sgd", …, {"sgd-momentum": 0.9})` raises ValueError — momentum is not an allowed key for `sgd`). (c) New `BuildOptimizerRMSPropTests`: `test_builds_without_error`, `test_step_runs`, `test_alpha_default_is_slides_value` (`opt.param_groups[0]["alpha"] == 0.9`), `test_custom_alpha` (`{"rmsprop-alpha": 0.99}` → 0.99), `test_per_group_eps_override` (`{"rmsprop-eps_other": 1e-6}` → `opt.param_groups[0]["eps"] == 1e-6`), `test_invalid_alpha_raises` (alpha=1.5 → ValueError from the constructor), `test_momentum_is_off` (`param_groups[0]["momentum"] == 0.0`). `tests/test_optimizer_base.py` needs no change (it tests the `base.py` helpers, not per-optimizer behavior). |
| 11 | `docs/bibliography/optimizers.bib` | **No change.** `hinton_csc321_2012` (line 492) and `tieleman_hinton2012` (line 516) already exist with the exact source URL, in the repo's established style. All lineage cites needed by the annexes are also present (`robbins_monro_1951:455`, `rumelhart1986learning:445`, `bottou_2010:465`, `ruder_2016:475`, `duchi_adagrad_2011:507`, `sutskever2013importance:523`). |
| 12 | `docs/paper/figures/optimizers/sgd.mmd`, `rmsprop.mmd` *(new)* | Two Mermaid flowcharts in the exact style of `sgd-momentum.mmd` (init header `%%{init: {"theme":"neutral",…}}%%`, `flowchart TD`, carry edges `-.->`): **sgd**: `g → {wd?} → gwd → upd["θₜ₊₁ ← θₜ − η·gₜ"]` with no state carry (contrast with the momentum figure — no `prev` node, no carry edges); **rmsprop**: `g → sq["v ← 0.9·v + 0.1·g²"] → div["θₜ₊₁ ← θₜ − η·gₜ/√(v+ε)"]` with `v` carried via `.-.` edges. Build at implementation time with `bash docs/paper/figures/build-figures.sh both` (produces the `.pdf` for LaTeX and the `.svg` the mirror consumes; needs pnpm + a Chromium — see Risks). **Fallback**: if the Mermaid CLI cannot run, drop the `\begin{figure}` floats from rows 13/16 and add the two optimizers to the figure-debt note; the annex paragraphs stand alone. |
| 13 | `docs/paper/appendices/annex-9-optimizers_adaptive.tex` | (a) New `\paragraph{Vanilla SGD (Mini-Batch Gradient Descent).}` **before** line 2 (`\paragraph{SGD with Momentum.}`): figure float `\fitfigure{\includegraphics{figures/optimizers/sgd.pdf}}` with `\label{fig:opt-sgd}`; body stating the deck's motivation (redundant datasets → mini-batch gradients; mini-batches beat online for GPU matmuls and class balance); equation `\theta_{t+1} = \theta_t - \eta g_t,\quad g_t = \frac{1}{|B|}\sum_{i\in B}\nabla_\theta E_i`; `algorithm` float (Require: θ₀, η, λ; loop: g ← ∇f(θ); if wd>0: g ← g + λθ; θ ← θ − ηg) styled like `alg:sgd-momentum`; Cons (single global rate, LR-schedule dependence — the deck's own heuristics; slow on ill-conditioned ravines) / Pros (zero optimizer state — the memory floor; fully reproducible; the reference point every other entry is measured against); cites `\citep{hinton_csc321_2012}` primary + `\citep{robbins_monro_1951,bottou_2010,rumelhart1986learning}` lineage. (b) New `\paragraph{RMSProp.}` **after** the SGD-with-Momentum block (before `\paragraph{Adam and AdamW.}`, line 47): figure float `figures/optimizers/rmsprop.pdf`, `\label{fig:opt-rmsprop}`; body quoting the rprop motivation verbatim ("The problem with mini-batch rprop is that we divide by a different number for each mini-batch. So why not force the number we divide by to be very similar for adjacent mini-batches?") and the two equations exactly as §2 above (`v_t = 0.9 v_{t-1} + 0.1 g_t^2`, `\theta_{t+1} = \theta_t - \eta g_t / \sqrt{v_t + \varepsilon}` with a note that ε is an implementation guard absent from the slides); algorithm float; Cons (no bias correction → early-step bias; hyperparameters still need tuning; superseded by Adam for transformers) / Pros (one buffer; per-coordinate scaling; robust to gradient-scale heterogeneity; the direct ancestor of Adam/AdamW — AdaGrad's diagonal idea made mini-batch-safe); cites `\citep{tieleman_hinton2012}` primary + `\citep{hinton_csc321_2012,duchi_adagrad_2011}` lineage. Matches the file's existing `\paragraph` / figure / `algorithm` / Cons / Pros rhythm (see lines 1-46). |
| 14 | `docs/paper/appendices/annex-summary-tables.tex` | Under the `\textbf{Classical \& Adaptive Baselines}` header (line ~131), before the `SGD+\allowbreak Momentum` row: `SGD & Classical & 0 & $\mathcal{O}(n)$ & lr, wd & \citep{hinton_csc321_2012} \\`; and after `SGD+Momentum` / before `AdamW`: `RMSProp & Adaptive & 1 ($v$) & $\mathcal{O}(n)$ & lr, decay $\alpha$ (0.9), eps, wd & \citep{tieleman_hinton2012} \\`. Census: line ~113 `all 36 supported optimizers` → `all 38 supported optimizers`. |
| 15 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: `thirty-six optimizer families in eight categories: classical baselines (SGD with momentum, AdamW, RAdam)` → `thirty-eight optimizer families in eight categories: classical baselines (vanilla SGD, SGD with momentum, RMSProp, AdamW, RAdam)`. No other sentence in the file needs the count (verified: single occurrence). |
| 16 | `docs/paper-es/appendices/annex-9-optimizadores_adaptativos.tex` | Faithful Spanish translations of both paragraphs from row 13, inserted at the mirrored anchors: `\paragraph{SGD Vainilla (Descenso de Gradiente por Mini-lotes).}` before line 2 (`\paragraph{SGD con Momentum.}`), `\paragraph{RMSProp.}` before `\paragraph{Adam y AdamW.}` (line 47). Figures point at `../paper/figures/optimizers/sgd.pdf` / `rmsprop.pdf` (ES convention, line 5). Equations identical to EN (math is not translated); labels `fig:opt-sgd-es`/`fig:opt-rmsprop-es`, `alg:sgd-es`/`alg:rmsprop-es` if the ES file uses distinct labels — copy whatever the file's existing convention is. Contras/Ventajas translated from the EN Cons/Pros. |
| 17 | `docs/paper-es/appendices/annex-summary-tables.tex` | Two mirrored rows under the ES "Classical & Adaptive Baselines" block (find the `SGD+\allowbreak Momentum` row): `SGD & Clásico & 0 & $\mathcal{O}(n)$ & lr, wd & \citep{hinton_csc321_2012} \\` and `RMSProp & Adaptativo & 1 ($v$) & $\mathcal{O}(n)$ & lr, decaimiento $\alpha$ (0.9), eps, wd & \citep{tieleman_hinton2012} \\`. Census: line 4 `36 optimizadores` → `38 optimizadores`; line 113 `los 36 optimizadores` → `los 38 optimizadores`. |
| 18 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: `treinta y seis familias de optimizadores en ocho categorías: líneas base clásicas (SGD con momentum, AdamW, RAdam)` → `treinta y ocho familias de optimizadores en ocho categorías: líneas base clásicas (SGD vainilla, SGD con momentum, RMSProp, AdamW, RAdam)`. |
| 19 | `docs/architecture.md`, `README.md`, `INSTALL.md`, summary-family tables | Census pass (grep the old count strings before editing): **`docs/architecture.md`** line 26 `| Optimizer families | **36** |` → `**38**`; line 875 heading `### 8.1 Optimizers (36)` → `(38)`; §8.1 family table — SGD row becomes ``| SGD | `sgd`, `sgd_momentum` | Baselines; plain mini-batch update and momentum/nesterov |`` and the Adaptive row gains `rmsprop` first in its class list with the character cell extended `…; `rmsprop` is the classical squared-gradient running average (slides' 0.9/0.1 split), the Adam ancestor`. **`README.md`** line 26 counts row `**36**` → `**38**`; line 36 `36 optimizer families` → `38`; line 126 `36 across 8 categories` → `38 across 8 categories`; line 197 state-profile sentence `from 0 (EA4LLM, gradient-free)` → `from 0 (SGD, EA4LLM gradient-free)`. **`INSTALL.md`** line 203 `36 optimizers` → `38 optimizers`. **`annex-9-optimizers_summary.tex`** line 8 `Classical baseline & SGD, AdamW, RAdam` → `Classical baseline & SGD, vanilla SGD, RMSProp, AdamW, RAdam`; **`annex-9-optimizadores_resumen.tex`** line 7 mirrored (`SGD, SGD vainilla, RMSProp, AdamW, RAdam`). |
| 20 | `docs/schema.assembled.yaml` | Regenerate: `uv run --extra cpu --extra train python -m src.registry.snapshot --out docs/schema.assembled.yaml`. Verify the diff contains the two new enum entries, two new `by_optimizer` entries and two new `if/then` rules — and nothing else. This snapshot is what the website mirror consumes. |

**Hard rules restated** (bind every step above): the schema is the contract —
`additionalProperties: false` holds everywhere and the optimizer enum, the
`by_optimizer` reference and the `^<name>-` `if/then` prefix rules are
*generated from the `OptimizerSpec`* by `src/registry/schema_assembly.py`
(`_inject_optimizers`, line 139) — never hand-edit them, and never read or
implement an undocumented key. Optimizer keys are prefixed
`<optimizer>-<group>_<param>` / `<optimizer>-<param>`; the allowed set comes
from `factory.OPTIMIZER_ALLOWED_KEYS` and unknown keys raise. Every component
is a declarative `ComponentSpec`; the registry is the single source of truth
(D13/D14). The papers are the theory source of truth (D11): the annex
formulations above must match the slide deck verbatim, and `paper-es` must
translate them faithfully. Examples are smoke-tested (D8): every new YAML
must validate and load.

## 5. Ordered execution checklist (implementation time — do NOT run during planning)

Implementation (files 1-9):

- [ ] `src/model/optimizer/sgd.py`
- [ ] `src/model/optimizer/rmsprop.py`
- [ ] `src/model/optimizer/__init__.py` (imports + `__all__`)
- [ ] `src/registry/builtin/optimizer.py` (two specs, alphabetical slots)
- [ ] `src/schema/_optimizer.yaml` (EN+ES table rows, EN+ES suffix bullet, EN+ES References lines)
- [ ] `configs/examples/es_arch_hybrid_sgd.yaml`
- [ ] `configs/examples/es_arch_hybrid_rmsprop.yaml`
- [ ] `configs/README.md`
- [ ] `full_tests/run_e2e.py` (`OPTIMIZERS` list)

Tests (file 10):

- [ ] `tests/test_optimizer_factory.py` (required set + two test classes)

Docs (files 11-19): bibliography (verify, no edit) → figures (`.mmd` +
`build-figures.sh both`, fallback omit) → EN annex → EN summary table → EN
family section → ES annex → ES summary table → ES family section → census
pass (`architecture.md`, `README.md`, `INSTALL.md`, both summary-family
tables) → regenerate `docs/schema.assembled.yaml` (file 20).

PDFs (preconditions verified: `pdflatex`, `bibtex`, `pandoc`, `node` all
present on this machine):

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

Fallback: if `pdflatex`/`bibtex` are unavailable on the implementing machine,
leave the sources updated and note that the rebuild must happen on a TeX
machine before the next release (the website sync copies prebuilt PDFs only).
Post-compilation verification: no `Fatal error`, no `Citation .* undefined`
(both new `\citep{hinton_csc321_2012}` / `\citep{tieleman_hinton2012}` must
resolve — keys pre-exist so `paper.bbl` should pick them up), `paper.pdf` /
`paper-es.pdf` newer than the edited `.tex`/`.bbl`, and
`git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf` changes.

Gates (CPU-first; quote, don't improvise):

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_registry_core.py tests/test_registry_binding.py tests/test_schema_assembly.py -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# enumerate the optimizer combos (no training) and then a fast smoke
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --list-combos | grep -F sgd
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --list-combos | grep -F rmsprop
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --only sgd
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --only rmsprop
# (--realonall is the default: the sweep trains on the real Borges corpus;
#  --no-realonall is the offline fallback; --limit N caps the sweep)
```

Website sync → §6. Commit/push → §7.

## 6. Website sync (planned, not executed)

Run `/sync-website` semantics manually at implementation time (see
`.opencode/command/sync-website.md`):

1. Regenerate the snapshot **first**: `uv run --extra cpu --extra train
   python -m src.registry.snapshot --out docs/schema.assembled.yaml` (file 20).
2. Copy into `erickfmm.github.io/frankenstein-transformer/`:
   `docs/schema.assembled.yaml` → `schema.yaml`; `src/schema/**` →
   `schema/**`; `src/registry/**` → `registry/**` (never `__pycache__`);
   `configs/examples/**` (the two new YAMLs) + `configs/README.md` →
   `examples/examples/**` + `examples/README.md`; `docs/paper/**` (new
   `.tex`, rebuilt `paper.bbl`/`paper.pdf`, figures `sgd.svg`/`rmsprop.svg`
   **and the `.mmd` sources** — the sync copies `*.svg` only, verify against
   the live file map); `docs/paper-es/**`; `docs/bibliography/**`;
   `docs/architecture.md`.
3. Rebuild the Paper-tab HTML: `flatten_tex.py --web` + `pandoc` +
   `postprocess_html.py`; refresh `paper/en.pdf` / `paper/es.pdf`.
4. Validation Gate: `full_tests/param_count_check.py` → `0 failures`
   (regenerates `full_tests/param_truth.json` — changes are intentional);
   `full_tests/param_estimate_check.mjs … --tol 0.01`; re-run
   `bash .opencode/command/sync-website-state.sh` and confirm every probe
   (`schema-refs`, `example-yaml`, `cli-flags`, `paper-figures-en`,
   `paper-figures-es`, estimator) reports PASS.
5. The mirror stays uncommitted until the commit phase below.

## 7. Commit plan (planned, not executed)

`/make-a-commit` conventions: English, conventional tag, structured body;
never stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json`
changes are intentional. One main-repo commit for both optimizers (the
census/table edits are shared; splitting would leave the docs half-counted).

1. Main repo — stage exactly:
   `src/model/optimizer/sgd.py src/model/optimizer/rmsprop.py
   src/model/optimizer/__init__.py src/registry/builtin/optimizer.py
   src/schema/_optimizer.yaml configs/examples/es_arch_hybrid_sgd.yaml
   configs/examples/es_arch_hybrid_rmsprop.yaml configs/README.md
   full_tests/run_e2e.py tests/test_optimizer_factory.py
   docs/paper/figures/optimizers/sgd.{mmd,pdf,svg}
   docs/paper/figures/optimizers/rmsprop.{mmd,pdf,svg}
   docs/paper/appendices/annex-9-optimizers_adaptive.tex
   docs/paper/appendices/annex-summary-tables.tex
   docs/paper/sections/05-optimizer-families.tex
   docs/paper/paper.bbl docs/paper/paper.pdf
   docs/paper-es/appendices/annex-9-optimizadores_adaptativos.tex
   docs/paper-es/appendices/annex-summary-tables.tex
   docs/paper-es/sections/05-familias-optimizadores.tex
   docs/paper-es/paper-es.bbl docs/paper-es/paper-es.pdf
   docs/architecture.md README.md INSTALL.md docs/schema.assembled.yaml`
   (drop the figure/PDF paths if the fallback in row 12 applied).
   Message: `feat(optimizer): add sgd and rmsprop (CSC321 lecture 6)` with a
   body covering: implementation notes (torch subclasses, slides' 0.9 alpha),
   new suffixes (`rmsprop-alpha`), the two registry specs (enum/by_optimizer/
   if-then generated), docs (EN+ES annexes, summary tables, family sections,
   census 36→38, rebuilt PDFs), the two example YAMLs, the e2e list, the
   factory tests, and the regenerated `docs/schema.assembled.yaml`. Then
   `git push origin HEAD:main`.
2. Mirror — `git -C erickfmm.github.io add …` (the paths synced in §6 plus
   `full_tests/param_truth.json` if the gate updated it here) and commit
   `docs(site): sync sgd and rmsprop optimizers (CSC321 lecture 6)`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

## 8. Verification matrix

| Gate | Command | Expected result |
|---|---|---|
| Targeted pytest | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_registry_core.py tests/test_registry_binding.py tests/test_schema_assembly.py -v --tb=short -p no:warnings` | All pass; `test_registry_contains_all_required` sees `sgd`+`rmsprop`; `test_optimizer_prefixes_match_conditional_rules` auto-covers the two new prefixes |
| Full suite | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | Green (84+ files); no regression in SBERT/Streamlit/training tests |
| e2e combos | `… run_e2e.py --category opt --list-combos \| grep -F sgd` / `grep -F rmsprop` | Both optimizers appear in the enumerated combos |
| e2e smoke | `… run_e2e.py --category opt --only sgd` and `--only rmsprop` | Each trains a short real-data run without unknown-key errors (keys auto-filtered by `OPTIMIZER_ALLOWED_KEYS`) |
| Snapshot diff | `python -m src.registry.snapshot --out docs/schema.assembled.yaml && git diff docs/schema.assembled.yaml` | Exactly: 2 enum entries, 2 `by_optimizer` entries, 2 `if/then` rules |
| PDF EN/ES | pdflatex+bibtex sequence above | No `Fatal error`, no `Citation .* undefined`, PDFs newer than sources, only intended files dirty |
| Param count | `full_tests/param_count_check.py` | `0 failures` (regenerates `param_truth.json`; both optimizers add no parameters — the model is unchanged, so counts must be identical apart from noise) |
| Estimator | `full_tests/param_estimate_check.mjs … --tol 0.01` | Within tolerance (unchanged model → unchanged estimate) |
| State probes | `bash .opencode/command/sync-website-state.sh` | All probes PASS (`schema-refs`, `example-yaml`, `cli-flags`, `paper-figures-en/es`, estimator) |

## 9. Risks, assumptions and open questions

- **No arXiv ID** (template deviation): the plan slug replaces
  `<arxiv-id>`; the spec `references=()` stays empty because the tuple is
  arXiv-shaped — the source URL lives in the descriptions, mirroring how
  `sgd_momentum` carries its DOI. Assumption to verify: nothing downstream
  assumes `references` is non-empty (the assembler only reads it for docs
  hints; grep `references` in `schema_assembly.py` at implementation time).
- **`rmsprop` lr default 1e-3 vs torch 1e-2**: the factory always injects
  per-group `lr`, so the constructor default is cosmetic; documented anyway.
- **`eps` placement**: torch RMSprop uses `√v + ε` (outside the root); the
  slides have no ε. Some implementations use `√(v+ε)`. We keep torch's —
  documented in the annex as an implementation guard.
- **`betas_*` on `rmsprop` groups**: with the default adaptive policy the
  factory writes a `betas` key into each group dict; torch RMSprop ignores
  unknown group keys — assumption to verify with
  `BuildOptimizerRMSPropTests.test_builds_without_error` on the pinned torch
  2.7.0.
- **Figures pipeline** needs pnpm + Chromium (`build-figures.sh` auto-detects
  google-chrome/chromium). If unavailable → fallback in row 12 (omit the two
  figure floats; note as figure debt).
- **Naming collisions**: `sgd` vs `sgd_momentum` — distinct registry names,
  distinct modules; `sgd-momentum` is deliberately *not* an allowed suffix
  for `sgd` (negative test included). `RMSProp` class vs `torch.optim.RMSprop`
  import — distinct casing, no shadowing.
- **Task compatibility**: no `training.task` constraints (unlike
  audio/vision-only components); both are `sbert_allowed=True`,
  `closure_based=False`, no grad-accumulation requirement — every task and
  model class can select them.
- **Open question (not blocking)**: the slides also sketch RMSProp+momentum
  and RMSProp+Nesterov ("needs more investigation") — explicitly **out of
  scope**; the user asked for the basic versions only.
- **Assumption**: `tests/test_optimizer_base.py` and the Streamlit picker
  need no edits (registry-derived) — verify at implementation time, do not
  pre-edit.
- **`param_truth.json`**: `param_count_check.py` may rewrite it with
  identical values or timestamp noise — the commit plan treats its change as
  intentional if it appears.

## 10. Out of scope

No code, schema, preset, test, doc, PDF, mirror or git change was made by
this planning command. This document is the only file written. Nothing in
`src/`, `configs/`, `tests/`, `docs/`, `full_tests/`, `erickfmm.github.io/`
or git history was touched.
