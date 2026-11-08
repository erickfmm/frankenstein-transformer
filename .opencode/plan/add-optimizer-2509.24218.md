# Plan: add Conda optimizer (arXiv:2509.24218)

**Paper:** *Conda: Column-Normalized Adam for Training Large Language Models Faster* —
Junjie Wang, Pan Zhou, Yiming Dong, Huan Li, Jia Li, Xun Zhou, Qicheng Lao, Cong Fang,
Zhouchen Lin (Peking University, SMU, Nankai, BNU, ByteDance Seed, BUPT).
arXiv:2509.24218v2 (v1 29 Sep 2025, v2 30 Sep 2025). Code: `github.com/jie040109/Conda`.

**One-paragraph summary.** Adam updates for 2-D transformer weights are ill-conditioned
and nearly low-rank; Muon fixes the spectrum (Newton–Schulz equalizes all singular
values) but throws away Adam's coordinate-wise adaptivity and overshoots tiny singular
values. Conda reformulates Muon as a first/second-moment method and keeps Adam's
element-wise adaptivity *inside* the momentum's singular subspace: it maintains the usual
first moment `M_t`, periodically takes the SVD of `M_t` (lazy, every `T` steps), and runs
a full Adam-style update — bias correction, epsilon and all — in that projected space,
with the second moment `N_t` being an EMA of the **squared projected gradients**
`(U_t^T G_t)^2`. The result is a *column-wise* (per-coordinate of the projected matrix)
adaptive normalizer that is milder than Muon's uniform `1/σ` scaling and preserves the
relative order of singular values. On LLaMA 60M–1B pretraining Conda beats AdamW, Muon,
SOAP and Adafactor, reaching AdamW's final loss with 2–2.5× fewer steps; it also wins on
GPT-2 (125M/355M) and LoRA fine-tuning. It belongs in this repo because it is exactly
the kind of single-family, matrix-aware optimizer the registry curates — a sibling of
`ada_muon`/`muon2`/`nor_muon` in the Orthogonalized bucket, with a published, complete
update rule and released reference code.

---

## 2. Paper extract

### 2.1 Update rule (verbatim, Eq. 5 + Algorithm 1 + released skeleton)

Main text Eq. 5 (for `W ∈ R^{m×n}`, `m ≤ n`):

```latex
\begin{cases}
\mathbf{M}_t = \beta_1 \mathbf{M}_{t-1} + (1-\beta_1)\mathbf{G}_t, \\
\mathbf{U}_t, \mathbf{\Sigma}_t, \mathbf{V}_t^{\top} = \texttt{SVD}(\mathbf{M}_t), \\
\mathbf{M}'_t = \mathbf{U}_t^{\top}\mathbf{M}_t, \\
\mathbf{N}_t = \beta_2 \mathbf{N}_{t-1} + (1-\beta_2)(\mathbf{U}_t^{\top}\mathbf{G}_t)^2, \\
\mathbf{W}_t = \mathbf{W}_{t-1} - \eta\,\mathbf{U}_t\left(\mathbf{M}'_t / \sqrt{\mathbf{N}_t}\right).
\end{cases}
```

Muon reformulation it builds on (Lemma 1, Eq. 3): Muon ≡ `N_t = diag(Σ_t) 1^T` (row-wise
`1/σ_i` normalization) + projection back through `U_t`. Conda replaces the uniform `1/σ`
denominator with the element-wise `√N_t` of the **projected gradients** — a per-column
adaptive denominator (Lemma 2, Eq. 6–7: Muon normalizes every column by the *same*
`Σ^{-1}U U^T`; Conda normalizes column `k` by `Σ_i 1/√(N_{t,i,k}) U^{(i)}_t`).

Appendix A.4, Algorithm 1 (complete pseudocode, incl. the details omitted in Eq. 5):

```
Input: W ∈ R^{m×n} (m ≤ n), lr η, RMS scale α, betas β1 β2, subspace update frequency T
Initialize t←0, M_0←0, V_0←0
repeat
  compute minibatch gradient G_t
  M_t ← β1 M_{t-1} + (1-β1) G_t
  if t mod T = 0:  U_t,Σ_t,V_t^T ← SVD(M_t);  Ū_t ← U_t
  else:            Ū_t ← Ū_{t-1}                     # lazy subspace
  M'_t ← Ū_t^T M_t
  N_t  ← β2 V_{t-1} + (1-β2)(Ū_t^T G_t)^2            # second moment of PROJECTED grad
  M'_t ← M'_t / (1-β1^t);  N_t ← N_t / (1-β2^t)      # bias correction
  W_t  ← W_{t-1} + η Ū_t M'_t / (√V_t + ε)           # (sign per skeleton: −η·update)
  t ← t+1
until convergence
```

Released PyTorch skeleton (Listing 1) — the authoritative implementation details:

* constructor: `lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0,
  correct_bias=True, update_proj_gap=2000, scale=1.0`;
* only `p.grad.ndim == 2` params take the Conda path (others `continue`, i.e. AdamW in
  the GaLore-derived codebase);
* state: `step:int`, `exp_avg` (full m×n), `exp_avg_sq_proj` (None initially, allocated
  like `G_proj`), `proj_basis`, `proj_type`;
* SVD trigger: `step % update_proj_gap == 0 or state["proj_basis"] is None`;
* basis choice: `grad.shape[0] <= grad.shape[1]` → `U`, `proj_type="left"`
  (`G_proj = P.T @ grad`, `M_proj = P.T @ exp_avg`); else `Vh`, `"right"`
  (`G_proj = grad @ P.T`, `M_proj = exp_avg @ P.T`) — §A.4 text: left `U^T` better when
  `m ≤ n`, right `V` when `m > n`;
* `exp_avg_sq_proj.mul_(β2).addcmul_(G_proj, G_proj, value=1-β2)`;
* bias corrections `bc1 = 1-β1^step`, `bc2 = 1-β2^step` (skipped when `correct_bias=False`);
  `denom = sqrt(exp_avg_sq_proj / bc2) + eps`; `update = P @ (M_proj/bc1 / denom)` (or
  right variant); `update *= scale`; `p -= lr * update`; decoupled
  `p -= lr * wd * p` when `weight_decay > 0`.

### 2.2 Hyperparameters

| Name | Type | Default (released code) | Paper experiments | Scope |
|---|---|---|---|---|
| `lr` η | float | 1e-3 | 1e-2 LLaMA 60M–1B; 6e-3/3e-3 GPT-2; 1e-4–3e-4 LoRA | per-group (`lr_*`) |
| `betas` (β1, β2) | (float,float) | (0.9, 0.99) | (0.9, 0.99) LLaMA+GPT-2 (Table 12); robust to β1∈{0.9,0.95}, β2∈{0.95…0.999} (Table 5) | per-group (`betas_*`) |
| `eps` ε | float | 1e-8 | — | per-group (`eps_*`) |
| `weight_decay` | float | 0.0 | 0 (<1B LLaMA), 0.1 (≥1B), 1e-2 (GPT-2) | per-group (`wd_*`, decoupled) |
| `update_proj_gap` T | int | 2000 | 2000 pretraining (Table 8/9: stable T∈[2,20000], sweet spot 500–1000); 200 LoRA | global |
| `scale` α | float | 1.0 | 0.25 pretraining (§A.5); 1.0 LoRA (Table 15) | global |
| `correct_bias` | bool | True | always on | global |

### 2.3 State buffers and per-step cost

Per 2-D parameter: `exp_avg` (m×n), `exp_avg_sq_proj` (r×n for left / m×r for right,
r = min(m,n)), `proj_basis` (m×r or r×n) → **3 tensors** + `step`, `proj_type` scalars.
Non-2-D parameters carry the AdamW-fallback state (`exp_avg`, `exp_avg_sq`, `step`).
Per-step cost: EMA updates O(mn) + two thin matmuls O(mn·r); SVD O(mn·r) amortized every
T steps (Table 8: negligible for T ≥ 100; T=2 costs +64% wall-clock vs T=2000 on 60M).
Memory ≈ AdamW + basis (Table 6: +0.03–1.3 GB vs AdamW across 60M–1B).

### 2.4 Category

Existing bucket **Orthogonalized** (`docs/architecture.md` §8.1 family table row,
`docs/paper/appendices/annex-summary-tables.tex` header *"Second-Order, Geometric &
Orthogonality"*). No new category needed.

### 2.5 BibTeX data

Authors: Junjie Wang, Pan Zhou, Yiming Dong, Huan Li, Jia Li, Xun Zhou, Qicheng Lao,
Cong Fang, Zhouchen Lin. Year 2025. eprint 2509.24218, cs.LG.
DOI: 10.48550/arXiv.2509.24218. Key (repo convention `<firstauthor>_<shortname>_<year>`):
`wang_conda_2025`.

### 2.6 Key evidence quotes (for the annex paragraph)

* "Conda projects updates into an orthogonal subspace and applies column-wise second
  moment normalization based on the projected gradients" (abstract).
* "the native second-moment estimator, which is not aligned with the subspace of the
  first moment, leads to instability and even divergence in the training of larger
  models" (§A.7; Table 4: no-proj fails at 1B).
* "on the LLaMA series, Conda achieves 2∼2.5× the convergence speed of AdamW, measured
  in both training steps and training time" (abstract); Table 1: 60M 28.32 / 130M 21.38 /
  350M 16.44 / 1B 13.59 val PPL vs AdamW 34.06/25.08/18.80/15.56, Muon 29.89/22.15/16.51/14.17.
* Table 7: Conda stable at lr 2e-2 where AdamW/Muon diverge ≥5e-3.

---

## 3. Decisions

| Decision | Value | Evidence / rationale |
|---|---|---|
| Registry name | `conda` | Paper short name, snake_case; no collision in `factory.OPTIMIZER_REGISTRY` (checked all 32 keys, `factory.py:58-91`) |
| Class name | `Conda` (`src/model/optimizer/conda.py`) | Released skeleton class name |
| Closest peer | `galore_adamw.py` (lazy periodic SVD projector, left/right basis by aspect ratio, per-group betas/eps, decoupled wd) **+** `muon.py`'s `adamw_fallback_update` for non-2-D params | The official Conda skeleton is GaLore-derived (same `update_proj_gap`/`scale`/`correct_bias` kwargs); repo convention requires non-matrix params to still train → `base.adamw_fallback_update` (`base.py:121-171`), exactly what the GaLore codebase's AdamW branch does for skipped params |
| Param routing | Conda path for **every 2-D parameter in every group**; `adamw_fallback_update` for every non-2-D parameter | Skeleton gates only on `p.grad.ndim != 2` — no group exclusion (unlike `muon.py:122` `ADAMW_FALLBACK_GROUPS`); the paper's LLaMA results were produced this way. Schema text must state this explicitly |
| Category | Orthogonalized (§8.1 row; annex "Second-Order, Geometric & Orthogonality") | Matrix-subspace hybrid, same bucket as muon/muon2/nor_muon |
| Suffixes | global: `update_proj_gap` (int, 2000, `to_int`), `scale` (float, 1.0, `to_float`), `correct_bias` (bool, True, `bool`). Shared per-group: `lr_*`, `wd_*`, `betas_*`, `eps_*` | Names copied verbatim from the released skeleton; `update_proj_gap`/`scale` already exist as suffix names for `galore_adamw`/`apollo` (precedent) |
| Defaults nuance | `scale=1.0` and `betas=(0.9,0.99)` are the *constructor* defaults (skeleton); effective factory betas default is `(0.9,0.95)` from `_build_param_groups_for_optimizer` (`factory.py:200`) unless `conda-betas_<group>` is set → document `(0.9,0.99)` in `configs/README.md` + set it in the example; `scale=0.25` is the paper's pretraining value → set it in the example | Faithful to released code; paper §A.5/Table 15 justify both scale values by regime |
| `supports_shared_per_group_suffixes` | `true` | Norm for the registry; conda consumes all four shared suffix families |
| Example preset | `configs/examples/es_arch_hybrid_conda.yaml` (task `mlm`, hybrid `retnet/titan_attn/mamba/ode` arch) | Exact mirror of `es_arch_hybrid_nor_muon.yaml` — the established pattern for every recent optimizer family (`es_arch_hybrid_{ada_muon,dion3,muon2,nor_muon,root,turbo_muon,eve,ea4llm}.yaml`); smallest *valid* config = same known-good arch, optimizer block swapped |
| Census bump | **32 → 33** | Actual current count read from repo: 32 registry entries (`factory.py:58-91`), README.md:27 "32 optimizer families", README.md:98 "32 across 8 categories", architecture.md:26/87/767; annex-summary-tables.tex:111 "all 32"; 05-optimizer-families.tex:4 "thirty-two"; ES mirrors. **Found stale counts to fix in the same pass:** annex-summary-tables.tex:4 (EN) and :4 (ES) still say "28 optimizers/optimizadores" — flagging to the user; they are the same census sentence, so the pass updates 28→33 there |
| Figure in annex | **Omitted** | `docs/paper/figures/optimizers/` has figures for only 25 of 32 optimizers; the two most recent Orthogonalized additions (`muon2`, `nor_muon`) shipped **without** figures — follow that precedent; do not fabricate a diagram |

---

## 4. File-by-file plan

Execution order = row order. All anchors verified against the current tree
(commit-clean except pre-existing untracked plan files).

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/conda.py` | **New file.** `torch.optim.Optimizer` subclass `Conda`; module docstring citing "Wang et al. 2025, arXiv:2509.24218" + reference block (mirror `ada_muon.py:1-15` / `galore_adamw.py:1-13`). `from __future__ import annotations`; Google-style docstrings; CPU-safe (no `.cuda()`); `@torch.no_grad() step`. Signature: `__init__(params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, correct_bias=True, update_proj_gap=2000, scale=1.0)` with `ValueError` guards (lr ≥ 0; 0 ≤ β < 1; eps ≥ 0; update_proj_gap ≥ 1). `step()` per parameter: **non-2-D** → `adamw_fallback_update(p, grad, self.state[p], lr=group["lr"], betas=group["betas"], eps=group["eps"], weight_decay=group["weight_decay"])` then `continue` (mirrors `muon.py:130-140`); **2-D** → Algorithm 1: (a) `exp_avg.mul_(β1).add_(grad, alpha=1-β1)`; (b) if `state["step"] % int(group["update_proj_gap"]) == 0 or state["proj_basis"] is None`: `U, _, Vh = torch.linalg.svd(exp_avg, full_matrices=False)`, store `proj_basis = U`/`"left"` when `m ≤ n` else `Vh`/`"right"` (skeleton §A.4); (c) `G_proj`, `M_proj` = `P.T @ X` (left) or `X @ P.T` (right); (d) lazily allocate `exp_avg_sq_proj = zeros_like(G_proj)`, `exp_avg_sq_proj.mul_(β2).addcmul_(G_proj, G_proj, value=1-β2)`; (e) `bc1/bc2 = 1-β^step` (1.0 when `correct_bias` False); (f) `denom = (exp_avg_sq_proj / bc2).sqrt() + group["eps"]`; (g) `update = P @ ((M_proj / bc1) / denom)` or right variant; (h) `update.mul_(group["scale"])`; (i) decoupled wd `p.mul_(1 - lr·wd)` then `p.add_(update, alpha=-group["lr"])` |
| 2 | `src/model/optimizer/__init__.py` | Insert `from .conda import Conda` between lines 11–12 (`from .cautious_adamw import CautiousAdamW` / `from .dion3 import Dion3`); add `"Conda",` to `__all__` between `"CautiousAdamW",` (line 45) and `"Dion3",` (line 46). (Note: `anon` is already absent from this file — pre-existing inconsistency, leave alone.) |
| 3 | `src/model/optimizer/factory.py` | (a) Import `from .conda import Conda` between lines 36–37 (`.cautious_adamw` / `.dion3`). (b) `OPTIMIZER_REGISTRY`: add `"conda": Conda,` after `"nor_muon": NorMuon,` (line 90). (c) `OPTIMIZER_ALLOWED_KEYS`: add after line 158 (`"nor_muon"` entry): `"conda": _COMMON_PER_GROUP_KEYS \| {"update_proj_gap", "scale", "correct_bias"},`. (d) Dispatch branch after the `nor_muon` branch (lines 483-492), before the final `return optimizer_cls(converted_groups)` (line 494): `if optimizer_name == "conda": update_proj_gap = to_int(scoped.get("update_proj_gap"), 2000); scale = to_float(scoped.get("scale"), 1.0); correct_bias = bool(scoped.get("correct_bias", True)); return optimizer_cls(converted_groups, update_proj_gap=update_proj_gap, scale=scale, correct_bias=correct_bias)` |
| 4 | `src/schema/_optimizer.yaml` | (a) `enum`: append `- conda` after `- nor_muon` (line 181). (b) EN use-case table: new row after the `nor_muon` row (line 66): `` | `conda` | Spectral + coordinate-wise adaptive pretraining | Adam update inside the SVD subspace of the first moment; second moment of *projected* gradients (column-wise adaptive); lazy SVD every `update_proj_gap` steps; AdamW fallback for non-matrix params (paper: arxiv:2509.24218) | ``. (c) ES table: new row after line 125: `` | `conda` | Preentrenamiento espectral + adaptativo por coordenada | Actualización de Adam dentro del subespacio SVD del primer momento; segundo momento de gradientes *proyectadas* (adaptativo por columna); SVD perezoso cada `update_proj_gap` pasos; respaldo AdamW para parámetros no matriciales (paper: arxiv:2509.24218) | ``. (d) EN `parameters.description` suffix list: bullet after the `nor_muon` bullet (line 215): `- `conda`: `update_proj_gap`, `scale`, `correct_bias`. Adam-style update in the SVD subspace of the first moment: the second moment is an EMA of the squared projected gradients `(P^T G)^2` (β from the shared `betas_*`, ε from the shared `eps_*`); the SVD basis is refreshed every `update_proj_gap` steps (default 2000); `scale` multiplies the update (0.25 in the paper's pretraining, 1.0 in its LoRA runs); every non-2-D parameter uses a decoupled AdamW fallback configured by the shared suffixes`. (e) ES `parameters.description_es`: same bullet after line 260: `- `conda`: `update_proj_gap`, `scale`, `correct_bias`. Actualización estilo Adam en el subespacio SVD del primer momento: el segundo momento es una EMA de los gradientes proyectados al cuadrado `(P^T G)^2` (β de los `betas_*` compartidos, ε de los `eps_*` compartidos); la base SVD se refresca cada `update_proj_gap` pasos (por defecto 2000); `scale` multiplica la actualización (0.25 en el preentrenamiento del paper, 1.0 en sus ejecuciones LoRA); todo parámetro no 2-D usa un respaldo AdamW desacoplado configurado por los sufijos compartidos`. (f) Both `**References:**` paragraphs: append ``; `conda` (paper: arxiv:2509.24218)`` after `` `nor_muon` (paper: arxiv:2510.05491)`` (EN line 88, ES line 148). (g) `x_parameter_reference.by_optimizer`: append after the `nor_muon` block (line 438): `conda:\n  prefix: "conda-"\n  supports_shared_per_group_suffixes: true\n  specific_global_suffixes: [update_proj_gap, scale, correct_bias]` |
| 5 | `src/schema/_training.yaml` | Append one `if/then` block after line 1637 (file end, after the `nor_muon` block): `- if:\n    properties:\n      optimizer:\n        properties:\n          optimizer_class:\n            const: conda\n  then:\n    properties:\n      optimizer:\n        properties:\n          parameters:\n            additionalProperties: false\n            patternProperties:\n              ^conda-: { type: [number, boolean, array, string] }` — exactly one block (parity test `test_schema_consistency.py::test_optimizer_prefixes_match_conditional_rules`, line 76) |
| 6 | `src/schema/_sbert.yaml` | Extend `optimizer_kwargs` EN description (line 400): after `group_size/pool_dim (for adam_pm: …)` add `, update_proj_gap/scale/correct_bias (for conda: subspace refresh period T, update scale factor, bias-correction toggle)` and after the nor_muon sentence add `For conda the paper's betas are (0.9, 0.99) and the shared per-group eps is the projected-denominator epsilon.` ES description (line 401): `, update_proj_gap/scale/correct_bias (para `conda`: periodo de refresco del subespacio T, factor de escala de la actualización, interruptor de corrección de sesgo)` + `Para `conda`, los betas del paper son (0.9, 0.99) y el `eps` por grupo compartido es el épsilon del denominador proyectado.` |
| 7 | `configs/examples/es_arch_hybrid_conda.yaml` | **New file** — full mirror of `configs/examples/es_arch_hybrid_nor_muon.yaml` with: `optimizer_class: conda`; all per-group keys `conda-lr_*` / `conda-wd_*` identical values; `conda-betas_*: [0.9, 0.99]` (paper Table 12); `conda-eps_*: 1e-8`; `conda-update_proj_gap: 2000`; `conda-scale: 0.25` (paper §A.5); `conda-correct_bias: true`; `csv_log_path: training_metrics_es_arch_hybrid_conda.yaml`'s `.csv` counterpart (`training_metrics_es_arch_hybrid_conda.csv`). Model block unchanged (same `retnet/titan_attn/mamba/ode` hybrid, `task: mlm`). Must pass `load_training_config` + strict schema validation (`tests/test_yaml_examples.py` globs this dir) |
| 8 | `configs/README.md` | (a) Optimizer Chooser list: append `- conda` after `- nor_muon` (line 165). (b) Optimizer Parameters Reference: entry after the `nor_muon` entry (line 219): `- `conda`: `update_proj_gap`, `scale`, `correct_bias` (Adam in the SVD subspace of the first moment; second moment EMA of squared projected gradients; lazy SVD every `update_proj_gap` steps — default 2000; paper pretraining uses `scale: 0.25`, `betas: [0.9, 0.99]`; non-2-D parameters use the decoupled AdamW fallback)`. (c) Example-keys block: after the `nor_muon-*` example keys (line 234) add `- `conda-update_proj_gap: 2000`\n- `conda-scale: 0.25`` (match surrounding bullet format) |
| 9 | `full_tests/run_e2e.py` | (a) `OPTIMIZERS` list: append `"conda",` after `"nor_muon",` (line 272). (b) `_with_optimizer`: add `"conda"` to the Adam-style betas/eps set (lines 662-665) so the smoke run sets per-group betas; add after the optimizer-specific blocks (~line 695+, next to the other Muon-family entries): `if optimizer_class == "conda": params[f"{prefix}update_proj_gap"] = 4; params[f"{prefix}scale"] = 0.25; params[f"{prefix}correct_bias"] = True` (gap 4 so the SVD-refresh path is actually exercised in a smoke run) |
| 10 | `tests/test_optimizer_factory.py` | (a) Add `"conda"` to the `required` set in `test_registry_contains_all_required` (line 31). (b) New class `BuildOptimizerCondaTests(unittest.TestCase)` (placed next to the other Muon-family classes, before/after `BuildOptimizerSophiaTests` region — follows the `BuildOptimizerAnonTests` pattern at line 787): `test_builds_without_error` (`build_optimizer("conda", _param_groups(), {})`); `test_step_runs` (4×4 param, `p.sum().backward()`, `opt.step()`); `test_matrix_uses_conda_state` (2-D param in group `attention` → after one step `state` has `exp_avg`, `exp_avg_sq_proj`, `proj_basis`, `proj_type`); `test_vector_param_uses_adamw_fallback` (1-D param → `exp_avg`/`exp_avg_sq`, no `proj_basis`); `test_custom_suffixes_parsed` (`{"conda-update_proj_gap": 10, "conda-scale": 0.25, "conda-correct_bias": False}` → `opt.defaults["update_proj_gap"] == 10`, `scale == 0.25`, `correct_bias is False`); `test_default_gap_2000` (defaults: `update_proj_gap == 2000`, `correct_bias is True`) |
| 11 | `docs/bibliography/optimizers.bib` | Append at end (Zotero arXiv style, matching `li_normuon_2025` block): `@misc{wang_conda_2025,\n      title={Conda: Column-Normalized Adam for Training Large Language Models Faster},\n      author={Junjie Wang and Pan Zhou and Yiming Dong and Huan Li and Jia Li and Xun Zhou and Qicheng Lao and Cong Fang and Zhouchen Lin},\n      year={2025},\n      eprint={2509.24218},\n      archivePrefix={arXiv},\n      primaryClass={cs.LG},\n      url={https://arxiv.org/abs/2509.24218},\n}` |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | Insert `\paragraph{Conda (Column-Normalized Adam).}` block after the NorMuon `Pros.` list (line 1610) and before `\subsection{Evolutionary and Population-Based Hybrids}` (line 1612) — last entry of the *Second-Order, Geometric and Orthogonality* subsection. **No figure** (precedent: `muon2`/`nor_muon` shipped figureless). Draft: intro sentence citing `\citep{wang_conda_2025}` ("Conda~\citep{wang_conda_2025} keeps Adam's coordinate-wise adaptivity and adds a milder, column-specific spectral normalization. It first reformulates Muon as a first/second-moment method whose second moment is the row-constant $\mathrm{diag}(\Sigma_t)\mathbf{1}^\top$; replacing that uniform $1/\sigma_i$ denominator with an element-wise second moment of the **projected** gradients yields a per-column adaptive normalizer that preserves the relative order of the singular values."); `align` block with the five update equations (M_t; lazy-SVD case split on $t \bmod T$; $M'_t = \bar U_t^\top M_t$; $N_t = \beta_2 N_{t-1} + (1-\beta_2)(\bar U_t^\top G_t)^2$; $W_t = W_{t-1} - \eta \bar U_t \frac{M'_t/(1-\beta_1^t)}{\sqrt{N_t/(1-\beta_2^t)}+\epsilon}$) with the note that for $m > n$ the right basis $V_t$ is used ($M'_t = M_t V_t$, update $(M'/\sqrt{N})V_t^\top$), matching the released skeleton; `algorithm[H]` float `\caption{Conda (Column-Normalized Adam)} \label{alg:conda}` mirroring `alg:normuon`'s structure (Require line with $\eta, \beta_1, \beta_2, T$, scale $\alpha$, $\epsilon$, $\lambda$; the 9 State lines of Algorithm 1 + the AdamW `\Else` branch for non-2-D parameters); results paragraph (2–2.5× convergence speed of AdamW on LLaMA 60M–1B, PPL table numbers 28.32/21.38/16.44/13.59 vs AdamW 34.06/25.08/18.80/15.56 and Muon 29.89/22.15/16.51/14.17; GPT-2 125M/355M 17.40/13.92; stable at lr 2e-2 where AdamW/Muon diverge; robust to $T \in [2, 20000]$); implementation-mapping paragraph (suffixes: `update_proj_gap` = $T$ (default 2000; 200 in the paper's LoRA runs), `scale` = $\alpha$ (1.0 default, 0.25 pretraining), `correct_bias` toggles the $1-\beta^t$ corrections, shared `betas_*`/`eps_*` = $(\beta_1,\beta_2)$/$\epsilon$ (paper: 0.9/0.99), shared `wd_*` decoupled; **every 2-D parameter takes the Conda path regardless of group** — unlike `muon`, there is no embedding/pos/head exclusion — and non-2-D parameters take the decoupled AdamW fallback); `Cons.` list (SVD cost spike every $T$ steps — Table 8/9 wall-clock; second moment must live in the same subspace: the unprojected estimator diverges at 1B, Table 4; state is Adam-sized plus the basis — +0.03–1.3 GB vs AdamW, Table 6; evaluated only up to 1B and without MoE); `Pros.` list (Adam's coordinate-wise adaptivity *and* improved spectral conditioning in one rule; structure-preserving normalization avoids Muon's overshoot of tiny singular values; single hyperparameter set across 60M–1B; stable at 10–20× AdamW's learning rate) |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Row in `tab:all-optimizers` after the NorMuon row (line 171), still under `\textbf{Second-Order, Geometric \& Orthogonality}`: `Conda & Orthogonality + column-wise adaptive & 3 ($m$, $v_{\mathrm{proj}}$, basis $P$); + $v$ on the AdamW fallback & $\mathcal{O}(mn\,r)$, $r=\min(m,n)$; SVD every $T$ steps & lr, $\beta_1$, $\beta_2$, eps, wd, gap $T$, scale $\alpha$ & \citep{wang_conda_2025} \\`. (b) Census: line 111 "catalogs all 32 supported optimizers" → "all 33". (c) **Stale-count fix (flagged):** line 4 header "… 28 optimizers …" → "… 33 optimizers …" (the intro sentence was already stale at 28 vs the table's 32; correcting it to 33 is part of the census pass) |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: "The system supports thirty-two optimizer families in eight categories" → "thirty-three"; append "Conda" to the geometry-oriented list "(Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon², NorMuon)" → "(…, NorMuon, Conda)" |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of row 12, inserted after the NorMuon Pros list (before `\subsection{Híbridos Evolutivos y Poblacionales}`, line 1632): `\paragraph{Conda (Adam Normalizado por Columnas).}` — "Conda~\citep{wang_conda_2025} conserva la adaptividad por coordenada de Adam y añade una normalización espectral más suave y específica por columna. Primero reformula Muon como un método de primer/segundo momento cuyo segundo momento es la matriz constante por filas $\mathrm{diag}(\Sigma_t)\mathbf{1}^\top$; sustituir ese denominador uniforme $1/\sigma_i$ por un segundo momento por elemento de los gradientes **proyectados** produce un normalizador adaptativo por columna que preserva el orden relativo de los valores singulares."; mismas cinco ecuaciones (los formulamos no cambian), mismo `algorithm` flotado `\caption{Conda (Adam Normalizado por Columnas)} \label{alg:conda-es}` si el anexo ES exige etiqueta propia — verificar contra `alg:normuon` en ES (usa `\label{alg:normuon}` igual que EN: reutilizar `\label{alg:conda}` si es el patrón, si no sufijar); párrafo de resultados (2–2,5× la velocidad de convergencia de AdamW en LLaMA 60M–1B; PPL 28.32/21.38/16.44/13.59 frente a 34.06/25.08/18.80/15.56 de AdamW; estable con lr 2e-2 donde AdamW/Muon divergen; robusto a $T \in [2, 20000]$); párrafo de mapeo (sufijos `update_proj_gap`, `scale`, `correct_bias`, `betas_*`/`eps_*` compartidos; **toda matriz 2-D toma la ruta Conda sin exclusión de grupos** — a diferencia de `muon` — y los parámetros no 2-D usan el respaldo AdamW desacoplado); `Contras.` (costo SVD cada $T$ pasos; el segundo momento debe vivir en el mismo subespacio: el estimador sin proyectar diverge en 1B; estado del tamaño de Adam más la base; evaluado solo hasta 1B y sin MoE); `Pros.` (adaptividad por coordenada + acondicionamiento espectral; normalización que preserva la estructura; un solo juego de hiperparámetros de 60M a 1B; estable con LR 10–20× la de AdamW). Figures live in `../paper/figures/optimizers/` — none referenced |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | (a) Row after NorMuon (line 170): `Conda & Ortogonalidad + adaptativa por columna & 3 ($m$, $v_{\mathrm{proj}}$, base $P$); + $v$ en el respaldo AdamW & $\mathcal{O}(mn\,r)$, $r=\min(m,n)$; SVD cada $T$ pasos & lr, $\beta_1$, $\beta_2$, eps, wd, hueco $T$, escala $\alpha$ & \citep{wang_conda_2025} \\`. (b) Line 111 "cataloga los 32 optimizadores soportados" → "los 33". (c) **Stale-count fix (flagged):** line 4 "… 28 optimizadores …" → "… 33 optimizadores …" |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: "El sistema soporta treinta y dos familias de optimizadores" → "treinta y tres"; "métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon², NorMuon)" → append ", Conda" |
| 18 | `docs/architecture.md`, `README.md`, `configs/README.md` | Census pass (grep the old count "32" in each file and update every optimizer-census occurrence): `docs/architecture.md` line 26 `| Optimizer families | **32** |` → `**33**`; line 87 `32 optimizers + registry + parameter routing` → `33`; line 767 heading `### 8.1 Optimizers (32)` → `(33)`; line 780 Orthogonalized row: add `conda` to the family list after `nor_muon` and append the clause `; `conda` runs a full Adam update inside the lazy SVD subspace of the first moment (second moment of the projected gradients, refreshed every `update_proj_gap` steps; AdamW fallback only for non-2-D parameters)`. `README.md` line 27 (mermaid) `"Train<br/>32 optimizer families<br/>…"` → `33`; line 98 `| Optimizer families | 32 across 8 categories |` → `33 across 8 categories`. `configs/README.md` — covered in row 8. Re-grep `grep -rn "32" README.md docs/architecture.md configs/README.md` at implementation time to catch any straggler |

**Hard rules restated.** The schema is the contract: `additionalProperties: false` on
`training.optimizer.parameters` per-optimizer blocks — every `conda-*` key must exist in
`x_parameter_reference.by_optimizer.conda` + `factory.OPTIMIZER_ALLOWED_KEYS["conda"]` +
exactly one `^conda-:` `if/then` rule. Optimizer keys are prefixed
`conda-<group>_<param>` / `conda-<param>`; never read an undocumented key. The papers
are the theory source of truth (AGENTS.md D11): the annex formulations above must match
arXiv:2509.24218v2 Eq. 5 + Algorithm 1 + Listing 1 verbatim, and `paper-es` must
translate them faithfully (same equations, same algorithm lines).

---

## 5. Ordered execution checklist

Implementation (rows 1–9) → tests (row 10) → docs (rows 11–18) → PDFs → website sync →
commit/push. CPU-first for everything except real training runs.

```bash
# --- implementation time only; NOT run by this planning command ---

# targeted gates (expect: all pass; test_optimizer_factory includes the new
# BuildOptimizerCondaTests; test_schema_consistency keeps registry ↔ enum ↔ by_optimizer
# ↔ if/then parity; test_yaml_examples validates the new es_arch_hybrid_conda.yaml)
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# fast e2e smoke. NOTE: conda is appended LAST in OPTIMIZERS, so --limit 3 will NOT
# reach it — raise the limit (e.g. --limit 40) or select the combo id directly:
#   uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --only opt_conda
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 40
```

Checklist:

- [ ] 1. `src/model/optimizer/conda.py` (new)
- [ ] 2. `src/model/optimizer/__init__.py`
- [ ] 3. `src/model/optimizer/factory.py` (import, registry, allowed keys, dispatch)
- [ ] 4. `src/schema/_optimizer.yaml` (enum, 2 table rows, 2 refs, 2 suffix bullets, by_optimizer block)
- [ ] 5. `src/schema/_training.yaml` (one `^conda-` if/then)
- [ ] 6. `src/schema/_sbert.yaml` (optimizer_kwargs EN+ES)
- [ ] 7. `configs/examples/es_arch_hybrid_conda.yaml` (new)
- [ ] 8. `configs/README.md` (list, suffix entry, example keys)
- [ ] 9. `full_tests/run_e2e.py` (OPTIMIZERS + `_with_optimizer`)
- [ ] 10. `tests/test_optimizer_factory.py` (required set + `BuildOptimizerCondaTests`) → run targeted pytest → full pytest
- [ ] 11. `docs/bibliography/optimizers.bib`
- [ ] 12. `docs/paper/appendices/annex-1-optimizers.tex`
- [ ] 13. `docs/paper/appendices/annex-summary-tables.tex`
- [ ] 14. `docs/paper/sections/05-optimizer-families.tex`
- [ ] 15. `docs/paper-es/appendices/annex-1-optimizadores.tex`
- [ ] 16. `docs/paper-es/appendices/annex-summary-tables.tex`
- [ ] 17. `docs/paper-es/sections/05-familias-optimizadores.tex`
- [ ] 18. Census pass: `docs/architecture.md`, `README.md` (+ re-grep "32" across the three docs)
- [ ] 19. PDF rebuild (both papers) — see below
- [ ] 20. Website sync (§6) — mirror only, uncommitted
- [ ] 21. Commits + pushes (§7)

### PDF rebuild (implementation time only)

Preconditions (verified available now): `/usr/bin/pdflatex`, `/usr/bin/bibtex`,
`/usr/bin/pandoc`. Run from the repository root:

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

Fallback: if `pdflatex`/`bibtex` are unavailable on the implementing machine, leave the
`.tex`/`.bib` sources updated and note that the rebuild must happen on a TeX machine
before the next release — the website sync copies **prebuilt** PDFs only.

Post-compilation verification (required):

- No `Fatal error` and no `Citation .* undefined` / `Reference .* undefined` in the logs
  (specifically `wang_conda_2025` and `alg:conda` must resolve);
- `paper.pdf` / `paper-es.pdf` mtimes newer than the edited `.tex` and `.bbl`;
- `git status --short docs/paper docs/paper-es` shows only the intended
  `.tex` / `.bbl` / `.pdf` changes (never stage `*.aux *.log *.out *.toc *.blg`).

---

## 6. Website sync (planned, not executed)

Per `.opencode/command/sync-website.md` (mirror = `erickfmm.github.io/frankenstein-transformer/`,
separate git repo, all git ops via `git -C erickfmm.github.io …`; start from the live
report `bash .opencode/command/sync-website-state.sh`):

1. Copy from main repo → mirror (respecting the file map):
   * `src/schema.yaml` → `schema.yaml`; `src/schema/**` → `schema/**`
     (includes the edited `_optimizer.yaml`, `_training.yaml`, `_sbert.yaml`);
   * `configs/examples/es_arch_hybrid_conda.yaml` → `examples/examples/**` and
     `configs/README.md` → `examples/README.md`;
   * `docs/paper/{paper.tex,paper.bbl,paper.pdf}` + `sections/*.tex` +
     `appendices/*.tex` → `paper/en/…` (edited `annex-1-optimizers.tex`,
     `annex-summary-tables.tex`, `05-optimizer-families.tex`);
   * `docs/paper-es/{paper-es.tex,paper-es.bbl,paper-es.pdf}` + `sections/*.tex` +
     `appendices/*.tex` → `paper/es/…`;
   * `docs/bibliography/optimizers.bib` → `paper/bibliography/`;
   * `docs/architecture.md` → `paper/specs/architecture.md`.
   No new figures, so `paper/figures/**` is untouched.
2. Rebuild the Paper-tab HTML: run `paper/flatten_tex.py` + `pandoc` +
   `paper/postprocess_html.py` (site-only infrastructure — never overwrite them) to
   regenerate `paper/paper-en.html` / `paper/paper-es.html`, and refresh
   `paper/en.pdf` / `paper/es.pdf` from the freshly compiled PDFs.
3. **Validation Gate** (from the repo root, CPU extras):
   * `uv run --extra cpu --extra train python full_tests/param_count_check.py` →
     `0 failures` (regenerates `full_tests/param_truth.json`; optimizer additions do
     not change model parameter counts, so any `param_truth.json` diff should be empty —
     if it changes, investigate before proceeding);
   * `node full_tests/param_estimate_check.mjs … --tol 0.01` → PASS (tolerance ≤ 1%);
   * state-script probes from `.opencode/command/sync-website-state.sh`:
     `schema-refs`, `example-yaml`, `cli-flags`, estimator → all PASS.
4. Leave the mirror **uncommitted** until §7's commit phase.

---

## 7. Commit plan (planned, not executed)

Follow `.opencode/command/make-a-commit.md` conventions: English, conventional tag,
structured body; never stage `*.aux *.log *.out *.toc *.blg` or caches;
`param_truth.json` changes (if any) are intentional.

1. **Main repo** — `feat(optimizer): add Conda (arXiv:2509.24218)`
   Body sections: implementation notes (routing: every 2-D param takes the Conda path,
   `adamw_fallback_update` for non-2-D; lazy SVD basis left/right by aspect ratio);
   new suffixes (`update_proj_gap`, `scale`, `correct_bias` + shared per-group keys);
   schema/registry changes (enum, use-case tables EN/ES, suffix lists, by_optimizer,
   `^conda-` rule, SBERT kwargs); docs (annex EN/ES, summary tables + census 32→33,
   bibliography `wang_conda_2025`, family sections, architecture/README counts,
   rebuilt PDFs); example `es_arch_hybrid_conda.yaml`; tests
   (`BuildOptimizerCondaTests`, registry list, e2e `OPTIMIZERS` + param map).
   Stage exactly:
   ```
   src/model/optimizer/conda.py src/model/optimizer/__init__.py src/model/optimizer/factory.py
   src/schema/_optimizer.yaml src/schema/_training.yaml src/schema/_sbert.yaml
   configs/examples/es_arch_hybrid_conda.yaml configs/README.md
   full_tests/run_e2e.py tests/test_optimizer_factory.py
   docs/bibliography/optimizers.bib
   docs/paper/appendices/annex-1-optimizers.tex docs/paper/appendices/annex-summary-tables.tex
   docs/paper/sections/05-optimizer-families.tex docs/paper/paper.tex  # only if touched
   docs/paper/paper.bbl docs/paper/paper.pdf
   docs/paper-es/appendices/annex-1-optimizadores.tex docs/paper-es/appendices/annex-summary-tables.tex
   docs/paper-es/sections/05-familias-optimizadores.tex docs/paper-es/paper-es.tex  # only if touched
   docs/paper-es/paper-es.bbl docs/paper-es/paper-es.pdf
   docs/architecture.md README.md
   ```
   Then `git push origin HEAD:main`.
2. **Mirror** — after the Validation Gate passes:
   `git -C erickfmm.github.io add schema.yaml schema/ examples/ paper/` then commit
   `docs(site): sync Conda optimizer (arXiv:2509.24218)` and
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected result |
|---|---|---|
| Targeted factory tests | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py -v` | All pass incl. 6 new `BuildOptimizerCondaTests` methods; `conda` in required-registry test |
| Base/helper tests | `… tests/test_optimizer_base.py -v` | Unchanged, all pass (no `base.py` edit) |
| Schema consistency | `… tests/test_schema_consistency.py -v` | Parity holds: enum ↔ registry ↔ `by_optimizer` ↔ exactly one `^conda-` if/then; presets validate |
| YAML examples | `… tests/test_yaml_examples.py -v` | `es_arch_hybrid_conda.yaml` validates strictly and loads |
| Full suite | `… python -m pytest tests/ --continue-on-collection-errors -v` | 76-file suite green, no regressions |
| e2e smoke | `… python full_tests/run_e2e.py --category opt --limit 40` (or `--only opt_conda`) | `opt_conda` combo trains a few steps on CPU without error (SVD refresh hit via `update_proj_gap: 4`) |
| PDF EN | `pdflatex`/`bibtex` cycle in `docs/paper` | No `Fatal error`, no `Citation wang_conda_2025 undefined`, `alg:conda` resolves, `paper.pdf` newer than sources |
| PDF ES | same in `docs/paper-es` | Same checks; Spanish paragraph/row present |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `0 failures`; `param_truth.json` unchanged (optimizer-only change) |
| Estimator | `node full_tests/param_estimate_check.mjs … --tol 0.01` | PASS (≤ 1% drift) |
| State probes | `bash .opencode/command/sync-website-state.sh` → schema-refs / example-yaml / cli-flags / estimator | All PASS after sync |
| Git hygiene | `git status --short` before commits | Only the intended files from §7 (plus pre-existing untracked plan files, untouched) |

---

## 9. Risks, assumptions and open questions

1. **Effective betas default is `(0.9, 0.95)`, not the paper's `(0.9, 0.99)`** when the
   factory builds groups without `conda-betas_*` (`_build_param_groups_for_optimizer`
   injects `(0.9, 0.95)` for every non-SGD optimizer, `factory.py:200`). Documented in
   configs/README + example sets `[0.9, 0.99]`. Verify at implementation time that no
   test asserts the constructor betas default through the factory.
2. **`scale` default `1.0` vs paper pretraining `0.25`** — constructor default follows
   the released skeleton (and the paper's own LoRA Table 15 uses 1.0). The example and
   README state 0.25 for pretraining. This is a documentation burden, not a bug.
3. **SVD on CPU in tests/e2e** — `torch.linalg.svd` on tiny matrices is fine; on large
   real runs it is the paper's own cost profile (amortized over T=2000). Assumption:
   params stay fp32 under AMP autocast, so `svd(exp_avg)` needs no dtype cast (the repo
   trains with fp32 master weights; the skeleton calls svd on the buffer directly).
4. **Routing difference vs `muon`** — Conda processes embedding/head 2-D matrices too
   (paper skeleton has no fallback groups). Tests must assert the 2-D-in-any-group
   behavior so a future refactor doesn't silently copy Muon's routing.
5. **Naming** — `conda` collides with the Anaconda package manager only as a string;
   no registry, schema, test or CLI collision exists (checked). If the user dislikes the
   name, `conda_adam` is the fallback candidate — flagged as an open question, default
   is `conda`.
6. **Stale census "28 optimizers"** in `annex-summary-tables.tex:4` (EN and ES) —
   pre-existing drift (table itself says 32). The census pass fixes both to 33; called
   out here so it is a visible, approved change rather than silent drift-fixing.
7. **`anon` missing from `optimizer/__init__.py`** — pre-existing inconsistency;
   out of scope, left untouched. Conda follows the recipe and *is* exported.
8. **Task/mixer compatibility** — Conda is a standard `step()` optimizer: no
   `gradient_accumulation_steps: 1` restriction, SBERT-available (unlike `eve`/`ea4llm`),
   works with every mixer (routing depends only on parameter rank, not module type).
   No `config_loader.py` rule needed.
9. **Paper ambiguities** — Algorithm 1 line 14 writes `W + η…` and references `V_t`
   where `N_t` is meant (typo); the released skeleton is internally consistent and is
   used as the tiebreaker (bias-corrected `M'/√V + ε`, `-lr·update`). The pseudocode's
   `M'_t/(1-β1^T)` renders as `1-β1^t`. Both noted in the annex mapping paragraph.
10. **e2e `--limit` pitfall** — conda is appended last in `OPTIMIZERS`; a small
    `--limit` truncates before reaching it (the checklist uses `--limit 40` or
    `--only opt_conda`).

---

## 10. Out of scope

This command is a planning command. **No code, schema, preset, test, doc, PDF,
website-mirror or git change was made.** Nothing outside `.opencode/plan/` was written;
no `pytest`/`pdflatex`/`uv sync`/training/sync script was executed; the two pre-existing
untracked plan files (`.opencode/plan/add-mixer-2305.05084.md`,
`.opencode/plan/add-optimizer-2510.05491.md`) were left untouched.
