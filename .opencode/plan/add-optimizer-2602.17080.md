# Plan: add NAMO optimizer (arXiv:2602.17080)

**Paper:** *Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum* — Minxin Zhang, Yuxuan Liu, Hayden Schaeffer (UCLA Mathematics), arXiv:2602.17080v2 [cs.LG], submitted 19 Feb 2026, revised 20 Feb 2026. Official code: `github.com/minxin-zhg/namo`. DOI: `10.48550/arXiv.2602.17080`.

**One-paragraph summary.** Muon updates weight matrices with Newton–Schulz-orthogonalized momentum, which is unbounded and can amplify stochastic noise, so its effective stepsize never shrinks under noise. NAMO (Norm-Based Adaptive Moment Estimation with Orthogonalized Momentum) fixes this with the first *principled* integration of orthogonalized momentum with Adam-type noise adaptation: it tracks a **single scalar** EMA of the squared Frobenius norm of the gradient and rescales the whole orthogonalized update by the bias-corrected norm ratio α_t = ‖M̂_t‖_F/(√v̂_t + ε_t) — an aggregate SNR statistic analogous to Adam's coordinate-wise ratio. Because the scaling is one scalar, the update direction stays exactly orthogonal, the state grows by only one scalar per matrix over Muon (zero extra memory), and the per-step overhead is O(mn). The paper proves optimal O(T^{-1/2}) deterministic and noise-adaptive stochastic rates, and beats both AdamW and Muon on GPT-2 (124M/355M) OpenWebText pretraining with wider learning-rate robustness. A diagonal extension, NAMO-D, right-multiplies by a clamped diagonal for neuron-wise adaptation; **both NAMO and NAMO-D were implemented** (registry names `namo` / `namo_d`). This repo's `Orthogonalized` bucket (Muon → TUON, 10 members) is the natural home: NAMO slots in as the 11th, following the same hybrid AdamW-fallback recipe as `muon`/`nor_muon`.

**Plan refresh (2026-11-08):** this plan was first drafted before `tuon` (arXiv:2610.35130) landed. Every census figure, line anchor and insertion point has been re-verified against current HEAD (34 optimizer families at planning time) and updated in place; the paper extract (§2) was re-checked verbatim against the arXiv v2 PDF and is unchanged.

**Implementation note (2026-11-08):** on the follow-up request the plan was executed for **both** methods — `src/model/optimizer/namo.py` and `namo_d.py`, the registry/schema/tests/e2e/docs rows below, examples `es_arch_hybrid_namo.yaml` and `es_arch_hybrid_namo_d.yaml`, and the census bumped **34 → 36** (examples 121 → 123). NAMO-D adds `namo_d-clamp` (default 0.1) on top of NAMO's suffix set.

---

## 2. Paper extract

### 2.1 Update rule (NAMO — Algorithm 1 and Eq. (3), verbatim)

For a matrix parameter Θ ∈ ℝ^{m×n}, stochastic gradient G_t, learning rate η, momenta μ₁, μ₂ ∈ [0,1), stability constant ε > 0, decoupled weight decay λ:

```
M_t = μ₁ M_{t-1} + (1 - μ₁) G_t                                  (momentum, Alg. 1 line 4)
v_t = μ₂ v_{t-1} + (1 - μ₂) ‖G_t‖²_F                             (scalar norm EMA, Alg. 1 line 5)
O_t = Orth(M_t) ≈ NS_K(M_t)                                      (orthogonalization, Alg. 1 line 6)
```

Adaptive stepsize (Eq. (3) / Alg. 1 line 7) — the two forms are algebraically identical:

```
           (1 - μ₂^t)^{1/2}      ‖M_t‖_F        ‖M̂_t‖_F
α_t :=  ────────────────────  ·  ───────────  =  ─────────────────
           (1 - μ₁^t)           √(v_t) + ε       √(v̂_t) + ε_t

with  M̂_t := M_t/(1 - μ₁^t),   v̂_t := v_t/(1 - μ₂^t),   ε_t := ε/√(1 - μ₂^t)
```

Update (Alg. 1 line 8):

```
Θ_t = Θ_{t-1} - η α_t O_t
```

Decoupled weight decay (paper §4.1) — the adaptive scaling is applied to the decay **as part of the effective stepsize**:

```
Θ_t = Θ_{t-1} - η α_t (O_t + λ Θ_{t-1})        ⇔  Θ ← Θ·(1 - η α_t λ) - η α_t O_t
```

**Non-matrix parameters:** "Since Muon, NAMO, and NAMO-D are designed specifically for matrix parameters, we use AdamW to optimize all remaining scalar and vector parameters across all models." (paper §4.1) → same hybrid AdamW fallback recipe as `muon.py` (`ADAMW_FALLBACK_GROUPS` + every non-2-D parameter).

**Newton–Schulz:** the paper uses Newton–Schulz "as in the Muon optimizer" for the experiments; exact orthogonalization only in the theory. Quintic coefficients (3.4445, −4.7750, 2.0315), K = 5 steps, bfloat16, Frobenius normalization + transpose-when-tall — identical to `src/model/optimizer/nor_muon.py::_newton_schulz` / `muon.py`.

**Momentum parameterization warning:** NAMO's Algorithm 1 includes the **(1 − μ₁) factor** in the momentum update (unlike `muon.py`/`nor_muon.py`, which accumulate `M ← βM + G`). For NAMO the factor does **not** cancel: α_t depends explicitly on ‖M_t‖_F and on the bias corrections (1−μ₁^t), (1−μ₂^t). Implement the paper's recursion exactly.

**ε placement:** add ε to the *biased* √v_t (Algorithm-1 form). This is identical to adding ε_t = ε/√(1−μ₂^t) to √v̂_t. No separate bias-corrected copy of v is needed.

**α_t boundedness:** by the paper's Lemma 7 argument (applied to the vectorized sequence), α_t ≤ √((1−μ₁)/(1−μ₂)) = √5 ≈ 2.24 with the default momenta, so no clamping is required for stability. Zero-gradient steps give α_t = 0 → the whole step (update *and* decay) is a no-op — a documented behavioral difference from `muon`, whose decoupled decay still applies on zero gradients.

### 2.2 Algorithm 1 (verbatim pseudocode)

```
Algorithm 1  NAMO: Norm-Based Adaptive Moment Estimation with Orthogonalized Momentum
Require: Learning rate η, momentum μ₁, μ₂ ∈ [0, 1), batch size b, ε > 0
1:  Initialize Θ₀ ∈ ℝ^{m×n}, M₀ = 0, v₀ = 0
2:  for t = 1, 2, …, T do
3:      Sample a minibatch of size b and compute stochastic gradient G_t = ∇L^t(Θ_{t-1})
4:      M_t ← μ₁ M_{t-1} + (1 − μ₁) G_t
5:      v_t ← μ₂ v_{t-1} + (1 − μ₂) ‖G_t‖²_F
6:      O_t ← Orth(M_t)
7:      α_t ← [√(1−μ₂^t) / (1−μ₁^t)] · ‖M_t‖_F / (√(v_t) + ε)
8:      Update parameters Θ_t ← Θ_{t−1} − η α_t O_t
9:  end for
10: return Θ_T
```

No warmup interaction is defined by the algorithm itself; the experiments use an external 2000-step linear warmup followed by a constant LR (scheduler-agnostic, standard `training.*` schedulers apply unchanged).

### 2.3 Hyperparameters

| Hyperparameter | Paper symbol | Type | Paper value (GPT-2 runs) | Our key | Scope |
|---|---|---|---|---|---|
| Learning rate | η | float | grid {0.005 … 0.015}; optimal **0.012** (124M) / **0.007** (355M) | shared `namo-lr_<group>` | per group |
| First momentum | μ₁ | float | **0.95** | `namo-betas[0]` | global |
| Norm-EMA decay | μ₂ | float | **0.99** | `namo-betas[1]` | global |
| Stability constant | ε | float | "small ε > 0" (official code `adamnorm_eps=1e-8`; theory ε = O(T^{-1/2})) | shared `namo-eps_<group>` (also the fallback ε) | per group |
| Weight decay | λ | float | **0.01** for all optimizers (paper §4.1; official Hydra config uses 0.1 on the Muon path and 0 on the AdamW backup) | shared `namo-wd_<group>`, decoupled, **scaled by α_t** | per group |
| NS iterations | K | int | 5 (official code) | `namo-ns_steps` | global |
| NS normalization ε | — | float | 1e-7 (official code) | `namo-ns_eps` | global |
| AdamW fallback betas | — | tuple | (0.9, 0.95) | shared `namo-betas_<group>` | per group |
| NAMO-D clamp (out of scope) | c | float | 0.1 (124M) / 0.9 (355M); official default 0.75 | — | — |

Scheduler: 2000-step linear warmup then constant. Effective batch 480×1024 tokens. Baselines: AdamW β=(0.9, 0.95) η=0.0013/0.0009; Muon β=0.95, same η as AdamW (the paper's Muon carries no 0.2√(mn) shape scaling — its LR grid matches AdamW's).

### 2.4 State buffers, cost, complexity

- **NAMO path (per 2-D parameter):** `momentum_buffer` M ∈ ℝ^{m×n} + `norm_second_moment` v (0-dim float32 scalar) + `step` counter → **mn + 1 numbers per matrix**: "no additional memory overhead" over Muon, ~½ of Adam's 2mn.
- **AdamW fallback (embeddings/pos/head groups + every non-2-D parameter):** `exp_avg` + `exp_avg_sq` (2 × param size) via `base.adamw_fallback_update`.
- **Per-step cost:** O(mn·K) for Newton–Schulz + **O(mn)** extra for the two norms and the scalar EMA — "negligible additional computational cost" vs Muon.
- **Convergence (paper §3):** optimal O(T^{-1/2}) deterministic; stochastic bound decomposes into O(T^{-1/4}) optimization term + variance term O(√σ·b^{-1/4}·T^{-1/8}); optimal O(T^{-1/4}) recovered when b = Ω(σ²T^{1/2}). (Under Assumption 1: nuclear-norm Lipschitz gradients; Assumption 2: unbiased, bounded-variance noise.)
- **Results (Table 2):** validation loss GPT-2 124M/355M — NAMO **3.0351/2.9516** vs AdamW 3.0643/2.9914, Muon 3.0435/2.9684; NAMO-D 3.0246/2.9507. NAMO/NAMO-D stay stable over a wider LR grid (Fig. 1).

### 2.5 NAMO-D (Algorithm 2) — recorded, NOT planned

For neuron-wise (column-wise) adaptation: `[v_t]_j = μ₂[v_{t-1}]_j + (1−μ₂)‖[G_t]_{:j}‖²` (operator N^c(M)_j = ‖M_{:j}‖); `d_t = N^c(M̂_t) ⊘ (√v̂_t + ε_t)` with ‖d_t‖_∞ ≤ √((1−μ₁)/(1−μ₂)) (Lemma 7); clamp toward the average `d̃_t = min(max(d_t, c·d̄_t·1), d̄_t/c·1)` where `d̄_t = ‖d_t‖₁/n`, c ∈ (0,1]; `D_t = diag(d̃_t)`; update `Θ_t = Θ_{t−1} − η(O_t + λΘ_{t−1})D_t`. Official code class `NAMO_D` with `col_state_clamp_c`. See Open questions (§9).

### 2.6 BibTeX data (repo style: `<firstauthor>_<shortname>_<year>`, `@misc` arXiv entries)

```bibtex
@misc{zhang_namo_2026,
      title={Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum},
      author={Minxin Zhang and Yuxuan Liu and Hayden Schaeffer},
      year={2026},
      eprint={2602.17080},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.17080},
}
```

(The authors' own README uses key `zhang2026namo`; the repo convention — cf. `li_normuon_2025`, `wang_conda_2025`, `tang_tuon_2026` at `docs/bibliography/optimizers.bib:403,413,423` — is `zhang_namo_2026`.)

---

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `namo` | Paper short name (NAMO); snake_case, no collision in `OPTIMIZER_REGISTRY` (34 entries, `factory.py:60-95`), no `namo` anywhere in `src/` |
| Class name | `NAMO` | Matches paper caps style; precedent `ROOT`, `EA4LLM` |
| Closest peer | `src/model/optimizer/nor_muon.py` | Same hybrid recipe (`ADAMW_FALLBACK_GROUPS` + `adamw_fallback_update` from `base.py`), module-level `_newton_schulz` helper (importable by tests, `test_optimizer_factory.py:12`), validation block in `__init__`, Google-style docstrings. Secondary reference: `muon.py` (routing logic), `q_apollo.py` (instance-attribute pattern for globals shadowed by group keys) |
| Family category | **Orthogonalized** (architecture.md §8.1 row, line 780) = "geometry-oriented" (paper §5) = "Second-Order, Geometric & Orthogonality" (summary-table header) | NAMO is an orthogonalized-momentum method; joins `muon … conda` |
| Figure in annex | **Omit** | No `namo.mmd/.pdf` can be produced in the established vector style; the last six optimizer additions (dion3, muon2, nor_muon, conda, tuon, adam_pm) all have no optimizer figure — `docs/paper/figures/optimizers/` (25 figures, newest `root`) has none of them |
| Scope | **NAMO only**; NAMO-D recorded under Open questions | NAMO is Algorithm 1 / the title method; NAMO-D is "a diagonal extension" (Algorithm 2). Not genuinely ambiguous |
| Fidelity | **Paper Algorithm 1 exactly** (D11: papers are the theory source of truth) | The official repo adds Nesterov momentum (`nesterov=True`), an effective-stepsize `clamp(max=1.0)` and a `0.2·√max(m,n)` shape scaling — none appear in Algorithm 1 → not implemented, documented in the module docstring |

### Suffix table

| Key | Type | Default | Coercion | Scope | EN description | ES description |
|---|---|---|---|---|---|---|
| `namo-betas` | [float, float] | [0.95, 0.99] | `to_betas` | global | (μ₁, μ₂) momenta of the norm-based adaptive scaling (first momentum + scalar norm-EMA decay); the paper's GPT-2 values | (μ₁, μ₂) del escalado adaptativo basado en normas (momentum + decaimiento de la EMA escalar de normas); los valores GPT-2 del artículo |
| `namo-ns_steps` | int | 5 | `to_int` | global | Newton–Schulz iteration count | Iteraciones de Newton–Schulz |
| `namo-ns_eps` | float | 1e-7 | `to_float` | global | Newton–Schulz Frobenius-normalization epsilon | Épsilon de la normalización Frobenius de Newton–Schulz |
| shared `lr_*`, `wd_*`, `betas_*`, `eps_*` | — | factory defaults | factory | per group | `wd` decoupled and scaled by α_t; `eps` is α_t's ε **and** the fallback ε; `betas`/`eps` configure the AdamW fallback | `wd` desacoplado y escalado por α_t; `eps` es el ε de α_t **y** el del respaldo; `betas`/`eps` configuran el respaldo AdamW |

- `supports_shared_per_group_suffixes: true` (the norm).
- `OPTIMIZER_ALLOWED_KEYS["namo"] = _COMMON_PER_GROUP_KEYS | {"betas", "ns_steps", "ns_eps"}`.
- **Shadowing pitfall (drives the implementation):** `_build_param_groups_for_optimizer` (`factory.py:202-204`) injects `betas`/`eps` into *every* group (defaults (0.9, 0.95) / 1e-8) for the AdamW fallback. If the NAMO momenta were stored in `defaults["betas"]`, the group-level injection would shadow them. Therefore the constructor stores the (μ₁, μ₂) pair as an **instance attribute** `self.betas` (precedent: `q_apollo.self.quant_bits`, `anon.self.*`), and the NAMO path reads `self.betas` while the fallback reads `group["betas"]`/`group["eps"]`.

### Example preset

`configs/examples/es_arch_hybrid_namo.yaml` — MLM task (`training.task: mlm`) on `frankenstein_text_encoder`, an exact structural copy of `es_arch_hybrid_nor_muon.yaml` (the closest structural peer — same shared-key layout; the newest family example is `es_arch_hybrid_tuon.yaml`) with `namo-` keys. Smallest valid config: same encoder/dims/block as every `es_arch_hybrid_<muon-family>.yaml` peer; `test_yaml_examples.py` auto-globs it and `test_schema_consistency.py::test_all_presets_validate` validates it — zero extra test wiring.

### Census bump

**34 → 35** optimizer families (actual old value re-read from the repo at this plan's refresh: `docs/architecture.md:767` heading `### 8.1 Optimizers (34)`; `OPTIMIZER_REGISTRY` has 34 entries — `tuon` was added after the first draft of this plan). Full census-pass list in file-plan row 18. No pre-existing census drift remains: README.md:20 says **34**, README.md:98 says `34 across 8 categories`, README.md:103 says `52 named presets + 121 example configs`, all matching the filesystem at HEAD (52 / 121); the example count goes to **122** with the new file.

---

## 4. File-by-file plan table

Execution order. Line anchors are current HEAD values; re-grep before editing.

| # | File | Planned change (be concrete) |
|---|---|---|
| 1 | `src/model/optimizer/namo.py` | **New module.** `torch.optim.Optimizer` subclass; module docstring cites the paper (mirror `nor_muon.py:1-29`): `Zhang, M., Liu, Y., & Schaeffer, H. (2026). Adam Improves Muon: ... arXiv:2602.17080.` + a "Reference-implementation differences" note (Nesterov/clamp/0.2√max(m,n) exist only in the official repo). `from __future__ import annotations`; imports `math`, `torch`, `from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update`. Module-level `_newton_schulz(matrix, steps, eps, coefficients=(3.4445, -4.7750, 2.0315))` copied verbatim from `nor_muon.py:47-87` (bfloat16, Frobenius normalize, transpose-when-tall). Constructor: `NAMO(params, lr=1e-3, betas=(0.95, 0.99), ns_steps=5, ns_eps=1e-7, weight_decay=0.0)`; validates `0<=lr`, `0<=betas[i]<1`, `1<=ns_steps`, `0<=ns_eps`, `0<=weight_decay` (ValueError, mirror `nor_muon.py:169-184`); stores `self.betas = (float(betas[0]), float(betas[1]))` as an **instance attribute** (NOT in defaults — group `betas` keys shadow defaults; see §3); `defaults = dict(lr=lr, ns_steps=ns_steps, ns_eps=ns_eps, weight_decay=weight_decay)`; `super().__init__(params, defaults)`. Google-style Args/Attributes docstrings. CPU-safe (no `.cuda()`, no `torch.compile`). `@torch.no_grad() step(closure=None)`: closure pattern mirror `muon.py:116-119`; then per group: `use_adamw = str(group.get("name", "other")) in ADAMW_FALLBACK_GROUPS`; per param: `if p.grad is None: continue`; **fallback** `if use_adamw or p.ndim != 2: adamw_fallback_update(p, p.grad, self.state[p], lr=group["lr"], betas=group.get("betas", (0.9, 0.95)), eps=group.get("eps", 1e-8), weight_decay=group["weight_decay"]); continue` (mirror `muon.py:130-140`). **NAMO path**, lazy state init `state["momentum_buffer"] = torch.zeros_like(grad)`, `state["norm_second_moment"] = torch.zeros((), dtype=torch.float32, device=grad.device)`, `state["step"] = 0`; then, mapping each line to the paper: line 4 `buf.mul_(mu1).add_(grad, alpha=1.0 - mu1)`; line 5 `grad_sq = grad.square().sum()` → `v.mul_(mu2).add_(grad_sq.to(v.dtype), alpha=1.0 - mu2)` (note: EMA tracks **‖G_t‖²_F of the raw gradient**); line 6 `orth = _newton_schulz(buf, int(group["ns_steps"]), float(group["ns_eps"]))`; line 7 `bc1 = 1.0 - mu1**t; bc2 = 1.0 - mu2**t; alpha = math.sqrt(bc2) / bc1 * float(buf.norm()) / (float(v.sqrt()) + float(group["eps"]))` (ε = shared group `eps`, default 1e-8); §4.1 update `if group["weight_decay"] > 0: p.mul_(1.0 - lr*alpha*wd)` then `p.add_(orth.to(p.dtype), alpha=-lr*alpha)` — decay **scaled by α_t**; zero-grad ⇒ α=0 ⇒ no-op (docstring note). Return `loss`. |
| 2 | `src/model/optimizer/__init__.py` | Insert `from .namo import NAMO` between lines 22 (`from .muon2 import Muon2`) and 23 (`from .nor_muon import NorMuon`) — alphabetical: muon2 < namo < nor_muon. Insert `"NAMO"` in `__all__` between `"Muon2"` (line 56) and `"NorMuon"` (line 57). |
| 3 | `src/model/optimizer/factory.py` | (a) Import `from .namo import NAMO` between lines 46 (`.muon2`) and 47 (`.nor_muon`). (b) `OPTIMIZER_REGISTRY`: add `"namo": NAMO,` after `"tuon": TUON,` (line 94, the registry's last entry; closing `}` at line 95). (c) `OPTIMIZER_ALLOWED_KEYS`: add `"namo": _COMMON_PER_GROUP_KEYS | {"betas", "ns_steps", "ns_eps"},` after the tuon entry (lines 164-173; closing `}` at 174). (d) Dispatch branch after tuon's (520-537), before the final `return optimizer_cls(converted_groups)` (539): `if optimizer_name == "namo": betas = to_betas(scoped.get("betas"), (0.95, 0.99)); ns_steps = to_int(scoped.get("ns_steps"), 5); ns_eps = to_float(scoped.get("ns_eps"), 1e-7); return optimizer_cls(converted_groups, betas=betas, ns_steps=ns_steps, ns_eps=ns_eps)`. |
| 4 | `src/schema/_optimizer.yaml` | (a) **enum**: append `- namo` after `- tuon` (line 189; `examples: [adamw]` follows at line 190). (b) **EN use-case table**: append after the tuon row (line 68): `` `namo` | Noise-adaptive orthogonal updates | Norm-based Adam-type adaptive stepsize α_t on Muon's orthogonalized momentum: scalar second-moment EMA of ‖G‖²_F, zero extra memory; AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2602.17080) ``. (c) **ES use-case table**: append after tuon (line 130 — note the ES table has no trailing blank line before `**Ejemplos explicados:**` at 131): `` `namo` | Actualizaciones ortogonales adaptativas al ruido | Paso adaptativo estilo Adam basado en normas sobre el momentum ortogonalizado de Muon: EMA escalar del segundo momento de ‖G‖²_F, sin memoria extra; respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2602.17080) ``. (d) **References lists**: EN line 91 and ES line 154 both end with `` `tuon` (paper: arxiv:2610.35130) `` → append `` ; `namo` (paper: arxiv:2602.17080) ``. (e) **EN "Optimizer-specific suffixes"** list: insert after the `tuon` bullet (line 225), before the `eve` bullet (line 226): `` - `namo`: `betas`, `ns_steps`, `ns_eps`. Norm-based adaptive stepsize on the orthogonalized momentum: `betas` is (μ₁, μ₂) of the momentum and the scalar norm EMA (default [0.95, 0.99], the paper's GPT-2 values); the shared `eps_*` is the α_t denominator epsilon; the shared `wd_*` is decoupled and scaled by α_t; same hybrid routing as `muon` — the shared `betas_*`/`eps_*` configure the AdamW fallback for the `embeddings`/`pos`/`head` groups and every non-2-D parameter ``. (f) **ES list**: after the `tuon` bullet (line 272), before `eve` (line 273): `` - `namo`: `betas`, `ns_steps`, `ns_eps`. Paso adaptativo basado en normas sobre el momentum ortogonalizado: `betas` es (μ₁, μ₂) del momentum y de la EMA escalar de normas (por defecto [0.95, 0.99], los valores GPT-2 del artículo); el `eps_*` compartido es el épsilon del denominador de α_t; el `wd_*` compartido es desacoplado y escalado por α_t; mismo enrutamiento híbrido que `muon` — los `betas_*`/`eps_*` compartidos configuran el respaldo AdamW para los grupos `embeddings`/`pos`/`head` y todo parámetro no 2-D ``. (g) **`x_parameter_reference.by_optimizer`**: append after the tuon entry (lines 455-458, the file's last block): `namo:` / `prefix: "namo-"` / `supports_shared_per_group_suffixes: true` / `specific_global_suffixes: [betas, ns_steps, ns_eps]`. |
| 5 | `src/schema/_training.yaml` | Append exactly one `if/then` block after tuon's (lines 1652-1665, the file's last block), same shape/indentation: `- if:` → `properties:` → `optimizer:` → `properties:` → `optimizer_class:` → `const: namo`; `then:` → `properties:` → `optimizer:` → `properties:` → `parameters:` → `additionalProperties: false` / `patternProperties:` / `^namo-: { type: [number, boolean, array, string] }`. The parity test `test_optimizer_prefixes_match_conditional_rules` (regex `^\s+\^([a-z0-9_]+)-:\s*\{`) requires exactly one such block per prefix. |
| 6 | `src/schema/_sbert.yaml` | Extend `optimizer_kwargs` descriptions (EN line 400, ES line 401): append `namo` to the `ns_steps` and `ns_eps` optimizer lists (`…/muon2/nor_muon/tuon` → `…/muon2/nor_muon/tuon/namo`; do **not** touch the `momentum` list — NAMO has no `momentum` suffix, μ₁ comes from the global `betas`); insert after the `momentum` clause a new clause `betas (for namo: the (mu1, mu2) momenta of the norm-based adaptive scaling, default [0.95, 0.99])`; append the closing sentence after conda's sentence ("…the projected-denominator epsilon.") and before "Parameters are passed as key-value pairs." (EN): "For namo the shared per-group `betas`/`eps` configure only the AdamW fallback and the shared `eps` doubles as the adaptive-stepsize epsilon, while the shared `wd` is decoupled and scaled by alpha_t." ES mirror: "Para `namo`, los `betas`/`eps` por grupo compartidos configuran solo el respaldo AdamW y el `eps` compartido sirve además como épsilon del paso adaptativo, mientras que el `wd` compartido es desacoplado y escalado por alpha_t." NAMO is available for SBERT (standard `step()`, no closure requirement — only `eve`/`ea4llm` are excluded). |
| 7 | `configs/examples/es_arch_hybrid_namo.yaml` | **New file.** Byte-for-byte structural copy of `es_arch_hybrid_nor_muon.yaml` (97 lines) with: `optimizer_class: namo`; all 40 shared keys re-prefixed `namo-lr_<g>` / `namo-wd_<g>` / `namo-betas_<g>: [0.9, 0.95]` / `namo-eps_<g>: 1e-8` for the 10 groups (same values as the nor_muon example, = `DEFAULT_GROUP_LR`/`DEFAULT_GROUP_WD`); specific globals `namo-betas: [0.95, 0.99]`, `namo-ns_steps: 5`, `namo-ns_eps: 1.0e-7`; `csv_log_path: training_metrics_es_arch_hybrid_namo.csv`. Everything else identical (frankenstein_text_encoder, mlm, seed 42, layer_pattern [retnet, titan_attn, mamba, ode], vocab 50000, hidden 768, 12 layers, 12 heads, scheduler_total_steps 12000, …). Full draft below (§4.1). |
| 8 | `configs/README.md` | (a) Optimizer chooser list: add `- namo` after `- tuon` (line 167). (b) Specific-suffix list: insert after the `tuon` bullet (line 223): `` - `namo`: `betas`, `ns_steps`, `ns_eps` (norm-based Adam-type adaptive stepsize α_t on the orthogonalized momentum; `betas` is the (μ₁, μ₂) pair of the momentum and the scalar norm EMA — default [0.95, 0.99]; the shared `eps_*` is the α_t epsilon, the shared `wd_*` is decoupled and scaled by α_t; same hybrid routing as `muon`) ``. (c) Examples list: add after `- adam_pm-pool_dim: col` (line 244): `- namo-betas: [0.95, 0.99]` and `- namo-ns_steps: 5`. |
| 9 | `full_tests/run_e2e.py` | (a) `OPTIMIZERS` list (267-273): append `"namo"` after `"tuon"` on line 272 → `"root", "muon2", "adam_pm", "nor_muon", "conda", "tuon", "namo",`. (b) `_optimizer_params` builder: add `"namo"` to the Adam-style per-group betas/eps set (lines 662-666, after `"tuon"` on line 665 — those keys are allowed for namo and configure the fallback), and append after the tuon block (722-725): `if optimizer_class == "namo": params[f"{prefix}betas"] = [0.95, 0.99]; params[f"{prefix}ns_steps"] = 5; params[f"{prefix}ns_eps"] = 1e-7`. Keys are filtered against `OPTIMIZER_ALLOWED_KEYS` (727-749) so nothing invalid can leak. |
| 10 | `tests/test_optimizer_factory.py` | (a) `test_registry_contains_all_required` (line 31): add `"namo"` to the `required` set. (b) New test class `BuildOptimizerNAMOTests` (docstring: `"""NAMO defaults, paper invariants and hybrid routing (arXiv:2602.17080)."""`), inserted after `BuildOptimizerTUONTests` (ends line 837), before `BuildOptimizerAdamPMTests` (838), mirroring the NorMuon class (411-542): `test_builds_without_error`; `test_step_runs` (4×4 param, `p.square().mean().backward()`, no NaN); `test_defaults_match_paper` (`opt.betas == (0.95, 0.99)`, `opt.defaults["ns_steps"] == 5`, `ns_eps == 1e-7`); `test_hidden_matrix_uses_namo_state` (attention group → state has `momentum_buffer` + `norm_second_moment` 0-dim, no `exp_avg`); `test_first_step_matches_algorithm1` — recompute `α₁ = √(1-μ₂)/(1-μ₁) · ‖M₁‖/(√v₁+ε)` from the imported `_newton_schulz` and the post-step state buffers, assert `‖Δp‖ ≈ lr·α₁·‖O₁‖_F` (wd=0); `test_weight_decay_is_scaled_by_adaptive_stepsize` (non-zero grad, wd=0.1 vs wd=0: assert `p_wd ≈ (1 − lr·α·wd)·p_zero_wd` with α recovered from state — note: zero-grad steps decay nothing, α=0); `test_embedding_group_uses_adamw_fallback`; `test_vector_param_is_not_frozen` (norms group 1-D param takes fallback, changes, no NaN); `test_specific_suffixes_parsed` (`{"namo-betas": [0.9, 0.98], "namo-ns_steps": "3", "namo-ns_eps": "1e-6"}` → `(0.9, 0.98)`/3/1e-6; bad strings → `(0.95, 0.99)`/5/1e-7); `test_unknown_group_suffix_raises` (`{"namo-lr_bogus": 1.0}` → ValueError mentioning `lr_bogus`); `test_per_group_lr_routes` (`{"namo-lr_ffn": 0.02}` → `param_groups[0]["lr"] == 0.02`). |
| 11 | `docs/bibliography/optimizers.bib` | Append after `tang_tuon_2026` (lines 423-431, the file's last entry) the entry drafted in §2.6 (`@misc{zhang_namo_2026, …}`), matching the Zotero arXiv style of the file. |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | Insert `\paragraph{NAMO (Norm-Based Adaptive Moment Estimation with Orthogonalized Momentum).}` block **after TUON's Pros list (ends line 1820; blank line 1821) and before `\subsection{Evolutionary and Population-Based Hybrids}` (line 1822)** — last member of the `Second-Order, Geometric, and Orthogonality Optimizers` subsection (starts line 895; TUON's paragraph currently spans 1709-1820). No figure (see §3). Full draft in §4.2 below: intro prose, `align` block with the five equations (momentum, scalar norm EMA, NS, α_t both forms with ε_t, update with α-scaled decay), `algorithm` float (`\label{alg:namo}`), implementation-notes paragraph (suffix mapping), results paragraph (GPT-2 losses, optimal LRs, theory rates), `\paragraph{Cons.}` + `\paragraph{Pros.}`, `\citep{zhang_namo_2026}`. |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Add a row after TUON (line 172), still under `\textbf{Second-Order, Geometric \& Orthogonality}` (line 158): `NAMO & Orthogonality + norm-based adaptive & 2 ($m$, scalar $v$); + $v$ on the AdamW fallback & $\mathcal{O}(n \cdot k)$ & lr, $\beta_1$, $\beta_2$, eps, wd, NS steps $k$ & \citep{zhang_namo_2026} \\`. (b) Census: line 4 `34 optimizers` → `35 optimizers`; line 111 `catalogs all 34 supported optimizers` → `all 35 supported optimizers`. |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: `The system supports thirty-four optimizer families in eight categories` → `thirty-five`; append `NAMO` to the geometry-oriented list: `(Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon², NorMuon, Conda, TUON, NAMO)`. |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of row 12, inserted **after TUON's Ventajas (ends line 1853) and before `\subsection{Híbridos Evolutivos y Poblacionales}` (line 1855)**, inside `Optimizadores de Segundo Orden, Geométricos y de Ortogonalidad` (subsection at line 903; TUON's paragraph currently spans 1735-1853). Terminology mirrors the file's conventions (`respaldo AdamW`, `momentum`, `pesos`, `Contras.`/`Ventajas.`). Full draft in §4.3. Figures live in `../paper/figures/optimizers/` (none referenced). |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | (a) Row after TUON (line 172): `NAMO & Ortogonalidad + adaptativa por norma & 2 ($m$, $v$ escalar); + $v$ en el respaldo AdamW & $\mathcal{O}(n \cdot k)$ & lr, $\beta_1$, $\beta_2$, eps, wd, pasos NS $k$ & \citep{zhang_namo_2026} \\`. (b) Census: line 4 `34 optimizadores` → `35 optimizadores`; line 111 `cataloga los 34 optimizadores soportados` → `los 35 optimizadores soportados`. |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: `El sistema soporta treinta y cuatro familias de optimizadores` → `treinta y cinco`; `métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon², NorMuon, Conda, TUON)` → append `, NAMO`. |
| 18 | `docs/architecture.md`, `README.md` (+ `configs/README.md` covered in row 8) | **Census pass, 34 → 35.** (a) `docs/architecture.md:26` `| Optimizer families | **34** |` → `**35**`; (b) `:87` `34 optimizers + registry + parameter routing` → `35 optimizers + registry + parameter routing`; (c) `:767` `### 8.1 Optimizers (34)` → `(35)`; (d) `:780` Orthogonalized row: add `namo` to the family list after `tuon` (list currently `muon … conda, tuon`), and append the clause `; \`namo\` rescales the orthogonalized momentum by a single Adam-type scalar α_t = ‖M̂_t‖_F/(√v̂_t+ε) built from a scalar second-moment EMA of ‖G_t‖²_F (μ₁/μ₂ from \`namo-betas\`, ε from the shared \`eps\`, decoupled decay scaled by α_t, one scalar of extra state)` at the end of the cell, after tuon's closing clause (`… the AdamW fallback is unchanged)`). (e) `README.md:20` At-a-Glance `| **48** | **34** | …` → `**35**` (currently accurate — the `tuon` commit fixed the old staleness); (f) `README.md:27` mermaid `34 optimizer families` → `35 optimizer families`; (g) `README.md:98` `Optimizer families | 34 across 8 categories` → `35 across 8 categories`; (h) `README.md:103` `52 named presets + 121 example configs` → `52 named presets + 122 example configs` (both numbers verified accurate at HEAD: 52 `configs/*.yaml`, 121 `configs/examples/*.yaml`; the namo example is the 122nd). Re-count at implementation time (`ls configs/*.yaml | wc -l`, `ls configs/examples/*.yaml | wc -l`) and write the actuals. Re-grep at implementation time: `grep -rn "34" README.md docs/architecture.md configs/README.md` and `grep -rn "(34)\|thirty-four\|treinta y cuatro" docs/paper docs/paper-es` to catch stragglers. |

**No change needed** (verified): `src/streamlit_gui/app.py` (schema-driven; `prefix_map.get(optimizer_class, optimizer_class)` fallback at line 398 covers new classes — the last five optimizer additions added no entry); `src/training/config_loader.py` (no optimizer list; grep `"conda"` in `src/**.py` hits only `factory.py`); `tests/test_optimizer_base.py` (base utilities only — a gate to *run*, not edit); `tests/test_yaml_examples.py` (auto-globs); SBERT enum (`$ref`s the canonical enum — `test_sbert_optimizer_enum_resolves_to_canonical` passes unchanged); `docs/paper/paper.tex`/`paper-es.tex` (annexes are `\input`).

### Hard rules (bind every row above)

1. **The schema is the contract** (`additionalProperties: false`): every `namo-*` key accepted at runtime must exist in `OPTIMIZER_ALLOWED_KEYS["namo"]` ∩ `_optimizer.yaml` ∩ the `_training.yaml` `if/then` rule; never read an undocumented key.
2. **Prefix contract:** `namo-<param>` (global) and `namo-<group>_<param>` (per group); groups are exactly `embeddings, pos, attention, ffn, norms, head, mhc, router, residual, other`.
3. **Papers are the theory source of truth (D11):** the EN annex formulations must match arXiv:2602.17080v2 verbatim; `paper-es` must translate them faithfully; the module implements Algorithm 1, not the official repo's extras.
4. CPU-first: implementation and all tests run on CPU (`uv run --extra cpu --extra train …`); no `.cuda()`, no CUDA-only ops.

### 4.1 Example YAML (row 7, full draft)

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
    optimizer_class: namo
    parameters:
      namo-lr_embeddings: 1e-6
      namo-lr_pos: 2e-6
      namo-lr_attention: 3e-6
      namo-lr_ffn: 2e-6
      namo-lr_norms: 5e-6
      namo-lr_head: 2e-6
      namo-lr_mhc: 3e-6
      namo-lr_router: 2e-6
      namo-lr_residual: 2e-6
      namo-lr_other: 2e-6
      namo-wd_embeddings: 0.01
      namo-wd_pos: 0.01
      namo-wd_attention: 0.01
      namo-wd_ffn: 0.01
      namo-wd_norms: 0.001
      namo-wd_head: 0.01
      namo-wd_mhc: 0.01
      namo-wd_router: 0.01
      namo-wd_residual: 0.01
      namo-wd_other: 0.01
      namo-betas_embeddings: [0.9, 0.95]
      namo-betas_pos: [0.9, 0.95]
      namo-betas_attention: [0.9, 0.95]
      namo-betas_ffn: [0.9, 0.95]
      namo-betas_norms: [0.9, 0.95]
      namo-betas_head: [0.9, 0.95]
      namo-betas_mhc: [0.9, 0.95]
      namo-betas_router: [0.9, 0.95]
      namo-betas_residual: [0.9, 0.95]
      namo-betas_other: [0.9, 0.95]
      namo-eps_embeddings: 1e-8
      namo-eps_pos: 1e-8
      namo-eps_attention: 1e-8
      namo-eps_ffn: 1e-8
      namo-eps_norms: 1e-8
      namo-eps_head: 1e-8
      namo-eps_mhc: 1e-8
      namo-eps_router: 1e-8
      namo-eps_residual: 1e-8
      namo-eps_other: 1e-8
      namo-betas: [0.95, 0.99]
      namo-ns_steps: 5
      namo-ns_eps: 1.0e-7
  scheduler_total_steps: 12000
  max_nan_retries: 4
  gradient_log_interval: 10
  csv_log_path: training_metrics_es_arch_hybrid_namo.csv
  telemetry_log_interval: 5
```

### 4.2 EN annex paragraph (row 12, full draft)

```latex
\paragraph{NAMO (Norm-Based Adaptive Moment Estimation with Orthogonalized Momentum).}
NAMO~\citep{zhang_namo_2026} keeps Muon's orthogonalized update direction and adds an
Adam-type \textbf{norm-based noise adaptation} on top of it. Matrix orthogonalization is
an unbounded operation that can amplify stochastic noise, so Muon's effective stepsize
does not shrink when gradients become noisy. NAMO tracks a single scalar EMA of the
squared Frobenius norm of the gradient and rescales the whole orthogonalized update by
the bias-corrected norm ratio $\lVert\widehat M_t\rVert_F/(\sqrt{\widehat v_t}+\epsilon_t)$
--- an aggregated signal-to-noise statistic analogous to Adam's coordinate-wise ratio.
Because the scaling is a single scalar, the update direction stays exactly orthogonal,
and the optimizer state grows by only one scalar per matrix over Muon. For a
hidden-layer weight matrix $W \in \mathbb{R}^{m\times n}$ with learning rate $\eta$,
momenta $\mu_1,\mu_2 \in [0,1)$, stability constant $\epsilon$ and decoupled weight
decay $\lambda$:
\begin{align}
M_t &= \mu_1 M_{t-1} + (1-\mu_1) G_t && \text{(momentum)} \\
v_t &= \mu_2 v_{t-1} + (1-\mu_2)\,\lVert G_t\rVert_F^2 && \text{(scalar norm EMA)} \\
O_t &= \mathrm{NS}_K(M_t) && \text{(orthogonalization)} \\
\alpha_t &= \frac{\sqrt{1-\mu_2^t}}{1-\mu_1^t}\cdot\frac{\lVert M_t\rVert_F}{\sqrt{v_t}+\epsilon}
   = \frac{\lVert\widehat M_t\rVert_F}{\sqrt{\widehat v_t}+\epsilon_t},
   \qquad \epsilon_t := \frac{\epsilon}{\sqrt{1-\mu_2^t}} && \text{(adaptive stepsize)} \\
W_t &= W_{t-1} - \eta\,\alpha_t\,\bigl(O_t + \lambda W_{t-1}\bigr) && \text{(update)}
\end{align}
with the same quintic Newton--Schulz coefficients $(a,b,c)=(3.4445,-4.7750,2.0315)$ as
Muon. Under nuclear-norm smoothness and bounded-variance noise, NAMO attains the optimal
deterministic rate $\mathcal{O}(T^{-1/2})$, and its stochastic bound decomposes into an
optimization term $\mathcal{O}(T^{-1/4})$ plus a variance term
$\mathcal{O}(\sqrt{\sigma}\,b^{-1/4}T^{-1/8})$, recovering the optimal $\mathcal{O}(T^{-1/4})$
once the batch size satisfies $b=\Omega(\sigma^2 T^{1/2})$.

\begin{algorithm}[H]
\caption{NAMO (Norm-Based Adaptive Moment Estimation with Orthogonalized Momentum)}
\label{alg:namo}
\begin{algorithmic}[1]
\Require Parameters $\theta$, learning rate $\eta$, momenta $\mu_1,\mu_2$, NS iters $K$, stability constant $\epsilon$, weight decay $\lambda$
\Ensure Updated parameters $\theta_T$
\For{each parameter $P$}
    \If{$P$ is a hidden-layer 2D matrix $W \in \mathbb{R}^{m\times n}$}
        \State $G \gets \nabla f(W)$ \Comment{Gradient}
        \State $M \gets \mu_1 M + (1-\mu_1) G$ \Comment{Momentum}
        \State $v \gets \mu_2 v + (1-\mu_2)\,\lVert G\rVert_F^2$ \Comment{Scalar second moment}
        \State $O \gets \mathrm{NS}_K(M)$ \Comment{Newton--Schulz}
        \State $\alpha \gets \frac{\sqrt{1-\mu_2^t}}{1-\mu_1^t}\cdot\frac{\lVert M\rVert_F}{\sqrt{v}+\epsilon}$ \Comment{Adaptive stepsize}
        \State $W \gets W - \eta\,\alpha\,(O + \lambda W)$ \Comment{Update}
    \Else
        \State $P \gets \operatorname{AdamW}(P, \nabla f(P); \eta, \beta_1, \beta_2, \epsilon, \lambda)$ \Comment{Non-matrix fallback}
    \EndIf
\EndFor
\Return $\theta_T$
\end{algorithmic}
\end{algorithm}

On GPT-2 pretraining over OpenWebText (124M/355M, 2000-step linear warmup then constant
LR), NAMO reaches validation losses of 3.0351/2.9516 versus 3.0643/2.9914 for AdamW and
3.0435/2.9684 for Muon, with optimal learning rates $\eta=0.012/0.007$ (roughly $9\times$
AdamW's) at $\mu_1=0.95$, $\mu_2=0.99$, $\lambda=0.01$, and it remains stable over a
wider learning-rate grid than both baselines. In this implementation the global
\texttt{namo-betas} suffix is $(\mu_1,\mu_2)$ (default $[0.95,0.99]$),
\texttt{ns\_steps}/\texttt{ns\_eps} configure Newton--Schulz, the shared per-group
\texttt{eps} is $\alpha_t$'s $\epsilon$, the shared \texttt{wd} is decoupled and scaled
by $\alpha_t$ as in the paper's update, and the shared \texttt{betas}/\texttt{eps}
configure the AdamW fallback for the embeddings/pos/head groups and every non-2-D
parameter. The reference release additionally exposes Nesterov momentum, an
effective-stepsize clamp and a $0.2\sqrt{\max(m,n)}$ shape scaling; Algorithm~1 of the
paper defines none of these, so this module follows the paper.

\paragraph{Cons.}
\begin{itemize}[leftmargin=*]
\item A single scalar stepsize carries no per-neuron information; the paper itself
  addresses this with the diagonal extension NAMO-D.
\item The update is not RMS-aligned, so Adam learning-rate schedules do not transfer
  unchanged: the paper's optima are roughly $9\times$ AdamW's.
\item The theory assumes exact orthogonalization while practice uses $K{=}5$
  Newton--Schulz iterations, and the empirical evidence is limited to GPT-2-scale
  pretraining.
\end{itemize}

\paragraph{Pros.}
\begin{itemize}[leftmargin=*]
\item First principled integration of orthogonalized momentum with Adam-type noise
  adaptation, with optimal deterministic and noise-adaptive stochastic convergence
  rates.
\item The scalar scaling preserves orthogonality exactly, and the effective stepsize
  shrinks automatically under noise or near stationarity.
\item Zero additional memory over Muon and negligible per-step overhead; outperforms
  AdamW and Muon on GPT-2 (124M/355M) over a wider stable learning-rate range.
\end{itemize}
```

### 4.3 ES annex paragraph (row 15, full draft — faithful translation)

```latex
\paragraph{NAMO (Estimación Adaptativa de Momentos Basada en Normas con Momentum Ortogonalizado).}
NAMO~\citep{zhang_namo_2026} conserva la dirección de actualización ortogonalizada de
Muon y le añade una \textbf{adaptación al ruido tipo Adam basada en normas}. La
ortogonalización de matrices es una operación no acotada que puede amplificar el ruido
estocástico, por lo que el tamaño de paso efectivo de Muon no se reduce cuando los
gradientes se vuelven ruidosos. NAMO rastrea una EMA escalar del cuadrado de la norma de
Frobenius del gradiente y reescala toda la actualización ortogonalizada mediante el
cociente de normas con corrección de sesgo
$\lVert\widehat M_t\rVert_F/(\sqrt{\widehat v_t}+\epsilon_t)$ --- un estadístico de
señal-ruido agregado análogo al cociente por coordenada de Adam. Como el escalado es un
único escalar, la dirección de actualización permanece exactamente ortogonal, y el
estado del optimizador crece en un solo escalar por matriz respecto de Muon. Para una
matriz de pesos de capa oculta $W \in \mathbb{R}^{m\times n}$ con tasa de aprendizaje
$\eta$, momenta $\mu_1,\mu_2 \in [0,1)$, constante de estabilidad $\epsilon$ y weight
decay desacoplado $\lambda$:
\begin{align}
M_t &= \mu_1 M_{t-1} + (1-\mu_1) G_t && \text{(momentum)} \\
v_t &= \mu_2 v_{t-1} + (1-\mu_2)\,\lVert G_t\rVert_F^2 && \text{(EMA escalar de normas)} \\
O_t &= \mathrm{NS}_K(M_t) && \text{(ortogonalización)} \\
\alpha_t &= \frac{\sqrt{1-\mu_2^t}}{1-\mu_1^t}\cdot\frac{\lVert M_t\rVert_F}{\sqrt{v_t}+\epsilon}
   = \frac{\lVert\widehat M_t\rVert_F}{\sqrt{\widehat v_t}+\epsilon_t},
   \qquad \epsilon_t := \frac{\epsilon}{\sqrt{1-\mu_2^t}} && \text{(paso adaptativo)} \\
W_t &= W_{t-1} - \eta\,\alpha_t\,\bigl(O_t + \lambda W_{t-1}\bigr) && \text{(actualización)}
\end{align}
con los mismos coeficientes quínticos de Newton--Schulz $(a,b,c)=(3.4445,-4.7750,2.0315)$
que Muon. Bajo suavidad en norma nuclear y ruido de varianza acotada, NAMO alcanza la
tasa determinista óptima $\mathcal{O}(T^{-1/2})$, y su cota estocástica se descompone en
un término de optimización $\mathcal{O}(T^{-1/4})$ más un término de varianza
$\mathcal{O}(\sqrt{\sigma}\,b^{-1/4}T^{-1/8})$, recuperando la óptima $\mathcal{O}(T^{-1/4})$
cuando el tamaño de lote cumple $b=\Omega(\sigma^2 T^{1/2})$.

\begin{algorithm}[H]
\caption{NAMO (Estimación Adaptativa de Momentos Basada en Normas con Momentum Ortogonalizado)}
\label{alg:namo-es}
\begin{algorithmic}[1]
\Require Parámetros $\theta$, tasa de aprendizaje $\eta$, momenta $\mu_1,\mu_2$, iteraciones NS $K$, constante de estabilidad $\epsilon$, weight decay $\lambda$
\Ensure Parámetros actualizados $\theta_T$
\For{cada parámetro $P$}
    \If{$P$ es una matriz 2D de capa oculta $W \in \mathbb{R}^{m\times n}$}
        \State $G \gets \nabla f(W)$ \Comment{Gradiente}
        \State $M \gets \mu_1 M + (1-\mu_1) G$ \Comment{Momentum}
        \State $v \gets \mu_2 v + (1-\mu_2)\,\lVert G\rVert_F^2$ \Comment{Segundo momento escalar}
        \State $O \gets \mathrm{NS}_K(M)$ \Comment{Newton--Schulz}
        \State $\alpha \gets \frac{\sqrt{1-\mu_2^t}}{1-\mu_1^t}\cdot\frac{\lVert M\rVert_F}{\sqrt{v}+\epsilon}$ \Comment{Paso adaptativo}
        \State $W \gets W - \eta\,\alpha\,(O + \lambda W)$ \Comment{Actualización}
    \Else
        \State $P \gets \operatorname{AdamW}(P, \nabla f(P); \eta, \beta_1, \beta_2, \epsilon, \lambda)$ \Comment{Respaldo no matricial}
    \EndIf
\EndFor
\Return $\theta_T$
\end{algorithmic}
\end{algorithm}

En el preentrenamiento de GPT-2 sobre OpenWebText (124M/355M, calentamiento lineal de
2000 pasos y LR constante después), NAMO alcanza pérdidas de validación de 3.0351/2.9516
frente a 3.0643/2.9914 de AdamW y 3.0435/2.9684 de Muon, con tasas óptimas
$\eta=0.012/0.007$ (unas $9\times$ la de AdamW) en $\mu_1=0.95$, $\mu_2=0.99$,
$\lambda=0.01$, y permanece estable en una rejilla de tasas más amplia que ambas líneas
base. En esta implementación el sufijo global \texttt{namo-betas} es $(\mu_1,\mu_2)$
(por defecto $[0.95,0.99]$), \texttt{ns\_steps}/\texttt{ns\_eps} configuran
Newton--Schulz, el \texttt{eps} compartido por grupo es el $\epsilon$ de $\alpha_t$, el
\texttt{wd} compartido es desacoplado y escalado por $\alpha_t$ como en la
actualización del artículo, y los \texttt{betas}/\texttt{eps} compartidos configuran el
respaldo AdamW para los grupos embeddings/pos/head y todo parámetro no 2-D. La
implementación de referencia expone además momentum de Nesterov, un tope del paso
efectivo y un escalado de forma $0.2\sqrt{\max(m,n)}$; el Algoritmo~1 del artículo no
define ninguno de ellos, así que este módulo sigue el artículo.

\paragraph{Contras.}
\begin{itemize}[leftmargin=*]
\item Un único paso escalar no porta información por neurona; el propio artículo lo
  aborda con la extensión diagonal NAMO-D.
\item La actualización no está alineada por RMS, por lo que los esquemas de tasa de
  Adam no se transfieren sin ajuste: los óptimos del artículo son unas $9\times$ los de
  AdamW.
\item La teoría asume ortogonalización exacta mientras la práctica usa $K{=}5$
  iteraciones de Newton--Schulz, y la evidencia empírica se limita al preentrenamiento
  a escala GPT-2.
\end{itemize}

\paragraph{Ventajas.}
\begin{itemize}[leftmargin=*]
\item Primera integración principiada del momentum ortogonalizado con la adaptación al
  ruido tipo Adam, con tasas de convergencia deterministas y estocásticas óptimas.
\item El escalado escalar preserva la ortogonalidad exactamente, y el paso efectivo se
  reduce automáticamente bajo ruido o cerca de la estacionariedad.
\item Cero memoria adicional sobre Muon y sobrecosto por paso insignificante; supera a
  AdamW y Muon en GPT-2 (124M/355M) en un rango estable de tasas más amplio.
\end{itemize}
```

---

## 5. Ordered execution checklist (implementation time — do NOT run now)

Environment (CPU-first, `AGENTS.md`): `uv sync --extra cpu --extra train`; every command below runs through `uv run --extra cpu --extra train`.

**Implementation (files 1–9):**
- [ ] 1. `src/model/optimizer/namo.py` (row 1 sketch, paper equations mapped line-by-line)
- [ ] 2. `src/model/optimizer/__init__.py` (row 2)
- [ ] 3. `src/model/optimizer/factory.py` — import, registry, allowed keys, dispatch (row 3)
- [ ] 4. `src/schema/_optimizer.yaml` — enum + both tables + both suffix lists + refs + `by_optimizer.namo` (row 4)
- [ ] 5. `src/schema/_training.yaml` — one `^namo-` if/then block (row 5)
- [ ] 6. `src/schema/_sbert.yaml` — `optimizer_kwargs` EN/ES (row 6)
- [ ] 7. `configs/examples/es_arch_hybrid_namo.yaml` (§4.1)
- [ ] 8. `configs/README.md` (row 8)
- [ ] 9. `full_tests/run_e2e.py` (row 9)

**Tests (file 10):**
- [ ] 10. `tests/test_optimizer_factory.py` — `required` set + `BuildOptimizerNAMOTests` (row 10)

**Docs (files 11–18):**
- [ ] 11. `docs/bibliography/optimizers.bib` (§2.6)
- [ ] 12. `docs/paper/appendices/annex-1-optimizers.tex` (§4.2, inserted at ~line 1821)
- [ ] 13. `docs/paper/appendices/annex-summary-tables.tex` — row + census lines 4/111
- [ ] 14. `docs/paper/sections/05-optimizer-families.tex` — line 4 count + list
- [ ] 15. `docs/paper-es/appendices/annex-1-optimizadores.tex` (§4.3, inserted at ~line 1854)
- [ ] 16. `docs/paper-es/appendices/annex-summary-tables.tex` — row + census lines 4/111
- [ ] 17. `docs/paper-es/sections/05-familias-optimizadores.tex` — line 4 count + list
- [ ] 18. `docs/architecture.md` + `README.md` census pass (row 18; re-grep the old count first)

**Verification (planned — expected results in §8):**
```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# fast e2e smoke — namo is appended LAST in OPTIMIZERS, so --limit 3 does NOT reach it;
# raise --limit to >= 34 for a category sweep, or target the combo directly:
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3   # does not include namo
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --list-combos  # confirm `opt_namo` exists
uv run --extra cpu --extra train python full_tests/run_e2e.py --only opt_namo            # the actual namo smoke
uv run --extra cpu --extra train python full_tests/run_e2e.py --dry-run                   # all combos validate, exit 0
```

**PDFs (only at implementation time):**
```bash
# PDF rebuild, from the repository root — only at implementation time
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

- **Preconditions (verified read-only today):** `/usr/bin/pdflatex` and `/usr/bin/bibtex` exist, so compilation is expected to work on this machine. (`sync-website.md`'s note "current machine: pdflatex=no" is stale — re-check with `command -v pdflatex bibtex` at implementation time.)
- **Fallback:** if `pdflatex`/`bibtex` are unavailable on the implementation machine, leave the `.tex`/`.bib` sources updated and note that the rebuild must happen on a TeX machine before the next release (the website sync copies prebuilt PDFs only, and the site probes `paper-en-copy`/`paper-es-copy` would then fail until the PDFs exist).
- **Post-compile verification:** no `Fatal error` in the logs; no `Citation .* undefined` warnings (in particular `zhang_namo_2026` must resolve — it appears in the new annex paragraphs and both summary-table rows); `docs/paper/paper.pdf` and `docs/paper-es/paper-es.pdf` are newer than the edited `.tex`/`.bbl` files; `git status --short docs/paper docs/paper-es` shows only intended `.tex`/`.bbl`/`.pdf` changes (no `*.aux *.log *.out *.toc *.blg`).

**Website sync → §6; commit/push → §7.** The mirror stays uncommitted until the commit phase.

---

## 6. Website sync (planned, not executed)

Follow `/sync-website` (`.opencode/command/sync-website.md`), driven by the read-only state script. Files that change for NAMO, per the file map:

1. `bash .opencode/command/sync-website-state.sh` — read the `[drift]`/`[probes]`/`[tools]` report first.
2. Copy (plain `cp`, no rsync; mirror = `erickfmm.github.io/frankenstein-transformer/`):
   - `src/schema.yaml` → `schema.yaml`; `src/schema/**` → `schema/**` (carries `_optimizer.yaml`, `_training.yaml`, `_sbert.yaml`),
   - `configs/examples/es_arch_hybrid_namo.yaml` → `examples/examples/`, `configs/README.md` → `examples/README.md`,
   - `docs/paper/{paper.tex, sections/*.tex, appendices/*.tex, paper.bbl, paper.pdf, figures/**}` → `paper/en/…`,
   - `docs/paper-es/{paper-es.tex, sections/*.tex, appendices/*.tex, paper-es.bbl, paper-es.pdf}` → `paper/es/…`,
   - `docs/bibliography/optimizers.bib` → `paper/bibliography/`,
   - `docs/architecture.md` → `paper/specs/architecture.md`.
3. Refresh the Paper-tab artifacts: `python3 flatten_tex.py en/paper.tex en/paper.bbl /tmp/paper-en-flat.tex --web --manifest /tmp/paper-en-figures.json` (+ ES counterpart), `pandoc … -o paper-en.html` / `paper-es.html`, `python3 postprocess_html.py --manifest … paper-en.html` / `paper-es.html` (exact commands in `sync-website.md` §3.3), and the top-level copies `cp paper/en/paper.pdf paper/en.pdf`, `cp paper/es/paper-es.pdf paper/es.pdf`.
4. **Validation Gate** (in order, all must pass):
   - `uv run --extra cpu --extra train python full_tests/param_count_check.py` → `… configs counted, 0 failures` (regenerates `full_tests/param_truth.json`; the new example config gets a model-side ground-truth entry — the intentional `param_truth.json` diff is committed with the main repo),
   - `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` → exit 0, `ALL PASS` (the estimator models the architecture, not the optimizer; the namo example reuses the nor_muon example's architecture, so no estimator change is expected),
   - state-script probes all PASS: `schema-refs`, `example-yaml`, `cli-flags`, `estimator`, `paper-en-copy`, `paper-es-copy`.
5. **Leave both repos uncommitted** until §7 (or hand off to `/make-a-commit` per the user's call).

---

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions: English, conventional tag, structured body; never stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json` changes are intentional.

1. **Main repo** — `feat(optimizer): add NAMO (arXiv:2602.17080)`

   Body sections: *Implementation* (new `src/model/optimizer/namo.py`: Algorithm 1 of the paper — scalar norm-EMA second moment, α_t-scaled orthogonalized update and decoupled decay, AdamW fallback for embeddings/pos/head and non-2-D params; instance-attribute betas to avoid group-key shadowing), *Schema* (`namo` enum, bilingual tables/suffix lists, `by_optimizer.namo`, `^namo-` if/then, SBERT `optimizer_kwargs` EN/ES), *Docs* (EN/ES annex paragraphs + algorithm floats, summary-table rows, family-section counts 34→35, bibliography `zhang_namo_2026`, architecture/README census, rebuilt PDFs), *Example* (`es_arch_hybrid_namo.yaml`), *Tests* (`BuildOptimizerNAMOTests`, registry required set, e2e list).

   ```bash
   git add src/model/optimizer/namo.py src/model/optimizer/__init__.py src/model/optimizer/factory.py \
           src/schema/_optimizer.yaml src/schema/_training.yaml src/schema/_sbert.yaml \
           configs/examples/es_arch_hybrid_namo.yaml configs/README.md \
           full_tests/run_e2e.py full_tests/param_truth.json \
           tests/test_optimizer_factory.py \
           docs/bibliography/optimizers.bib \
           docs/paper/appendices/annex-1-optimizers.tex docs/paper/appendices/annex-summary-tables.tex \
           docs/paper/sections/05-optimizer-families.tex docs/paper/paper.bbl docs/paper/paper.pdf \
           docs/paper-es/appendices/annex-1-optimizadores.tex docs/paper-es/appendices/annex-summary-tables.tex \
           docs/paper-es/sections/05-familias-optimizadores.tex docs/paper-es/paper-es.bbl docs/paper-es/paper-es.pdf \
           docs/architecture.md README.md
   git commit -m "feat(optimizer): add NAMO (arXiv:2602.17080)"   # with the structured body above
   git push origin HEAD:main
   ```

2. **Mirror** — `docs(site): sync NAMO optimizer (arXiv:2602.17080)` (exact drift list comes from the state script at implementation time; expected paths below)

   ```bash
   git -C erickfmm.github.io add \
           frankenstein-transformer/schema.yaml frankenstein-transformer/schema/_optimizer.yaml \
           frankenstein-transformer/schema/_training.yaml frankenstein-transformer/schema/_sbert.yaml \
           frankenstein-transformer/examples/README.md \
           frankenstein-transformer/examples/examples/es_arch_hybrid_namo.yaml \
           frankenstein-transformer/paper/en/appendices/annex-1-optimizers.tex \
           frankenstein-transformer/paper/en/appendices/annex-summary-tables.tex \
           frankenstein-transformer/paper/en/sections/05-optimizer-families.tex \
           frankenstein-transformer/paper/en/paper.bbl frankenstein-transformer/paper/en/paper.pdf \
           frankenstein-transformer/paper/es/appendices/annex-1-optimizadores.tex \
           frankenstein-transformer/paper/es/appendices/annex-summary-tables.tex \
           frankenstein-transformer/paper/es/sections/05-familias-optimizadores.tex \
           frankenstein-transformer/paper/es/paper-es.bbl frankenstein-transformer/paper/es/paper-es.pdf \
           frankenstein-transformer/paper/bibliography/optimizers.bib \
           frankenstein-transformer/paper/specs/architecture.md \
           frankenstein-transformer/paper/paper-en.html frankenstein-transformer/paper/paper-es.html \
           frankenstein-transformer/paper/en.pdf frankenstein-transformer/paper/es.pdf
   git -C erickfmm.github.io commit -m "docs(site): sync NAMO optimizer (arXiv:2602.17080)"
   git -C erickfmm.github.io push origin HEAD:main
   ```

---

## 8. Verification matrix

| Gate | Command (implementation time) | Expected result |
|---|---|---|
| Registry/allowed-keys/step | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py -v` | All pass, incl. the 10 new `BuildOptimizerNAMOTests` methods and `test_registry_contains_all_required` with `namo` |
| Base utilities | `… -m pytest tests/test_optimizer_base.py -v` | All pass (unchanged file, must stay green) |
| Schema parity | `… -m pytest tests/test_schema_consistency.py -v` | `test_optimizer_prefixes_match_conditional_rules`, `test_all_presets_validate` (incl. the new example), `test_sbert_optimizer_enum_resolves_to_canonical` all pass |
| Example smoke | `… -m pytest tests/test_yaml_examples.py -v` | Auto-injected `test_load_es_arch_hybrid_namo` passes (`load_training_config` OK) |
| Full suite | `… -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | No failures (76 test files stay green; no regression in naming-conventions or defaults tests) |
| e2e config validation | `… python full_tests/run_e2e.py --dry-run` | Exit 0; `opt_namo` combo validates |
| e2e smoke | `… python full_tests/run_e2e.py --only opt_namo` | `opt_namo` trains and reports PASS in `results.json`/`results.md` (CPU, default timeout) |
| PDF (EN) | `pdflatex`×2 + `bibtex` sequence (§5) | No `Fatal error`; no `Citation .* undefined`; `zhang_namo_2026` resolves; `paper.pdf` newer than the edited `.tex`/`.bbl` |
| PDF (ES) | Same in `docs/paper-es` | Same checks for `paper-es.pdf` |
| PDF git hygiene | `git status --short docs/paper docs/paper-es` | Only intended `.tex`/`.bbl`/`.pdf` files; no aux artifacts |
| Param ground truth | `… python full_tests/param_count_check.py` | `… configs counted, 0 failures`; `param_truth.json` gains the new example's entry |
| JS estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | Exit 0, `ALL PASS` (≤ 1% drift) |
| Site probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, `estimator`, `paper-en-copy`, `paper-es-copy` all PASS; drift list empty for the synced files |

---

## 9. Risks, assumptions and open questions

1. **NAMO-D status:** implemented in the follow-up as registry `namo_d` with the extra `namo_d-clamp` suffix (c ∈ (0,1]; default 0.1, the paper's GPT-2 124M value; the 355M sweep selects 0.9 and the reference implementation defaults to 0.75). Its state adds one ℝⁿ column second moment per matrix, and its tests/docs mirror NAMO's.
2. **Official-code extras are not implemented** (design decision, D11): the reference repo defaults `nesterov=True`, clamps the adaptive LR at 1.0 and multiplies by `0.2·√max(m,n)` (`adjust_lr_for_muon`); none of these appear in Algorithm 1. If the user later wants official-parity knobs (`nesterov`, `scale_coeff`), that is a separate schema/registry extension. Assumption to re-verify: the boundedness argument (α_t ≤ √((1−μ₁)/(1−μ₂))) holds for the exact recursion, making the clamp unnecessary.
3. **Momentum parameterization:** NAMO must keep the paper's (1−μ₁) factor (unlike `muon.py`'s `M ← βM + G`), because α_t reads ‖M_t‖ and the bias corrections directly. A wrong parameterization silently rescales α_t — the `test_first_step_matches_algorithm1` test guards this.
4. **`namo-betas` (global) vs `namo-betas_<group>` (shared)** are distinct keys consumed by different code paths (`self.betas` vs `_build_param_groups_for_optimizer`). Mechanically safe (verified against `extract_prefixed_parameters` + `_COMMON_PER_GROUP_KEYS`), but document the overloading everywhere the suffix is described (schema EN/ES, configs/README, SBERT kwargs).
5. **Zero-gradient steps are no-ops** (α_t = 0 ⇒ no update, no decay). Documented divergence from `muon` (whose decoupled decay fires unconditionally); covered by an explicit test with non-zero gradients.
6. **LR scale:** NAMO is *not* RMS-aligned (unlike `nor_muon`/`ada_muon`); the paper's optima are ≈9× AdamW's. Class default `lr=1e-3` (factory injects group LR anyway); the example preset keeps the `DEFAULT_GROUP_LR`-scale values like every other Muon-family example (they are schema smoke-tests, not training recipes — the paper-suggested η≈0.01 for hidden matrices is recorded in the schema/README prose).
7. **Census state (re-verified at this plan's refresh):** HEAD is consistent — `README.md:20/27/98` and `docs/architecture.md:26/87/767` all say **34** optimizers, `README.md:103` says 52 presets + 121 examples, matching the filesystem. This change bumps the optimizer census to **35** and the example count to **122** (row 18 lists every occurrence). The first draft of this plan (written pre-`tuon`, when README said 32/33 and examples were 117/120) is superseded. Still re-run `ls … | wc -l` at implementation time before writing counts.
8. **No task/mixer/model-class restrictions:** NAMO is a standard `step()` optimizer — valid for all nine `training.task` values, all four model classes, and SBERT (`optimizer_kwargs`); only `eve`/`ea4llm` carry the closure/`gradient_accumulation_steps: 1` restrictions. 4-D parameters (ViT/audio conv kernels) take the AdamW fallback (`p.ndim != 2`), matching `muon.py`/`nor_muon.py` — note the official repo instead flattens 4-D to 2-D and orthogonalizes; the repo hybrid convention wins (documented in the module docstring).
9. **PDF tooling assumption:** `pdflatex`/`bibtex` verified present (`/usr/bin/…`), but `sync-website.md` claims otherwise (stale note) — re-check at implementation time and use the §5 fallback if the TeX machine differs.
10. **Assumptions to verify at implementation time:** none of the commands in §5/§6/§7/§8 were executed during planning; all line anchors were re-verified against HEAD (post-`tuon`) on 2026-11-08 and must still be re-grepped before editing (files drift).

---

## 10. Out of scope

- **This planning command changed nothing** outside `.opencode/plan/add-optimizer-2602.17080.md`: no code, schema, preset, test, doc, PDF, website-mirror or git change was made (see `git status --short` in the delivery message). This plan file already existed (written by an earlier session before `tuon` landed); this run refreshed it in place against current HEAD. The other four untracked plan files (`add-mixer-2305.05084.md`, `add-optimizer-2509.24218.md`, `add-optimizer-2510.05491.md`, `add-optimizer-2610.35130.md`) were left untouched.
- Not planned: NAMO-D (see §9.1), official-code extras (§9.2), a `namo` annex figure (§3), Streamlit GUI edits (schema-driven, no entry needed), estimator changes (architecture-independent), any destructive/renaming step (none implied by the paper).
