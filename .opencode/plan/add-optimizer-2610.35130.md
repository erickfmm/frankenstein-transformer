# Plan: add TUON optimizer (arXiv:2610.35130)

**Paper:** *TUON: A Tensor-Aware Optimizer Utilizing Tensor-SVD and Polarization for Large Language Model Pretraining* — Anda Tang, Xuanlin Yang, Junjie Wang, Zhouchen Lin (State Key Lab of General AI, Peking University; Pazhou Laboratory), arXiv:2610.35130v1 [cs.LG], 26 Oct 2026. Anonymous code: `anonymous.4open.science/r/tuon-802A/`.

**One-paragraph summary.** Matrix-aware optimizers (Muon) orthogonalize each layer's momentum independently and lose cross-layer information; existing tensor-aware optimizers (TEON) matricize/unfold the layer stack and degrade beyond two layers or beyond QKV modules. TUON replaces the unfolding with a **transform-based tensor algebra** derived from the Tensor-SVD (T-SVD): stack the K compatible momentum matrices of a group into a tensor `M ∈ R^{m×n×K}`, mix the slices with a unitary layer transform `Q` (`M̃ = M ×₃ Q`), polarize each transformed slice independently (Newton–Schulz), invert the transform (`P = P̃ ×₃ Qᴴ`), rescale the whole stack to RMS `s = 1/√max(m,n)` with a shared factor `γ_t = s/(RMS(P_t)+ε)`, and apply the decoupled-decay update `W ← (1−ηλ)W − ηγ_t P`. The default **DecorQ** transform is data-dependent: `Q = Uᵀ` from the eigendecomposition of the K×K layer Gram matrix `C_t = M̂ M̂ᵀ` (refreshed every 200 iterations), which makes the transformed slices mutually Frobenius-orthogonal so the per-slice polar maps act in disjoint subspaces. The paper proves TUON is steepest descent under the Q-transformed tensor spectral norm, equals the tensor polar factor of a T-product SVD, distorts the Muon geometry by at most √K, and dominates the TEON oracle at equal radius. Empirically it beats AdamW, Muon and TEON on GPT-124M/350M and Qwen-0.6B/4B at ≤12.5% step-time overhead over AdamW, is the only method that *improves* when stacking goes from K=2 to K=full, and transfers from QKV to QKVO+MLP (Module_All). This repo's `Orthogonalized` bucket (Muon → Conda, 9 members) is the natural home: TUON slots in as the 10th, following the same hybrid AdamW-fallback recipe as `muon`, with K=1 buckets reducing exactly to `muon` (paper Prop. 2: "For K = 1, this reduces to matrix Muon").

---

## 2. Paper extract

### 2.1 Update rule (Algorithm 1 + §2.1–2.2 + Appendices E/F/G, verbatim)

For one **compatible group** of K same-shape matrices `W⁽ᵏ⁾ ∈ R^{m×n}`, gradients `G⁽ᵏ⁾`, momentum coefficient β, learning rate η_t, decoupled weight decay λ, NS steps J, refresh interval I:

```
G_t  ← stack₃(G⁽¹⁾, …, G⁽ᴷ⁾)                                  (Alg. 1 line 1)
M_t  = β M_{t-1} + (1−β) G_t ∈ R^{m×n×K}                       (Alg. 1 line 2; Eq. of §2.1)
C_t  := M̂_t M̂_tᵀ ∈ R^{K×K},  (C_t)_{kℓ} = ⟨M_t(:,:,k), M_t(:,:,ℓ)⟩_F
                                                               (Alg. 1 line 3; Eq. 2/92; M̂ = mode-3 unfolding, rows vec(M⁽ᵏ⁾)ᵀ)
C_t  = U_t Λ_t U_tᵀ                                            (Alg. 1 line 4, eigendecomposition)
Q_t  ← U_tᵀ                     ▷ Refresh per 200 iterations   (Alg. 1 line 5; I.3: "refreshed every 200 iterations")
M̃_t  = M_t ×₃ Q_t,   M̃_{t,j} = Σ_k (Q_t)_{jk} M⁽ᵏ⁾_t           (Alg. 1 line 6; Eq. 1/77)
for j = 1..K in parallel: P̃_t(:,:,j) ← polarization(M̃_{t,j}) ▷ Newton–Schulz, J steps
                                                               (Alg. 1 lines 7–9; Eq. 4)
P_t  = P̃_t ×₃ Q_tᴴ                                             (Alg. 1 line 10; Eq. 5)
W_{t+1} ← (1 − η_t λ) W_t − η_t γ_t P_t,  γ_t ← 1/(RMS(P_t) + ε)
                                                               (Alg. 1 line 11)
```

Slice polarization via Newton–Schulz (Appendix F, Eqs. 97–99): transpose the slice when `m > n` (keep the Gram small); initialize `X_{0,j} = M̃_{t,j}/(‖M̃_{t,j}‖_F + ε)`; per step `A_{ℓ,j} = X_{ℓ,j} X_{ℓ,j}^H` and `X_{ℓ+1,j} = aX_{ℓ,j} + (bA_{ℓ,j} + cA²_{ℓ,j}) X_{ℓ,j}` with the quintic coefficients `(a, b, c) = (3.4445, −4.7750, 2.0315)`; `J = 5` in the implementation ("In practice, J = 4 or 5 suffices"; I.3 uses J = 5). Exact polarization is `polar(M̃_j) = U_{t,j} Vᵀ_{t,j}` (Eq. 4).

Stack scaling (Appendix G, Eqs. 100–103): `RMS(P_t) = √( (mnK)⁻¹ Σᵢ Σⱼ Σₖ |P_t(i,j,k)|² )`, `γ_t = s/(RMS(P_t) + ε)` with target RMS `s = 1/√max(m,n)` (I.3: "The scaling factor is s = 1/√max(m,n) with ϵ = 10⁻⁸"), scaled update `P̃ᵤpdate = γ_t P_t`, and the decoupled update `W_{t+1} = (1 − η_t λ) W_t − η_t γ_t P_t`. Because γ_t is shared across all layers it "changes the effective trust-region radius but not the relative direction across slices". By Prop. 3, full-rank slices give `RMS(P_t) = 1/√max(m,n)` independently of K, so `γ_t ≈ 1` and TUON's per-matrix update magnitude matches Muon's (Muon's NS output also has RMS `1/√max(m,n)`) — learning rates transfer from `muon`.

**DecorQ (default transform; §2.1 + App. E):** `Q = U₃ᵀ` from the SVD of the mode-3 unfolding `M̂_t = U₃ Σ₃ V₃ᵀ` — equivalently the eigenvectors of the K×K Gram `C_t` (Algorithm 1 lines 3–5 use `eigendecomposition`; `torch.linalg.eigh` suffices). It diagonalizes the transformed Gram (`Ĉ_t = Σ₃²`), so transformed slices are mutually Frobenius-orthogonal, their polar factors stay orthogonal (`⟨P̃_i, P̃_j⟩_F = 0`), and the update slices evolve in disjoint subspaces — "K independent single-channel problems". Fixed unitary alternatives (App. E): **DCT-II** (orthonormal, zero-based `α₀ = K^{-1/2}, α_j = (2/K)^{1/2}`), **Haar** (recursive `H_{2n} = 2^{-1/2} [[H_n⊗[1,1]], [I_n⊗[1,−1]]]`), **DFT/FFT** (`Q_{jk} = K^{-1/2} e^{−2πijk/K}`; "Frontal slices and NS products are complex-valued, with Hermitian rather than ordinary transposes"). Ablation order (Qwen3-4B, K=full): DecorQ 2.7598 < DFT 2.7631 < Haar 2.7635 < Residual-DecorQ 2.7640 < DecorQ-static 2.7658 < DCT 2.8100. Non-adaptive (static) DecorQ freezes the basis at initialization and is slightly worse; the refresh is what helps.

**Scope of the tensor path:** the paper's `Module_All` (QKVO + MLP — "covering all major weight tensors in the network") matches or beats `Module_QKV`, so the natural mapping is the repo's Muon hybrid routing: **every hidden-layer 2-D matrix takes the TUON stack path; the `embeddings`/`pos`/`head` groups and every non-2-D parameter take the decoupled AdamW fallback** (`base.ADAMW_FALLBACK_GROUPS` + `adamw_fallback_update`). The paper's future-work note ("extending TUON to other parameter groups") concerns cross-*parameter-category* stacking, not the hybrid recipe.

### 2.2 Algorithm 1 (verbatim pseudocode)

```
Algorithm 1  TUON
Require: compatible matrices {W⁽ᵏ⁾_t}ₖ₌₁ᴷ, gradients {G⁽ᵏ⁾_t} and momentum {M⁽ᵏ⁾_t};
         learning rate η_t, momentum coefficient β, weight decay coefficient λ, NS steps J
1:  G_t ← stack₃(G⁽¹⁾_t, …, G⁽ᴷ⁾_t)
2:  M_t ← β M_{t-1} + (1−β) G_t
3:  Compute the Gram matrix C_t := M̂_t M̂_tᵀ ∈ R^{K×K}, (C_t)_{kℓ} = ⟨M_t(:,:,k), M_t(:,:,ℓ)⟩_F
4:  Compute eigendecomposition C_t = U_t Λ_t U_tᵀ
5:  Q_t ← U_tᵀ                    ▷ Refresh per 200 iterations
6:  M̃_t ← M_t ×₃ Q_t
7:  for j = 1, …, K in parallel do
8:      P̃_t(:,:,j) ← polarization(M̃_t(:,:,j))   ▷ Newton–Schulz, J steps
9:  end for
10: P_t ← P̃_t ×₃ Qᴴ_t
11: W_{t+1} ← (1 − η_t λ) W_t − η_t γ_t P_t  with γ_t ← 1/(RMS(P_t) + ε)
```

No warmup/schedule interaction is defined by the algorithm (external schedulers apply unchanged); the experiments use 2000-step linear warmup + cosine decay (GPT-2) under standard `training.*` schedulers.

### 2.3 Hyperparameters

| Hyperparameter | Paper symbol | Type | Paper value | Our key | Scope |
|---|---|---|---|---|---|
| Learning rate | η_t | float | peak 1e-3 (GPT-2 124M/350M, Qwen3-4B), 2e-3 (Qwen3-0.6B) — Table 5; "Moonlight adjusted-LR" | shared `tuon-lr_<group>` | per group |
| Momentum | β | float | required by Alg. 1; value not stated (repo Muon-family convention **0.95** — assumption §9.4) | `tuon-momentum` | global |
| Weight decay | λ | float | decoupled (Eq. 103); value not stated | shared `tuon-wd_<group>` (decoupled) | per group |
| NS steps | J | int | **5** (I.3 "J = 5"; Table 11 "Newton–Schulz (J = 5)") | `tuon-ns_steps` | global |
| NS normalization ε | ε | float | "small constant" (App. F); repo convention **1e-7** | `tuon-ns_eps` | global |
| Refresh interval | I | int | **200** (Alg. 1 line 5 "Refresh per 200 iterations"; I.3 "refreshed every 200 iterations"; App. G.1 says "e.g., 100" — Algorithm 1 wins) | `tuon-refresh_interval` | global |
| RMS-scale ε | ε | float | **1e-8** (I.3 "with ϵ = 10⁻⁸") | `tuon-rms_eps` | global |
| Target RMS | s | float | **1/√max(m,n)**, fixed by Prop. 3 / I.3 | not exposed (constant) | — |
| Layer transform | Q | enum | **DecorQ** (default, best); DCT / Haar / DFT alternates (App. E, Table 7) | `tuon-transform` ∈ {decorq, dct, haar, dft} | global |
| Stacking size | K | enum | **full** (best with DecorQ; Table 3/8) or pairs (K=2, the TEON-comparable setting) | `tuon-stack` ∈ {full, pairs} | global |
| AdamW fallback β, ε | — | tuple/float | repo defaults (0.9, 0.95) / 1e-8 | shared `tuon-betas_<group>`, `tuon-eps_<group>` | per group |

### 2.4 State buffers, cost, complexity

- **TUON path:** one `momentum_buffer` (m×n) per 2-D parameter — identical to Muon (mn numbers per matrix, ~½ of Adam) — plus one cached **K×K layer basis** + step counter per same-shape bucket (K² scalars, negligible; refreshed every I steps). No per-element second moment.
- **AdamW fallback** (`embeddings`/`pos`/`head` + non-2-D): `exp_avg` + `exp_avg_sq` via `base.adamw_fallback_update`.
- **Per-step cost (App. G.1, Table 1):** slice-wise NS `O(J·K·mn·min(m,n)) = O(K·mn·min(m,n))` (dominant) + transform/inverse `O(K²mn)` each + Gram `O(K²mn)` + `eigh` of K×K `O(K³)` amortized over I steps. Total `O(K·mn·min(m,n) + K²mn)`, same dominant term as Muon (Table 1: Muon O(Kmn·min(m,n)); TEON O(Kmn·min(m,nK))).
- **Theory:** steepest descent under `‖·‖_{Q,Ts}` with dual `‖·‖_{Q,Tn}` (Thm 1, Prop 1); update = tensor polar factor `tpolar_Q(M) = U *_Q J_supp *_Q Vᵀ` of the T-product T-SVD (Prop 2; K=1 → Muon, Q=I → independent per-layer polar); transform non-redundancy under slice-wise polar (Thm 2); Muon-geometry distortion ≤ √K (Thm 3, Cor. 1: smoothness gap ≤ K); ≥ TEON oracle, up to √K stronger (Thm 4); O(T^{-1/2}) convergence (Thm 5).
- **Results (Table 2/6, validation loss):** TUON **2.9357 / 2.6925 / 2.8887 / 2.7598** vs AdamW 2.9230/2.7307/2.9652/2.8108, Muon 3.0488/2.7404/2.9056/2.7636, TEON 2.9442/2.6935/2.8978/2.7604 (GPT-124M/350M, Qwen3-0.6B/4B); beats Muon and TEON on all four; Δ vs AdamW −0.0382/−0.0765/−0.0510 on the three ≥350M scales. Step time 162.12/411.64/3378.3/6150.7 ms → 0.2–1.1% over TEON, ≤12.5% over AdamW (Table 4/12). K=full beats K=2 with DecorQ on all four models (only method that improves with K); Module_All ≥ Module_QKV; NS(J=5) ≈ PolarExpress (differences ≤0.003, an order of magnitude below the transform effect).

### 2.5 Other optimizers in the paper — recorded, NOT planned

TUON is the paper's single optimizer; the alternatives are *transform choices* (DCT/Haar/DFT/Residual-DecorQ/static DecorQ) and *polarization methods* (PolarExpress), all exposed as suffixes (`tuon-transform`) or noted as future work (PolarExpress as a drop-in polar solver; Residual-DecorQ as a basis source). Nothing else needs a registry entry.

### 2.6 BibTeX data (repo style: `<firstauthor>_<shortname>_<year>`, `@misc` arXiv entries)

```bibtex
@misc{tang_tuon_2026,
      title={TUON: A Tensor-Aware Optimizer Utilizing Tensor-SVD and Polarization for Large Language Model Pretraining},
      author={Anda Tang and Xuanlin Yang and Junjie Wang and Zhouchen Lin},
      year={2026},
      eprint={2610.35130},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.35130},
}
```

(Repo convention — cf. `wang_conda_2025`, `li_normuon_2025`, `liu_muon2_2026` at `docs/bibliography/optimizers.bib:383,403,413` — is `firstauthor_shortname_year`; first author Anda Tang → `tang_tuon_2026`.)

---

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `tuon` | Paper short name (TUON), lower-cased snake_case; no collision — `grep -rn "tuon\|TUON" src/ docs/ configs/ tests/ full_tests/` = 0 hits today; registry has 33 entries (`factory.py:59-93`) |
| Class name | `TUON` | Paper caps-acronym style; precedent `ROOT`, `EA4LLM` in the same `__init__.py` `__all__` |
| Closest peer | `src/model/optimizer/muon.py` (hybrid routing, NS iteration, decoupled decay, momentum buffer) + `nor_muon.py` (module-level `_newton_schulz` helper importable by tests — `test_optimizer_factory.py:12`) + `conda.py` (cached-basis-in-state pattern: `proj_basis` refreshed every `update_proj_gap`, cf. its state keys in `test_optimizer_factory.py:558-579`) | TUON's `step()` is the first *two-pass* member: per-param momentum update, then per-(group, shape)-bucket stack processing |
| Family category | **Orthogonalized** (architecture.md §8.1 row, line 780) = "geometry-oriented" (paper §5) = "Second-Order, Geometric & Orthogonality" (summary-table header, line 158) | TUON is a polar/orthogonalized method generalized to tensors; joins `muon … conda` |
| Figure in annex | **Omit** | No `tuon.mmd/.pdf` in the established vector style; the last five optimizer additions (dion3, muon2, nor_muon, conda, adam_pm) have no figure — `docs/paper/figures/optimizers/` has none of them (verified read-only) |
| Scope | **TUON only**, with `transform` (decorq/dct/haar/dft) and `stack` (full/pairs) as schema suffixes | The paper's ablation axes become config knobs; Residual-DecorQ and PolarExpress recorded under Open questions (§9) |
| Fidelity | **Paper Algorithm 1 exactly** (D11) | Implement `(1−β)` in the momentum update (Alg. 1 line 2) even though it cancels through the scale-invariant polar map (Frobenius-normalized NS + γ_t RMS rescale) — a code comment must state the equivalence so the deviation from `muon.py`'s `M ← βM + G` is deliberate |
| Bucketing | Within each param group, stack 2-D params by **exact shape**, preserving registration order; `stack: full` = one bucket per shape (paper default); `stack: pairs` = consecutive pairs (+ possible K=1 leftover) | The factory delivers unnamed per-category param lists (`base.categorize_param_groups`), so shape-bucketing is the only faithful realization of "one compatible group"; same-shape matrices across layers of a homogeneous pattern give the paper's K=full stacks |

### Suffix table

| Key | Type | Default | Coercion | Scope | EN description | ES description |
|---|---|---|---|---|---|---|
| `tuon-momentum` | float | 0.95 | `to_float` | global | momentum coefficient β of the momentum tensor | coeficiente de momentum β del tensor de momentum |
| `tuon-ns_steps` | int | 5 | `to_int` | global | Newton–Schulz iterations per transformed slice | iteraciones de Newton–Schulz por slice transformado |
| `tuon-ns_eps` | float | 1e-7 | `to_float` | global | Newton–Schulz Frobenius-normalization epsilon | épsilon de la normalización Frobenius de Newton–Schulz |
| `tuon-refresh_interval` | int | 200 | `to_int` | global | DecorQ basis refresh period I (paper default 200) | periodo de refresco I de la base DecorQ (200 por defecto) |
| `tuon-rms_eps` | float | 1e-8 | `to_float` | global | epsilon of the stack RMS scaling γ_t = s/(RMS(P)+ε) | épsilon del escalado RMS de la pila γ_t = s/(RMS(P)+ε) |
| `tuon-transform` | enum str | `decorq` | `str.strip().lower()` | global | layer transform Q: `decorq` (eigenvectors of the layer Gram matrix), `dct`, `haar`, `dft` | transformada de capas Q: `decorq` (autovectores de la matriz de Gram de capas), `dct`, `haar`, `dft` |
| `tuon-stack` | enum str | `full` | `str.strip().lower()` | global | stacking: `full` (every same-shape matrix of the group) or `pairs` (K=2) | apilamiento: `full` (toda matriz de la misma forma del grupo) o `pairs` (K=2) |
| shared `lr_*`, `wd_*`, `betas_*`, `eps_*` | — | factory defaults | factory | per group | `wd` decoupled on both paths; `betas`/`eps` configure the AdamW fallback only (the TUON path has no second moment) | `wd` desacoplado en ambas rutas; `betas`/`eps` configuran solo el respaldo AdamW |

- `supports_shared_per_group_suffixes: true` (the norm).
- `OPTIMIZER_ALLOWED_KEYS["tuon"] = _COMMON_PER_GROUP_KEYS | {"momentum", "ns_steps", "ns_eps", "refresh_interval", "rms_eps", "transform", "stack"}`.
- **No shadowing pitfall:** unlike the NAMO plan's global `betas`, TUON's specific globals (`momentum`, `refresh_interval`, `transform`, `stack`, …) are *not* injected per group by `_build_param_groups_for_optimizer` (`factory.py:193-207` only injects `lr`/`weight_decay`/`betas`/`eps`/`name`), so storing them in `defaults` is safe; the shared group `betas`/`eps` feed only `adamw_fallback_update` (exact `muon.py` pattern).

### Example preset

`configs/examples/es_arch_hybrid_tuon.yaml` — MLM task (`training.task: mlm`) on `frankenstein_text_encoder`, a structural copy of `es_arch_hybrid_muon2.yaml` with `tuon-` keys (the whole `es_arch_hybrid_<muon-family>.yaml` convention: turbo_muon, ada_muon, dion3, muon2, nor_muon, root, conda all live there). Smallest valid config: same encoder/dims/block as every peer; `test_yaml_examples.py` auto-globs it and `test_schema_consistency.py`'s preset validation covers it — zero extra test wiring. A `causal_lm_tuon.yaml` is not needed: TUON is architecture-agnostic and the repo's Muon-family examples are all `es_arch_hybrid_*`.

### Census bump

**33 → 34** optimizer families (actual old value read from the repo: `docs/architecture.md:767` `### 8.1 Optimizers (33)`; registry has 33 entries; `docs/architecture.md:26`, `:87`). Full census-pass list in file-plan row 18, including two **pre-existing drifts** to fix while bumping: `README.md:20` At-a-Glance says **32** (stale even before this change), and `README.md:103` says "51 named presets + 117 example configs" while the filesystem has **52** root presets and **120** examples today (verified read-only: `ls configs/*.yaml | wc -l` = 52, `ls configs/examples/*.yaml | wc -l` = 120). ⚠️ **Collision:** the untracked NAMO plan (`.opencode/plan/add-optimizer-2602.17080.md`) also targets 33→34 — whichever lands second must re-grep the actual current count and bump from there (34→35). Same for the annex insertion point (both insert after Conda) and `configs/README.md` example lists.

---

## 4. File-by-file plan table

Execution order. Line anchors are current-HEAD values; re-grep before editing.

| # | File | Planned change (be concrete) |
|---|---|---|
| 1 | `src/model/optimizer/tuon.py` | **New module.** `torch.optim.Optimizer` subclass; module docstring cites the paper (mirror `muon.py:1-17`): `Tang, A., Yang, X., Wang, J., & Lin, Z. (2026). TUON: A Tensor-Aware Optimizer Utilizing Tensor-SVD and Polarization for Large Language Model Pretraining. arXiv:2610.35130.` plus a mapping note (shape-bucket = "compatible group"; K=1 reduces to Muon; the `(1−β)` momentum factor is kept per Algorithm 1 even though it cancels through the scale-invariant polar + RMS rescale). `from __future__ import annotations`; imports `math`, `torch`, `from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update`. Module-level helpers (importable by tests, mirror `nor_muon.py`'s `_newton_schulz` and `root.py`'s `_shape_coefficients` precedent): `_newton_schulz(x, steps, eps, coefficients=(3.4445, -4.7750, 2.0315))` (bfloat16, Frobenius normalize, transpose-when-tall, real path); `_newton_schulz_complex` or a dtype-agnostic variant using `x @ x.mH` for the `dft` path; `_build_transform(name, K, flat)` returning the unitary Q (decorq: `torch.linalg.eigh(C)` on `C = flat @ flat.T`, `Q = U.T`; dct: orthonormal DCT-II matrix; haar: recursive `H`, block-embedded with identity when K is not a power of two — documented deviation; dft: `Q[j,k] = K**-0.5 * exp(-2πi jk/K)` complex). Constructor: `TUON(params, lr=0.02, momentum=0.95, ns_steps=5, ns_eps=1e-7, refresh_interval=200, rms_eps=1e-8, transform="decorq", stack="full", weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)`; validates `0<=lr`, `0<=momentum<1`, `ns_steps>=1`, `ns_eps>=0`, `refresh_interval>=1`, `rms_eps>=0`, `transform in {decorq,dct,haar,dft}`, `stack in {full,pairs}` (ValueError, mirror `nor_muon.py`'s validation block); `defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, ns_eps=ns_eps, refresh_interval=refresh_interval, rms_eps=rms_eps, transform=transform, stack=stack, weight_decay=weight_decay, betas=betas, eps=eps)`; `super().__init__(params, defaults)`. Google-style Args/Attributes docstrings; CPU-safe (no `.cuda()`). `@torch.no_grad() step(closure=None)`: closure pattern mirror `muon.py:116-119`; then per group: `use_adamw = str(group.get("name", "other")) in ADAMW_FALLBACK_GROUPS`; **pass 1 per param**: `if p.grad is None: continue`; fallback `if use_adamw or p.ndim != 2: adamw_fallback_update(p, p.grad, self.state[p], lr=group["lr"], betas=group.get("betas", (0.9, 0.95)), eps=group.get("eps", 1e-8), weight_decay=group["weight_decay"]); continue` (mirror `muon.py:130-140`); TUON path lazy state `state["momentum_buffer"] = torch.zeros_like(grad)`, then `buf.mul_(group["momentum"]).add_(grad, alpha=1.0 - group["momentum"])` (Alg. 1 line 2 with (1−β)). Collect `(group_index, shape) -> [params]` buckets in registration order; `stack="pairs"` chunks each bucket into consecutive pairs (leftover K=1). **Pass 2 per bucket** (paper lines 3–11 mapped): `M = torch.stack([self.state[p]["momentum_buffer"] for p in bucket], dim=-1)` (m×n×K); refresh basis on `p0 = bucket[0]`: `self.state[p0]["tuon_step"] = n+1`; refresh if `tuon_basis` missing / wrong shape / `n % group["refresh_interval"] == 0` → `Q = _build_transform(...)` (fixed transforms are rebuilt when the cache is missing, then cached identically), store `self.state[p0]["tuon_basis"] = Q`; `Mtil = torch.einsum("jk,mnk->mnj", Q, M)` (complex for dft); per slice transpose-when-tall + `X0 = slice/(‖slice‖_F + group["ns_eps"])`, J NS steps (Alg. lines 7–9; batched bmm over slices); inverse `P = torch.einsum("jk,mnj->mnk", Q.conj(), Ptil)` (Alg. line 10; `.real` for dft — conjugate symmetry of the DFT of a real stack makes P real, documented); `rms = P.float().pow(2).mean().sqrt()` (Eq. 100 over the whole stack), `gamma = (1.0/math.sqrt(max(m,n))) / (rms + group["rms_eps"])` (Eq. 101 with s from I.3); decoupled decay per param `p.mul_(1.0 - group["lr"]*group["weight_decay"])` (Eq. 103, mirror `muon.py:155-156`); update `p.add_(P[..., k].to(p.dtype), alpha=-group["lr"]*gamma)`. Return loss. |
| 2 | `src/model/optimizer/__init__.py` | Insert `from .tuon import TUON` between `from .soap import SOAP` (line 32) and `from .turbo_muon import TurboMuon` (line 33) — alphabetical: soap < tuon < turbo_muon. Insert `"TUON"` in `__all__` between `"Sophia"` (line 67) and `"TurboMuon"` (line 68). |
| 3 | `src/model/optimizer/factory.py` | (a) Import `from .tuon import TUON` between `.soap` (line 56) and `.turbo_muon` (line 57). (b) `OPTIMIZER_REGISTRY`: add `"tuon": TUON,` after `"conda": Conda,` (line 92). (c) `OPTIMIZER_ALLOWED_KEYS`: add `"tuon": _COMMON_PER_GROUP_KEYS \| {"momentum", "ns_steps", "ns_eps", "refresh_interval", "rms_eps", "transform", "stack"},` after the conda line (161). (d) Dispatch branch after conda's (497–506), before the final `return optimizer_cls(converted_groups)` (508): `if optimizer_name == "tuon": momentum = to_float(scoped.get("momentum"), 0.95); ns_steps = to_int(scoped.get("ns_steps"), 5); ns_eps = to_float(scoped.get("ns_eps"), 1e-7); refresh_interval = to_int(scoped.get("refresh_interval"), 200); rms_eps = to_float(scoped.get("rms_eps"), 1e-8); transform = str(scoped.get("transform", "decorq")).strip().lower(); stack = str(scoped.get("stack", "full")).strip().lower(); return optimizer_cls(converted_groups, momentum=momentum, ns_steps=ns_steps, ns_eps=ns_eps, refresh_interval=refresh_interval, rms_eps=rms_eps, transform=transform, stack=stack)`. |
| 4 | `src/schema/_optimizer.yaml` | (a) **enum**: append `- tuon` after `- conda` (line 184). (b) **EN use-case table**: append after the conda row (line 67): `` `tuon` | Cross-layer orthogonal updates | Tensor-aware Muon: stacks same-shape hidden matrices per group, mixes them with a unitary layer transform (DecorQ from the layer Gram matrix, refreshed every `refresh_interval` steps), polarizes each slice with Newton-Schulz, inverts the transform and rescales the stack to RMS 1/√max(m,n); K=1 reduces to `muon`; AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2610.35130) ``. (c) **ES use-case table**: append after conda (line 127): `` `tuon` | Actualizaciones ortogonales entre capas | Muon consciente de tensores: apila matrices ocultas de la misma forma por grupo, las mezcla con una transformación unitaria de capas (DecorQ desde la matriz de Gram de capas, refrescada cada `refresh_interval` pasos), polariza cada slice con Newton-Schulz, invierte la transformación y reescala la pila a RMS 1/√max(m,n); K=1 reduce a `muon`; respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2610.35130) ``. (d) **References lists**: EN line 89 and ES line 150 both end with `` `conda` (paper: arxiv:2509.24218) `` → append `` ; `tuon` (paper: arxiv:2610.35130) ``. (e) **EN "Optimizer-specific suffixes"** list: insert after the `conda` bullet (line 219), before `Others` (222): `` - `tuon`: `momentum`, `ns_steps`, `ns_eps`, `refresh_interval`, `rms_eps`, `transform`, `stack`. Tensor-aware Muon: within each parameter group, same-shape hidden 2-D matrices are stacked along the layer mode; a unitary transform Q (`transform: decorq` diagonalizes the layer Gram matrix via eigendecomposition and refreshes every `refresh_interval` steps — default 200; `dct`/`haar`/`dft` are fixed transforms) mixes the slices, each transformed slice is polarized with Newton-Schulz (`ns_steps`/`ns_eps`), the transform is inverted, and the whole stack is rescaled to RMS 1/√max(m,n) via `gamma_t = s/(RMS(P_t) + rms_eps)` (default 1e-8); `stack: full` groups every same-shape matrix of the group (paper default), `stack: pairs` stacks consecutive pairs (K=2); the momentum β is `momentum` (default 0.95) and `wd_*` stays decoupled; same hybrid routing as `muon` — the shared `betas_*`/`eps_*` configure the AdamW fallback for the `embeddings`/`pos`/`head` groups and every non-2-D parameter ``. (f) **ES list**: same position after conda (line 265), faithful translation (draft above the EN one translated; see §4.3 terminology). (g) **`x_parameter_reference.by_optimizer`**: append after conda (lines 444–447): `tuon:` / `  prefix: "tuon-"` / `  supports_shared_per_group_suffixes: true` / `  specific_global_suffixes: [momentum, ns_steps, ns_eps, refresh_interval, rms_eps, transform, stack]`. |
| 5 | `src/schema/_training.yaml` | Append exactly one `if/then` block after conda's (lines 1638–1651), same shape/indentation: `- if:` → `properties:` → `optimizer:` → `properties:` → `optimizer_class:` → `const: tuon`; `then:` → `properties:` → `optimizer:` → `properties:` → `parameters:` → `additionalProperties: false` / `patternProperties:` / `^tuon-: { type: [number, boolean, array, string] }`. The parity test `test_optimizer_prefixes_match_conditional_rules` (regex `^\s+\^([a-z0-9_]+)-:\s*\{`, `tests/test_schema_consistency.py:76-90`) requires exactly one such block per prefix. |
| 6 | `src/schema/_sbert.yaml` | Extend `optimizer_kwargs` descriptions (EN line 400, ES line 401): append `tuon` to the `momentum` optimizer list (`…/dion3/root/nor_muon` → `…/dion3/root/nor_muon/tuon`), to the `ns_steps` and `ns_eps` lists (`…/muon2/nor_muon` → `…/muon2/nor_muon/tuon`); append a clause after the conda one: `refresh_interval/rms_eps/transform/stack (for tuon: DecorQ basis refresh period, stack-RMS epsilon, layer transform choice decorq|dct|haar|dft, stacking full|pairs)`. ES mirror. TUON is available for SBERT (standard `step()`, no closure requirement — only `eve`/`ea4llm` are excluded). |
| 7 | `configs/examples/es_arch_hybrid_tuon.yaml` | **New file.** Structural copy of `es_arch_hybrid_muon2.yaml` (97 lines) with: `optimizer_class: tuon`; all 40 shared keys re-prefixed `tuon-lr_<g>` / `tuon-wd_<g>` / `tuon-betas_<g>: [0.9, 0.95]` / `tuon-eps_<g>: 1e-8` for the 10 groups (same values = `DEFAULT_GROUP_LR`/`DEFAULT_GROUP_WD`); specific globals `tuon-momentum: 0.95`, `tuon-ns_steps: 5`, `tuon-ns_eps: 1.0e-7`, `tuon-refresh_interval: 200`, `tuon-rms_eps: 1.0e-8`, `tuon-transform: decorq`, `tuon-stack: full`; `csv_log_path: training_metrics_es_arch_hybrid_tuon.csv`. Full draft in §4.1. |
| 8 | `configs/README.md` | (a) Optimizer chooser list: add `- tuon` after `- conda` (line 166). (b) Specific-suffix list: insert after the conda bullet (line 221): `` - `tuon`: `momentum`, `ns_steps`, `ns_eps`, `refresh_interval`, `rms_eps`, `transform`, `stack` (tensor-aware Muon: same-shape hidden matrices per group are stacked, mixed with a unitary layer transform — `decorq` eigendecomposes the layer Gram matrix refreshed every `refresh_interval` steps, `dct`/`haar`/`dft` are fixed — slice-polarized with Newton-Schulz, inverse-transformed and rescaled to RMS 1/√max(m,n) via `rms_eps`; `stack: full|pairs`; same hybrid routing as `muon`) ``. (c) Examples list: add after `- adam_pm-pool_dim: col` (line 240): `- tuon-refresh_interval: 200` and `- tuon-transform: decorq`. |
| 9 | `full_tests/run_e2e.py` | (a) `OPTIMIZERS` list (267–273): append `"tuon"` after `"conda"` → `"root", "muon2", "adam_pm", "nor_muon", "conda", "tuon",`. (b) `_with_optimizer`: add `"tuon"` to the Adam-style per-group betas/eps set (lines 662–666; those shared keys are allowed and configure the fallback), and append after the conda block (718–721): `if optimizer_class == "tuon": params[f"{prefix}momentum"] = 0.95; params[f"{prefix}ns_steps"] = 5; params[f"{prefix}refresh_interval"] = 4` (small refresh gap so the DecorQ-refresh path is exercised inside the short e2e run — same rationale as conda's `update_proj_gap: 4`). Keys are filtered against `OPTIMIZER_ALLOWED_KEYS` (726–734) so nothing invalid can leak. |
| 10 | `tests/test_optimizer_factory.py` | (a) `test_registry_contains_all_required` (line 31): add `"tuon"` to the `required` set. (b) New test class `BuildOptimizerTUONTests` (docstring: `"""TUON defaults, paper invariants and hybrid routing (arXiv:2610.35130)."""`), inserted after `BuildOptimizerCondaTests` (ends line 599), mirroring the Muon2/NorMuon classes: `test_builds_without_error`; `test_step_runs` (4×4 attention param, `p.square().mean().backward()`, no NaN); `test_step_runs_stack` (two same-shape 4×8 params in one attention group → K=2 bucket exercises the transform + inverse path); `test_defaults_match_paper` (`momentum == 0.95`, `ns_steps == 5`, `ns_eps == 1e-7`, `refresh_interval == 200`, `rms_eps == 1e-8`, `transform == "decorq"`, `stack == "full"`, `betas == (0.9, 0.95)`, `eps == 1e-8`); `test_hidden_matrix_uses_tuon_state` (attention group → state has `momentum_buffer`, no `exp_avg`); `test_embedding_group_uses_adamw_fallback`; `test_vector_param_is_not_frozen` (norms 1-D param takes fallback, changes, no NaN); `test_k1_matches_muon` (single 4×4 attention matrix, wd=0: build `muon` and `tuon` on identical params/grads, step once, `torch.allclose` — guards Prop. 2's K=1 reduction and the γ_t ≈ 1 scale); `test_update_rms_matches_stack_scale` (K=2 bucket of 4×8, wd=0: `‖ΔP_stack‖²_F ≈ η²·s²·mnK` with `s = 1/√8`, tolerance 1e-3 — guards Eq. 100–102); `test_decorq_basis_orthogonalizes_gram` (K=3 bucket, `refresh_interval=1`: recompute `M̃ = M ×₃ Q` from `state[p0]["tuon_basis"]` and the momentum buffers, assert `⟨M̃_i, M̃_j⟩_F ≈ 0` for i≠j — guards the eigh path); `test_fixed_transforms_step_runs` (`transform: dct`, `haar`, `dft` each: real, finite update; dft takes the real part); `test_pairs_stack_step_runs` (`stack: pairs`, 3 same-shape params → one K=2 bucket + one K=1 leftover); `test_specific_suffixes_parsed` (`{"tuon-momentum": "0.9", "tuon-ns_steps": "3", "tuon-ns_eps": "1e-6", "tuon-refresh_interval": "50", "tuon-rms_eps": "1e-7", "tuon-transform": "DCT", "tuon-stack": "PAIRS"}` → 0.9/3/1e-6/50/1e-7/`dct`/`pairs`; bad values → defaults); `test_unknown_group_suffix_raises` (`{"tuon-lr_bogus": 1.0}` → ValueError mentioning `lr_bogus`); `test_per_group_lr_routes` (`{"tuon-lr_ffn": 0.02}` → `param_groups[0]["lr"] == 0.02`). Import the module helpers at the top like the existing `from src.model.optimizer.nor_muon import _newton_schulz` (line 12) if the basis/NS assertions need them. |
| 11 | `docs/bibliography/optimizers.bib` | Append after `wang_conda_2025` (lines 413–421) the entry drafted in §2.6 (`@misc{tang_tuon_2026, …}`), matching the Zotero arXiv style of the file. |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | Insert `\paragraph{TUON (Tensor-Aware Optimization via T-SVD and Polarization).}` block **after Conda's Pros list (ends ~line 1707) and before `\subsection{Evolutionary and Population-Based Hybrids}` (line 1709)** — last member of the `Second-Order, Geometric, and Orthogonality Optimizers` subsection (starts line 895). No figure (see §3). Full draft in §4.2: intro prose, `align` block with the five equation groups (momentum tensor, Gram + eigendecomposition + Q, transform + slice polarization, inverse transform, RMS-scaled update), NS appendix-F note, `algorithm` float (`\label{alg:tuon}`), implementation-notes paragraph (suffix mapping: shape-bucket = compatible group, `momentum` = β, `refresh_interval` = I, `rms_eps` = ε of γ_t, `transform`/`stack` = Q and K, shared betas/eps = AdamW fallback, K=1 ≡ `muon`), results paragraph (Table 2/3/4 numbers), `\paragraph{Cons.}` + `\paragraph{Pros.}`, `\citep{tang_tuon_2026}`. |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Add a row after Conda (line 171), still under `\textbf{Second-Order, Geometric \& Orthogonality}` (line 158): `TUON & Orthogonality (tensor, cross-layer) & 1 ($m$ per slice); + $K{\times}K$ basis per stack (refresh $T$); + $v$ on the AdamW fallback & $\mathcal{O}(Kmn\cdot\min(m,n) + K^2 mn)$; eigh every $T$ steps & lr, momentum, wd, NS steps $k$, refresh $T$, transform $Q$, stack $K$ & \citep{tang_tuon_2026} \\`. (b) Census: line 4 `33 optimizers` → `34 optimizers`; line 111 `catalogs all 33 supported optimizers` → `all 34 supported optimizers`. |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: `The system supports thirty-three optimizer families in eight categories` → `thirty-four`; append `TUON` to the geometry-oriented list: `(Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon², NorMuon, Conda, TUON)`. |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of row 12, inserted **after Conda's Ventajas (ends ~line 1733) and before `\subsection{Híbridos Evolutivos y Poblacionales}` (line 1735)**, inside `Optimizadores de Segundo Orden, Geométricos y de Ortogonalidad` (subsection at line 903). Terminology mirrors the file's conventions (`respaldo AdamW`, `momentum`, `matriz de Gram`, `Contras.`/`Ventajas.`; figures live in `../paper/figures/optimizers/`, none referenced). Full draft in §4.3. |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | (a) Row after Conda (line 171): `TUON & Ortogonalidad (tensorial, entre capas) & 1 ($m$ por slice); + base $K{\times}K$ por pila (refresco $T$); + $v$ en el respaldo AdamW & $\mathcal{O}(Kmn\cdot\min(m,n) + K^2 mn)$; eigh cada $T$ pasos & lr, momentum, wd, pasos NS $k$, refresco $T$, transformada $Q$, pila $K$ & \citep{tang_tuon_2026} \\`. (b) Census: line 4 `33 optimizadores` → `34 optimizadores`; line 111 `cataloga los 33 optimizadores soportados` → `los 34 optimizadores soportados`. |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: `El sistema soporta treinta y tres familias de optimizadores` → `treinta y cuatro`; `métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon², NorMuon, Conda)` → append `, TUON`. |
| 18 | `docs/architecture.md`, `README.md` (+ `configs/README.md` covered in row 8) | **Census pass, 33 → 34.** (a) `docs/architecture.md:26` `| Optimizer families | **33** |` → `**34**`; (b) `:87` `33 optimizers + registry + parameter routing` → `34 optimizers + …`; (c) `:767` `### 8.1 Optimizers (33)` → `(34)`; (d) `:780` Orthogonalized row: add `tuon` to the family list after `conda`, and append the clause `; \`tuon\` stacks same-shape hidden matrices of a group, mixes them with a unitary layer transform (DecorQ: eigenvectors of the layer Gram matrix, refreshed every \`refresh_interval\` steps), polarizes each transformed slice with Newton-Schulz, inverts the transform and rescales the stack to RMS 1/√max(m,n) (K=1 buckets reduce to \`muon\`; AdamW fallback unchanged)`. (e) `README.md:20` At-a-Glance Optimizers cell — **stale at 32** (actual 33 pre-change) → set to `**34**`; (f) `README.md:27` mermaid `33 optimizer families` → `34 optimizer families`; (g) `README.md:98` `Optimizer families | 33 across 8 categories` → `34 across 8 categories`; (h) `README.md:103` `51 named presets + 117 example configs` — **stale** (filesystem today: 52 root presets, 120 examples) → re-count at implementation time (`ls configs/*.yaml \| wc -l`, `ls configs/examples/*.yaml \| wc -l`) and write the actuals (expected **52 + 121** after this change). Re-grep at implementation time: `grep -rn "33" README.md docs/architecture.md configs/README.md` and `grep -rn "(33)\|thirty-three\|treinta y tres" docs/paper docs/paper-es` to catch stragglers. |

**No change needed** (verified read-only): `src/streamlit_gui/app.py` (schema-driven; new classes need no entry — the last five optimizer additions added none); `src/training/config_loader.py` (no optimizer list; the dispatcher is `factory.build_optimizer`); `tests/test_optimizer_base.py` (base utilities only — a gate to *run*, not edit); `tests/test_yaml_examples.py` (auto-globs); SBERT enum (`$ref`s the canonical enum at `_sbert.yaml:395` — `test_sbert_optimizer_enum_resolves_to_canonical` passes unchanged); `full_tests/param_count_check.py` (no optimizer dependency; grep "optimizer" = 0 hits — the estimator models the architecture, not the optimizer); `docs/paper/paper.tex`/`paper-es.tex` (annexes are `\input`).

### Hard rules (bind every row above)

1. **The schema is the contract** (`additionalProperties: false`): every `tuon-*` key accepted at runtime must exist in `OPTIMIZER_ALLOWED_KEYS["tuon"]` ∩ `_optimizer.yaml` (enum + tables + suffix list + `by_optimizer.tuon`) ∩ the `_training.yaml` `if/then` rule; never read an undocumented key.
2. **Prefix contract:** `tuon-<param>` (global) and `tuon-<group>_<param>` (per group); groups are exactly `embeddings, pos, attention, ffn, norms, head, mhc, router, residual, other`.
3. **Papers are the theory source of truth (D11):** the EN annex formulations must match arXiv:2610.35130v1 verbatim (Algorithm 1, Eqs. 1–5, 77, 92–96, 97–99, 100–103); `paper-es` must translate them faithfully; the module implements Algorithm 1 exactly (including the `(1−β)` momentum factor), not the anonymous repo's extras if any diverge.
4. CPU-first: implementation and all tests run on CPU (`uv run --extra cpu --extra train …`); no `.cuda()`, no CUDA-only ops; complex DFT tensors are supported on CPU.

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
    optimizer_class: tuon
    parameters:
      tuon-lr_embeddings: 1e-6
      tuon-lr_pos: 2e-6
      tuon-lr_attention: 3e-6
      tuon-lr_ffn: 2e-6
      tuon-lr_norms: 5e-6
      tuon-lr_head: 2e-6
      tuon-lr_mhc: 3e-6
      tuon-lr_router: 2e-6
      tuon-lr_residual: 2e-6
      tuon-lr_other: 2e-6
      tuon-wd_embeddings: 0.01
      tuon-wd_pos: 0.01
      tuon-wd_attention: 0.01
      tuon-wd_ffn: 0.01
      tuon-wd_norms: 0.001
      tuon-wd_head: 0.01
      tuon-wd_mhc: 0.01
      tuon-wd_router: 0.01
      tuon-wd_residual: 0.01
      tuon-wd_other: 0.01
      tuon-betas_embeddings: [0.9, 0.95]
      tuon-betas_pos: [0.9, 0.95]
      tuon-betas_attention: [0.9, 0.95]
      tuon-betas_ffn: [0.9, 0.95]
      tuon-betas_norms: [0.9, 0.95]
      tuon-betas_head: [0.9, 0.95]
      tuon-betas_mhc: [0.9, 0.95]
      tuon-betas_router: [0.9, 0.95]
      tuon-betas_residual: [0.9, 0.95]
      tuon-betas_other: [0.9, 0.95]
      tuon-eps_embeddings: 1e-8
      tuon-eps_pos: 1e-8
      tuon-eps_attention: 1e-8
      tuon-eps_ffn: 1e-8
      tuon-eps_norms: 1e-8
      tuon-eps_head: 1e-8
      tuon-eps_mhc: 1e-8
      tuon-eps_router: 1e-8
      tuon-eps_residual: 1e-8
      tuon-eps_other: 1e-8
      tuon-momentum: 0.95
      tuon-ns_steps: 5
      tuon-ns_eps: 1.0e-7
      tuon-refresh_interval: 200
      tuon-rms_eps: 1.0e-8
      tuon-transform: decorq
      tuon-stack: full
  scheduler_total_steps: 12000
  max_nan_retries: 4
  gradient_log_interval: 10
  csv_log_path: training_metrics_es_arch_hybrid_tuon.csv
  telemetry_log_interval: 5
```

### 4.2 EN annex paragraph (row 12, full draft)

```latex
\paragraph{TUON (Tensor-Aware Optimization via T-SVD and Polarization).}
TUON~\citep{tang_tuon_2026} lifts Muon's layer-wise orthogonalization to
\textbf{cross-layer tensors} using a transform-based tensor algebra derived from the
Tensor-SVD. Layer-wise polar maps discard cross-layer correlations, while
unfolding-based tensor optimizers (TEON) suffer dimension imbalance and cross-layer
misalignment and degrade beyond two stacked layers or QKV-only modules. TUON instead
\emph{mixes then separates}: for one compatible group of $K$ same-shape matrices with
stacked momentum $\mathcal{M}_t \in \mathbb{R}^{m \times n \times K}$:
\begin{align}
\mathcal{M}_t &= \beta \mathcal{M}_{t-1} + (1-\beta)\, \mathcal{G}_t
  && \text{(momentum tensor, Alg.~1 line 2)} \\
C_t &= \mathsf{M}_t \mathsf{M}_t^{\top} \in \mathbb{R}^{K \times K}, \qquad
C_t = U_t \Lambda_t U_t^{\top}, \qquad Q_t = U_t^{\top}
  && \text{(layer Gram + eigendecomposition, lines 3--5)} \\
\widetilde{\mathcal{M}}_t &= \mathcal{M}_t \times_3 Q_t, \qquad
\widetilde{P}_{t,j} = \operatorname{polar}\!\big(\widetilde{\mathcal{M}}_{t,j}\big)
  && \text{(transform + slice polarization, lines 6--9)} \\
\mathcal{P}_t &= \widetilde{\mathcal{P}}_t \times_3 Q_t^{H}
  && \text{(inverse transform, line 10)} \\
W_{t+1} &= (1 - \eta_t \lambda)\, W_t - \eta_t \gamma_t \mathcal{P}_t, \qquad
\gamma_t = \frac{s}{\operatorname{RMS}(\mathcal{P}_t) + \epsilon}
  && \text{(RMS-scaled update, line 11)}
\end{align}
where $\mathsf{M}_t$ is the mode-3 unfolding ($k$-th row $= \operatorname{vec}(\mathcal{M}_t^{(k)})^{\top}$),
the mode-3 product mixes slices as
$\widetilde{\mathcal{M}}_{t,j} = \sum_k (Q_t)_{jk} \mathcal{M}_t^{(k)}$,
$\operatorname{RMS}(\mathcal{P}_t) = \sqrt{(mnK)^{-1} \sum_{i,j,k} |\mathcal{P}_t(i,j,k)|^2}$,
and the target stack RMS is $s = 1/\sqrt{\max(m,n)}$ (the rank--energy identity: full-rank
slices give $\operatorname{RMS} = 1/\sqrt{\max(m,n)}$, so $\gamma_t \approx 1$ and per-matrix
update magnitudes match Muon). The default transform \textbf{DecorQ} takes the
eigenvectors of the layer Gram matrix ($Q_t = U_t^{\top}$) and is refreshed every $T$
steps (default 200); it diagonalizes the transformed Gram, so the transformed slices are
mutually Frobenius-orthogonal and each polar map acts in its own subspace --- the coupled
$K$-layer problem decouples into $K$ independent single-channel problems. Fixed unitary
transforms (DCT, Haar, DFT) are valid but impose a prior mismatched to layer statistics;
in the paper's ablation DecorQ (2.7598) beats DFT (2.7631), Haar (2.7635), static DecorQ
(2.7658) and DCT (2.8100) on Qwen3-4B. Slice polarization uses the same quintic
Newton--Schulz iteration as Muon, $(a,b,c) = (3.4445, -4.7750, 2.0315)$ with $J = 5$
steps after a Frobenius normalization and a transpose-when-tall.

\begin{algorithm}[H]
\caption{TUON (Tensor-Aware Optimization via T-SVD and Polarization)}
\label{alg:tuon}
\begin{algorithmic}[1]
\Require Compatible matrices $\{W^{(k)}\}_{k=1}^{K}$, learning rate $\eta$, momentum $\beta$, weight decay $\lambda$, NS iters $J$, refresh interval $T$, RMS $\epsilon$
\Ensure Updated parameters
\For{each compatible group}
    \State $\mathcal{G} \gets \operatorname{stack}_3(\nabla f(W^{(1)}), \ldots, \nabla f(W^{(K)}))$ \Comment{Gradient tensor (line 1)}
    \State $\mathcal{M} \gets \beta \mathcal{M} + (1-\beta)\, \mathcal{G}$ \Comment{Momentum tensor (line 2)}
    \State $C \gets \mathsf{M}\mathsf{M}^{\top}$; $C = U \Lambda U^{\top}$; $Q \gets U^{\top}$ \Comment{Gram + eigendecomposition, refreshed every $T$ steps (lines 3--5)}
    \State $\widetilde{\mathcal{M}} \gets \mathcal{M} \times_3 Q$ \Comment{Mix slices (line 6)}
    \For{$j = 1, \ldots, K$ in parallel}
        \State $\widetilde{P}_{:, :, j} \gets \mathrm{NS}_J(\widetilde{\mathcal{M}}_{:, :, j})$ \Comment{Slice polarization (lines 7--9)}
    \EndFor
    \State $\mathcal{P} \gets \widetilde{\mathcal{P}} \times_3 Q^{H}$ \Comment{Inverse transform (line 10)}
    \State $\gamma \gets s / (\operatorname{RMS}(\mathcal{P}) + \epsilon)$ \Comment{Shared stack scale (line 11)}
    \State $W \gets (1-\eta\lambda) W - \eta \gamma \mathcal{P}$ \Comment{Decoupled update (line 11)}
\EndFor
\end{algorithmic}
\end{algorithm}

This implementation maps the paper's ``one compatible group'' to each
\emph{(parameter group, matrix shape)} bucket: same-shape hidden 2-D matrices of a group
are stacked along the layer mode in registration order
(\texttt{stack: full}); \texttt{stack: pairs} restricts the stack to consecutive pairs
($K=2$). The \texttt{transform} suffix selects $Q$ (\texttt{decorq} default;
\texttt{dct}/\texttt{haar}/\texttt{dft} fixed transforms), \texttt{momentum} is $\beta$,
\texttt{refresh\_interval} is $T$, \texttt{ns\_steps}/\texttt{ns\_eps} drive the
Newton--Schulz iteration, and \texttt{rms\_eps} is the $\epsilon$ of $\gamma_t$. For
$K=1$ the transform cancels exactly and TUON reduces to Muon (paper Prop.~2); following
the hybrid recipe of Muon, only hidden-layer 2-D matrices take the TUON path ---
embedding tables, learned positional encodings, output heads and every non-matrix
parameter are trained with decoupled AdamW using the group-level learning rate, weight
decay, $\beta_1, \beta_2$ and $\epsilon$ settings. Theory: TUON is steepest descent under
the $Q$-transformed tensor spectral norm (dual norm pair $\|\cdot\|_{Q,Ts}$ /
$\|\cdot\|_{Q,Tn}$), its update equals the tensor polar factor of a T-product T-SVD, no
unitary layer transform distorts the Muon geometry by more than $\sqrt{K}$, and at equal
operator radius the TUON oracle is never weaker than TEON's and up to $\sqrt{K}$
stronger, with an $O(T^{-1/2})$ convergence certificate. Empirically TUON beats Muon and
TEON on all of GPT-124M/350M and Qwen-0.6B/4B (validation loss
2.9357/2.6925/2.8887/2.7598), is the only method that improves from $K{=}2$ to
$K{=}\mathrm{full}$ stacking on all four models, transfers from QKV to QKVO+MLP, and
costs 0.2--1.1\% step-time over TEON and at most 12.5\% over AdamW; per-iteration
complexity is $\mathcal{O}(Kmn\min(m,n) + K^2 mn)$, the same dominant term as Muon.

\paragraph{Cons.}
\begin{itemize}[leftmargin=*]
\item Cross-layer mixing assumes the stacked matrices are genuinely compatible; this
  implementation stacks by exact shape within a parameter group, which is the closest
  faithful realization available from the unnamed factory param lists.
\item The Gram/eigendecomposition adds $\mathcal{O}(K^2 mn)$ per step and a $K \times K$
  cached basis per bucket; negligible for $K \ll \min(m, n)$, but a per-bucket cost that
  Muon does not pay.
\item $\beta$ is required by Algorithm 1 but its default value is not stated in the
  paper; this implementation inherits the repo Muon-family convention ($0.95$).
\item The \texttt{dft} transform runs the Newton--Schulz iteration over complex slices
  (Hermitian Gram); the real part is taken after the inverse transform, justified by the
  conjugate symmetry of the DFT of a real stack.
\end{itemize}

\paragraph{Pros.}
\begin{itemize}[leftmargin=*]
\item Exploits cross-layer structure that layer-wise Muon discards, with a convergence
  guarantee and exact steepest-descent interpretation under the induced tensor norm.
\item DecorQ is data-dependent and refreshed every $T$ steps; stacking $K{=}\mathrm{full}$
  \emph{improves} with depth instead of degrading as unfolding methods do.
\item Memory stays at Muon's level (one momentum buffer per matrix plus a $K \times K$
  basis), and the RMS stack scaling keeps updates at Muon's magnitude so existing
  learning rates transfer.
\item Applies uniformly beyond QKV (attention, MLP and output projections), where prior
  tensor-aware optimizers fail to transfer.
\end{itemize}
```

### 4.3 ES annex paragraph (row 15, full draft — faithful translation)

```latex
\paragraph{TUON (Optimización Consciente de Tensores vía T-SVD y Polarización).}
TUON~\citep{tang_tuon_2026} eleva la ortogonalización por capa de Muon a
\textbf{tensores entre capas} mediante un álgebra tensorial basada en transformaciones,
derivada del Tensor-SVD. Los mapas polares por capa descartan las correlaciones entre
capas, mientras que los optimizadores tensoriales basados en despliegue (TEON) sufren
desequilibrio de dimensiones y desalineación entre capas, y degradan más allá de dos
capas apiladas o de los módulos QKV. TUON en cambio \emph{mezcla y luego separa}: para
un grupo compatible de $K$ matrices de la misma forma con momentum apilado
$\mathcal{M}_t \in \mathbb{R}^{m \times n \times K}$:
\begin{align}
\mathcal{M}_t &= \beta \mathcal{M}_{t-1} + (1-\beta)\, \mathcal{G}_t
  && \text{(tensor de momentum, Alg.~1 línea 2)} \\
C_t &= \mathsf{M}_t \mathsf{M}_t^{\top} \in \mathbb{R}^{K \times K}, \qquad
C_t = U_t \Lambda_t U_t^{\top}, \qquad Q_t = U_t^{\top}
  && \text{(Gram de capas + autodescomposición, líneas 3--5)} \\
\widetilde{\mathcal{M}}_t &= \mathcal{M}_t \times_3 Q_t, \qquad
\widetilde{P}_{t,j} = \operatorname{polar}\!\big(\widetilde{\mathcal{M}}_{t,j}\big)
  && \text{(transformación + polarización por slice, líneas 6--9)} \\
\mathcal{P}_t &= \widetilde{\mathcal{P}}_t \times_3 Q_t^{H}
  && \text{(transformación inversa, línea 10)} \\
W_{t+1} &= (1 - \eta_t \lambda)\, W_t - \eta_t \gamma_t \mathcal{P}_t, \qquad
\gamma_t = \frac{s}{\operatorname{RMS}(\mathcal{P}_t) + \epsilon}
  && \text{(actualización escalada por RMS, línea 11)}
\end{align}
donde $\mathsf{M}_t$ es el despliegue por modo 3 (la $k$-ésima fila es
$\operatorname{vec}(\mathcal{M}_t^{(k)})^{\top}$), el producto por modo 3 mezcla los
slices como $\widetilde{\mathcal{M}}_{t,j} = \sum_k (Q_t)_{jk} \mathcal{M}_t^{(k)}$,
$\operatorname{RMS}(\mathcal{P}_t) = \sqrt{(mnK)^{-1} \sum_{i,j,k} |\mathcal{P}_t(i,j,k)|^2}$
y el RMS objetivo de la pila es $s = 1/\sqrt{\max(m,n)}$ (identidad rango--energía: slices
de rango completo dan $\operatorname{RMS} = 1/\sqrt{\max(m,n)}$, de modo que $\gamma_t
\approx 1$ y la magnitud de la actualización por matriz coincide con Muon). La
transformación por defecto \textbf{DecorQ} toma los autovectores de la matriz de Gram de
capas ($Q_t = U_t^{\top}$) y se refresca cada $T$ pasos (200 por defecto); diagonaliza la
Gram transformada, por lo que los slices transformados son mutuamente ortogonales en
Frobenius y cada mapa polar actúa en su propio subespacio: el problema acoplado de $K$
capas se desacopla en $K$ problemas de un solo canal. Las transformaciones unitarias
fijas (DCT, Haar, DFT) son válidas pero imponen un prior ajeno a la estadística de las
capas; en la ablacíon del artículo DecorQ (2.7598) supera a DFT (2.7631), Haar (2.7635),
DecorQ estático (2.7658) y DCT (2.8100) en Qwen3-4B. La polarización por slice usa la
misma iteración quíntica de Newton--Schulz que Muon, $(a,b,c) = (3.4445, -4.7750,
2.0315)$ con $J = 5$ pasos tras una normalización Frobenius y una transposición cuando la
matriz es alta.

[Float de algoritmo: traducción fiel del de la §4.2, \label{alg:tuon} — misma estructura
de líneas y comentarios traducidos.]

Esta implementación mapea el «grupo compatible» del artículo a cada cubeta
\emph{(grupo de parámetros, forma de matriz)}: las matrices 2-D ocultas de la misma forma
de un grupo se apilan por el modo de capas en orden de registro (\texttt{stack: full});
\texttt{stack: pairs} restringe la pila a pares consecutivos ($K=2$). El sufijo
\texttt{transform} selecciona $Q$ (\texttt{decorq} por defecto;
\texttt{dct}/\texttt{haar}/\texttt{dft} fijas), \texttt{momentum} es $\beta$,
\texttt{refresh\_interval} es $T$, \texttt{ns\_steps}/\texttt{ns\_eps} dirigen
Newton--Schulz y \texttt{rms\_eps} es el $\epsilon$ de $\gamma_t$. Con $K=1$ la
transformación se cancela exactamente y TUON se reduce a Muon (Prop.~2 del artículo);
siguiendo la receta híbrida de Muon, solo las matrices 2-D de capas ocultas toman la ruta
TUON: las tablas de embeddings, los encoding posicionales aprendidos, los heads de salida
y todo parámetro no matricial se entrenan con AdamW desacoplado usando el learning rate,
weight decay, $\beta_1, \beta_2$ y $\epsilon$ del grupo. Teoría: TUON es descenso máximo
bajo la norma espectral tensorial transformada por $Q$ (par dual
$\|\cdot\|_{Q,Ts}$ / $\|\cdot\|_{Q,Tn}$), su actualización es el factor polar tensorial
de una T-SVD de producto-T, ninguna transformación unitaria de capas distorsiona la
geometría de Muon en más de $\sqrt{K}$, y a radio de operador común el oráculo de TUON
nunca es más débil que el de TEON y puede ser hasta $\sqrt{K}$ más fuerte, con un
certificado de convergencia $O(T^{-1/2})$. Empíricamente TUON supera a Muon y TEON en
GPT-124M/350M y Qwen-0.6B/4B (pérdida de validación 2.9357/2.6925/2.8887/2.7598), es el
único método que mejora al pasar de $K{=}2$ a $K{=}\mathrm{full}$ en los cuatro modelos,
se transfiere de QKV a QKVO+MLP, y cuesta 0,2--1,1\% de tiempo de paso sobre TEON y a lo
sumo 12,5\% sobre AdamW; la complejidad por iteración es
$\mathcal{O}(Kmn\min(m,n) + K^2 mn)$, el mismo término dominante que Muon.

\paragraph{Contras.}
\begin{itemize}[leftmargin=*]
\item La mezcla entre capas asume matrices genuinamente compatibles; esta implementación
  apila por forma exacta dentro de un grupo de parámetros, la realización fiel más
  cercana disponible a partir de las listas de parámetros sin nombre de la factoría.
\item La Gram/autodescomposición añade $\mathcal{O}(K^2 mn)$ por paso y una base
  $K \times K$ por cubeta; despreciable para $K \ll \min(m, n)$, pero es un costo por
  cubeta que Muon no paga.
\item $\beta$ es requerido por el Algoritmo 1 pero su valor por defecto no aparece en el
  artículo; esta implementación hereda la convención de la familia Muon del repositorio
  ($0.95$).
\item La transformación \texttt{dft} ejecuta Newton--Schulz sobre slices complejos
  (Gram hermítica); se toma la parte real tras la transformación inversa, justificado por
  la simetría conjugada de la DFT de una pila real.
\end{itemize}

\paragraph{Ventajas.}
\begin{itemize}[leftmargin=*]
\item Explota la estructura entre capas que el Muon por capa descarta, con garantía de
  convergencia e interpretación exacta de descenso máximo bajo la norma tensorial
  inducida.
\item DecorQ depende de los datos y se refresca cada $T$ pasos; apilar
  $K{=}\mathrm{full}$ \emph{mejora} con la profundidad en lugar de degradar como ocurre
  con los métodos de despliegue.
\item La memoria se mantiene al nivel de Muon (un búfer de momentum por matriz más una
  base $K \times K$) y el escalado RMS de la pila conserva la magnitud de Muon, por lo que
  los learning rates existentes se transfieren.
\item Se aplica uniformemente más allá de QKV (atención, MLP y proyecciones de salida),
  donde los optimizadores tensoriales previos no se transfieren.
\end{itemize}
```

---

## 5. Ordered execution checklist (implementation time — do NOT run now)

Environment (CPU-first, `AGENTS.md`): `uv sync --extra cpu --extra train`; every command below runs through `uv run --extra cpu --extra train`.

**Implementation (files 1–9):**
- [ ] 1. `src/model/optimizer/tuon.py` (row 1 sketch, Algorithm 1 mapped line-by-line; `(1−β)` momentum; per-(group, shape) buckets; cached DecorQ basis in `state[p0]`; NS quintic; γ_t RMS stack scale)
- [ ] 2. `src/model/optimizer/__init__.py` (row 2)
- [ ] 3. `src/model/optimizer/factory.py` — import, registry, allowed keys, dispatch (row 3)
- [ ] 4. `src/schema/_optimizer.yaml` — enum + both tables + both suffix lists + refs + `by_optimizer.tuon` (row 4)
- [ ] 5. `src/schema/_training.yaml` — one `^tuon-` if/then block (row 5)
- [ ] 6. `src/schema/_sbert.yaml` — `optimizer_kwargs` EN/ES (row 6)
- [ ] 7. `configs/examples/es_arch_hybrid_tuon.yaml` (§4.1)
- [ ] 8. `configs/README.md` (row 8)
- [ ] 9. `full_tests/run_e2e.py` (row 9)

**Tests (file 10):**
- [ ] 10. `tests/test_optimizer_factory.py` — `required` set + `BuildOptimizerTUONTests` (row 10)

**Docs (files 11–18):**
- [ ] 11. `docs/bibliography/optimizers.bib` (§2.6)
- [ ] 12. `docs/paper/appendices/annex-1-optimizers.tex` (§4.2, inserted at ~line 1708)
- [ ] 13. `docs/paper/appendices/annex-summary-tables.tex` — row + census lines 4/111
- [ ] 14. `docs/paper/sections/05-optimizer-families.tex` — line 4 count + geometry-oriented list
- [ ] 15. `docs/paper-es/appendices/annex-1-optimizadores.tex` (§4.3, inserted at ~line 1734)
- [ ] 16. `docs/paper-es/appendices/annex-summary-tables.tex` — row + census lines 4/111
- [ ] 17. `docs/paper-es/sections/05-familias-optimizadores.tex` — line 4 count + list
- [ ] 18. `docs/architecture.md` + `README.md` census pass (row 18; re-grep the old count first)

**Verification (planned — expected results in §8):**
```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# fast e2e smoke — tuon is appended LAST in OPTIMIZERS, so --limit 3 does NOT reach it;
# raise --limit to >= 34 for a category sweep, or target the combo directly:
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3   # does not include tuon
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --list-combos  # confirm `opt_tuon` exists
uv run --extra cpu --extra train python full_tests/run_e2e.py --only opt_tuon            # the actual tuon smoke
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

- **Preconditions (verified read-only today):** `/usr/bin/pdflatex` and `/usr/bin/bibtex` exist, so compilation is expected to work on this machine (`command -v pdflatex bibtex` → both present).
- **Fallback:** if `pdflatex`/`bibtex` are unavailable on the implementation machine, leave the `.tex`/`.bib` sources updated and note that the rebuild must happen on a TeX machine before the next release (the website sync copies prebuilt PDFs only, and the site probes `paper-en-copy`/`paper-es-copy` would then fail until the PDFs exist).
- **Post-compile verification:** no `Fatal error` in the logs; no `Citation .* undefined` warnings (in particular `tang_tuon_2026` must resolve — it appears in both new annex paragraphs and both summary-table rows); `docs/paper/paper.pdf` and `docs/paper-es/paper-es.pdf` are newer than the edited `.tex`/`.bbl` files; `git status --short docs/paper docs/paper-es` shows only intended `.tex`/`.bbl`/`.pdf` changes (no `*.aux *.log *.out *.toc *.blg`).

**Website sync → §6; commit/push → §7.** The mirror stays uncommitted until the commit phase.

---

## 6. Website sync (planned, not executed)

Follow `/sync-website` (`.opencode/command/sync-website.md`), driven by the read-only state script `bash .opencode/command/sync-website-state.sh`. Files that change for TUON, per the drift map in the script (lines 105–133):

1. Run the state script first; read `[drift]`/`[probes]`/`[tools]`.
2. Copy (plain `cp`; mirror = `erickfmm.github.io/frankenstein-transformer/`):
   - `src/schema.yaml` → `schema.yaml`; `src/schema/**` → `schema/**` (carries `_optimizer.yaml`, `_training.yaml`, `_sbert.yaml`),
   - `configs/examples/es_arch_hybrid_tuon.yaml` → `examples/examples/`, `configs/README.md` → `examples/README.md`,
   - `docs/paper/{paper.tex, sections/*.tex, appendices/*.tex, paper.bbl, paper.pdf}` → `paper/en/…` (annex-1, annex-summary-tables, 05-optimizer-families),
   - `docs/paper-es/{paper-es.tex, sections/*.tex, appendices/*.tex, paper-es.bbl, paper-es.pdf}` → `paper/es/…`,
   - `docs/bibliography/optimizers.bib` → `paper/bibliography/`,
   - `docs/architecture.md` → `paper/specs/architecture.md`.
3. Refresh the Paper-tab artifacts: `flatten_tex.py en/paper.tex en/paper.bbl … --web --manifest …` (+ ES counterpart), `pandoc … -o paper-en.html` / `paper-es.html`, `postprocess_html.py --manifest …` (exact commands in `sync-website.md` §3.3), and the top-level copies `cp paper/en/paper.pdf paper/en.pdf`, `cp paper/es/paper-es.pdf paper/es.pdf` (probes `paper-en-copy`/`paper-es-copy` at script lines 282–291 compare them).
4. **Validation Gate** (in order, all must pass):
   - `uv run --extra cpu --extra train python full_tests/param_count_check.py` → `… configs counted, 0 failures` (regenerates `full_tests/param_truth.json`; the new example config gets a model-side ground-truth entry — the intentional `param_truth.json` diff is committed with the main repo),
   - `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` → exit 0 / `ALL PASS` (the estimator models the architecture, not the optimizer; the tuon example reuses the muon2 example's architecture, so no estimator change is expected),
   - state-script probes all PASS: `schema-refs`, `example-yaml`, `cli-flags`, `estimator-drift`, `paper-en-copy`, `paper-es-copy` (script lines 336, 365, 392, 411, 285, 290).
5. **Leave both repos uncommitted** until §7 (or hand off to `/make-a-commit` per the user's call).

---

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions: English, conventional tag, structured body; never stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json` changes are intentional.

1. **Main repo** — `feat(optimizer): add TUON (arXiv:2610.35130)`

   Body sections: *Implementation* (new `src/model/optimizer/tuon.py`: Algorithm 1 of the paper — per-(group, shape) momentum stacks, DecorQ layer transform from the Gram eigendecomposition refreshed every `refresh_interval` steps, slice-wise Newton–Schulz quintic, inverse transform, RMS stack scaling `γ_t = s/(RMS(P)+rms_eps)`, decoupled decay, AdamW fallback for embeddings/pos/head and non-2-D params; K=1 reduces to Muon), *Schema* (`tuon` enum, bilingual tables/suffix lists, `by_optimizer.tuon`, `^tuon-` if/then, SBERT `optimizer_kwargs` EN/ES), *Docs* (EN/ES annex paragraphs + algorithm floats, summary-table rows, family-section counts 33→34, bibliography `tang_tuon_2026`, architecture/README census, rebuilt PDFs), *Example* (`es_arch_hybrid_tuon.yaml`), *Tests* (`BuildOptimizerTUONTests`, registry required set, e2e list).

   ```bash
   git add src/model/optimizer/tuon.py src/model/optimizer/__init__.py src/model/optimizer/factory.py \
           src/schema/_optimizer.yaml src/schema/_training.yaml src/schema/_sbert.yaml \
           configs/examples/es_arch_hybrid_tuon.yaml configs/README.md \
           full_tests/run_e2e.py full_tests/param_truth.json \
           tests/test_optimizer_factory.py \
           docs/bibliography/optimizers.bib \
           docs/paper/appendices/annex-1-optimizers.tex docs/paper/appendices/annex-summary-tables.tex \
           docs/paper/sections/05-optimizer-families.tex docs/paper/paper.bbl docs/paper/paper.pdf \
           docs/paper-es/appendices/annex-1-optimizadores.tex docs/paper-es/appendices/annex-summary-tables.tex \
           docs/paper-es/sections/05-familias-optimizadores.tex docs/paper-es/paper-es.bbl docs/paper-es/paper-es.pdf \
           docs/architecture.md README.md
   git commit -m "feat(optimizer): add TUON (arXiv:2610.35130)"   # with the structured body above
   git push origin HEAD:main
   ```

2. **Mirror** — `docs(site): sync TUON optimizer (arXiv:2610.35130)` (exact drift list comes from the state script at implementation time; expected paths below)

   ```bash
   git -C erickfmm.github.io add \
           frankenstein-transformer/schema.yaml frankenstein-transformer/schema/_optimizer.yaml \
           frankenstein-transformer/schema/_training.yaml frankenstein-transformer/schema/_sbert.yaml \
           frankenstein-transformer/examples/README.md \
           frankenstein-transformer/examples/examples/es_arch_hybrid_tuon.yaml \
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
   git -C erickfmm.github.io commit -m "docs(site): sync TUON optimizer (arXiv:2610.35130)"
   git -C erickfmm.github.io push origin HEAD:main
   ```

---

## 8. Verification matrix

| Gate | Command (implementation time) | Expected result |
|---|---|---|
| Registry/allowed-keys/step | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py -v` | All pass, incl. the ~15 new `BuildOptimizerTUONTests` methods and `test_registry_contains_all_required` with `tuon` |
| Base utilities | `… -m pytest tests/test_optimizer_base.py -v` | All pass (unchanged file, must stay green) |
| Schema parity | `… -m pytest tests/test_schema_consistency.py -v` | `test_optimizer_prefixes_match_conditional_rules` (exactly one `^tuon-` block), preset validation (incl. the new example), `test_sbert_optimizer_enum_resolves_to_canonical` all pass |
| Example smoke | `… -m pytest tests/test_yaml_examples.py -v` | Auto-injected `test_load_es_arch_hybrid_tuon` passes (`load_training_config` OK) |
| Full suite | `… -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | No failures (76 test files stay green; no regression in naming-conventions or defaults tests) |
| e2e config validation | `… python full_tests/run_e2e.py --dry-run` | Exit 0; `opt_tuon` combo validates |
| e2e smoke | `… python full_tests/run_e2e.py --only opt_tuon` | `opt_tuon` trains and reports PASS in `results.json`/`results.md` (CPU, default timeout; refresh_interval=4 exercises the DecorQ refresh within the run) |
| PDF (EN) | `pdflatex`×2 + `bibtex` sequence (§5) | No `Fatal error`; no `Citation .* undefined`; `tang_tuon_2026` resolves; `paper.pdf` newer than the edited `.tex`/`.bbl` |
| PDF (ES) | Same in `docs/paper-es` | Same checks for `paper-es.pdf` |
| PDF git hygiene | `git status --short docs/paper docs/paper-es` | Only intended `.tex`/`.bbl`/`.pdf` files; no aux artifacts |
| Param ground truth | `… python full_tests/param_count_check.py` | `… configs counted, 0 failures`; `param_truth.json` gains the new example's entry |
| JS estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | Exit 0, `ALL PASS` (≤ 1% drift; architecture identical to the muon2 example) |
| Site probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, `estimator-drift`, `paper-en-copy`, `paper-es-copy` all PASS; drift list empty for the synced files |

---

## 9. Risks, assumptions and open questions

1. **Census collision with the NAMO plan (untracked, same target numbers).** `.opencode/plan/add-optimizer-2602.17080.md` also plans 33→34, the same annex insertion point (after Conda), the same `configs/README.md` example-list position and the same README preset/example-count fixes. Whichever plan is implemented **second** must re-grep the actual counts (34→35, insertion after the first-comer's paragraph, 121→122 examples). Do not hardcode this plan's numbers without re-running the greps.
2. **"Compatible group" interpretation.** The paper stacks K same-shape matrices sharing a semantic role; the factory delivers unnamed per-category param lists, so this implementation buckets by exact `(group, shape)` in registration order. For homogeneous layer patterns this reproduces the paper's K=full stacks; for heterogeneous patterns (`es_arch_hybrid_*` examples) stacks are smaller — the algorithm is still correct for any K (K=1 ≡ Muon). Assumption to verify at implementation time: `categorize_param_groups` order is stable across steps (it iterates `named_parameters`, which is deterministic).
3. **β (momentum) default is an assumption.** Algorithm 1 requires β but the paper never states its value (the released code is the anonymous repo; not consulted beyond the paper). 0.95 follows the repo Muon-family convention. The `(1−β)` factor is kept per Algorithm 1; it cancels through the scale-invariant polar map and RMS rescale (documented in the module), so the choice interacts weakly.
4. **Refresh-interval inconsistency in the paper.** Algorithm 1 line 5 says "Refresh per 200 iterations" and I.3 says "refreshed every 200 iterations", but Appendix G.1 says "every I (e.g., 100) steps". Default **200** (the algorithm wins); the suffix makes it configurable either way.
5. **DFT is the trickiest suffix.** Complex slices need Hermitian NS (`X @ X^H`), and the inverse-transform result is real only up to conjugate symmetry — take `.real` and document. If complex-CPU matmul proves fragile at implementation time, the fallback is to drop `dft` from the `transform` enum before merging (schema bullet, README, tests all shrink accordingly) — an explicit user decision point, recorded here, not made by the implementer unilaterally.
6. **Haar for non-power-of-two K.** The recursive Haar matrix is defined for `N = 2ⁿ`; real stacks have arbitrary K (12, 24 layers). Planned handling: build `H` for `K' = 2^⌊log₂K⌋` and embed block-diagonally with identity (still unitary), documented as a deviation. If the implementer finds the paper's released code does something else, follow the code's behavior and cite it in the annex note.
7. **LR scale.** With `s = 1/√max(m,n)` and full-rank slices, `γ_t ≈ 1` and TUON's update magnitude equals Muon's — the class default `lr=0.02` matches `muon.py`, and the paper's 1e-3–2e-3 peaks (Moonlight-adjusted) are recorded in the schema prose but not used as defaults (the factory injects group LRs anyway; the examples are schema smoke-tests, not training recipes).
8. **No task/mixer/model-class restrictions.** TUON is a standard `step()` optimizer — valid for all nine `training.task` values, all four model classes, and SBERT; only `eve`/`ea4llm` carry the closure/`gradient_accumulation_steps: 1` restrictions. 4-D parameters (ViT/audio conv kernels) take the AdamW fallback (`p.ndim != 2`), matching `muon.py`/`nor_muon.py`.
9. **Basis caching and checkpoint resume.** The DecorQ basis lives in `self.state[p0]` so it serializes with the optimizer; buckets whose membership changes (varying grads) trigger a refresh on shape mismatch. A stale basis for ≤ `refresh_interval` steps after resume is harmless (γ_t rescales the stack regardless).
10. **PDF tooling assumption.** `pdflatex`/`bibtex` verified present (`/usr/bin/…`) — re-check at implementation time and use the §5 fallback if the TeX machine differs.
11. **Open questions (not planned):** PolarExpress as an alternative polar solver (paper §4.2: comparable results; could become `tuon-polar_solver` later); Residual-DecorQ basis (App. I.6); cross-parameter-category stacking (paper's future work). None block this change.

---

## 10. Out of scope

- **This planning command changed nothing** outside `.opencode/plan/add-optimizer-2610.35130.md`: no code, schema, preset, test, doc, PDF, website-mirror or git change was made (see `git status --short` in the delivery message; the only new untracked path is this plan file, alongside the pre-existing plan files, which were left untouched).
- Not planned: fixed-transform-only variants as separate registry entries, PolarExpress/Residual-DecorQ knobs (§9.11), a `tuon` annex figure (§3 — no vector-style source exists), Streamlit GUI edits (schema-driven), estimator changes (architecture-independent), any destructive or renaming step (none implied by the paper). The pre-existing README census drifts (32 at line 20, 51/117 at line 103) are *fixed* by row 18 because this change touches the same numbers — they are not silently rewritten elsewhere.
