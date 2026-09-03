# Fast-Weight Attention for Continual Learning: A Literature Review

## Executive Summary

Recurrent fast-weight memories and selective state-space models compress an
expanding context into a fixed-size recurrent state, which makes the state
transition an **online learning rule**. The Falcon framework (Zhang et al.,
2026, arXiv:2608.27763, "Fast Weight Attention for Continual Learning",
ByteDance Seed / Princeton / Tsinghua / UCLA) studies this rule under
**read-after-write (RAW)** autoregressive semantics and makes a central
alignment observation: for the prefix-prediction objective, the local
fast-memory example revealed at step `t` is the **prefix-aligned pair**
`(x_t, y_t) = (φ(k_{t−1}), v_t)` — one step shifted relative to the
same-step association `(φ(k_t), v_t)` used by DeltaNet and standard linear
attention. The same-step pairing remains causal but optimizes a different
internal objective. The framework derives **normalized first-order
updates** for two objectives (squared-error regression and negative inner
product) and three step-size structures (scalar, per-column, sliding
window), giving six variants: **Falcon-1/2/3** (regression family) and
**Falcon-1A/2A/3A** (inner-product family). Every variant ships a
recurrent form, a masked-parallel form, and an SSD-style chunk-parallel
form, together with a numerically stable **positive-decay renormalization**
(clamping the per-step shrinkage `α_t = min(η_t·λ_t, 1 − ε_γ)` and working
in fp32 `log1p(−α)` chunk-local log-prefix space). The framework separates
four previously entangled knobs: **temporal alignment** (which pairing the
write optimizes), **plasticity** (the dimensionless NLMS gain `β_t`),
**forgetting** (the ridge decay `λ_t`), and **bounded rehearsal** (the
sliding window `B` of Falcon-3/3A).

## The Two Objectives

**Squared-error regression** (delta-rule family):

```
ℓ_t(S) = 1/2 ||v_t − S^T x_t||² + (λ_t/2)||S||_F²
∇_S ℓ_t = x_t(S^T x_t − v_t)^T + λ_t S          (L_t-smooth with L_t = ||x_t||² + λ_t)
```

**Negative inner product** (additive / linear-attention family):

```
ℓ_t^{ip}(S) = −⟨S^T x_t, v_t⟩ + (λ_t/2)||S||_F²
∇_S ℓ_t^{ip} = −x_t v_t^T + λ_t S
```

A normalized first-order step with the objective-matched denominator gives
`η_t = β_t/(statistic_t + λ_t + ε)` where the statistic is `||x_t||²`
(Falcon-1/1A/2/2A), the window spectral statistic `μ_t^(B) =
λ_max(X^T X)/B_t` (Falcon-3) or the window mean energy `Ē_t^(B)`
(Falcon-3A).

## 1. Falcon-1 — Scalar NLMS Regression

### Description

The scalar delta-rule variant. The state update is the classical NLMS
recursion of adaptive filtering, upgraded with the prefix-aligned write
stream and the explicit ridge decay:

### Mathematical Formulation

```
x_t = φ(k_{t−1}), x_1 = 0, η_1 = 0
η_t = β_t / (||x_t||² + λ_t + ε),  β_t ∈ (0, 2)
r_t = v_t − S_{t−1}^T x_t
S_t = (1 − η_t λ_t) S_{t−1} + η_t x_t r_t^T
o_t = S_t^T φ(q_t)                      (read-after-write)
```

With `ε = 0`, `λ_t = 0` and the unshifted assignment `x_t = φ(k_t)` this
recovers DeltaNet exactly. The eigenstructure of the transition
`A_t = γ_t I − η_t x_t x_t^T` shows the write direction contracts by
`1 − β_t` while the orthogonal subspace carries `γ_t`: `β_t > 1` yields a
stable sign-flip along the write direction (useful for state tracking).

Chunk-parallel form: the WY representation `Π(I − η_s x_s x_s^T) = I −
U T U^T` with a single-residual triangular solve (`L = tril(G, −1) + I`,
one multi-RHS solve), plus the positive-decay renormalization that maps
the decayed recurrence exactly onto the no-ridge kernel via
`η̃_t = η_t/γ_t` and `ṽ_t = v_t/c_{t−1}` (Alg. 8 of the paper).

### Pros and Cons

| Pros | Cons |
|------|------|
| Objective-matched normalization guarantees per-step descent (β ∈ (0,2)) | Regression family extrapolates worse on arithmetic tasks than the additive family |
| Best LM perplexity of the paper (Falcon-1.3: FineWeb-Edu 17.10 at 124M, beating Gated DeltaNet 17.32) | Same-step→shifted alignment requires the x_1 = 0 feature-space boundary sentinel |
| Reduces to DeltaNet / classical NLMS under ablations | — |

## 2. Falcon-2 — Per-Column NLMS Regression

The ridge loss decomposes across value coordinates, so each value channel
`j` gets its own step size sharing the NLMS normalizer across columns:

```
η_{j,t} = β_{j,t} / (||x_t||² + λ_t + ε)
S_t = S_{t−1}(I − λ_t Diag(η_t)) + x_t (η_t ⊙ r_t)^T
```

Chunk-parallel form: a shared key Gram `G = X^T X` with per-channel
unit-lower-triangular systems `L_j = I + tril(G ⊙ (Σ_j Σ_j^T), −1)`
(`Σ = √η̃`) solved as a batch (Alg. 1). Defined but not separately
benchmarked in the paper.

## 3. Falcon-3 — Sliding-Window Minibatch Regression

Bounded rehearsal: each step regresses the fast memory on the last
`B_t ≤ B` prefix-aligned pairs with **all window residuals evaluated at
the pre-update state** `S_{t−1}`:

```
I_t = {max(2, t−B+1), …, t}
μ_t = λ_max(X_t^T X_t)/B_t           (exact, from the B_t × B_t Gram)
η_t = β_t/(μ_t + λ_t + ε)
S_t = (1 − η_t λ_t) S_{t−1} + (η_t/B_t) Σ_{j∈I_t} x_j (v_j − S_{t−1}^T x_j)^T
```

Window-averaging keeps the update magnitude invariant to `B`: each pair is
replayed in exactly `B` consecutive updates at weight `1/B`, so its
cumulative injection is `η` independent of `B`. `S_t` alone is **not
Markov**: exact continuation requires the FIFO tail of the last `B−1`
pairs (the chunk kernels gather an extended slice `|J| ≤ C + B − 1`).
Chunk-parallel form: a rank-B affine recurrence solved by block forward
substitution over the (time × rank) extended index space — same-time rank
components uncoupled — the ParallelFlow `tensorInv` structure of Alg. 3.

## 4. Falcon-1A / Falcon-2A — Inner-Product (Additive) Variants

One gradient step on the inner-product objective gives a purely additive
(Hebbian) write with an energy-normalized gain:

```
Falcon-1A:  S_t = γ_t S_{t−1} + η_t x_t v_t^T,           γ_t = 1 − min(η_t λ_t, 1 − ε_γ)
Falcon-2A:  S_t = S_{t−1} Diag(γ_t) + x_t (η_t ⊙ v_t)^T  (per-channel decays)
```

With `λ_t = 0` and the unshifted assignment this is exactly standard
linear attention / Mamba-2 accumulation; the shifted write stream makes it
the one-step-shifted next-latent variant (Eq. 2.3). The A-family's
denominator is not required by curvature (the objective is linear) — it
retains the energy-normalized write as a magnitude stabilizer.
Chunk-parallel form: decay-mask linear attention
`M_{t,i} = η_i Π_{r=i+1}^t γ_r` (Fig. 7B), per-channel for 2A (Fig. 8B).

## 5. Falcon-3A — Sliding-Window Inner-Product

The windowed additive variant — the paper's **best length-extrapolation
configuration** (Falcon-3A.3: 87.2% mean accuracy on 33–48-digit addition
vs 65.8% for a RoPE Transformer and 85.9% for Falcon-1A.3):

```
N̄_t^(B) = (1/B_t) Σ_{j∈I_t} x_j v_j^T
Ē_t^(B) = (1/B_t) Σ_{j∈I_t} ||x_j||²
η_t = β_t/(Ē_t^(B) + λ_t + ε)
S_t = γ_t S_{t−1} + η_t N̄_t^(B)
```

Chunk-parallel form: the window-banded decay mask factors as
`M = D · Diag(η) · A` where `A` is the B-banded window operator and `D`
the causal decay kernel, evaluated over the extended slice (Alg. 4) —
"moving-sum then exponential-tail" in the stationary case.

## Experimental Results (paper)

124M-parameter models on FineWeb-Edu (49.2B tokens, bfloat16, AdamW):

| Model | WikiText ppl | LMB. ppl | FineEdu ppl | 0-shot avg |
|---|---|---|---|---|
| Transformer (RoPE) | 33.25 | 47.43 | 17.38 | 48.16 |
| Mamba-2 | 34.53 | 48.74 | 17.70 | 48.80 |
| DeltaNet | 34.19 | 52.84 | 17.84 | 48.88 |
| Gated DeltaNet | 30.99 | 46.70 | 17.32 | 48.78 |
| Falcon-1A.3 | 34.02 | 49.84 | 17.40 | 48.95 |
| **Falcon-1.3** | **33.00** | 48.70 | **17.10** | 49.18 |
| **Falcon-3A.3** (arith.) | — | — | — | 49.00 |

Variable-digit addition length extrapolation (33–48 digits, mean accuracy):
Falcon-3A.3 **87.2** > Falcon-1A.3 85.9 > RetNet 82.9 > Falcon-1A.2 85.2 >
Transformer 65.8 > Falcon-1.3 68.8. The inner-product family carries
arithmetic; the regression family carries perplexity.

## Relation to Prior Work

- **DeltaNet**: the delta rule is the squared-error OGD step with the
  same-step pairing; Falcon recovers it with "the critical index shift"
  plus objective-matched normalization and explicit `λ_t` shrinkage.
- **Gated DeltaNet**: adds a free learned carry gate; Falcon **derives**
  the carry `γ_t = 1 − η_t λ_t` from the local objective and the
  normalized step-size parameterization instead of adding an independent
  gate, and defaults to QK-RMSNorm rather than ℓ2 normalization.
- **Linear Attention / Mamba-2**: the additive numerator update is one
  gradient step on the inner-product objective with the unshifted
  assignment; Mamba-2's SSD chunkwise scheme is what the Falcon
  chunk-parallel forms reuse.
- **TTT / Titans / ATLAS**: fast-weight update as a constrained
  single-step test-time-training instance; Falcon-3 is a sliding-window
  specialization of the internal-objective view with strict
  next-latent alignment.
- **MesaNet**: solves the cumulative least-squares problem offline in the
  forward pass; Falcon is its online first-order counterpart.
- **Adaptive filtering (LMS/NLMS/RLS)**: Falcon imports the stability and
  normalization principles of NLMS into fast-weight attention under
  strict causality; fixed learning rates are scale-mismatched for
  regression-style updates.

## Summary Comparison Table

| Variant | Family | Step-size statistic | Chunk kernel | Paper status |
|---|---|---|---|---|
| **Falcon-1** | Regression (delta rule) | `‖x_t‖²` | WY triangular solve + decay renorm (Alg. 8) | Benchmarked (.3 best overall ppl) |
| **Falcon-2** | Regression, per-column | `‖x_t‖²` shared | Batched per-channel triangular solves (Alg. 1) | Defined, unbenchmarked |
| **Falcon-3** | Regression, window B | `μ_t^(B) = λ_max(X^TX)/B_t` | Rank-B block substitution (Alg. 3) | Defined, unbenchmarked |
| **Falcon-1A** | Inner product (additive) | `‖x_t‖²` | Decay-mask linear attention (Fig. 7B) | Benchmarked (.3 second-best extrap.) |
| **Falcon-2A** | Inner product, per-column | `‖x_t‖²` shared | Per-channel decay masks (Fig. 8B) | Defined, unbenchmarked |
| **Falcon-3A** | Inner product, window B | `Ē_t^(B)` | Window-banded decay mask (Alg. 4) | Benchmarked (.3 best extrapolation) |

## References

- Zhang, Y., Ta, S., Zhang, J., Feng, J., Li, S., Zhang, Y., Liu, Y.,
  Yuan, H., Wang, M., Gu, Q., Yao, A. C.-C. (2026). "Fast Weight
  Attention for Continual Learning". arXiv:2608.27763.
  Project page: https://github.com/yifanzhang-pro/fast-weight-attention
- Schlag, I., Irie, K., Schmidhuber, J. (2021). "Linear Transformers
  Are Secretly Fast Weight Programmers". arXiv:2102.11174.
- Yang, S. et al. (2024). "Gated Delta Networks: Improving Mamba-based
  Models with Delta Rules". arXiv:2412.06464.
- Kimi Team (2025). "Kimi Linear: An Expressive, Efficient Attention
  Architecture". arXiv:2510.26692.