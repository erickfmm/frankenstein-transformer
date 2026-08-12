# mHC: Manifold-Constrained Hyper-Connections

> Cross-references: [Architecture](architecture.md) · [Schema Reference](schema-reference.md) · [Training Safety](training-safety.md)

This spec documents the Frankenstein integration of **mHC**
(Manifold-Constrained Hyper-Connections), arXiv:2512.24880 (DeepSeek-AI).

## What problem does mHC solve?

A normal residual connection (the `x + F(x)` pattern in every transformer
layer) is what lets gradients flow cleanly through a deep network. When you
expand this to an **n-stream residual** (keeping `n` copies of the hidden
state instead of one), you gain representational power but lose the
identity-mapping property: the network can now distort or lose information
between layers, which causes unstable gradients and poor scaling.

mHC fixes this by **constraining the within-stream mixing matrix to the
Birkhoff polytope** (doubly-stochastic matrices). This restores the
identity/conservation property, so you get the benefits of a wider stream
without the gradient-instability cost. In plain terms: it lets the model use
a wider, more expressive residual while guaranteeing the math stays stable.

## Overview

mHC replaces the standard residual connection with an **n-stream residual**.
The residual stream width is expanded by a factor `n`, so the stream carried
across layers is `x ∈ R^{n×C}` while each layer's internal function `F`
(attention and FFN) still operates at dimension `C`. Three learnable mappings
read, write and mix the stream each layer:

- `H[pre] ∈ R^{1×n}` — aggregates the `n·C`-dim stream into the `C`-dim layer input.
- `H[post] ∈ R^{1×n}` — maps the layer output back onto the stream.
- `H[res] ∈ R^{n×n}` — mixes features *within* the residual stream.

Unconstrained Hyper-Connections (HC) lose the identity-mapping property of the
residual connection, causing unstable gradients and restricted scalability. mHC
**constrains `H[res]` to the Birkhoff polytope** (the set of doubly stochastic
matrices) via the Sinkhorn-Knopp projection, which restores the identity-mapping
/conservation property: the composite product `Π_l H_l[res]` stays doubly
stochastic across all depth, spectral norm `‖H[res]‖₂ ≤ 1` (non-expansive), and
`H[res] x` becomes a convex combination of the stream features.

When `n = 1` the doubly-stochastic condition degenerates to scalar `1`, exactly
recovering the identity mapping.

## Mathematical formulation

Per layer `l`, with stream `x_l ∈ R^{n×C}`:

```
x̃_l  = vec(x_l) ∈ R^{1×nC}
H̃    = (1/r)·(α ⊙ (x̃_l φ_l)) + b_l      # r = ‖x̃_l‖₂ / √(nC); α gating (init 0.01)
H_l[pre]  = σ(H̃_pre)                     # non-negative
H_l[post] = 2σ(H̃_post)                   # non-negative, range [0, 2]
H_l[res]  = SinkhornKnopp(exp(H̃_res))    # doubly stochastic
Fpre      = H_l[pre] @ x_l                # [1, C]
x_{l+1}   = H_l[res] @ x_l + H_l[post]ᵀ ⊗ F(Fpre, W_l)
```

- `φ_l ∈ R^{nC × (n²+2n)}` — learned linear projection (full precision).
- `b_l ∈ R^{1 × (n²+2n)}` — learned bias.
- `α_pre, α_post, α_res` — learnable scalar gates, initialised small (0.01).
- **Sinkhorn-Knopp**: starting from `exp(H̃_res)`, alternate row and column
  normalisations (default `t_max = 20` rounds) to converge to a doubly
  stochastic matrix. The backward pass recomputes the iteration on-chip and
  differentiates through it (exact Jacobian-vector product).

## Implementation (Frankenstein)

New module `src/model/mhc.py`:

- `SinkhornKnoppFunction(torch.autograd.Function)` — differentiable projection.
- `ManifoldHyperConnections(nn.Module)` — holds `φ_l` (`proj`), `b_l` (`bias`)
  and the three gating scalars; exposes `fpre`, `recombine` and `mappings`.

Wiring in `src/model/frankenstein_model.py`:

- `HybridLayer` gains `mhc_attn` and `mhc_ffn` (one module per layer function).
  `_forward_dense_mhc` runs attention then FFN as layer functions over the
  shared `(B, S, n, C)` stream.
- `FrankensteinTransformer` expands the `C`-dim embedding to `(B, S, n, C)`
  via `mhc_in_proj` and collapses back via `mhc_out_proj` before the head.
- `mhc_checkpoint` optionally applies gradient checkpointing per layer to
  mitigate the ~`n`× activation-memory increase of the n-stream residual.

## Config reference

The `model.mhc` sub-object (hierarchical schema) or flat keys:

| YAML (`model.mhc.*`) | Flat key | Type | Default | Meaning |
|---|---|---|---|---|
| `enabled` | `use_mhc` | bool | `false` | Enable the mHC n-stream residual. |
| `expansion_rate` | `mhc_expansion_rate` | int ≥ 1 | `4` | Stream expansion factor `n`. |
| `sinkhorn_iters` | `mhc_sinkhorn_iters` | int ≥ 1 | `20` | Sinkhorn-Knopp normalisation rounds. |
| `gating_init` | `mhc_gating_init` | float > 0 | `0.01` | Initial value of the gating scalars `α`. |
| `checkpoint` | `mhc_checkpoint` | bool | `false` | Gradient checkpointing on mHC layers. |
| `full_prec_under_bitnet` | `mhc_full_prec_under_bitnet` | bool | `true` | Keep `φ_l` full-precision under BitNet. |

Example config: `configs/examples/es_arch_mhc_adamw.yaml`.

### Enabling mHC in your config

The nested `model.mhc` block and its flat equivalent are equivalent. To turn
mHC on with a wider stream and checkpointing:

```yaml
model:
  mhc:
    enabled: true
    expansion_rate: 4      # stream width = 4 × hidden_size
    sinkhorn_iters: 20
    gating_init: 0.01
    checkpoint: true       # trade compute for lower activation memory
```

### Choosing `expansion_rate`

`expansion_rate` (`n`) is the width multiplier of the residual stream. Larger
`n` gives more representational capacity but multiplies activation memory by
roughly `n×` (this is why `mhc_checkpoint` exists). As a rule of thumb:

| Goal | Suggested `n` |
|---|---|
| Minimal overhead, near-standard residual | `2` |
| Balanced capacity vs. memory (paper default) | `4` |
| Maximum expressiveness (large memory budget) | `8` |

When `n = 1`, mHC degenerates to the plain identity residual and adds no
benefit. If you are memory-constrained, start at `n = 2` with `checkpoint:
true`.

## Constraints

- mHC is **incompatible with `use_mixture_of_depths`** (MoD token routing
  operates on a single `C`-dim stream, conflicting with the n-stream residual).
  A `ValueError` is raised if both are enabled.
- The `φ_l` projection stays full-precision under BitNet by default
  (`full_prec_under_bitnet: true`) to avoid ternary-quantisation noise on the
  small mHC coefficients. Set it to `false` to use `BitLinear`.
