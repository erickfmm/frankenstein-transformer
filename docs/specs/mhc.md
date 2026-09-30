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

## Hyperloop mode (arXiv:2604.21254)

Setting `model.mhc.hyperloop: true` replaces the per-sublayer wiring above with
the **Hyperloop Transformer** (Zeitoun, Torroba-Hennigen & Kim, MIT): the layer
stack is partitioned **middle-cycle** into a begin block (runs once), a looped
middle block, and an end block (runs once), and hyper-connections fire **once
per loop iteration** with **per-loop** parameters `{W_l, b_l, α_l, e_l}`
instead of shared per-sublayer modules:

```
x → begin → copy×n → { read(H^pre) → middle → +e_l → update(H^res, H^post) } × num_loops → avg → end → head

y_t^{(l+1)} = H_l^res · y_t^{(l)} + H_l^post ⊗ ( F(H_l^pre · y_t^{(l)}) + e_l )

H_l^pre  = σ(α_pre · (W_pre z_t) + b_pre)          # z_t = RMSNorm(flatten(y_t^{(l)}))
H_l^post = 2σ(α_post · (W_post z_t) + b_post)
H_l^res  = diag(σ(α_res · (W_res z_t) + b_res))    # "diagonal" (paper default)
```

Why it matters (paper results): matches depth-matched Transformers at ~50%
fewer parameters (136M Hyperloop PPL 14.40 vs 238M Transformer 14.65 on
FineWeb-Edu), survives INT4/GPTQ quantization, and — because the loop-level
`H^res` uses a diagonal sigmoid and fires once per loop — adds only ~5%
throughput overhead in plain PyTorch (per-layer mHC loses ~33%).

Deltas vs per-layer mHC implemented above:

1. Hyper-connections fire once per **loop**, not after every sub-layer — the
   paper's Table 5 ablation also shows every-loop placement is the most
   performant.
2. The n-stream residual is created by **copying** the begin-block output and
   collapsed by **averaging** — no learned `mhc_in_proj`/`mhc_out_proj`.
3. Parameters (`W/b/α/e`) are **per loop**, relaxing strict weight sharing so
   looped representations can deviate across iterations (the paper's Table 8
   cosine-similarity analysis supports this as the source of the gains).
4. `H^res` defaults to a **diagonal sigmoid** (`diagonal`), the paper's best
   ablation (14.40 vs 14.59 Sinkhorn / 14.61 identity, Table 6); `"sinkhorn"`
   and `"identity"` remain available.

New module `src/model/hyperloop.py`:

- `HyperloopConnections` — per-loop `projs`/`biases`/`alphas_*`/`loop_pos_embs`
  with `expand` (copy), `collapse` (average), `mappings`, `apply_fpre`
  (stream → C) and `apply_update` (write `F(...) + e_l`, mix with `H^res`).

Wiring:

- `FrankensteinEncoder.__init__` builds `self.hyperloop` instead of
  `mhc_in_proj`/`mhc_out_proj` when `mhc_hyperloop` is set; `forward` runs the
  begin slice, copies the stream, then per loop: read `H^pre` → middle layers
  → `apply_update` (mix `H^res`, write with `H^post`, add `e_l`) — and
  finally averages the streams before the end slice.
- `HybridLayer` disables its per-layer `mhc_attn`/`mhc_ffn` under Hyperloop
  (layers run the standard C-dim path), which **restores
  `use_mixture_of_depths` compatibility** — MoD token routing lives inside
  middle-block layers and never sees the n-stream residual.
- `FrankensteinViT` wires the same loop-level path in `_run_encoder`.

### Hyperloop config reference

| YAML (`model.mhc.*`) | Flat key | Type | Default | Meaning |
|---|---|---|---|---|
| `hyperloop` | `mhc_hyperloop` | bool | `false` | Enable loop-level hyper-connections over the middle-cycle looped block. |
| `hyperloop_begin_layers` | `mhc_hyperloop_begin_layers` | int ≥ 0 | `0` | Begin-block layers (run once before the loop). `0` = loop the whole stack. |
| `hyperloop_end_layers` | `mhc_hyperloop_end_layers` | int ≥ 0 | `0` | End-block layers (run once after the loop). |
| `hyperloop_res_parameterization` | `mhc_hyperloop_res_parameterization` | enum | `diagonal` | `H^res` mixing: `diagonal`, `sinkhorn`, or `identity`. |

Shared with per-layer mHC: `enabled`, `expansion_rate` (`n`), `gating_init`,
`checkpoint`, `full_prec_under_bitnet`, `sinkhorn_iters` (only used by the
`sinkhorn` parameterization). `hyperloop` requires `enabled: true` and
`dims.num_loops >= 2`. Parameter budget per loop: `3n²C + 3n + 3 + C`
(diagonal) — e.g. the paper's 136M model (C=1024, n=4, 3 loops) spends only
~150K extra parameters vs the plain looped Transformer.

Paper middle-cycle presets (paper uses ~25% / 50% / 25% parameter split over
begin/middle/end; unrolled depth = begin + middle×loops + end):

| Paper model | Begin | Middle (×3) | End |
|---|---|---|---|
| 136M (depth 16) | 2L | 4L | 2L |
| 580M (depth 18) | 3L | 4L | 3L |
| 990M (depth 38) | 4L | 10L | 4L |

Example config: `configs/examples/hyperloop_mhc_adamw.yaml` (scaled-down
`2L → 2L(x3) → 2L` middle cycle).

### Hyperloop constraints

- Requires `use_mhc: true`; plain `mhc.enabled` without `hyperloop` keeps the
  per-sublayer wiring (the two modes are mutually exclusive at runtime).
- Requires `num_loops >= 2` (the paper's models use 3).
- `hyperloop_begin_layers + hyperloop_end_layers < num_layers` (the looped
  middle block must contain at least one layer).
- **Incompatible with AttnRes residuals** (`residuals.type: full_attn` /
  `block_attn`) — AttnRes applies depth-wise attention per logical layer on
  the n-stream residual, while Hyperloop only mixes at loop boundaries.
- Unlike per-layer mHC, **Mixture-of-Depths is compatible** (middle-block
  layers run the standard C-dim path).
