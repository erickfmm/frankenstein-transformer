# Activation Functions Specification

> Cross-references: [Schema Reference](schema-reference.md) · [Architecture](architecture.md) · Paper Annex 8

## Overview

The system supports **43 feed-forward activation functions** (40 elementwise
plus 3 gated-FFN variants) across five families. Activations are selected via
the `model.ffn_activation` schema field and dispatched at runtime by the
`get_activation` factory in `src/model/activation_function/factory.py`,
mirroring the `get_norm` pattern used for normalization layers.

### What is an activation function?

An activation function is the non-linear "decision" applied after a linear
projection inside the feed-forward network (FFN). Without one, stacking
linear layers would collapse into a single linear map and the model could not
learn complex patterns. The choice of activation shapes *how* the model
transforms its hidden representations and can affect training stability,
final quality, and inference speed.

### How to think about the choice

There is no single "best" activation — the right pick depends on your goal:

| Goal | Suggested activation |
|---|---|
| Safe, modern default | `silu` (SiLU / Swish₁) |
| Classic, well-understood baseline | `gelu` / `relu` |
| Smooth, non-monotonic curves | `mish` / `swish` |
| Cheap on mobile / edge hardware | `hardswish` / `relu6` |
| Learn the shape per task | `raf`, `prelu`, `pelu`, `swish_trainable`, `maxout` |
| Gated FFN with its own projections | `swiglu` / `geglu` / `reglu` |

The easiest way to experiment is to change one field in your YAML config and
re-run — see [Switching an activation](#switching-an-activation) below.

## Family Decision Tree

```
Choose activation objective
├── Safe default → silu (SiLU/Swish₁)  [recommended]
├── Classic stability → gelu / relu
├── Smooth non-monotonic → mish / swish
├── Cheap mobile-friendly → hardswish / relu6
├── Learnable per-task shape → raf (Rational Activation Function)
│                           → prelu / pelu / swish_trainable / maxout
└── Gated FFN (own projections) → swiglu / geglu / reglu
```

## Selection Contract

```yaml
model:
  ffn_activation: <name>        # one of the 43 enum values
  ffn_activation_config:        # optional; only for learnable/parametric activations
    raf_degrees: [5, 4]
    raf_version: A
    raf_approx_func: gelu
    ...
```

### Enum (source of truth)

Mirrored in `src/schema/_model.yaml` (`ffn_activation.enum`) and
`src/model/activation_function/factory.py` (`ELEMENTWISE_ACTIVATIONS`).

The table below adds a plain-English description of **what each activation
does** and **when you might choose it**. All names are accepted values for
`model.ffn_activation`.

| Family | Name | What it does / When to use it |
|---|---|---|
| Classical / Sigmoid-Tanh | `silu` | `x·σ(x)` — smooth, non-negative, the modern default. Best general-purpose choice. |
| | `gelu` | `x·Φ(x)` — smooth approximation of a stochastic gate. Strong on BERT-class models. |
| | `gelu_tanh` | GELU computed with the `tanh` approximation — slightly faster, near-identical output. |
| | `relu` | `max(0, x)` — classic, cheapest, but can "die" on negative inputs. |
| | `sigmoid` | `1/(1+e⁻ˣ)` — maps to `(0, 1)`. Rarely used directly in hidden layers. |
| | `tanh` | Maps to `(−1, 1)` — symmetric, good for bounded hidden states. |
| | `arctan` | `atan(x)` — smooth, saturating, mild. |
| | `softsign` | `x/(1+∣x∣)` — saturating, no vanishing-gradient cliff. |
| | `elliott` | `x/(1+∣x∣)` variant — a cheap, saturating alternative. |
| | `identity` | `x` — passes through unchanged. Useful for linear FFN experiments. |
| | `softplus` | `log(1+eˣ)` — smooth, always positive. |
| | `mish` | `x·tanh(softplus(x))` — smooth, slightly better gradients than SiLU on some tasks. |
| Rectified | `leaky_relu` | ReLU with a small negative slope (`0.01`) — avoids dying neurons. |
| | `relu6` | `min(max(0,x), 6)` — ReLU capped for quantized / mobile-friendly models. |
| | `hardswish` | Cheap piecewise approximation of SiLU — ideal for edge devices. |
| | `prelu` | Parametric ReLU — learns its own negative slope during training. |
| | `abs_relu` | Absolute-value based ReLU variant. |
| | `nl_relu` | Non-linear variant of ReLU. |
| | `brelu` | Bounded ReLU (caps positive values). |
| | `vrelu` | Variance-based ReLU. |
| | `hexpo` | Hexponential — smooth, tunable curvature. |
| | `ptanh` | Penalized tanh — tanh with a learnable penalty. |
| | `dis_relu` | Displaced ReLU. |
| | `lisht` | `x·tanh(x)` — smooth, self-gated, stable. |
| Exponential / ELU | `elu` | Smooth, can output negatives (mean-shift toward zero) — `elu_alpha` tunable. |
| | `selu` | Self-normalizing ELU — good for very deep nets (keeps activations normalized). |
| | `celu` | Continuously differentiable ELU — smooth everywhere. |
| | `pelu` / `mpelu` | Parametric ELU variants — learn their shape. |
| | `felu` / `eelu` | ELU variants with learnable exponential parameters. |
| | `pdelu` | Parametric displaced ELU. |
| | `preu` | Parametric ELU variant with learnable `α`, `β`. |
| | `softexp` | Soft exponential — interpolates between linear and exponential. |
| | `elish` / `hardelish` | ELU + SiLU hybrids — smooth self-gated curves. |
| Learnable / Adaptive | `swish` | `x·σ(βx)` — learnable `swish_beta`. Flexible generalization of SiLU. |
| | `swish_trainable` | Swish whose `β` is trained as a parameter. |
| | `maxout` | Takes the max over `maxout_pieces` linear projections — universal approximator. |
| | `raf` | Learnable rational (Padé) function — the most flexible option. See [RAF](#rational-activation-function-raf). |
| Gated FFN | `swiglu` | Swish-gated FFN — two projections gated together. State-of-the-art in LLMs. |
| | `geglu` | GELU-gated FFN. |
| | `reglu` | ReLU-gated FFN — cheaper, slightly less expressive than SwiGLU. |

> The three **gated FFN** names are not elementwise activations: they change
> the whole FFN block. See [GLU Variants](#glu-variants).

## `ffn_activation_config` Keys

All optional; ignored for stateless activations. Enforced by
`FrankensteinModelConfig.__post_init__` and the JSON-Schema (`additionalProperties: false`).

| Key | Type | Default | Applies to |
|---|---|---|---|
| `raf_degrees` | `[int, int]` ≥1 | `[5, 4]` | `raf` |
| `raf_version` | enum `A\|B\|C\|D\|N` | `A` | `raf` |
| `raf_approx_func` | enum | `gelu` | `raf` |
| `raf_trainable` | bool | `true` | `raf` |
| `raf_input_scaling` | bool | `false` | `raf` |
| `prelu_init` | number | `0.25` | `prelu` |
| `elu_alpha` | number | `1.0` | `elu` |
| `celu_alpha` | number | `1.0` | `celu` |
| `swish_beta` | number | `1.0` | `swish`, `swish_trainable` |
| `leaky_relu_slope` | number | `0.01` | `leaky_relu` |
| `maxout_pieces` | int ≥1 | `2` | `maxout` |

The parametric ELU family also uses per-activation alpha/beta keys. They are
grouped here (instead of a separate row each) for brevity:

| Key | Default | Applies to |
|---|---|---|
| `pelu_alpha` | — | `pelu` |
| `mpelu_alpha`, `mpelu_beta` | — | `mpelu` |
| `felu_alpha` | — | `felu` |
| `eelu_alpha`, `eelu_beta` | — | `eelu` |
| `pdelu_alpha` | — | `pdelu` |
| `preu_alpha`, `preu_beta` | — | `preu` |
| `softexp_alpha` | — | `softexp` |

## Switching an activation

Because the activation is just a config field, swapping it is a one-line
change. The example below switches from the default SiLU to a learnable
rational activation and tunes its shape, while keeping everything else
unchanged:

```yaml
model:
  ffn_activation: raf          # was: silu
  ffn_activation_config:
    raf_degrees: [6, 5]        # numerator/denominator degree
    raf_version: A
    raf_trainable: true
```

To use a gated FFN instead (which also swaps the FFN block structure):

```yaml
model:
  ffn_activation: swiglu       # becomes a GatedFFN with its own gate projection
```

There is no schema change required — both are valid enum values. Re-validate
by loading the config; unknown keys or invalid values fail fast with a clear
message.

## Rational Activation Function (RAF)

From *Transformers with Learnable Activation Functions* (Fang et al.,
arXiv:2208.14111). A learnable Padé ratio `P(x)/Q(x)`:

- **Default:** degree `(5, 4)`, version `A` (safe, per-term-abs denominator so
  `Q(x) ≥ 1`), initialized by a least-squares fit to GELU on `[-3, 3]`.
- **RAFT input scaling:** `raf_input_scaling=true` applies per-token min-max
  scaling to `[-3, 3]` before the rational (keeps inputs in the fitted range).
- **Freezing:** `raf_trainable=false` for parameter-efficient fine-tuning.

> **Naming note:** RAF = *Rational* Activation Function, not "Rectified".

## GLU Variants

SwiGLU / GEGLU / ReGLU (Shazeer, arXiv:2002.05202) are gated FFN units, not
elementwise activations. When `ffn_activation` is one of these, `HybridLayer`
swaps the dense/MoE FFN block for a `GatedFFN`:

```
GatedFFN(x) = act(x W_gate) ⊙ (x W_up) W_down
```

built with the same projection class (BitLinear under BitNet).

## Key Files

| File | Role |
|---|---|
| `src/model/activation_function/factory.py` | `get_activation(config, dim=None)` dispatch + enum constants |
| `src/model/activation_function/common.py` | Classical activations |
| `src/model/activation_function/rectified.py` | ReLU family |
| `src/model/activation_function/exponential.py` | ELU family |
| `src/model/activation_function/learnable.py` | RAF, SwishTrainable, Maxout |
| `src/model/activation_function/glu.py` | `GatedFFN`, `make_gated_ffn` |
| `src/model/frankenstein_model.py` | `FrankensteinModelConfig.ffn_activation[_config]`, `HybridLayer` FFN wiring, `_validate_ffn_activation_config` |
| `src/schema/_model.yaml` | `ffn_activation` enum + `ffn_activation_config` object |
| `tests/test_activation_functions.py` | 45 tests: shape/gradient/range/correctness/factory/validation |
| `configs/examples/activation_*.yaml` | Example presets (raf, swiglu, mish) |

## References

- Dubey et al. (2021), arXiv:2109.14545 — survey & benchmark.
- Lederer (2021), arXiv:2101.09957 — systematic overview + derivatives.
- Fang et al. (2023), arXiv:2208.14111 — Rational Activation Functions (RAF).
- See `docs/paper/appendices/annex-8-activation-functions.tex` for full formulas.
