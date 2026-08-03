# Activation Functions Specification

> Cross-references: [Schema Reference](schema-reference.md) · [Architecture](architecture.md) · Paper Annex 8

## Overview

The system supports **43 feed-forward activation functions** (40 elementwise
plus 3 gated-FFN variants) across five families. Activations are selected via
the `model.ffn_activation` schema field and dispatched at runtime by the
`get_activation` factory in `src/model/activation_function/factory.py`,
mirroring the `get_norm` pattern used for normalization layers.

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

| Family | Names |
|---|---|
| Classical / Sigmoid-Tanh | `silu`, `gelu`, `gelu_tanh`, `relu`, `sigmoid`, `tanh`, `arctan`, `softsign`, `elliott`, `identity`, `softplus`, `mish` |
| Rectified | `leaky_relu`, `relu6`, `hardswish`, `prelu`, `abs_relu`, `nl_relu`, `brelu`, `vrelu`, `hexpo`, `ptanh`, `dis_relu`, `lisht` |
| Exponential / ELU | `elu`, `selu`, `celu`, `pelu`, `mpelu`, `felu`, `eelu`, `pdelu`, `preu`, `softexp`, `elish`, `hardelish` |
| Learnable / Adaptive | `swish`, `swish_trainable`, `maxout`, `raf` |
| Gated FFN | `swiglu`, `geglu`, `reglu` |

## `ffn_activation_config` Keys

All optional; ignored for stateless activations. Enforced by
`UltraConfig.__post_init__` and the JSON-Schema (`additionalProperties: false`).

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

Plus `pelu_alpha`, `mpelu_alpha`, `mpelu_beta`, `felu_alpha`, `eelu_alpha`,
`eelu_beta`, `pdelu_alpha`, `preu_alpha`, `preu_beta`, `softexp_alpha` for the
ELU parametric family.

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
| `src/model/tormented_bert_frankestein.py` | `UltraConfig.ffn_activation[_config]`, `HybridLayer` FFN wiring, `_validate_ffn_activation_config` |
| `src/schema/_model.yaml` | `ffn_activation` enum + `ffn_activation_config` object |
| `tests/test_activation_functions.py` | 45 tests: shape/gradient/range/correctness/factory/validation |
| `configs/examples/activation_*.yaml` | Example presets (raf, swiglu, mish) |

## References

- Dubey et al. (2021), arXiv:2109.14545 — survey & benchmark.
- Lederer (2021), arXiv:2101.09957 — systematic overview + derivatives.
- Fang et al. (2023), arXiv:2208.14111 — Rational Activation Functions (RAF).
- See `docs/paper/appendices/annex-8-activation-functions.tex` for full formulas.
