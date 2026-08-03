# System Architecture Specification

> Cross-references: [Attention Mixers](attention-mixers.md) · [Schema Reference](schema-reference.md) · [Training Safety](training-safety.md) · [Deployment](deployment.md)

## System Design Overview

Frankestein Transformer is a **schema-first**, configuration-driven experimentation toolkit. The authoritative contract is `src/schema.yaml`, which enforces three top-level objects: `model_class`, `model`, and `training`. All nested objects set `additionalProperties: false` — unknown keys fail fast instead of being silently ignored.

### System Architecture Diagram

```
YAML Config → Validation (schema + rules) → Web (Streamlit) / CLI (train/deploy/infer)
                                              ↓
                    Model (17 mixers)  Training (AMP + scheduler)  Optimizer (23 families)
                         ↓                    ↓                          ↓
                    Model Build          Train Loop                  Opt Step
                    (pattern+loops)     (checks+logs)            (group routing)
                         ↓                    ↓                          ↓
                    Deploy (quantization)          SBERT (search/cluster)
```

Configuration flows through validation, partitions into model/training/optimizer, executes runtime, and produces deployment/SBERT artifacts.

## Model Classes

| Class | Type | Description |
|---|---|---|
| `frankenstein` | Mixed-architecture encoder | Full-featured encoder with all 19 mixer types, MoE, advanced normalization. Optimized for bidirectional MLM. |
| `mini` | Simplified encoder | Reduced parameter overhead for rapid prototyping and small-scale experimentation. |
| `frankesteindecoder` | Autoregressive causal decoder | LLM-style next-token generation. **Runtime forces `mode='decoder'`**. Enables causal attention masking. |

## Training Modes

| Mode | Attention Masking | Task | Model Class Compatibility |
|---|---|---|---|
| `encoder` | Bidirectional (all tokens attend to all) | MLM (masked language modeling) | `frankenstein`, `mini` |
| `decoder` | Causal (token attends only to previous) | AR (autoregressive next-token) | `frankesteindecoder` (forced), `frankenstein` |

When `model_class='frankesteindecoder'`, the system automatically forces `mode='decoder'` at runtime.

## Looped Depth Formula

```
L_logical = num_layers × num_loops
```

Physical layers are defined by `num_layers`. Each physical layer is executed `num_loops` times per forward pass, creating logical depth without increasing parameter count. Weights are shared across loop iterations by position.

## Layer Pattern Dispatcher

The `model.layer_pattern` array defines the ordered sequence of layer types. Pattern length must equal `num_layers`. The dispatcher routes each position to the corresponding mixer implementation:

| Category | Code Names | Count |
|---|---|---|
| Dense | `standard_attn`, `sigmoid_attn` | 2 |
| Recurrent | `retnet`, `retnet_attn`, `mamba`, `ode`, `titan_attn` | 5 |
| Sparse | `sparse_transformer_attn`, `longformer_attn`, `bigbird_attn`, `sparsek_attn`, `nsa_attn`, `sparge_attn`, `fasa_attn` | 7 |
| Gated | `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `hgrn2_attn`, `fox_attn`, `gated_softmax_attn` | 6 |
| Memory | `engram_attn` | 1 |

**Training-free policy**: `fasa_attn` and `sparge_attn` are eval/inference-only. Using them during training raises a runtime error.

## Mixture-of-Depths (MoD) Routing

When `use_mixture_of_depths=true`, each transformer block scores tokens with a lightweight router. Only the top `capacity_ratio` subset is updated; skipped tokens pass through unchanged. This allocates more depth to salient tokens, reducing per-layer compute.

| Field | Type | Range | Description |
|---|---|---|---|
| `use_mixture_of_depths` | bool | — | Enable MoD token routing |
| `mixture_of_depths_capacity_ratio` | float | (0, 1] | Fraction of tokens updated per layer |
| `mixture_of_depths_router_aux_loss_weight` | float | ≥ 0 | Auxiliary loss weight for router regularization |

## Engram Conditional Memory

Engram (`engram_attn`) implements conditional memory via scalable N-gram lookup (arXiv:2601.07372). It builds hash-based lookup tables for N-gram contexts from size 2 up to `engram_max_ngram_size`, with independent hash heads per N-gram order.

| Field | Type | Default | Description |
|---|---|---|---|
| `engram_max_ngram_size` | int | 3 | Highest N-gram order |
| `engram_n_heads_per_ngram` | int | 4 | Hash heads per N-gram order |
| `engram_embed_dim_per_head` | int | 32 | Embedding dim per hash head |
| `engram_kernel_size` | int | 4 | Causal depthwise conv kernel width |
| `engram_seed` | int | 42 | Hash seed for reproducibility |

Total Engram hidden size = `(max_ngram_size - 1) × n_heads_per_ngram × embed_dim_per_head`.

## BitNet Path

When `use_bitnet=true`, BitLinear layers replace standard `nn.Linear` with ternary weight quantization (−1, 0, +1). This enables significant compression (~32× vs FP32) but is experimental and may affect model quality. Best suited for inference deployment.

## Factorized Embeddings

When `use_factorized_embedding=true`, the embedding matrix is factorized into two smaller matrices, reducing parameters from `O(V × H)` to `O(V × R + R × H)` where `R = factorized_embedding_dim`.

| Field | Type | Range | Description |
|---|---|---|---|
| `use_factorized_embedding` | bool | — | Enable factorization |
| `factorized_embedding_dim` | int | ≥ 1 | Intermediate dimension (typical: 64–256) |

## Embedding Convolution

When `use_embedding_conv=true`, a 1D convolution is applied over token embeddings before the transformer stack to capture local n-gram patterns.

| Field | Type | Range | Description |
|---|---|---|---|
| `use_embedding_conv` | bool | — | Enable Conv1d over embeddings |
| `embedding_conv_kernel` | int | ≥ 1 | Kernel size (typical: 3) |

## Normalization Variants

| Code Name | Formula | Stats Needed | Notes |
|---|---|---|---|
| `layer_norm` | Standard LayerNorm (subtract mean, divide by std) | Mean + std | Stable, widely used baseline |
| `dynamic_tanh` | `tanh(α·x)` with learned α | None | Normalization-free bounded transform (DyT) |
| `derf` | `erf(α·x + s)` with learned α, s | None | Normalization-free; improves over DyT |
| `rms_norm` | `g_i · x_i / √(mean(x²) + ε)` | RMS only | No mean subtraction; 7%–64% faster than LayerNorm (arXiv:1910.07467) |
| `prms_norm` | `g_i · x_i / √(mean(x[:k]²) + ε)`, `k=⌈n·p⌉` | Partial RMS | RMS from first `p`% of dims via `prms_partial_ratio` (default 6.25%; arXiv:1910.07467 §5) |

## Positional Encodings

| Encoding | Code Name | Mechanism | Key Parameters |
|---|---|---|---|
| Rotary Position Embedding | `rope` | Rotates Q/K vectors by position-dependent angles | `rope_base` (default 10000), `rope_scaling` (default 1.0) |
| Hyperbolic Rotary PE | `hope` | Lorentz rotations with monotonic attention decay | `hope_base` (default 10000), `hope_damping` (default 0.01) |

Positional encoding is configured via `model.positional_encoding` and applies to `titan_attn` layers. The legacy `use_hope` flag is deprecated.

## MoE FFN Routing

When `use_moe=true`, standard FFN layers are replaced with Mixture of Experts routing. A learned router selects `top_k_experts` from `num_experts` specialized FFN sub-networks per token.

| Field | Type | Range | Description |
|---|---|---|---|
| `use_moe` | bool | — | Enable MoE in FFN layers |
| `num_experts` | int | ≥ 1 | Total expert count (typical: 4–8) |
| `top_k_experts` | int | ≥ 1 | Experts activated per token (typical: 1–2) |
| `ffn_hidden_size` | int | ≥ 1 | Hidden size per expert FFN |
| `ffn_activation` | enum | `silu`, `gelu` | FFN non-linearity |
