# Sequence Mixer Families Specification

> Cross-references: [Architecture](architecture.md) · [Schema Reference](schema-reference.md) · [Training Safety](training-safety.md)

## Taxonomy Overview

The system implements **20 sequence mixer architectures** organized into five functional categories, with two additional latent attention variants documented for reference. The taxonomy figure from the paper:

```
Sequence Mixer Registry (20 variants)
├── Dense (2): standard_attn, sigmoid_attn
├── GQA (1): gqa_attn
├── Recurrent (5): retnet/retnet_attn, mamba, ode, titan_attn, engram_attn
├── Sparse (7): sparse_transformer_attn, longformer_attn, bigbird_attn,
│                sparsek_attn, nsa_attn, sparge_attn, fasa_attn
├── Gated (6): gla_attn, deltanet_attn, gated_deltanet_attn,
│              hgrn2_attn, fox_attn, gated_softmax_attn
└── Latent (documented): cca_attn, ccgqa_attn
```

## Training-Free Policy

**`fasa_attn` and `sparge_attn` are eval/inference-only.** The implementation raises a runtime error if either is used while the model is in training mode. These methods require pretrained checkpoints from full-attention models or specific fine-tuning procedures.

## Dense Attention Baselines (2)

### `standard_attn` — Standard Softmax Attention

| Attribute | Value |
|---|---|
| Paper | Vaswani et al. (2017) — arXiv:1706.03762 |
| Core Equation | `Attn(Q,K,V) = softmax(QK^⊤/√d_k) V` |
| Training Complexity | O(n²·d) time, O(n²) space |
| Inference Complexity | O(n) per step, O(n·d) KV cache |
| Key Characteristics | Full global context, perfect historical recall, highly parallelizable training |
| Pros | Unparalleled expressiveness; induction heads for in-context learning; de facto baseline |
| Cons | Quadratic bottleneck prohibits very long contexts; KV cache dominates memory during generation |

### `sigmoid_attn` — Sigmoid Self-Attention

| Attribute | Value |
|---|---|
| Paper | Ramapuram et al. (2024) — arXiv:2409.04431 |
| Core Equation | `SigmoidAttn(Q,K,V) = σ(QK^⊤/√d_k + b) V` |
| Training Complexity | O(n²·d) (identical to standard theoretically) |
| Inference Complexity | O(n) per step, O(n·d) KV cache |
| Key Characteristics | Element-wise sigmoid replaces row-wise softmax; eliminates zero-sum token competition |
| Pros | 17% kernel speedup via FlashSigmoid; superior sample complexity (MoE analysis); hardware-friendly element-wise ops |
| Cons | Requires hybrid-norm stabilization at scale; training instability without auxiliary loss |

## Grouped-Query Attention (1)

### `gqa_attn` — Grouped-Query Attention

| Attribute | Value |
|---|---|
| Paper | Ainslie et al. (2023) — arXiv:2305.13245 |
| Core Equation | `GQA(Q,K,V) = softmax(Q K^⊤/√d_k) V` with `K,V ∈ ℝ^{B×G×T×d_k}` where G = number of KV head groups |
| Training Complexity | O(n²·d) quadratic |
| Inference Complexity | O(n) per step, O(n·d_kv·d) KV cache (d_kv = num_kv_heads × head_dim) |
| Key Characteristics | Num. KV heads is configurable between 1 (MQA) and num_heads (MHA); each KV head broadcasts to num_heads/num_kv_heads query heads |
| Pros | Quality close to MHA with speed almost matching MQA; reduces KV-cache memory by num_heads/num_kv_heads × ; can be uptrained from MHA checkpoints via mean-pooling |
| Cons | Still O(n²) quadratic; KV cache larger than pure MQA; optimal group count is model-size dependent |

## Recurrent and Retentive Architectures (5)

### `retnet` / `retnet_attn` — Retentive Network

| Attribute | Value |
|---|---|
| Paper | Sun et al. (2023) — arXiv:2307.08621 |
| Core Equation | `Retention(X) = (QK^⊤ ⊙ D) V` where D_nm = γ^(n−m); recurrent: S_n = γ S_{n−1} + k_n^⊤ v_n |
| Training Complexity | O(n²·d) parallel, or O(n·c·d) chunkwise |
| Inference Complexity | O(1) per step, O(d²) state |
| Key Characteristics | Triple computation paradigm (parallel/recurrent/chunkwise); multi-scale decay per head |
| Pros | Constant-time inference; no KV cache; solves the "impossible triangle" |
| Cons | Fixed decay imposes rigid inductive bias; may truncate long-range dependencies |

### `mamba` — Selective State Space Model

| Attribute | Value |
|---|---|
| Paper | Gu & Dao (2023) — arXiv:2312.00752 |
| Core Equation | h_t = Ā_t h_{t−1} + B̄_t x_t, y_t = C_t h_t (input-dependent Ā_t, B̄_t, C_t) |
| Training Complexity | O(n·d) via hardware-aware parallel scan |
| Inference Complexity | O(1) per step, O(d) state |
| Key Characteristics | Input-dependent selectivity; SRAM-fused parallel scan; linear training + constant inference |
| Pros | 5× inference throughput vs Transformers; scales to million-length sequences |
| Cons | State compression weakens exact copying and dense associative recall |

### `ode` — ODE-style Continuous Depth Block

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2021) — AAAI 2021 |
| Core Equation | dh(t)/dt = f_θ(h(t), t); discrete via RK4 or Euler integration |
| Training Complexity | O(k·n²·d) where k = RK order |
| Inference Complexity | O(k·n) per step |
| Key Characteristics | Depth as numerical integration; weight sharing across RK stages; learned coefficient gating |
| Pros | Higher accuracy via reduced truncation error; parameter-efficient via weight sharing |
| Cons | Higher per-step compute; inference latency ×k; complex gating for gradient stability |

### `titan_attn` — Titans Memory-Augmented Attention

| Attribute | Value |
|---|---|
| Paper | Behrouz et al. (2025) — arXiv:2501.00663 |
| Core Equation | M_t = (1−α_t)M_{t−1} + S_t; S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t) |
| Training Complexity | Approx. O(n·d) |
| Inference Complexity | Retrieval-centric, O(1) memory |
| Key Characteristics | Test-time neural memorization; surprise-driven gradient updates; bifurcated short/long-term memory |
| Pros | Handles >2M token contexts; near-perfect needle-in-haystack; true associative recall |
| Cons | Gradient-based weight updates during inference add systems complexity |

### `engram_attn` — Engram Conditional Memory

| Attribute | Value |
|---|---|
| Paper | Cheng et al. (2026) — arXiv:2601.07372 |
| Core Equation | Hash-based N-gram lookup with multi-head embedding retrieval + causal depthwise conv |
| Training Complexity | O(n·d) with hash table overhead |
| Inference Complexity | O(1) lookup per N-gram |
| Key Characteristics | Scalable N-gram conditional memory; independent hash heads per N-gram order; prime-modulus collision handling |
| Pros | New axis of sparsity; constant-time context retrieval; complementary to attention |
| Cons | Hash collisions; embedding table count grows quadratically with max N-gram size |

## Sparse Attention Patterns (7)

### Sparse Attention Design Strategies

```
Token sequence
├── Local / windowed (Longformer)
├── Hybrid sparse graph (BigBird / Sparse Transformer)
└── Selective pruning (SparseK / NSA / FASA / SpargeAttn)
```

### `sparse_transformer_attn` — Sparse Transformer

| Attribute | Value |
|---|---|
| Paper | Child et al. (2019) — arXiv:1904.10509 |
| Core Equation | Attn_i = softmax(q_i K_{A_i}^⊤/√d_k) V_{A_i} with factorized strided + fixed masks |
| Training Complexity | O(n√n·d) |
| Inference Complexity | O(n) per step |
| Key Characteristics | Factorized strided + fixed patterns; two complementary heads; O(n√n) complexity |
| Pros | Proven on images, audio, text; enables 10K+ token sequences |
| Cons | Fixed patterns may miss important long-range dependencies; requires custom CUDA kernels |

### `longformer_attn` — Longformer

| Attribute | Value |
|---|---|
| Paper | Beltagy et al. (2020) — arXiv:2004.05150 |
| Core Equation | A_i = {j : |i−j| ≤ w/2} ∪ G (sliding window + global tokens) |
| Training Complexity | O(n·w·d) linear in n for fixed w |
| Inference Complexity | O(n) per step |
| Key Characteristics | Sliding window + dilated window + global tokens; receptive field grows to L×w |
| Pros | Linear scaling; drop-in replacement; handles 4096+ tokens |
| Cons | Window size limits local context per layer; global tokens must be task-specifically chosen |

### `bigbird_attn` — BigBird

| Attribute | Value |
|---|---|
| Paper | Zaheer et al. (2020) — NeurIPS 2020 |
| Core Equation | A_i = A_i^window ∪ A_i^random ∪ A_i^global |
| Training Complexity | O(n) near-linear |
| Inference Complexity | O(n) per step |
| Key Characteristics | Random + local + global sparse graph; provably Turing complete and universal approximator |
| Pros | Theoretical completeness proof; handles 8× longer sequences than BERT |
| Cons | Random patterns introduce non-determinism; block-level granularity may waste computation |

### `sparsek_attn` — SparseK Attention

| Attribute | Value |
|---|---|
| Paper | Lou et al. (2024) — arXiv:2406.16747 |
| Core Equation | Differentiable top-k: SparseK(u,k)_j = max(u_j − τ(u), 0); attention on selected KV only |
| Training Complexity | O(n·d) linear |
| Inference Complexity | O(k) constant memory per step |
| Key Characteristics | Learned scoring network + differentiable top-k operator; end-to-end trainable |
| Pros | Seamless integration into existing LLMs; constant memory at generation |
| Cons | Scoring network adds overhead; fixed k may not be optimal for all layers |

### `nsa_attn` — Native Sparse Attention (NSA)

| Attribute | Value |
|---|---|
| Paper | Yuan et al. / DeepSeek (2025) — arXiv:2502.11089 |
| Core Equation | o_t = Σ_c g_t^c · Attn(q_t, K̃_t^c, Ṽ_t^c) for c ∈ {cmp, sel, win} |
| Training Complexity | O(t/d + n·l' + w) tokens per query |
| Inference Complexity | Reduced-token multi-branch |
| Key Characteristics | Three-branch design: compressed + selected + sliding window; learned gating; hardware-aligned |
| Pros | 11.6× decode speedup, 9× forward speedup at 64k; outperforms full attention on many benchmarks |
| Cons | Requires custom Triton kernels; complex multi-branch architecture |

### `sparge_attn` — SpargeAttn ⚠️ TRAINING-FREE

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2025) — arXiv:2502.18137 |
| Core Equation | Two-stage block filtering: prediction → softmax-aware pruning |
| Training Complexity | N/A (eval-only) |
| Inference Complexity | O(n²·s) where s = sparsity fraction |
| Key Characteristics | Universal training-free method; works on LLMs, image/video diffusion models |
| Pros | 2.5–5× speedup; compatible with quantization; plug-and-play |
| Cons | Speedup depends on inherent sparsity; block granularity; threshold tuning per model |

### `fasa_attn` — FASA (Frequency-Aware Sparse Attention) ⚠️ TRAINING-FREE

| Attribute | Value |
|---|---|
| Paper | Wang et al. (2026) — arXiv:2602.03152 |
| Core Equation | TIP: dominant RoPE FCs → token importance; FAC: full attention on selected tokens |
| Training Complexity | N/A (eval-only) |
| Inference Complexity | O(t·N_tip + N_fac·d) |
| Key Characteristics | Exploits RoPE frequency chunk sparsity; <1% of FCs are dominant; universal across model scales |
| Pros | Near-oracle accuracy with ≤256 tokens; 2.56× speedup; 8× KV cache compression |
| Cons | Requires RoPE-based models; offline calibration step needed |

## Gated Attention Mechanisms (6)

### Generic Gating Template

```
Previous state S_{t−1} → Gate(s) α_t, β_t, G_t, f_t → New key/value or SDPA output
                                                         ↓
                                              Updated memory/output S_t or O_t
```

### `gla_attn` — Gated Linear Attention

| Attribute | Value |
|---|---|
| Paper | Yang et al. (2023) — arXiv:2312.06635 |
| Core Equation | S_t = G_t ⊙ S_{t−1} + v_t k_t^⊤, o_t = S_t q_t |
| Training Complexity | O(Ld²) sub-quadratic chunkwise |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | Data-dependent diagonal gating on linear attention; low-rank gate projection; logsigmoid activation |
| Pros | Sub-quadratic training; constant-memory inference; good length generalization (2K→20K+) |
| Cons | Underperforms softmax on retrieval-heavy tasks; gate structure limits per-KV selectivity |

### `deltanet_attn` — DeltaNet

| Attribute | Value |
|---|---|
| Paper | Yang et al. (2024) — arXiv:2406.06484 |
| Core Equation | S_t = S_{t−1}(I − β_t k_t k_t^⊤) + β_t v_t k_t^⊤ |
| Training Complexity | O(Ld²) sub-quadratic |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | Delta learning rule for error-correcting state updates; WY representation for parallelism |
| Pros | Perfect MQAR associative recall; theoretically principled (online MSE optimization) |
| Cons | No global forgetting → memory crowding on long sequences; requires L2-normalized keys |

### `gated_deltanet_attn` — Gated DeltaNet

| Attribute | Value |
|---|---|
| Paper | Yang et al. (2024) — arXiv:2412.06464 (ICLR 2025) |
| Core Equation | S_t = α_t S_{t−1}(I − β_t k_t k_t^⊤) + β_t v_t k_t^⊤ |
| Training Complexity | O(Ld²) sub-quadratic |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | Unifies gating (rapid erasure via α_t) + delta rule (targeted updates via β_t); complementary gates |
| Pros | Best-in-class among pure linear recurrent models; integrated into Qwen3-Next/3.5 |
| Cons | Slightly lower throughput than Mamba2; two gates increase hyperparameter sensitivity |

### `hgrn2_attn` — HGRN2

| Attribute | Value |
|---|---|
| Paper | Qin et al. (2024) — arXiv:2404.07904 |
| Core Equation | S_t = diag(g_t) · S_{t−1} + v_t k_t^⊤, with hierarchically lower-bounded forget gates |
| Training Complexity | O(Ld²) sub-quadratic |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | Outer-product state expansion; hierarchical bounds enforce multi-scale temporal modeling |
| Pros | Efficient state expansion without extra params; competitive with Mamba at 3B scale |
| Cons | Vector-valued gating less flexible than delta-rule targeting; requires layer-wise bound tuning |

### `fox_attn` — Forgetting Transformer (FoX)

| Attribute | Value |
|---|---|
| Paper | Lin et al. (2025) — arXiv:2503.02130 (ICLR 2025) |
| Core Equation | O = softmax(QK^⊤/√d_k + D) V where D_ij = Σ_{l=j+1}^i log f_l |
| Training Complexity | O(n²·d) quadratic |
| Inference Complexity | O(n) per step, O(n·d) KV cache |
| Key Characteristics | Data-dependent forget gate in softmax logit space; FlashAttention-compatible; no positional embeddings needed |
| Pros | Full softmax expressiveness retained; superior length extrapolation; near-perfect needle-in-haystack |
| Cons | Still O(n²) quadratic; forget gate adds recency bias; model size ablations only up to 760M |

### `gated_softmax_attn` — Gated Softmax Attention

| Attribute | Value |
|---|---|
| Paper | Qiu et al. / Qwen Team (2025) — arXiv:2505.06708 (NeurIPS 2025 Best Paper) |
| Core Equation | Y' = SDPA(Q,K,V) ⊙ σ(X W_g) |
| Training Complexity | O(n²·d) quadratic |
| Inference Complexity | O(n) per step, O(n·d) KV cache |
| Key Characteristics | Post-SDPA sigmoid gate; headwise variant adds ~1.6M params for 15B model; sparse gates (mean ≈ 0.116) |
| Pros | Eliminates attention sink; improves training stability; <2% latency overhead; drop-in improvement |
| Cons | Still O(n²) quadratic; marginal benefit for short-context tasks |

## Latent Attention Mechanisms (2)

### `cca_attn` — Compressed Convolutional Attention

| Attribute | Value |
|---|---|
| Paper | Figliola et al. (2025) — arXiv:2510.04476 |
| Core Equation | q̃=W̃_Q x, k̃=W̃_K x, v=[W̃_V1 x_t ‖ W̃_V2 x_{t−1}]; dual causal convs; qk-mean; QK L2-norm + learnable β; RoPE in latent; Y=W̃_O·softmax(q̂k̂^⊤/√d_h)v |
| Training Complexity | O(n²·d/C) — C× fewer FLOPs than MHA |
| Inference Complexity | O(n) per step, O(ẽ·n) KV cache (ẽ = E/C) |
| Key Characteristics | Attention performed ENTIRELY in compressed latent (no Q/K/V up-projections, only W̃_O); dual causal convs (depthwise seq + grouped channel); query-key mean; value-shift; QK L2-norm + learnable temperature β; RoPE native in latent |
| Config Knobs | `compression_factor` (C ≥ 1), `cca_qk_mean` (bool), `cca_value_shift` (bool), `cca_learnable_temp` (bool), `cca_conv_kernel_size`, `cca_conv_groups` |
| Pros | Reduces params + KV cache + FLOPs simultaneously; native RoPE; >2× fewer params than MLA |
| Cons | Does not remove quadratic S² (divides by C); conv/qk-mean/v-shift add inductive bias; fused kernel needed |

### `ccgqa_attn` — Compressed Convolutional Grouped Query Attention

| Attribute | Value |
|---|---|
| Paper | Figliola et al. (2025) — arXiv:2510.04476 |
| Core Equation | CCA + GQA head-sharing in latent; decoupled Q compression C₁ and KV compression C₂ (C₂ ≥ C₁); grouped qk-mean via B_group (replicate KV→Q) and E_group (average Q→KV) |
| Training Complexity | O(n²·d/C₁) |
| Inference Complexity | O(n) per step, O(ẽ_kv·n) KV cache (ẽ_kv = E/C₂) |
| Key Characteristics | CCA extended with GQA-style head sharing inside the latent; decoupled query (C₁) and KV (C₂) compression; constraint C₂/C₁ = num_heads/num_kv_heads; grouped qk-mean with B_group/E_group mixing matrices; best reported loss at 8× cache reduction |
| Config Knobs | `query_compression` (C₁), `kv_compression` (C₂), `num_kv_heads` (must satisfy C₂/C₁ = num_heads/num_kv_heads), `cca_qk_mean`, `cca_value_shift`, `cca_learnable_temp`, `cca_conv_kernel_size`, `cca_conv_groups` |
| Pros | Decoupled compression; smooth Pareto; same arithmetic intensity as GQA; best loss at 8× cache reduction |
| Cons | Still quadratic in S²; fused kernel required; C₂/C₁ ratio constraint on num_heads/num_kv_heads |

## Comprehensive Comparison Table

| Mixer | Category | Train Complexity | Infer Complexity | Trainable | Key Strength |
|---|---|---|---|---|---|
| `standard_attn` | Dense | O(n²d) | O(n)/step | Yes | Full expressiveness baseline |
| `sigmoid_attn` | Dense | O(n²d) | O(n)/step | Yes | Element-wise gating, 17% kernel speedup |
| `gqa_attn` | GQA | O(n²d) | O(n)/step | Yes | Configurable KV heads, quality-speed tradeoff |
| `retnet` | Recurrent | O(n²d) / chunkwise | O(1)/step | Yes | Triple computation paradigm |
| `mamba` | Recurrent | O(nd) | O(1)/step | Yes | Linear training + constant inference |
| `ode` | Recurrent | O(k·n²d) | O(k·n)/step | Yes | Numerical integration refinement |
| `titan_attn` | Recurrent | ~O(nd) | O(1) memory | Yes | Test-time memory, >2M context |
| `engram_attn` | Memory | O(nd) | O(1) lookup | Yes | N-gram conditional memory |
| `sparse_transformer_attn` | Sparse | O(n√n·d) | O(n)/step | Yes | Factorized strided + fixed |
| `longformer_attn` | Sparse | O(n·w·d) | O(n)/step | Yes | Linear scaling with dilation |
| `bigbird_attn` | Sparse | O(n) | O(n)/step | Yes | Theoretical completeness proof |
| `sparsek_attn` | Sparse | O(nd) | O(k)/step | Yes | Differentiable top-k selection |
| `nsa_attn` | Sparse | Reduced-token | Reduced-token | Yes | Hardware-aligned 3-branch |
| `sparge_attn` | Sparse | N/A (eval-only) | O(n²·s) | **No** | Universal training-free filtering |
| `fasa_attn` | Sparse | N/A (eval-only) | O(t·N_tip+N_fac·d) | **No** | RoPE frequency insight |
| `gla_attn` | Gated | O(Ld²) | O(d²) memory | Yes | Data-dependent diagonal gating |
| `deltanet_attn` | Gated | O(Ld²) | O(d²) memory | Yes | Error-correcting delta rule |
| `gated_deltanet_attn` | Gated | O(Ld²) | O(d²) memory | Yes | Gating + delta rule synthesis |
| `hgrn2_attn` | Gated | O(Ld²) | O(d²) memory | Yes | Hierarchical forget gates |
| `fox_attn` | Gated | O(n²d) | O(n)/step | Yes | Forget gate in softmax logits |
| `gated_softmax_attn` | Gated | O(n²d) | O(n)/step | Yes | Post-SDPA sigmoid gating |
| `cca_attn` | Latent | O(n²d/C) | O(n)/step | Yes | Latent-space attention, C× FLOP reduction |
| `ccgqa_attn` | Latent | O(n²d/C₁) | O(n)/step | Yes | Decoupled latent compression + GQA head sharing |
