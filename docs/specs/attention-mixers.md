# Sequence Mixer Families Specification

> Cross-references: [Architecture](architecture.md) · [Schema Reference](schema-reference.md) · [Training Safety](training-safety.md)

## Taxonomy Overview

The system implements **44 sequence mixer architectures** (the full
`layer_pattern` enum) organized into seven functional categories. The taxonomy
figure from the paper:

```
Sequence Mixer Registry (44 variants)
├── Dense (2): standard_attn, sigmoid_attn
├── GQA (1): gqa_attn
├── Recurrent / Retentive (6): retnet/retnet_attn, mamba, ode, titan_attn,
│                               engram_attn
├── Sparse (9): sparse_transformer_attn, longformer_attn, bigbird_attn,
│               sparsek_attn, nsa_attn, sparge_attn, fasa_attn, msa_attn,
│               sparda_attn
├── Gated (8): gla_attn, deltanet_attn, gated_deltanet_attn,
│              gated_deltanet2_attn, hgrn2_attn, fox_attn, gated_softmax_attn,
│              kda_attn
├── Fast-Weight / Online Learning (6): falcon1_attn, falcon2_attn,
│              falcon3_attn, falcon1a_attn, falcon2a_attn, falcon3a_attn
├── Latent / KV-compression (10): mla_attn, gqla_attn, mlra_attn, tucker_attn,
│                                  iha_attn, gta_attn, mtla_attn, cca_attn,
│                                  ccgqa_attn, gma_attn
└── Geometric Field (1): ssog_attn
```

> The detailed per-mixer attribute tables below cover the most commonly used
> and best-documented families. The authoritative list of all 44 names is the
> `layer_pattern` enum in `src/schema/_model/_dims.yaml`.

### How to pick a mixer (plain English)

The core question is: **how much context do you need, and how much
memory/speed budget do you have?**

| Your situation | Suggested mixers |
|---|---|
| Standard-length text, max quality | `standard_attn`, `sigmoid_attn`, `gqa_attn` |
| Very long sequences, tight memory | `retnet`, `mamba`, `gla_attn`, `deltanet_attn` |
| Linear training + constant inference | `mamba`, `retnet`, `gated` family |
| >1M-token context / associative recall | `titan_attn`, `engram_attn` |
| Long-context but want softmax quality | `longformer_attn`, `bigbird_attn`, `nsa_attn`, `sparsek_attn` |
| Principled forgetting + length extrapolation | `falcon3a_attn` (windowed additive), `falcon1_attn` (best LM perplexity) |
| Shrink KV cache / params in latent space | `mla_attn`, `cca_attn`, `ccgqa_attn`, `gma_attn` |
| Vision / grid data, or small-data geometric prior | `ssog_attn` (encoder-only; needs a `grid_h × grid_w` row-major raster) |
| Inference-only speedup (no training) | `sparge_attn`, `fasa_attn` (eval-only) |

You can mix families freely within a `layer_pattern` — the dispatcher routes
each position independently.

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

## Recurrent and Retentive Architectures (6)

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

## Sparse Attention Patterns (9)

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

## Gated Attention Mechanisms (8)

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

## Fast-Weight Attention / Online Continual Learning (6)

The Falcon family (arXiv:2608.27763, "Fast Weight Attention for Continual
Learning", Zhang et al. 2026) treats the recurrent fast-weight state
transition as an **online continual-learning rule** under read-after-write
(RAW) autoregressive semantics: the local fast-memory example revealed at
step `t` is the **prefix-aligned** pair `(x_t, y_t) = (phi(k_{t-1}), v_t)` —
one step shifted relative to DeltaNet's same-step association. All six
variants share normalized first-order step sizes
`eta_t = beta_t / (statistic_t + lambda_t + eps)`, the positive-decay
renormalization `alpha_t = min(eta_t*lambda_t, 1 - eps_gamma)` (fp32
`log1p` chunk-local log-prefixes in the chunk-parallel kernels), the
boundary sentinels `x_1 = 0`, `eta_1 = 0`, `(alpha_1, gamma_1) = (0, 1)`,
QK-RMSNorm by default, optional causal short convolutions on the
projections, and a recurrent + chunk-parallel dual implementation
(gated by `falcon_chunk_size`, `null` = recurrent reference scan).
Config knobs live under `model.attention.falcon` (shared by all six).

### `falcon1_attn` — Falcon-1 (scalar NLMS regression)

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2026) — arXiv:2608.27763 |
| Core Equation | S_t = (1 - eta_t*lambda_t) S_{t-1} + eta_t x_t (v_t - S_{t-1}^T x_t)^T, o_t = S_t^T q_t |
| Step Size | eta_t = beta_t / (\|x_t\|^2 + lambda_t + eps) |
| Training Complexity | O(Ld²) chunk-parallel (WY triangular solve, Alg. 8) |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | Read-after-write prefix-aligned delta rule with NLMS normalization; with eps = 0, lambda = 0 and the unshifted assignment recovers DeltaNet |
| Config Knobs | `falcon_chunk_size`, `falcon_qk_norm`, `falcon_beta_mode`, `falcon_lambda_mode`, `falcon_beta`, `falcon_lambda`, `falcon_short_conv`, `falcon_conv_kernel`, `falcon_eps`, `falcon_eps_gamma` |
| Pros | Best LM perplexity of the paper (Falcon-1.3: FineEdu 17.10, beats Gated DeltaNet); principled forgetting tied to the local objective |
| Cons | Regression family extrapolates worse on arithmetic than the additive family |

### `falcon2_attn` — Falcon-2 (per-column NLMS regression)

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2026) — arXiv:2608.27763 |
| Core Equation | S_t = S_{t-1}(I - lambda_t Diag(eta_t)) + x_t (eta_t r_t)^T (per-value-channel step sizes) |
| Step Size | eta_{j,t} = beta_{j,t} / (\|x_t\|^2 + lambda_t + eps) |
| Training Complexity | O(Ld²) chunk-parallel (batched per-channel triangular solves, Alg. 1) |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | The ridge loss decomposes across value coordinates — d_v independent scalar-step updates sharing the NLMS normalizer; defined-but-unbenchmarked in the paper |
| Config Knobs | same shared `falcon.*` block; per-column plasticity gates |
| Pros | Finer-grained plasticity than Falcon-1 |
| Cons | Heavier chunk kernels (per-channel triangular systems); unbenchmarked |

### `falcon3_attn` — Falcon-3 (sliding-window minibatch regression)

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2026) — arXiv:2608.27763 |
| Core Equation | S_t = (1 - eta_t*lambda_t) S_{t-1} + (eta_t/B_t) sum_{j in I_t} x_j (v_j - S_{t-1}^T x_j)^T (all residuals at the pre-update state) |
| Step Size | eta_t = beta_t / (mu_t^(B) + lambda_t + eps), mu_t = lambda_max(X^T X)/B_t |
| Training Complexity | O(N·C·B) chunk-parallel (rank-B block forward substitution, Alg. 3) |
| Inference Complexity | O(d²) constant memory + O(B) FIFO tail |
| Key Characteristics | Bounded rehearsal: each pair is replayed in B consecutive window-averaged updates (invariant to B); S_t alone is not Markov — exact continuation needs the last B-1 pairs |
| Config Knobs | same shared `falcon.*` block + `falcon_window` (paper default 4) |
| Pros | Controlled rehearsal horizon; exact window statistics |
| Cons | Most complex chunk kernel; eigendecomposition per step in the recurrent scan |

### `falcon1a_attn` — Falcon-1A (scalar inner-product / additive)

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2026) — arXiv:2608.27763 |
| Core Equation | S_t = gamma_t S_{t-1} + eta_t x_t v_t^T, gamma_t = 1 - min(eta_t lambda_t, 1 - eps_gamma) |
| Step Size | eta_t = beta_t / (\|x_t\|^2 + lambda_t + eps) |
| Training Complexity | O(Ld²) chunk-parallel (decay-mask linear attention, Fig. 7B) |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | One gradient step on the negative inner-product objective; with lambda = 0 and the unshifted assignment it is standard linear attention / Mamba-2 accumulation — the shifted stream makes it the next-latent variant |
| Pros | Simple and stable; competitive LM quality (Falcon-1A.3) |
| Cons | Purely additive write cannot erase stale associations |

### `falcon2a_attn` — Falcon-2A (per-column inner-product)

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2026) — arXiv:2608.27763 |
| Core Equation | S_t = S_{t-1} Diag(gamma_t) + x_t (eta_t v_t)^T (per-channel decays) |
| Step Size | eta_{j,t} = beta_{j,t} / (\|x_t\|^2 + lambda_t + eps) |
| Training Complexity | O(Ld²) chunk-parallel (per-channel decay masks, Fig. 8B) |
| Inference Complexity | O(d²) constant memory |
| Key Characteristics | The per-column analogue of Falcon-1A; value channels carry independent decays |
| Pros | Channel-wise forgetting granularity |
| Cons | O(C²·d) chunk masks; unbenchmarked in the paper |

### `falcon3a_attn` — Falcon-3A (sliding-window inner-product)

| Attribute | Value |
|---|---|
| Paper | Zhang et al. (2026) — arXiv:2608.27763 |
| Core Equation | S_t = gamma_t S_{t-1} + eta_t Nbar_t^(B), Nbar = (1/B_t) sum_{j in I_t} x_j v_j^T |
| Step Size | eta_t = beta_t / (Ebar_t^(B) + lambda_t + eps), Ebar = window mean key energy |
| Training Complexity | O(N·C) chunk-parallel (window-banded decay mask M = D·Diag(eta)·A over the extended slice, Alg. 4) |
| Inference Complexity | O(d²) constant memory + O(B) FIFO tail |
| Key Characteristics | Bounded-rehearsal additive update; the paper's best length extrapolation (Falcon-3A.3: 87.2% mean accuracy on 33-48-digit addition vs 65.8% for a RoPE Transformer) |
| Config Knobs | same shared `falcon.*` block + `falcon_window` |
| Pros | Best-in-family length extrapolation; carries arithmetic |
| Cons | Window statistics add per-step overhead; worse perplexity than Falcon-1 |

## Latent Attention Mechanisms (10)

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

### `gma_attn` — Gaussian Mixture Attention

| Attribute | Value |
|---|---|
| Paper | Huang & Raza (2026) — arXiv:2606.18283 |
| Core Equation | Γ^Q, Γ^K = posterior responsibilities over K learned Gaussian components; Ṽ = (Γ^K)ᵀ V_X; Z = (Γ^K)ᵀ 1_N; O = Γ^Q Ṽ / (Γ^Q Z + ε); causal via prefix cumsums |
| Training Complexity | O(NKd_r + NKd_v) — linear in sequence length N for fixed K |
| Inference Complexity | O(K) per step, O(NK) activation storage |
| Key Characteristics | Replaces pairwise QKᵀ with probabilistic routing through K per-head Gaussian-mixture components (means μ, diagonal covariances σ²=softplus(ω)+ε_σ, priors π=softmax(α)); writes values into a K-slot latent memory via key responsibilities, reads via query responsibilities; implicit normalised affinity A^GMA = diag(d)⁻¹ Γ^Q (Γ^K)ᵀ never materialised; bidirectional + causal variants; fully differentiable GMM params (no EM loop) |
| Config Knobs | `num_components` (K ≥ 1), `routing_dim` (d_r, default head_dim), `epsilon` (read normalizer ε > 0), `sigma_eps` (covariance floor ε_σ > 0), `init_mean_std` (μ init std) |
| Pros | Fixed-K linear-time sequence mixing; probabilistic interpretable routing; non-negative low-rank affinity; bidirectional + causal |
| Cons | Quality depends on K and d_r; causal GMA trails optimised SDPA and state-space models on WikiText-103 in the paper; not a universal softmax-attention replacement |

## Geometric Field Attention (1)

### `ssog_attn` — SSOG (Separable Sum of Gaussians)

| Attribute | Value |
|---|---|
| Paper | Pisoni (2026) — https://www.pisoni.ai/posts/ssog/ |
| Core Equation | A(p,q) = softmax_q(τ⁻¹ · logsumexp_r(log λ_r + log N(p−q; μ_r, σ_r))); applied as two 1D softmaxed kernel passes per atom (rows × columns) + λ-mix; steering: μ←μ₀+s_μ·max_offset·tanh(W_μx), σ←σ₀·e^{s_σ tanh(W_σx)}, λ←softmax(log λ₀+s_λ tanh(W_λx)) |
| Training Complexity | O(R·N·√N·d) for R atoms — the N×N matrix never exists |
| Inference Complexity | O(R·N·√N·d) per forward (no KV cache semantics; encoder-only) |
| Key Characteristics | No Q/K projections (2d² instead of 4d² per layer); each head owns R Gaussian atoms over *relative position* (5 numbers per atom: μy, μx, σy, σx, λ); same shared field for every input; content only *steers* via zero-init probes behind cold-started softplus gates (softplus(−8) ≈ 3e-4); geometry is the positional encoding (no PE consumed); field lives on coordinates → zero-shot resolution transfer; `grid_h=1` degenerates to a 1D positional field for sequences |
| Config Knobs | `num_atoms` (R ≥ 1, default 4), `lookat` (bool, default true), `max_offset` (cells, default 4.0), `cold_init` (bool, default true), `sigma_floor` (default 0.25), `grid_h`/`grid_w` (default: derived from image/patch size; set explicitly for non-vision grids) |
| Pros | Near-linear cost by geometry alone; +17 pts over SDPA on small data (CIFAR-100, matched recipe); beats SDPA on ImageNet-1k at 12M params while 20% smaller / 30% cheaper; fully interpretable (each head is a few plottable blobs); free zero-shot resolution transfer |
| Cons | Encoder-only (no causal formulation); requires a fixed `grid_h × grid_w` raster (ViT needs `cls_token: false`; NLP needs an explicit 1×L grid); per-query steering kernels add memory in the steered path; language results unproven (content scoring is likely load-bearing there) |

**Constraints** (enforced at config load and runtime):
- `model.dims.mode` must be `encoder` (`ssog_attn` + decoder raises).
- Sequence length must equal `grid_h * grid_w` (clear error otherwise).
- With `frankenstein_vit`: `image.cls_token: false` (a prepended [CLS]
  token breaks the raster); use `pooling_mode: gap` for pooling.

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
| `falcon1_attn` | Fast-weight | O(Ld²) | O(d²) memory | Yes | Prefix-aligned NLMS delta rule; best LM ppl of the family |
| `falcon2_attn` | Fast-weight | O(Ld²) | O(d²) memory | Yes | Per-value-channel NLMS step sizes |
| `falcon3_attn` | Fast-weight | O(Ld²B) | O(d²) + O(B) tail | Yes | Sliding-window minibatch rehearsal |
| `falcon1a_attn` | Fast-weight | O(Ld²) | O(d²) memory | Yes | Shifted additive linear attention with ridge decay |
| `falcon2a_attn` | Fast-weight | O(Ld²) | O(d²) memory | Yes | Per-channel additive decays |
| `falcon3a_attn` | Fast-weight | O(Ld²) | O(d²) + O(B) tail | Yes | Windowed additive; best length extrapolation |
| `cca_attn` | Latent | O(n²d/C) | O(n)/step | Yes | Latent-space attention, C× FLOP reduction |
| `ccgqa_attn` | Latent | O(n²d/C₁) | O(n)/step | Yes | Decoupled latent compression + GQA head sharing |
| `gma_attn` | Latent | O(nKd_r+nKd_v) | O(K)/step | Yes | Probabilistic Gaussian-mixture routing, linear-time for fixed K |
| `ssog_attn` | Geometric field | O(R·n√n·d) | O(R·n√n·d)/forward | Yes | Separable Gaussian field; content steers, never scores |

## Config knobs and example assembly

Mixers are selected by name in `layer_pattern` and tuned by per-mixer config
keys that live under `model.attention.<mixer>`. Here is a small reference for
the most common knobs (see [Schema Reference](schema-reference.md) for the
full list):

| Mixer | Example config keys |
|---|---|
| `gqa_attn` | `num_kv_heads` |
| `retnet` / `retnet_attn` | `retention_heads`, `retnet_chunk_size` |
| `titan_attn` | `positional_encoding` (`rope`/`hope`), RoPE/HoPE base |
| `ode` | `ode_solver` (`rk4`/`euler`), `ode_steps` |
| `engram_attn` | `engram_max_ngram_size`, `engram_n_heads_per_ngram`, `engram_embed_dim_per_head`, `engram_kernel_size` |
| `sparsek_attn` | `sparsek_k` |
| `longformer_attn` | window size |
| `cca_attn` | `compression_factor`, `cca_qk_mean`, `cca_value_shift`, `cca_learnable_temp`, `cca_conv_kernel_size`, `cca_conv_groups` |
| `ccgqa_attn` | `query_compression`, `kv_compression`, `num_kv_heads`, CCA flags |
| `ssog_attn` | `num_atoms`, `lookat`, `max_offset`, `cold_init`, `sigma_floor`, `grid_h`, `grid_w` |
| `falcon{1,2,3,1a,2a,3a}_attn` | shared `falcon.*` block: `chunk_size`, `qk_norm`, `beta_mode`, `lambda_mode`, `beta`, `lambda`, `window`, `short_conv`, `conv_kernel`, `eps`, `eps_gamma` |

### Building a hybrid network

You can combine a dense baseline, a linear-recurrent mixer, and a sparse
mixer in one model. For a 4-layer encoder:

```yaml
model:
  dims:
    num_layers: 4
    hidden_size: 512
    num_heads: 8
    layer_pattern:
      - standard_attn   # full attention at the bottom
      - gla_attn        # linear gated attention for long-range
      - longformer_attn # windowed attention to keep memory low
      - standard_attn   # full attention near the head
  attention:
    gqa_attn:
      num_kv_heads: 4   # (only applies to gqa_attn layers)
```

Every position reads its own mixer class, so per-layer mixing is trivial.
