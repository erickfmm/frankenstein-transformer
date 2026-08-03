# Gated Attention Blocks in Transformers: A Literature Review

## Executive Summary

The standard softmax attention mechanism in Transformers, while powerful, suffers from quadratic complexity in sequence length and lacks explicit mechanisms for memory management. Over 2023–2025, a family of **gated attention** architectures has emerged to address these limitations by incorporating data-dependent gating into the attention computation. These mechanisms draw inspiration from forget gates in recurrent neural networks (LSTMs, GRUs) and apply them to linear attention, state-space models, or even softmax attention itself. This review covers eight key architectures: **Gated Linear Attention (GLA)**, **DeltaNet**, **Gated DeltaNet**, **RetNet**, **HGRN2**, **Forgetting Transformer (FoX)**, **Gated Attention (NeurIPS 2025)**, and **Kimi Delta Attention (KDA)**. For each, we provide a description, the mathematical formulation, advantages and limitations, and a PyTorch-compatible reference implementation. A summary comparison table concludes the review.

## 1. Gated Linear Attention (GLA)

### Description

Gated Linear Attention (GLA), proposed by Yang et al. (2023), augments vanilla linear attention with a data-dependent gating mechanism. Linear attention reformulates the standard attention as a recurrence with a matrix-valued hidden state \(\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top\), enabling constant-memory inference. However, this purely additive accumulation leads to "memory overload"—old associations can never be erased. GLA introduces a diagonal gating matrix \(\mathbf{G}_t\) to selectively decay historical information, significantly narrowing the performance gap with softmax attention.[1][2][3]

The hardware-efficient FLASHLINEARATTENTION algorithm achieves sub-quadratic training complexity via a chunk-wise block-parallel strategy that maximizes tensor core utilization. GLA Transformer performs competitively with LLaMA-architecture Transformers and Mamba on language modeling, and generalizes from 2K training length to over 20K without significant perplexity degradation.[2][4]

### Mathematical Formulation

The recurrent form of GLA is:

\[
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
\]

where \(\mathbf{G}_t \in \mathbb{R}^{d \times d}\) is parameterized with an outer-product structure \(\mathbf{G}_t = \mathbf{1} \boldsymbol{\alpha}_t^\top\), and \(\boldsymbol{\alpha}_t\) is computed via a low-rank projection followed by logsigmoid activation and normalization:[3][1]

\[
\boldsymbol{\alpha}_t = \frac{\log \sigma(\mathbf{W}_{gk} \mathbf{x}_t)}{c}
\]

where \(c\) is a normalizer (typically 16). In the parallel (chunkwise) form:[5]

\[
\mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}}_{[t]} \mathbf{S}_{[t]}^\top + \left(\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{\Gamma}_{[t]}\right) \mathbf{V}_{[t]}
\]

where \(\mathbf{\Gamma}\) is the decay-aware causal mask.

### Pros and Cons

| Pros | Cons |
|------|------|
| Sub-quadratic training via chunkwise parallelism | Underperforms softmax attention on retrieval-heavy tasks |
| Constant O(d²) memory at inference | Gate structure limits per-key-value selectivity |
| Good length generalization (2K → 20K+) | Still a gap vs. Transformers on long-context downstream |
| Higher throughput than Mamba at similar scale | Requires custom CUDA kernels for optimal speed |

### PyTorch Implementation

A simplified recurrent-form reference implementation is provided in the attached code file (`gated_attention_implementations.py`, class `GatedLinearAttention`). The production-grade implementation is available in the `flash-linear-attention` library.[6]



***

## 2. DeltaNet

### Description

DeltaNet applies the classical **delta learning rule**—an error-correction principle—to linear attention. Instead of merely accumulating key-value outer products, DeltaNet updates its state by computing the difference between the predicted value (retrieved from memory via the current key) and the target value, then correcting accordingly. This is mathematically equivalent to performing a single step of stochastic gradient descent on an online MSE loss at each timestep, connecting DeltaNet to the test-time training (TTT) paradigm.[7][3]

DeltaNet demonstrates **perfect performance on Multi-Query Associative Recall (MQAR)**, a critical synthetic benchmark, and outperforms gated linear models on in-context retrieval tasks. However, without a global forgetting mechanism, it struggles with memory saturation in real-world settings with diverse contexts.[8][3]

### Mathematical Formulation

The DeltaNet state update (delta rule) is:[3]

\[
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
\]

where \(\beta_t \in (0,1)\) is a data-dependent "writing strength" (sigmoid of a linear projection), and keys are L2-normalized. This can be decomposed as an erase-write operation:[3]

\[
\mathbf{v}_t^{\text{new}} = (1 - \beta_t)\mathbf{v}_t^{\text{old}} + \beta_t \mathbf{v}_t, \quad \mathbf{S}_t = \mathbf{S}_{t-1} - \mathbf{v}_t^{\text{old}} \mathbf{k}_t^\top + \mathbf{v}_t^{\text{new}} \mathbf{k}_t^\top
\]

The connection to online learning: DeltaNet minimizes \(\mathcal{L}_t(\mathbf{S}) = \frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2\) via gradient descent with learning rate \(\beta_t\)[3].

The transition matrix \(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\) is a generalized Householder matrix, and the chunkwise training algorithm uses the **WY representation** to parallelize computation.[7]

### Pros and Cons

| Pros | Cons |
|------|------|
| Superior in-context associative recall (perfect MQAR) | No global forgetting → memory crowding on long real-world sequences |
| Theoretically principled (online MSE optimization) | Slightly slower than Mamba2 due to richer transitions |
| Strong state-tracking capabilities | Moderate real-world retrieval performance |
| Efficient chunkwise training via WY representation | Requires L2-normalized keys for stability |

***

## 3. Gated DeltaNet

### Description

Gated DeltaNet (Yang et al., 2024) is a natural synthesis of the gating mechanism from Mamba2/GLA and the delta rule from DeltaNet. The key insight is that these two mechanisms are **complementary**: gating enables rapid memory erasure (setting \(\alpha_t \to 0\) clears the state), while the delta rule enables targeted, per-key-value updates (\(\alpha_t \to 1\) recovers pure delta rule). Published at ICLR 2025, Gated DeltaNet consistently outperforms both Mamba2 and DeltaNet across language modeling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. It has been integrated into Alibaba's Qwen3-Next and Qwen3.5 models.[9][10][8]

### Mathematical Formulation

The **gated delta rule** state update is:[8]

\[
\mathbf{S}_t = \mathbf{S}_{t-1} \left(\alpha_t (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)\right) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
\]

where \(\alpha_t \in (0,1)\) is the decay gate and \(\beta_t \in (0,1)\) is the writing strength. From the online learning perspective, this corresponds to the objective:[8]

\[
\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \mathbf{k}_t, \beta_t(\mathbf{v}_t - \alpha_t \mathbf{S}_{t-1} \mathbf{k}_t)\rangle
\]

This introduces an adaptive weight decay \(\alpha_t\) into the SGD-like update, analogous to decoupled weight decay in deep learning optimization.

The hardware-efficient chunkwise algorithm extends DeltaNet's WY representation with per-chunk cumulative decay products \(\gamma_j^{[t]} = \prod_{i=tC+1}^{tC+j} \alpha_i\), enabling matmul-rich parallelism on tensor cores.[8]

### Pros and Cons

| Pros | Cons |
|------|------|
| Best-in-class among pure linear recurrent models | Slightly lower throughput than Mamba2 (richer transition matrices) |
| Complementary gating + delta rule for memory management | Still bounded by fixed state size for exact retrieval |
| Hardware-efficient chunkwise training | Requires custom triton/CUDA kernels for production |
| Hybrid variants (with SWA) outperform Transformers | Two gates (α, β) increase hyperparameter sensitivity |

***

## 4. RetNet (Retentive Network)

### Description

RetNet, proposed by Sun et al. (2023) at Microsoft Research, introduces a **retention mechanism** that unifies the benefits of recurrence (efficient inference) with attention (parallel training). The key innovation is replacing softmax attention with a position-dependent exponential decay, enabling three computation paradigms: parallel, recurrent, and chunkwise recurrent. RetNet uses **multi-scale retention** where each head has a different decay rate \(\gamma_h\), allowing different heads to capture dependencies at different time scales.[11][12]

RetNet achieves O(1) inference memory and latency, comparable performance to Transformers on language modeling, and is compatible with fully parallelizable training.[11]

### Mathematical Formulation

The retention mechanism in **parallel form** is:[11]

\[
\mathbf{O} = (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{D}) \mathbf{V}
\]

where the decay matrix \(D_{nm} = \gamma^{n-m}\) for \(n \geq m\), and 0 otherwise. In **recurrent form**:[12]

\[
\mathbf{S}_t = \gamma \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
\]

For multi-scale retention, each head \(h\) uses a different \(\gamma_h\), and positional information is encoded via xPos-style complex exponential embeddings. The **chunkwise recurrent** form processes chunks in parallel while propagating states recurrently across chunks.[11]

### Pros and Cons

| Pros | Cons |
|------|------|
| Three computation paradigms (parallel/recurrent/chunkwise) | Fixed decay rate (not data-dependent) limits adaptivity |
| O(1) inference memory; no KV cache needed | Underperforms data-dependent gating models (GLA, Mamba) |
| No softmax → simpler computation | Limited expressiveness for retrieval tasks |
| Compatible with parallel training | Complex positional encoding scheme |

***

## 5. HGRN2 (Hierarchically Gated Recurrent Network 2)

### Description

HGRN2, by Qin et al. (2024), extends HGRN by introducing an **outer-product-based state expansion** mechanism that significantly enlarges the recurrent state size without adding parameters. While HGRN uses a vector-valued hidden state with hierarchical forget gates (lower bounds increase for upper layers), HGRN2 adopts a matrix-valued state via key-value outer products, providing a **linear attention interpretation** that enables hardware-efficient training. HGRN2 at 3B scale slightly outperforms Mamba and LLaMA-architecture Transformers on language modeling.[13][14]

### Mathematical Formulation

The HGRN2 state update is:[13]

\[
\mathbf{S}_t = \text{diag}(\mathbf{g}_t) \cdot \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
\]

where \(\mathbf{g}_t \in \mathbb{R}^d\) is a diagonal forget gate with **hierarchically lower-bounded values**: the lower bound increases monotonically in upper layers, ensuring upper layers capture long-range dependencies while lower layers model local patterns. The forget gate is computed as:[15]

\[
\mathbf{g}_t = b + (1 - b) \cdot \sigma(\mathbf{W}_g \mathbf{x}_t)
\]

where \(b\) is the layer-dependent lower bound.

### Pros and Cons

| Pros | Cons |
|------|------|
| Efficient state expansion via outer products (no extra params) | Vector-valued gating less flexible than delta-rule targeting |
| Hierarchical bounds enforce multi-scale temporal modeling | Less effective on recall tasks than DeltaNet variants |
| Linear attention interpretation → efficient chunkwise training | Performance gap vs. Transformers persists on long-context |
| Competitive with Mamba at 3B scale | Requires careful layer-wise bound tuning |

***

## 6. Forgetting Transformer (FoX)

### Description

The Forgetting Transformer (FoX), by Lin et al. (2025), takes a different approach from the linear-recurrent models above: it embeds a **forget gate directly into softmax attention** by adding a data-dependent logit bias to the attention scores. This preserves the full expressiveness and quadratic-time computation of softmax attention while gaining the benefits of recency control. FoX outperforms Transformers on long-context language modeling, length extrapolation, and short-context tasks, while maintaining near-perfect needle-in-haystack retrieval. It is compatible with FlashAttention and requires no positional embeddings.[16][17][18]

### Mathematical Formulation

At each timestep, a scalar forget gate is computed:[19]

\[
f_t = \sigma(\mathbf{w}_f^\top \mathbf{x}_t + b_f)
\]

The **Forgetting Attention** output is:[18]

\[
\mathbf{o}_i = \frac{\sum_{j=1}^{i} \exp(\mathbf{q}_i^\top \mathbf{k}_j + d_{ij}) \mathbf{v}_j}{\sum_{j=1}^{i} \exp(\mathbf{q}_i^\top \mathbf{k}_j + d_{ij})}
\]

where \(d_{ij} = \sum_{l=j+1}^{i} \log f_l\). In matrix form:

\[
\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top + \mathbf{D}) \mathbf{V}, \quad \text{where } \mathbf{D} = \log \mathbf{F}
\]

This is equivalent to a **data-dependent, learnable version of ALiBi**. The implementation adds cumulative log-forget-gate sums to FlashAttention's attention logit computation, with negligible overhead.[20]

The **FoX (Pro)** variant adds output gates, QK-norm, output normalization, and data-dependent KV-shift.[18]

### Pros and Cons

| Pros | Cons |
|------|------|
| Full softmax expressiveness retained | Still O(L²) time and memory (quadratic) |
| FlashAttention-compatible (minimal overhead) | Forget gate adds recency bias that may hurt global retrieval |
| No positional embeddings needed | Model size ablations only up to 760M parameters |
| Superior length extrapolation vs. Transformer | Not as memory-efficient at inference as linear models |

***

## 7. Gated Attention (Sigmoid after SDPA)

### Description

The NeurIPS 2025 Best Paper by the Alibaba Qwen team systematically investigates gating-augmented softmax attention through over 30 variants of 15B MoE and 1.7B dense models trained on 3.5T tokens. Their central finding is that applying a **head-specific sigmoid gate after SDPA** consistently improves performance, training stability, and scaling properties. This simple modification introduces **non-linearity** (breaking the low-rank bottleneck between \(W_V\) and \(W_O\)) and **query-dependent sparsity** that eliminates the attention sink phenomenon. Already integrated into Qwen3-Next.[21][22][23]

### Mathematical Formulation

The gating mechanism is formalized as:[23]

\[
\mathbf{Y}' = \mathbf{Y} \odot \sigma(\mathbf{X}\mathbf{W}_\theta)
\]

where \(\mathbf{Y}\) is the SDPA output and \(\sigma\) is the sigmoid function. With the full attention pipeline:

\[
\mathbf{O} = \left[\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}\right] \odot \sigma(\mathbf{X}\mathbf{W}_g)
\]

The gate score shape is \(\mathbb{R}^{n \times q \times d_k}\) for elementwise gating (where \(q\) is the number of query heads), or \(\mathbb{R}^{n \times q}\) for headwise gating (only 1.6M extra parameters for a 15B MoE model). Key insight: effective gates are **sparse** (mean score ≈ 0.116), acting as query-dependent filters.[23]

### Pros and Cons

| Pros | Cons |
|------|------|
| Extremely simple modification (< 2% latency overhead) | Still O(L²) quadratic complexity |
| Eliminates attention sink phenomenon | Requires softmax attention (not applicable to linear variants) |
| Improves training stability; enables larger learning rates | Marginal benefit for short-context tasks |
| Headwise variant adds only ~1.6M params for 15B model | Effect partially overlaps with existing normalization tricks |

***

## 8. Kimi Delta Attention (KDA)

### Description

KDA is the core attention module of Kimi Linear (Kimi Team, 2025, arXiv:2510.26692). It extends Gated DeltaNet with a finer-grained gating mechanism: it replaces Gated DeltaNet's scalar per-head write gate `beta_t` with a **channel-wise** decay `alpha_t` (per key channel) computed via a log-decay parameterisation `g = a - softplus(delta)` and applied as a diagonal matrix `D_t = diag(alpha_t)` before the delta-rule update. The delta rule keeps a scalar write gate `beta_t` controlling the erase-write strength along the key axis. A bespoke chunkwise algorithm uses a specialised Diagonal-Plus-Low-Rank (DPLR) transition matrix variant for hardware efficiency, substantially reducing computation versus the general DPLR formulation while remaining consistent with the classical delta rule. KDA reduces to Gated DeltaNet when `alpha_t` collapses to a scalar per head. Kimi Linear — 3B activated / 48B total parameters, a layerwise hybrid of KDA and Multi-Head Latent Attention (MLA) — outperforms full MLA across all evaluated tasks (short-context, long-context, and RL scaling), while reducing KV cache usage by up to 75% and achieving up to 6x decoding throughput for a 1M context. Cite as [arXiv:2510.26692](https://arxiv.org/abs/2510.26692).

### Mathematical Formulation

The state update is

```
S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t v_t k_t^T
o_t = S_t^T q_t
```

where `D_t = diag(alpha_t)`, `alpha_t = exp(a - softplus(delta))` (computed in fp32 to avoid precision loss over long cumulative products), `beta_t = sigmoid(W_beta x_t)` is the scalar per-head write gate, queries and keys are L2-normalised, and values are silu-activated. The chunkwise training algorithm extends DeltaNet's WY representation by absorbing the channel-wise decay into an asymmetric delta recurrence on a decay-normalised state, enabling matmul-rich parallelism on tensor cores. Because `beta_t` is per-channel diagonal varying per row, the scalar post-scaling shortcut of KDA breaks: the backward pass requires a gate-aware WY inversion.

### Pros and Cons

| Pros | Cons |
|------|------|
| Channel-wise decay gives finer-grained memory control than Gated DeltaNet | Requires a new gate-aware WY Triton backward kernel |
| First linear attention to outperform full attention under fair comparisons | Slightly slower than Mamba-2 per-token due to richer transitions |
| Up to 75% KV cache reduction; up to 6x decode throughput at 1M context | Two parameter families (alpha, beta) increase hyperparameter sensitivity |
| Strong multi-key needle-in-a-haystack (RULER) recall | DPLR chunkwise algorithm adds engineering complexity |

***

## Summary Comparison Table

| Architecture | Year | Venue | Complexity (Training) | Complexity (Inference per step) | Memory (Inference) | Key Innovation | Reference |
|---|---|---|---|---|---|---|---|
| **GLA** | 2023 | ICML 2024 | O(Ld²) sub-quadratic | O(d²) | O(d²) | Data-dependent diagonal gating on linear attention; hardware-efficient chunkwise training | [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)[2] |
| **DeltaNet** | 2024 | NeurIPS 2024 | O(Ld²) sub-quadratic | O(d²) | O(d²) | Delta learning rule for error-correcting state updates; WY representation for parallelism | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)[7] |
| **Gated DeltaNet** | 2024 | ICLR 2025 | O(Ld²) sub-quadratic | O(d²) | O(d²) | Unifies gating (rapid erasure) + delta rule (targeted updates); gated delta rule | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)[8] |
| **RetNet** | 2023 | — | O(Ld²) sub-quadratic | O(d) | O(d²) | Multi-scale retention with exponential decay; parallel/recurrent/chunkwise triple paradigm | [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)[11] |
| **HGRN2** | 2024 | COLM 2024 | O(Ld²) sub-quadratic | O(d²) | O(d²) | Outer-product state expansion; hierarchical lower-bounded forget gates | [arXiv:2404.07904](https://arxiv.org/abs/2404.07904)[13] |
| **FoX** | 2025 | ICLR 2025 | O(L²d) quadratic | O(Ld) | O(L) KV cache | Data-dependent forget gate in softmax attention (logit bias); FlashAttention-compatible | [arXiv:2503.02130](https://arxiv.org/abs/2503.02130)[18] |
| **Gated Attention** | 2025 | NeurIPS 2025 | O(L²d) quadratic | O(Ld) | O(L) KV cache | Sigmoid gate after SDPA; non-linearity + sparsity; attention-sink-free | [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)[24] |
| **KDA** | 2025 | — | O(Ld²) sub-quadratic | O(d²) | O(d²) | Channel-wise decay + scalar write gate; DPLR chunkwise; first linear attn to beat full attention | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) |

### Additional Comparison Dimensions

| Architecture | State Type | Gate Type | Forgetting Mechanism | Positional Encoding | In-Context Recall Strength |
|---|---|---|---|---|---|
| **GLA** | Matrix \(d \times d\) | Diagonal data-dependent | Multiplicative decay | Learnable / none | Moderate |
| **DeltaNet** | Matrix \(d \times d\) | Scalar β per head | Erase-write (per-key) | None (via L2 norm) | Strong (perfect MQAR) |
| **Gated DeltaNet** | Matrix \(d \times d\) | Scalar α (decay) + β (write) | Global decay + per-key erase | None (via L2 norm) | Very strong |
| **RetNet** | Matrix \(d \times d\) | Fixed scalar per head | Exponential decay (data-independent) | xPos / complex exponential | Weak |
| **HGRN2** | Matrix \(d \times d\) | Diagonal with lower bounds | Hierarchical decay | None | Moderate |
| **FoX** | Full attention (no fixed state) | Scalar per head per token | Logit-space decay on softmax | Not needed | Very strong (near-perfect) |
| **Gated Attention** | Full attention (no fixed state) | Element/head-wise sigmoid | Sparse gating of SDPA output | RoPE (standard) | Strong (attention-sink-free) |
| **KDA** | Matrix \(d \times d\) | Channel-wise diagonal decay + scalar β write | Channel-wise erase + scalar write | None (via L2 norm) | Very strong (RULER multi-key) |

## PyTorch Reference Implementations

Simplified, pedagogical PyTorch (`nn.Module`) implementations of all eight architectures are provided in the attached code file. These use the recurrent form for clarity (production systems use chunkwise parallel algorithms). For optimized training kernels, refer to:

- **flash-linear-attention** (`fla-org/flash-linear-attention`): GLA, DeltaNet, Gated DeltaNet, RetNet, HGRN2[6]
- **Gated DeltaNet official**: `NVlabs/GatedDeltaNet`[10]
- **Forgetting Transformer**: `zhixuan-lin/forgetting-transformer`[25]



## Taxonomy and Evolution

The progression of gated attention blocks can be understood through three "generations" of state update rules:[26]

1. **Generation 1 — Fixed/Simple Gating**: RetNet (fixed decay per head), RWKV (data-independent). Limited adaptivity but simple.
2. **Generation 2 — Data-Dependent Gating**: GLA, Mamba2, HGRN2 (data-dependent diagonal gates). Better memory management but purely additive updates.
3. **Generation 3 — Delta Rule + Gating**: DeltaNet, Gated DeltaNet, **KDA** (targeted erase-write with optional gating and channel-wise decay). Best recall and memory management but more complex transitions.

Orthogonally, **FoX** and **Gated Attention** represent a parallel direction: rather than replacing softmax with linear recurrence, they enhance softmax attention with gating mechanisms, preserving its quadratic expressiveness while gaining forgetting/sparsity benefits.[16][21]

The field is converging toward **hybrid architectures** that interleave linear recurrent layers (e.g., Gated DeltaNet) with sparse softmax attention layers, achieving the best of both worlds. Architectures like Gated DeltaNet-H1 (Gated DeltaNet + sliding window attention) consistently outperform both pure recurrent and pure attention models.[26][8]

```python

# ==========================================
# Simplified PyTorch Reference Implementations
# of Gated Attention Blocks in Transformers
# ==========================================
# These are pedagogical implementations meant for
# understanding. For production use, refer to the
# flash-linear-attention library (fla-org/flash-linear-attention).

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================
# 1. GLA: Gated Linear Attention
# =========================================
class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention (Yang et al., 2023)
    arXiv:2312.06635
    Recurrent form: S_t = G_t * S_{t-1} + v_t k_t^T
    where G_t is a data-dependent diagonal gate matrix.
    """
    def __init__(self, d_model, num_heads, head_dim=None, gate_low_rank=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.key_dim = self.head_dim * num_heads
        self.value_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)  # output gate
        self.gk_proj = nn.Sequential(
            nn.Linear(d_model, gate_low_rank, bias=False),
            nn.Linear(gate_low_rank, self.key_dim, bias=True)
        )  # key gate (low-rank)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, self.value_dim)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        gk = F.logsigmoid(self.gk_proj(x)).view(B, L, self.num_heads, self.head_dim) / 16.0

        # Recurrent computation (for clarity; chunk-wise is faster)
        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            gate = torch.exp(gk[:, t])  # [B, H, dk]
            # Decay: S = diag(gate) @ S + v_t @ k_t^T
            S = S * gate.unsqueeze(-1) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
            o_t = (S * q[:, t].unsqueeze(-2)).sum(-1)  # S @ q_t
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)  # [B, L, H, dv]
        o = self.group_norm(o.reshape(B * L, -1)).reshape(B, L, -1)
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)


# =========================================
# 2. DeltaNet: Linear Attention with Delta Rule
# =========================================
class DeltaNet(nn.Module):
    """
    DeltaNet (Schlag et al., 2021; Yang et al., 2024)
    arXiv:2406.06484
    S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    """
    def __init__(self, d_model, num_heads, head_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        total_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        self.g_proj = nn.Linear(d_model, total_dim, bias=False)  # output gate
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, total_dim)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.beta_proj(x))  # [B, L, H]

        # L2-normalize q, k for stability
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            b_t = beta[:, t, :, None, None]  # [B, H, 1, 1]
            k_t = k[:, t]  # [B, H, dk]
            v_t = v[:, t]
            kk = k_t.unsqueeze(-1) * k_t.unsqueeze(-2)  # k_t k_t^T
            # Delta rule: S = S(I - beta*kk) + beta*v*k^T
            S = S * (1 - b_t * kk) + b_t * (v_t.unsqueeze(-1) * k_t.unsqueeze(-2))
            o_t = (S * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)
        o = self.group_norm(o.reshape(B * L, -1)).reshape(B, L, -1)
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)


# =========================================
# 3. Gated DeltaNet
# =========================================
class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet (Yang et al., 2024)
    arXiv:2412.06464 (ICLR 2025)
    S_t = S_{t-1} * alpha_t * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    Combines gating (alpha) with delta rule (beta).
    """
    def __init__(self, d_model, num_heads, head_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        total_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.alpha_proj = nn.Linear(d_model, num_heads, bias=True)  # decay gate
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)   # write gate
        self.g_proj = nn.Linear(d_model, total_dim, bias=False)
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, total_dim)

    def forward(self, x):
        B, L, _ = x.shape
        q = F.normalize(self.q_proj(x).view(B, L, self.num_heads, self.head_dim), dim=-1)
        k = F.normalize(self.k_proj(x).view(B, L, self.num_heads, self.head_dim), dim=-1)
        v = F.silu(self.v_proj(x)).view(B, L, self.num_heads, self.head_dim)
        alpha = torch.sigmoid(self.alpha_proj(x))  # [B, L, H]
        beta = torch.sigmoid(self.beta_proj(x))     # [B, L, H]

        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            a_t = alpha[:, t, :, None, None]
            b_t = beta[:, t, :, None, None]
            k_t = k[:, t]
            v_t = v[:, t]
            kk = k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            # Gated delta rule
            S = a_t * S * (1 - b_t * kk) + b_t * (v_t.unsqueeze(-1) * k_t.unsqueeze(-2))
            o_t = (S * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)
        o = self.group_norm(o.reshape(B * L, -1)).reshape(B, L, -1)
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)


# =========================================
# 4. RetNet: Retentive Network (Retention)
# =========================================
class MultiScaleRetention(nn.Module):
    """
    RetNet Retention Mechanism (Sun et al., 2023)
    arXiv:2307.08621
    Parallel: O = (QK^T ⊙ D)V where D_nm = gamma^(n-m)
    Recurrent: S_t = gamma * S_{t-1} + v_t k_t^T
    """
    def __init__(self, d_model, num_heads, head_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        total_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.g_proj = nn.Linear(d_model, total_dim, bias=False)
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, total_dim)

        # Multi-scale: each head has different decay gamma
        gammas = 1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), num_heads))
        self.register_buffer('gammas', gammas)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Build decay matrix D: D[i,j] = gamma^(i-j) for i >= j, 0 otherwise
        pos = torch.arange(L, device=x.device).float()
        decay = pos.unsqueeze(0) - pos.unsqueeze(1)  # [L, L]
        mask = (decay >= 0).float()
        # Per-head decay
        D = self.gammas.view(1, 1, -1).pow(decay.unsqueeze(-1).abs()) * mask.unsqueeze(-1)
        # D: [L, L, H]

        # Parallel form: O = (Q K^T ⊙ D) V
        # Compute per head
        q = q.permute(0, 2, 1, 3)  # [B, H, L, dk]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        D = D.permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]

        attn = torch.matmul(q, k.transpose(-1, -2)) * D
        o = torch.matmul(attn, v)  # [B, H, L, dv]
        o = o.permute(0, 2, 1, 3).reshape(B * L, -1)
        o = self.group_norm(o).reshape(B, L, -1)
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)


# =========================================
# 5. HGRN2: Hierarchically Gated Recurrent Network 2
# =========================================
class HGRN2(nn.Module):
    """
    HGRN2 (Qin et al., 2024)
    arXiv:2404.07904 (COLM 2024)
    Uses outer-product state expansion with hierarchical forget gates.
    S_t = diag(g_t) @ S_{t-1} + v_t k_t^T
    where g_t has lower-bounded forget gates.
    """
    def __init__(self, d_model, num_heads, head_dim=None, lower_bound=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        total_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.g_proj = nn.Linear(d_model, total_dim, bias=False)
        self.forget_proj = nn.Linear(d_model, total_dim, bias=True)
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, total_dim)
        self.lower_bound = lower_bound

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        fg = torch.sigmoid(self.forget_proj(x)).view(B, L, self.num_heads, self.head_dim)
        fg = self.lower_bound + (1 - self.lower_bound) * fg

        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            gate = fg[:, t]  # [B, H, dk]
            # State expansion via outer product with gating
            S = S * gate.unsqueeze(-1) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
            o_t = (S * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)
        o = self.group_norm(o.reshape(B * L, -1)).reshape(B, L, -1)
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)


# =========================================
# 6. FoX: Forgetting Transformer
# =========================================
class ForgettingAttention(nn.Module):
    """
    Forgetting Transformer / FoX (Lin et al., 2025)
    arXiv:2503.02130 (ICLR 2025)
    O = softmax(QK^T + D) V
    where D_ij = sum_{l=j+1}^{i} log(f_l), f_t = sigmoid(w_f^T x_t + b_f)
    """
    def __init__(self, d_model, num_heads, head_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        total_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.f_proj = nn.Linear(d_model, num_heads, bias=True)  # forget gate
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        f = torch.sigmoid(self.f_proj(x))  # [B, L, H]
        log_f = torch.log(f + 1e-6).permute(0, 2, 1)  # [B, H, L]
        cum_log_f = torch.cumsum(log_f, dim=-1)  # [B, H, L]

        # D_ij = cum_log_f_i - cum_log_f_j (for i >= j)
        D = cum_log_f.unsqueeze(-1) - cum_log_f.unsqueeze(-2)  # [B, H, L, L]
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        D = D.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = (q @ k.transpose(-1, -2)) * self.scale + D
        attn = F.softmax(attn, dim=-1)
        o = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)


# =========================================
# 7. Gated Attention (Sigmoid after SDPA)
# =========================================
class GatedSoftmaxAttention(nn.Module):
    """
    Gated Attention for LLMs (Qiu et al., 2025)
    arXiv:2505.06708 (NeurIPS 2025 Best Paper)
    Standard softmax attention + sigmoid gate after SDPA output:
    Y' = Attention(Q,K,V) ⊙ sigmoid(X W_gate)
    """
    def __init__(self, d_model, num_heads, head_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        total_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, total_dim, bias=False)  # head-specific gate
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Standard causal softmax attention
        attn = (q @ k.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        sdpa_out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, -1)

        # Head-specific sigmoid gate after SDPA
        gate = torch.sigmoid(self.gate_proj(x))
        gated_out = sdpa_out * gate

        return self.o_proj(gated_out)


# =========================================
# 8. KDA: Kimi Delta Attention
# =========================================
class KDA(nn.Module):
    """
    Kimi Delta Attention (Kimi Team, 2025)
    arXiv:2510.26692
    S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t v_t k_t^T
    where D_t = diag(alpha_t), alpha_t = exp(a - softplus(delta)).
    Channel-wise decay (KDA-style) + scalar write gate (Gated DeltaNet-style).
    """
    def __init__(self, d_model, num_heads, head_dim=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        total_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(d_model, total_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)  # scalar write gate
        # Channel-wise log-decay parameters (KDA-style)
        self.log_decay_base = nn.Parameter(torch.zeros(total_dim))
        self.log_decay_delta = nn.Parameter(torch.zeros(total_dim))
        self.g_proj = nn.Linear(d_model, total_dim, bias=False)
        self.o_proj = nn.Linear(total_dim, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, total_dim)

    def forward(self, x):
        B, L, _ = x.shape
        q = F.normalize(self.q_proj(x).view(B, L, self.num_heads, self.head_dim), dim=-1)
        k = F.normalize(self.k_proj(x).view(B, L, self.num_heads, self.head_dim), dim=-1)
        v = F.silu(self.v_proj(x)).view(B, L, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.beta_proj(x))  # [B, L, H]

        # Channel-wise decay in fp32: g = base - softplus(delta); alpha = exp(g)
        log_decay = (self.log_decay_base - F.softplus(self.log_decay_delta)).to(torch.float32)
        alpha = torch.exp(log_decay).view(1, 1, self.num_heads, self.head_dim).to(x.dtype)

        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            b_t = beta[:, t, :, None, None]
            k_t = k[:, t]
            v_t = v[:, t]
            a_t = alpha[0, 0]  # [H, dk]
            # Decay state along key axis: D_t @ S
            S = S * a_t.unsqueeze(0).unsqueeze(-1)
            # Delta rule with scalar write gate
            read = (S * k_t.unsqueeze(-1)).sum(-2)
            S = S - b_t * k_t.unsqueeze(-1) * read.unsqueeze(-2)
            S = S + b_t * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            o_t = (S * q[:, t].unsqueeze(-1)).sum(-2)
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1).reshape(B, L, -1)
        o = self.group_norm(o.reshape(B * L, -1)).reshape(B, L, -1)
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)

```