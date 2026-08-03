# Review of Sparse Attention Blocks in Transformers

## Executive Summary

Standard self-attention in Transformers computes pairwise interactions for all tokens, resulting in \(O(n^2)\) computational and memory complexity. This becomes prohibitive for long sequences. Sparse attention mechanisms address this by restricting the set of token interactions, exploiting the empirical observation that most learned attention weights are near zero. This report reviews nine prominent sparse attention blocks—from foundational methods like the Sparse Transformer to the latest approaches such as FASA, NSA, MiniMax Sparse Attention (MSA), and SparDA—providing for each: a description, mathematical formulation, pros and cons, and PyTorch implementation code.[^1][^2][^3][^4]

***

## Sparse Transformer (Child et al., 2019)

### Description

The Sparse Transformer was one of the first works to introduce **factorized sparse attention**, reducing the \(O(n^2)\) complexity to \(O(n\sqrt{n})\). Rather than computing a full attention matrix, it factors the attention pattern into two complementary sparse patterns—**strided attention** and **fixed attention**—each attending to \(O(\sqrt{n})\) positions. By composing these over two attention heads, every position can attend to every other position through a path of length at most \(p+1\), where \(p\) is the number of factorized heads.[^5][^6]

The method was applied to images, audio, and text, setting state-of-the-art results on Enwik8, CIFAR-10, and ImageNet-64 density modeling benchmarks.[^6]

### Mathematical Formulation

Let \(n\) be the sequence length and \(l = \lfloor\sqrt{n}\rfloor\) be the stride. Two factorized attention heads are defined:

**Strided head** — each position \(i\) attends to positions in \(\{j : (i - j) \bmod l = 0\}\), i.e., every \(l\)-th previous position:

\[
A_i^{(\text{strided})} = \{j : j \leq i,\; (i - j) \bmod l = 0\}
\]

**Fixed head** — each position \(i\) attends to a local window plus fixed column positions:

\[
A_i^{(\text{fixed})} = \{j : \lfloor j/l \rfloor = \lfloor i/l \rfloor\} \cup \{j : j \bmod l \in \{l{-}c, \ldots, l{-}1\}\}
\]

where \(c\) is a hyperparameter for the number of summary columns. The overall attention output per head follows scaled dot-product attention restricted to each set \(A_i\):

\[
\text{Attention}(Q, K, V)_i = \text{softmax}\!\left(\frac{q_i \cdot K_{A_i}^\top}{\sqrt{d_k}}\right) V_{A_i}
\]

### Pros and Cons

| Pros | Cons |
|------|------|
| Reduces complexity from \(O(n^2)\) to \(O(n\sqrt{n})\)[^6] | Fixed patterns may miss important long-range dependencies |
| Proven on images, audio, text[^5] | Requires custom CUDA kernels for efficiency |
| Enables sequences of 10K+ tokens with hundreds of layers | Stride pattern is data-agnostic |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseTransformerAttention(nn.Module):
    """Factorized Sparse Attention (Strided + Fixed pattern)."""
    def __init__(self, d_model, n_heads, seq_len, stride=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.stride = stride or int(math.sqrt(seq_len))
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _strided_mask(self, n, device):
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        for i in range(n):
            indices = list(range(max(0, i - self.stride + 1), i + 1))  # local
            indices += list(range(i % self.stride, i + 1, self.stride))  # strided
            for j in set(indices):
                mask[i, j] = True
        return mask

    def _fixed_mask(self, n, device):
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        c = 1  # summary columns
        for i in range(n):
            block_start = (i // self.stride) * self.stride
            indices = list(range(block_start, min(i + 1, block_start + self.stride)))
            indices += [j for j in range(n) if j <= i and (j % self.stride >= self.stride - c)]
            for j in set(indices):
                mask[i, j] = True
        return mask

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Split heads: half strided, half fixed
        half = self.n_heads // 2
        strided_mask = self._strided_mask(N, x.device)
        fixed_mask = self._fixed_mask(N, x.device)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply masks
        scores[:, :half].masked_fill_(~strided_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        scores[:, half:].masked_fill_(~fixed_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.Wo(out)
```

***

## Longformer (Beltagy et al., 2020)

### Description

The Longformer introduces an attention mechanism that **scales linearly** with sequence length by combining three patterns: (1) **sliding window attention**, where each token attends to a fixed window of \(w\) neighbors; (2) **dilated sliding window attention**, which introduces gaps of size \(d\) to expand the receptive field; and (3) **global attention** on task-specific tokens (e.g., `[CLS]`) that attend to and are attended by all tokens. With \(L\) layers and window size \(w\), the top-layer receptive field is \(L \times w\), covering the full sequence efficiently.[^7][^8][^9][^10]

### Mathematical Formulation

For a token at position \(i\) with a sliding window of size \(w\):

\[
A_i^{(\text{slide})} = \{j : |i - j| \leq w/2\}
\]

The dilated variant with dilation \(d\):

\[
A_i^{(\text{dilated})} = \{j : |i - j| \leq w/2 \cdot (d + 1),\; (i - j) \bmod (d+1) = 0 \}
\]

For global tokens in set \(\mathcal{G}\):

\[
A_i^{(\text{global})} = \{1, \ldots, n\} \quad \text{if } i \in \mathcal{G}, \qquad A_i^{(\text{slide})} \cup \mathcal{G} \quad \text{otherwise}
\]

The Longformer uses separate projections \(Q_s, K_s, V_s\) for sliding window and \(Q_g, K_g, V_g\) for global attention:[^11]

\[
\text{Attention}(Q, K, V)_i = \text{softmax}\!\left(\frac{q_i \cdot K_{A_i}^\top}{\sqrt{d_k}}\right) V_{A_i}
\]

Overall complexity is \(O(n \cdot w)\), which is linear in \(n\) for fixed \(w\).[^8]

### Pros and Cons

| Pros | Cons |
|------|------|
| Linear complexity \(O(n \cdot w)\)[^8] | Window size limits local context per layer |
| Handles sequences up to 4096+ tokens[^7] | Global tokens must be task-specifically chosen |
| Drop-in replacement for standard attention[^8] | Dilated patterns may miss fine-grained local detail |
| Flexible: different \(w\) per layer | Requires custom sparse CUDA kernels for efficiency |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LongformerAttention(nn.Module):
    """Sliding Window + Global Attention (simplified Longformer)."""
    def __init__(self, d_model, n_heads, window_size=256, global_tokens=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w = window_size
        self.global_tokens = global_tokens or  # e.g., [CLS]
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _build_mask(self, n, device):
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        half_w = self.w // 2
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            mask[i, start:end] = True
        # Global attention
        for g in self.global_tokens:
            if g < n:
                mask[g, :] = True
                mask[:, g] = True
        return mask

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = self._build_mask(N, x.device)
        scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.Wo(out)
```

***

## BigBird (Zaheer et al., 2020)

### Description

BigBird combines three sparse attention patterns—**random**, **sliding window (local)**, and **global** tokens—to achieve **linear complexity** \(O(n)\) while preserving the Turing completeness and universal approximation properties of full attention. The theoretical contribution proves that \(O(1)\) global tokens suffice to maintain these properties. BigBird handles sequences up to 8× longer than BERT on the same hardware and achieves state-of-the-art on question answering, summarization, and genomics tasks.[^12][^13][^14][^15][^16]

### Mathematical Formulation

For each token \(i\), the attended set is:

\[
A_i = A_i^{(\text{random})} \cup A_i^{(\text{window})} \cup A_i^{(\text{global})}
\]

where:

- **Random**: \(r\) randomly chosen positions per token
- **Window**: \(A_i^{(\text{window})} = \{j : |i - j| \leq w/2\}\) for window size \(w\)
- **Global**: a set of \(g\) tokens that attend to/from all positions

The overall mask \(M\) is:

\[
M_{ij} = \mathbf{1}\!\left[j \in A_i^{(\text{random})} \cup A_i^{(\text{window})} \cup A_i^{(\text{global})}\right]
\]

The attention is computed as:

\[
\text{Attention}(Q, K, V)_i = \text{softmax}\!\left(\frac{q_i K_{A_i}^\top}{\sqrt{d_k}}\right) V_{A_i}
\]

In practice, BigBird uses **block-sparse** implementation: tokens are grouped into blocks of size \(b\), and attention decisions are made at the block level.[^15]

### Pros and Cons

| Pros | Cons |
|------|------|
| Proven universal approximator & Turing complete[^12] | Random patterns introduce non-determinism |
| Linear \(O(n)\) complexity[^13] | Block-level granularity may waste computation |
| Handles 8× longer sequences than BERT[^14] | Requires careful tuning of \(r, w, g\) hyperparameters |
| Strong on QA, summarization, genomics | Less effective on tasks needing dense global attention |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BigBirdAttention(nn.Module):
    """BigBird: Random + Window + Global sparse attention."""
    def __init__(self, d_model, n_heads, window_size=128,
                 num_random=64, num_global=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w = window_size
        self.num_random = num_random
        self.num_global = num_global
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _build_mask(self, n, device):
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        half_w = self.w // 2
        # Sliding window
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            mask[i, start:end] = True
        # Random connections
        for i in range(n):
            rand_idx = torch.randint(0, n, (self.num_random,), device=device)
            mask[i, rand_idx] = True
        # Global tokens
        for g in range(min(self.num_global, n)):
            mask[g, :] = True
            mask[:, g] = True
        return mask

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = self._build_mask(N, x.device)
        scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.Wo(out)
```

***

## FASA — Frequency-Aware Sparse Attention (Wang et al., 2026)

### Description

FASA is a **training-free** sparse attention framework that leverages a novel insight into RoPE (Rotary Position Embeddings): only a small subset of **frequency chunks** (FCs) in each attention head contribute to contextual awareness, while the majority encode positional patterns. These "dominant FCs" are **sparse** (less than 1% of all FCs), **universal** across model scales, and **task-agnostic**. FASA operates in two stages: (1) **Token Importance Prediction (TIP)** uses dominant FCs to cheaply estimate which tokens matter, and (2) **Focused Attention Computation (FAC)** performs full-precision attention only on the selected subset.[^17][^18][^19]

FASA achieves nearly 100% of full-KV performance on LongBench while keeping only 256 tokens, and delivers 2.56× speedup with 18.9% cache usage.[^18]

### Mathematical Formulation

**Frequency-Chunk Decomposition under RoPE.** Each \(d\)-dimensional vector is split into \(d/2\) 2D chunks \(\mathbf{v}^{[i]} = (v_{2i}, v_{2i+1})^\top\). The RoPE rotation is block-diagonal:

\[
\mathbf{R}_{\Delta t} = \bigoplus_{i=1}^{d/2} \mathbf{R}_{\Delta t, \theta_i}, \quad \theta_i = B^{-2(i-1)/d}
\]

**Contextual Agreement (CA)** measures dominance of FC \(i\) in head \((l,h)\):

\[
\text{CA}_{\mathcal{K}}^{l,h,i}(\mathbf{q}_t, \mathbf{K}_{1:t}) = \frac{|\text{TopK-I}(\boldsymbol{\alpha}_{l,h}, \mathcal{K}) \cap \text{TopK-I}(\boldsymbol{\alpha}_{l,h}^{(i)}, \mathcal{K})|}{\mathcal{K}}
\]

**TIP Stage** — Online importance scoring using only dominant FCs \(\mathcal{I}_{\text{dom}}\):

\[
\mathbf{S}_t^{l,h} = \sum_{i \in \mathcal{I}_{\text{dom}}^{l,h}} \boldsymbol{\alpha}^{l,h,i}(\mathbf{q}_t, \mathbf{K}_{1:t}), \quad \mathcal{T}_t = \text{TopK-I}(\mathbf{S}_t^{l,h}, N_{\text{fac}})
\]

**FAC Stage** — Full-precision attention on selected tokens:

\[
\hat{\boldsymbol{\alpha}}_{\text{FAC}}^{l,h} = \text{softmax}\!\left(\frac{\mathbf{q}_t \mathbf{K}_{\mathcal{T}_t}^\top}{\sqrt{d}}\right), \quad \mathbf{O}_t^{l,h} = \hat{\boldsymbol{\alpha}}_{\text{FAC}}^{l,h} \, \mathbf{V}_{\mathcal{T}_t}
\]

**Overall complexity**: \(O(2t \cdot N_{\text{tip}} + 2 N_{\text{fac}} \cdot d)\) vs. \(O(2td)\) for full attention. Speedup \(\approx d / N_{\text{tip}}\) when \(N_{\text{fac}} \ll t\).[^17]

### Pros and Cons

| Pros | Cons |
|------|------|
| Training-free, plug-and-play[^18] | Requires RoPE-based models (extended to ALiBi/MLA)[^17] |
| Near-oracle accuracy with ≤256 tokens[^18] | Offline calibration step needed (though one-time) |
| Two variants: FASA-M (memory), FASA-C (compute)[^17] | FC dominance analysis adds implementation complexity |
| Up to 8× KV cache compression and 2.56× speedup[^18] | Performance depends on quality of dominant FC identification |
| Orthogonal to other KV compression methods | Limited to decoding phase optimization |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FASA(nn.Module):
    """Frequency-Aware Sparse Attention (simplified decoding)."""
    def __init__(self, d_model, n_heads, n_tip=16, n_fac=256, rope_base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_tip = n_tip  # number of dominant FCs
        self.n_fac = n_fac  # number of tokens for focused attention
        self.rope_base = rope_base
        # Dominant FC indices per head (pre-calibrated offline)
        self.register_buffer(
            'dominant_fcs',
            torch.zeros(n_heads, n_tip, dtype=torch.long)
        )
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _apply_rope(self, x, positions):
        """Apply RoPE to input tensor."""
        d = x.shape[-1]
        freqs = 1.0 / (self.rope_base ** (torch.arange(0, d, 2, device=x.device).float() / d))
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        cos_a, sin_a = angles.cos(), angles.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)

    def calibrate_dominant_fcs(self, sample_q, sample_K, top_k_ca=64):
        """One-time offline calibration to find dominant FCs per head."""
        with torch.no_grad():
            d_half = self.d_k // 2
            for h in range(self.n_heads):
                q_h = sample_q[:, h]
                K_h = sample_K[:, h]
                full_scores = torch.matmul(q_h, K_h.transpose(-2, -1))
                _, full_topk = full_scores.topk(top_k_ca, dim=-1)
                ca_scores = torch.zeros(d_half, device=q_h.device)
                for fc in range(d_half):
                    q_fc = q_h[..., 2*fc:2*fc+2]
                    K_fc = K_h[..., 2*fc:2*fc+2]
                    fc_scores = torch.matmul(q_fc, K_fc.transpose(-2, -1))
                    _, fc_topk = fc_scores.topk(top_k_ca, dim=-1)
                    overlap = sum(
                        len(set(ft.tolist()) & set(fk.tolist()))
                        for ft, fk in zip(full_topk.unbind(0), fc_topk.unbind(0))
                    ) / (full_topk.shape * top_k_ca)
                    ca_scores[fc] = overlap
                self.dominant_fcs[h] = ca_scores.topk(self.n_tip).indices

    def forward(self, q_t, K_cache, V_cache, positions):
        """
        q_t: (B, 1, d_model) - current query
        K_cache, V_cache: (B, T, d_model) - cached keys/values
        positions: (T,) - position indices
        """
        B, T, _ = K_cache.shape
        q = self.Wq(q_t).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(K_cache).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(V_cache).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        outputs = []
        for h in range(self.n_heads):
            q_h = q[:, h]       # (B, 1, d_k)
            K_h = K[:, h]       # (B, T, d_k)
            V_h = V[:, h]       # (B, T, d_k)
            # TIP: compute importance using dominant FCs only
            fc_idx = self.dominant_fcs[h]
            dim_idx = torch.stack([fc_idx * 2, fc_idx * 2 + 1], dim=-1).flatten()
            q_sub = q_h[..., dim_idx]       # (B, 1, 2*n_tip)
            K_sub = K_h[..., dim_idx]       # (B, T, 2*n_tip)
            importance = torch.matmul(q_sub, K_sub.transpose(-2, -1)).squeeze(1)  # (B, T)
            # Select top-N_fac tokens
            n_sel = min(self.n_fac, T)
            _, top_idx = importance.topk(n_sel, dim=-1)  # (B, n_sel)
            # FAC: gather and compute full attention
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, self.d_k)
            K_sel = K_h.gather(1, top_idx_exp)
            V_sel = V_h.gather(1, top_idx_exp)
            attn_scores = torch.matmul(q_h, K_sel.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = F.softmax(attn_scores, dim=-1)
            out_h = torch.matmul(attn_weights, V_sel)
            outputs.append(out_h)

        out = torch.cat(outputs, dim=-1).view(B, 1, self.d_model)
        return self.Wo(out)
```

***

## NSA — Native Sparse Attention (Yuan et al., 2025)

### Description

NSA, developed by DeepSeek, is a **natively trainable** sparse attention mechanism that integrates hardware-aligned optimizations with a **dynamic hierarchical sparse strategy**. Unlike inference-only methods, NSA supports end-to-end training, reducing pretraining computation without sacrificing performance. It processes keys and values through three parallel attention branches:[^20][^21][^22]

1. **Compressed attention**: coarse-grained tokens via a learnable MLP that aggregates blocks
2. **Selected attention**: fine-grained top-\(n\) block selection based on compressed scores
3. **Sliding window**: local context from the most recent \(w\) tokens

Outputs are combined through a learned gating mechanism. NSA achieves up to 11.6× decoding speedup and 9× forward speedup on 64k-length sequences.[^21][^20]

### Mathematical Formulation

The overall attention replaces full \((\mathbf{k}_{1:t}, \mathbf{v}_{1:t})\) with compact representations via three strategies:

\[
o_t^* = \sum_{c \in \{\text{cmp}, \text{slc}, \text{win}\}} g_t^c \cdot \text{Attn}(q_t, \tilde{K}_t^c, \tilde{V}_t^c)
\]

where \(g_t^c \in [0,1]\) are learned gate scores from an MLP + sigmoid.

**Compression** — A learnable MLP \(\varphi\) maps blocks of \(l\) keys with stride \(d\):

\[
\tilde{K}_t^{\text{cmp}} = \left(\varphi(k_{id+1:id+l})\right)_{1 \leq i \leq \lfloor(t-l)/d\rfloor}
\]

**Selection** — Block importance from compressed attention scores:

\[
p_t^{\text{cmp}} = \text{softmax}(q_t^\top \tilde{K}_t^{\text{cmp}}), \quad I_t = \{i : \text{rank}(p_t^{\text{slc}'}[i]) \leq n\}
\]

\[
\tilde{K}_t^{\text{slc}} = \text{Cat}\!\left(\{k_{il'+1:(i+1)l'} \mid i \in I_t\}\right)
\]

**Sliding window**: \(\tilde{K}_t^{\text{win}} = k_{t-w:t}\).

**Total tokens attended per query**: \(N_t = \lfloor t/d \rfloor + n \cdot l' + w \ll t\).

### Pros and Cons

| Pros | Cons |
|------|------|
| Natively trainable end-to-end[^22] | Requires custom Triton kernels |
| Hardware-aligned design (Tensor Core utilization)[^21] | Complex multi-branch architecture |
| 11.6× decode, 9× forward speedup at 64k[^20] | Needs GQA/MQA backbone for best results |
| Outperforms full attention on many benchmarks[^21] | Compression MLP adds parameters |
| Hierarchical design captures both local and global patterns | Higher implementation complexity vs simpler methods |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NSAAttention(nn.Module):
    """Native Sparse Attention: Compress + Select + Sliding Window."""
    def __init__(self, d_model, n_heads, block_size=32, stride=16,
                 select_block_size=64, n_select=16, window_size=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size
        self.stride = stride
        self.select_block_size = select_block_size
        self.n_select = n_select
        self.window_size = window_size

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        # Compression MLP per head
        self.compress_k = nn.Linear(block_size * self.d_k, self.d_k)
        self.compress_v = nn.Linear(block_size * self.d_k, self.d_k)
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(self.d_k, 3),
            nn.Sigmoid()
        )

    def _compress(self, K, V):
        """Compress key/value blocks using learned MLP."""
        B, H, T, D = K.shape
        n_blocks = (T - self.block_size) // self.stride + 1
        comp_K, comp_V = [], []
        for i in range(n_blocks):
            start = i * self.stride
            end = start + self.block_size
            k_block = K[:, :, start:end, :].reshape(B, H, -1)
            v_block = V[:, :, start:end, :].reshape(B, H, -1)
            comp_K.append(self.compress_k(k_block))
            comp_V.append(self.compress_v(v_block))
        return torch.stack(comp_K, dim=2), torch.stack(comp_V, dim=2)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # 1. Compressed attention
        comp_K, comp_V = self._compress(K, V)
        comp_scores = torch.matmul(Q, comp_K.transpose(-2, -1)) / math.sqrt(self.d_k)
        comp_attn = F.softmax(comp_scores, dim=-1)
        out_comp = torch.matmul(comp_attn, comp_V)

        # 2. Selected attention (top-n blocks)
        block_importance = comp_attn.mean(dim=1).mean(dim=1)  # (B, n_blocks)
        n_sel = min(self.n_select, block_importance.shape[-1])
        _, top_blocks = block_importance.topk(n_sel, dim=-1)
        # Gather selected blocks
        sel_indices = []
        for b_idx in top_blocks.unbind(-1):
            start = b_idx * self.stride
            indices = torch.arange(self.select_block_size, device=x.device).unsqueeze(0) + start.unsqueeze(-1)
            sel_indices.append(indices.clamp(max=N-1))
        sel_indices = torch.cat(sel_indices, dim=-1)  # (B, n_sel * select_block_size)
        sel_indices = sel_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_heads, -1, self.d_k)
        K_sel = K.gather(2, sel_indices)
        V_sel = V.gather(2, sel_indices)
        sel_scores = torch.matmul(Q, K_sel.transpose(-2, -1)) / math.sqrt(self.d_k)
        sel_attn = F.softmax(sel_scores, dim=-1)
        out_sel = torch.matmul(sel_attn, V_sel)

        # 3. Sliding window attention
        win_size = min(self.window_size, N)
        K_win = K[:, :, -win_size:, :]
        V_win = V[:, :, -win_size:, :]
        win_scores = torch.matmul(Q, K_win.transpose(-2, -1)) / math.sqrt(self.d_k)
        win_attn = F.softmax(win_scores, dim=-1)
        out_win = torch.matmul(win_attn, V_win)

        # Gated combination
        gates = self.gate(Q.mean(dim=2))  # (B, H, 3)
        g_comp = gates[..., 0:1].unsqueeze(2)
        g_sel = gates[..., 1:2].unsqueeze(2)
        g_win = gates[..., 2:3].unsqueeze(2)
        out = g_comp * out_comp + g_sel * out_sel + g_win * out_win

        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.Wo(out)
```

***

## SparseK Attention (Lou et al., 2024)

### Description

SparseK Attention introduces a **differentiable top-k operator** for sparse attention that enables gradient-based optimization. A learned scoring network evaluates the importance of each key-value pair, and the SparseK operator selects a constant number \(k\) of KV pairs per query. This yields **linear time complexity** during training and **constant memory footprint** during autoregressive generation. It integrates seamlessly into pre-trained LLMs with minimal fine-tuning.[^23][^24][^25]

### Mathematical Formulation

For each query \(q\), a scoring network produces importance scores \(u \in \mathbb{R}^n\) for all KV pairs. The **SparseK operator** computes a threshold \(\tau(u)\) such that the sum of normalized scores equals \(k\):

\[
\text{SparseK}(u, k)_j = \max(u_j - \tau(u), 0)
\]

where \(\tau\) is chosen such that \(\sum_j \max(u_j - \tau, 0) = k\). This is a differentiable relaxation of top-k. The attention is then:

\[
m_j = \text{SparseK}(u, k)_j, \quad \text{Attention}(q, K, V) = \text{softmax}\!\left(\frac{qK_{\text{sel}}^\top}{\sqrt{d_k}}\right) V_{\text{sel}}
\]

where \(K_{\text{sel}}, V_{\text{sel}}\) contain only the top-\(k\) entries. During generation, the operator supports **incremental evaluation**, maintaining constant memory.[^25][^26]

### Pros and Cons

| Pros | Cons |
|------|------|
| Differentiable, supports end-to-end training[^23] | Scoring network adds overhead |
| Linear time, constant memory at generation[^24] | Requires fine-tuning when applied to pre-trained models |
| Seamless integration into existing LLMs[^25] | Fixed \(k\) may not be optimal for all layers/heads |
| Outperforms previous sparse attention methods[^23] | Top-k selection not hardware-aligned (scattered access) |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseKOperator(torch.autograd.Function):
    """Differentiable SparseK: projects onto the k-simplex."""
    @staticmethod
    def forward(ctx, scores, k):
        sorted_scores, _ = scores.sort(dim=-1, descending=True)
        cumsum = sorted_scores.cumsum(dim=-1)
        arange = torch.arange(1, scores.shape[-1] + 1, device=scores.device).float()
        threshold = (cumsum - k) / arange
        mask = sorted_scores > threshold
        # Find the last valid index
        rho = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        tau = (cumsum.gather(-1, rho - 1) - k) / rho.float()
        output = (scores - tau).clamp(min=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        supp = (output > 0).float()
        n_supp = supp.sum(dim=-1, keepdim=True).clamp(min=1)
        grad = supp * (grad_output - (grad_output * supp).sum(dim=-1, keepdim=True) / n_supp)
        return grad, None

class SparseKAttention(nn.Module):
    """SparseK Attention with differentiable top-k selection."""
    def __init__(self, d_model, n_heads, k=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k = k
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.score_net = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.ReLU(),
            nn.Linear(self.d_k, 1)
        )

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Score each KV pair
        kv_scores = self.score_net(K).squeeze(-1)  # (B, H, N)
        # Apply differentiable SparseK
        selection = SparseKOperator.apply(kv_scores, self.k)  # (B, H, N)
        # Select top-k indices
        k_actual = min(self.k, N)
        _, top_idx = selection.topk(k_actual, dim=-1)  # (B, H, k)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        K_sel = K.gather(2, top_idx_exp)
        V_sel = V.gather(2, top_idx_exp)

        scores = torch.matmul(Q, K_sel.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_sel)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.Wo(out)
```

***

## SpargeAttn (Zhang et al., 2025)

### Description

SpargeAttn is a **universal, training-free** sparse attention method that accelerates diverse models—language, image, and video generation—using a **two-stage online filter** built atop FlashAttention. The first stage rapidly predicts which blocks of the attention map will contain near-zero values and skips the corresponding \(Q_iK_j^\top\) multiplications. The second stage applies an **online softmax-aware filter** at no extra overhead to further eliminate unnecessary \(\tilde{P}_{ij}V_j\) computations. SpargeAttn achieves 2.5–5× acceleration while preserving end-to-end metrics.[^27][^28][^29]

### Mathematical Formulation

SpargeAttn operates at the block level. For query block \(Q_i\) and key block \(K_j\):

**Stage 1 — Sparse Prediction**: Estimate whether \(\text{Attn}(Q_i, K_j)\) will be negligible. This is done by computing a low-cost proxy (e.g., using compressed representations or token self-similarity):

\[
\hat{s}_{ij} = f_{\text{pred}}(Q_i, K_j) \quad \Rightarrow \quad \text{skip if } \hat{s}_{ij} < \epsilon_1
\]

**Stage 2 — Softmax-Aware Filter**: After computing \(\tilde{P}_{ij} = \text{softmax}(Q_i K_j^\top / \sqrt{d_k})\), check if the block's contribution is negligible relative to the running online softmax maximum:

\[
\text{skip } \tilde{P}_{ij} V_j \quad \text{if } \max(\tilde{P}_{ij}) \ll e^{m_{\text{old}} - m_{\text{new}}}
\]

where \(m_{\text{old}}, m_{\text{new}}\) are online softmax running maxima. The overall output remains equivalent to or very close to exact FlashAttention.

### Pros and Cons

| Pros | Cons |
|------|------|
| Universal: works on LLMs, image and video diffusion models[^28] | Speedup depends on inherent sparsity of model |
| Training-free, plug-and-play[^27] | Block granularity may miss fine-grained patterns |
| 2.5–5× faster than dense/existing sparse attention[^28] | Two-stage filter adds some constant overhead |
| Enhances long-context LLM performance[^29] | Requires Triton/CUDA kernel integration |
| Compatible with quantization (SageAttention)[^29] | Threshold \(\epsilon\) tuning needed per model family |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpargeAttention(nn.Module):
    """Simplified SpargeAttn: two-stage block-sparse attention."""
    def __init__(self, d_model, n_heads, block_size=64, sparsity_threshold=0.01):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size
        self.threshold = sparsity_threshold
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        bs = self.block_size
        n_blocks = (N + bs - 1) // bs
        # Pad if needed
        pad = n_blocks * bs - N
        if pad > 0:
            Q = F.pad(Q, (0, 0, 0, pad))
            K = F.pad(K, (0, 0, 0, pad))
            V = F.pad(V, (0, 0, 0, pad))
        L = n_blocks * bs

        Q_blocks = Q.view(B, self.n_heads, n_blocks, bs, self.d_k)
        K_blocks = K.view(B, self.n_heads, n_blocks, bs, self.d_k)
        V_blocks = V.view(B, self.n_heads, n_blocks, bs, self.d_k)

        # Stage 1: predict block importance using mean Q/K
        Q_mean = Q_blocks.mean(dim=3)  # (B, H, n_blocks, d_k)
        K_mean = K_blocks.mean(dim=3)  # (B, H, n_blocks, d_k)
        block_scores = torch.matmul(Q_mean, K_mean.transpose(-2, -1)) / math.sqrt(self.d_k)
        block_mask = (block_scores > self.threshold)  # (B, H, n_blocks, n_blocks)

        # Expand block mask to full size
        full_mask = block_mask.unsqueeze(3).unsqueeze(5)
        full_mask = full_mask.expand(-1, -1, -1, bs, -1, bs)
        full_mask = full_mask.reshape(B, self.n_heads, L, L)

        scores = torch.matmul(Q.view(B, self.n_heads, L, self.d_k),
                              K.view(B, self.n_heads, L, self.d_k).transpose(-2, -1)) / math.sqrt(self.d_k)
        scores.masked_fill_(~full_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        # Stage 2: softmax-aware filter (zero out negligible blocks)
        attn_blocks = attn.view(B, self.n_heads, n_blocks, bs, n_blocks, bs)
        block_max = attn_blocks.amax(dim=(3, 5))  # (B, H, n_blocks, n_blocks)
        softmax_mask = (block_max > self.threshold)
        softmax_mask_full = softmax_mask.unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, bs, -1, bs)
        softmax_mask_full = softmax_mask_full.reshape(B, self.n_heads, L, L)
        attn.masked_fill_(~softmax_mask_full, 0.0)

        out = torch.matmul(attn, V.view(B, self.n_heads, L, self.d_k))
        out = out[:, :, :N, :]  # remove padding
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.Wo(out)
```

***

## MSA — MiniMax Sparse Attention (Lai et al., 2026)

### Description

MSA is a blockwise sparse attention built on Grouped Query Attention (GQA), proposed by MiniMax (Lai et al., 2026). A lightweight **Index Branch** scores key-value blocks and independently selects a Top-k subset for each GQA group, enabling group-specific sparse retrieval while maintaining efficient block-level execution; the **Main Branch** then performs exact block-sparse softmax attention over only the selected blocks. The Index Branch input is detached (stopgrad); a KL alignment loss trains the indexer. The block containing the query is always forced into the selected set. Deployed on a 109B-parameter natively multimodal model (MiniMax-M3), MSA performs on par with full GQA while reducing per-token attention compute by 28.4x at 1M context and delivering 14.2x prefill / 7.6x decode wall-clock speedups on H800. Cite as [arXiv:2606.13392](https://arxiv.org/abs/2606.13392).

### Mathematical Formulation

Tokens are partitioned into contiguous blocks of size `B_k` (default 128). The Index Branch computes `Q_idx = stopgrad(X) W_q^idx` (one index query head per GQA group, `H_kv` heads) and `K_idx = stopgrad(X) W_k^idx` (single shared index key head). Token-level scores `S^idx,(r)_{i,j} = (Q_idx)^(r)_i · (K_idx)_j^T / sqrt(d_idx)` are max-pooled within each block (causal) to give block scores `M^idx,(r)_{i,b}`. Per-group Top-k selection `I^(r)_i = TopK(M^idx,(r)_{i,·}, k)` with the local block forced in. The Main Branch runs standard scaled dot-product softmax attention restricted to causally visible tokens in selected blocks: `O^(h)_i = softmax(Q^(h)_i · K^(r)[I^(r)_i]^T / sqrt(d_h)) · V^(r)[I^(r)_i]`. Complexity: `O(H_kv · d_idx · N^2)` for the index branch + `O(H_q · d_h · N · k · B_k)` for the main branch; per-query budget fixed at `k · B_k` tokens regardless of N.

### Pros and Cons

| Pros | Cons |
|------|------|
| 28.4x per-token compute reduction at 1M context | Requires custom block-sparse kernels for full speedup |
| 14.2x prefill, 7.6x decode speedup on H800 | Index Branch adds parameters (W_q^idx, W_k^idx) |
| Per-group selection preserves GQA's KV-cache savings | KL alignment loss and indexer warmup add training complexity |
| Forced local block prevents degenerate omission | Top-k selection is non-differentiable (trained via KL loss) |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSAAttention(nn.Module):
    """MiniMax Sparse Attention (simplified): Index Branch + Main Branch."""
    def __init__(self, d_model, n_heads, n_kv_heads=None, block_size=128,
                 topk_blocks=16, index_dim=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 8)
        self.group_size = n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.index_dim = index_dim
        kv_dim = self.n_kv_heads * self.d_k
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, kv_dim, bias=False)
        self.Wv = nn.Linear(d_model, kv_dim, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.q_idx = nn.Linear(d_model, self.n_kv_heads * index_dim, bias=False)
        self.k_idx = nn.Linear(d_model, index_dim, bias=False)
        self.scale = self.d_k ** -0.5
        self.idx_scale = index_dim ** -0.5

    def forward(self, x):
        B, N, _ = x.shape
        bs = self.block_size
        num_blocks = (N + bs - 1) // bs
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        with torch.no_grad():
            x_det = x.detach()
        qi = self.q_idx(x_det).view(B, N, self.n_kv_heads, self.index_dim)
        ki = self.k_idx(x_det).view(B, N, 1, self.index_dim)
        token_scores = torch.einsum('bshd,btd->bsht', qi, ki.squeeze(2)) * self.idx_scale
        pad = (bs - N % bs) % bs
        if pad:
            token_scores = F.pad(token_scores, (0, pad), value=float('-inf'))
        padded = N + pad
        blk_scores = token_scores.view(B, N, self.n_kv_heads, padded // bs, bs).max(dim=-1).values[:, :, :, :num_blocks]
        k_sel = min(self.topk_blocks, num_blocks)
        _, sel = torch.topk(blk_scores, k_sel, dim=-1)
        G = self.group_size
        K_exp = K.unsqueeze(2).expand(B, self.n_kv_heads, G, N, self.d_k).reshape(B, self.n_heads, N, self.d_k)
        V_exp = V.unsqueeze(2).expand(B, self.n_kv_heads, G, N, self.d_k).reshape(B, self.n_heads, N, self.d_k)
        out = torch.zeros_like(Q)
        for b in range(num_blocks):
            s, e = b * bs, min(b * bs + bs, N)
            if e <= s: continue
            kb, vb = K_exp[:, :, s:e, :], V_exp[:, :, s:e, :]
            sc = torch.matmul(Q, kb.transpose(-2, -1)) * self.scale
            sel_b = (sel == b).any(dim=-1).view(B, N, self.n_kv_heads, 1).expand(B, N, self.n_kv_heads, G).reshape(B, N, self.n_heads).transpose(0, 1).unsqueeze(-1)
            attn = F.softmax(sc, dim=-1)
            out = torch.where(sel_b, torch.matmul(attn, vb), out)
        return self.Wo(out.transpose(1, 2).contiguous().view(B, N, self.d_model))
```

***

## SparDA — Sparse Decoupled Attention (Fu et al., 2026)

### Description

SparDA (Fu et al., 2026, NVIDIA) is a decoupled sparse attention architecture that introduces a fourth per-layer projection — the **Forecast** — alongside Query, Key, and Value. The Forecast predicts the KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch with current-layer execution. Because the Forecast is decoupled from the attention query, the GQA implementation uses one Forecast head per GQA group, reducing selection overhead versus the original multi-head selector. SparDA adds <0.5% parameters and trains only the Forecast projections by matching the original selector's attention distribution. On two sparse-pretrained 8B models, SparDA matches or slightly improves accuracy and delivers up to 1.25x prefill speedup and 1.7x decode speedup over the sparse-attention offload baseline; by enabling larger feasible batch sizes on a single GPU, SparDA reaches up to 5.3x higher decode throughput than the non-offload sparse baseline. Cite as [arXiv:2606.04511](https://arxiv.org/abs/2606.04511).

### Mathematical Formulation

The Forecast projection `F = W_F x` produces one head per GQA group (shape `N × H_kv × d_f`). Block-level scores are computed via `S[i, b] = F[i] · block_rep[b]` where `block_rep` is an average-pooled representation of the KV blocks. Top-k block selection per GQA group. The Main Branch then runs standard block-sparse softmax attention over selected blocks. Complexity: `O(H_kv · d_f · N · num_blocks)` for the forecast + `O(H_q · d_h · N · k · B_k)` for the main branch.

### Pros and Cons

| Pros | Cons |
|------|------|
| Lookahead selection overlaps CPU-GPU prefetch with compute | CPU-offload prefetch overlap is a runtime/kernel concern |
| <0.5% added parameters; trains only Forecast projections | Requires sparse-pretrained backbone for best results |
| Up to 5.3x decode throughput via larger feasible batch sizes | Block-level granularity may miss fine-grained patterns |
| 1.25x prefill, 1.7x decode speedup over offload baseline | Top-k selection is non-differentiable (trained via distillation) |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparDAAttention(nn.Module):
    """SparDA (simplified): Forecast projection + block-sparse attention."""
    def __init__(self, d_model, n_heads, n_kv_heads=None, block_size=128,
                 topk_blocks=16, forecast_dim=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 8)
        self.group_size = n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.forecast_dim = forecast_dim
        kv_dim = self.n_kv_heads * self.d_k
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, kv_dim, bias=False)
        self.Wv = nn.Linear(d_model, kv_dim, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.forecast = nn.Linear(d_model, self.n_kv_heads * forecast_dim, bias=False)
        self.scale = self.d_k ** -0.5
        self.f_scale = forecast_dim ** -0.5

    def forward(self, x):
        B, N, _ = x.shape
        bs = self.block_size
        num_blocks = (N + bs - 1) // bs
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        f = self.forecast(x).view(B, N, self.n_kv_heads, self.forecast_dim)
        block_rep = F.avg_pool1d(
            f.transpose(1, 2).reshape(B * self.n_kv_heads, self.forecast_dim, N),
            kernel_size=bs, stride=bs, ceil_mode=True,
        ).reshape(B, self.n_kv_heads, self.forecast_dim, -1).permute(0, 3, 1, 2)
        blk_scores = torch.einsum('bshd,bthd->bsht', f, block_rep) * self.f_scale
        k_sel = min(self.topk_blocks, num_blocks)
        _, sel = torch.topk(blk_scores, k_sel, dim=-1)
        G = self.group_size
        K_exp = K.unsqueeze(2).expand(B, self.n_kv_heads, G, N, self.d_k).reshape(B, self.n_heads, N, self.d_k)
        V_exp = V.unsqueeze(2).expand(B, self.n_kv_heads, G, N, self.d_k).reshape(B, self.n_heads, N, self.d_k)
        out = torch.zeros_like(Q)
        for b in range(num_blocks):
            s, e = b * bs, min(b * bs + bs, N)
            if e <= s: continue
            kb, vb = K_exp[:, :, s:e, :], V_exp[:, :, s:e, :]
            sc = torch.matmul(Q, kb.transpose(-2, -1)) * self.scale
            sel_b = (sel == b).any(dim=-1).view(B, N, self.n_kv_heads, 1).expand(B, N, self.n_kv_heads, G).reshape(B, N, self.n_heads).transpose(0, 1).unsqueeze(-1)
            attn = F.softmax(sc, dim=-1)
            out = torch.where(sel_b, torch.matmul(attn, vb), out)
        return self.Wo(out.transpose(1, 2).contiguous().view(B, N, self.d_model))
```

***

## Summary Comparison Table

| Method | Year | Complexity (per query) | Memory Footprint | Trainable | Training-Free | Key Innovation | Reference |
|--------|------|----------------------|------------------|-----------|---------------|----------------|-----------|
| **Sparse Transformer** | 2019 | \(O(n\sqrt{n})\)[^6] | \(O(n\sqrt{n})\) | Yes | No | Factorized strided + fixed patterns | arXiv:1904.10509 / DOI: 10.48550/arXiv.1904.10509[^6] |
| **Longformer** | 2020 | \(O(n \cdot w)\)[^8] | \(O(n \cdot w)\) | Yes | No | Sliding window + dilated + global attention | arXiv:2004.05150 / DOI: 10.48550/arXiv.2004.05150[^8] |
| **BigBird** | 2020 | \(O(n)\)[^13] | \(O(n)\) | Yes | No | Random + window + global; Turing complete proof | arXiv:2007.14062 / DOI: 10.48550/arXiv.2007.14062[^13] |
| **SparseK Attention** | 2024 | \(O(n)\) train / \(O(k)\) gen[^23] | \(O(k)\) constant at gen | Yes | No | Differentiable top-k operator | arXiv:2406.16747 / DOI: 10.48550/arXiv.2406.16747[^23] |
| **NSA** | 2025 | \(O(t/d + n l' + w)\)[^21] | \(O(t/d + n l' + w)\) | Yes | No | Hierarchical compress + select + window; hardware-aligned | arXiv:2502.11089 / DOI: 10.48550/arXiv.2502.11089[^22] |
| **SpargeAttn** | 2025 | \(O(n^2 \cdot s)\), \(s\) = sparsity[^28] | \(O(n)\) (FlashAttn-based) | No | Yes | Two-stage online filter: prediction + softmax-aware | arXiv:2502.18137 / DOI: 10.48550/arXiv.2502.18137[^28] |
| **FASA** | 2026 | \(O(t \cdot N_{\text{tip}} + N_{\text{fac}} \cdot d)\)[^18] | \(O(N_{\text{fac}} \cdot d)\) per head | No | Yes | Frequency-chunk sparsity in RoPE; dominant FCs | arXiv:2602.03152 / DOI: 10.48550/arXiv.2602.03152[^18] |
| **MSA** | 2026 | \(O(k \cdot B_k)\) fixed per query | \(O(n)\) GQA-sized KV cache | Yes | No | Block-sparse on GQA; per-group Top-k index branch; exp-free selection | arXiv:2606.13392 / DOI: 10.48550/arXiv.2606.13392 |
| **SparDA** | 2026 | \(O(k \cdot B_k)\) fixed per query | \(O(n)\) GQA-sized KV cache | Yes | No | Decoupled Forecast projection; lookahead block selection; CPU-GPU prefetch overlap | arXiv:2606.04511 / DOI: 10.48550/arXiv.2606.04511 |

> **Notes:** \(n\) = sequence length; \(w\) = window size; \(k\) = selected KV pairs; \(t\) = context length; \(d\) = head dimension; \(N_{\text{tip}}\) = number of dominant FCs; \(N_{\text{fac}}\) = number of selected tokens for focused attention; \(l'\) = selection block size; \(s\) = fraction of non-sparse blocks.

---

## References

1. [Sparse Transformer Algorithms (FlashAttention) - Emergent Mind](https://www.emergentmind.com/topics/sparse-transformer-algorithms-flash-attention) - Explore sparse transformer algorithms like FlashAttention that reduce computational costs and improv...

2. [Efficient Sparse Attention - Emergent Mind](https://www.emergentmind.com/topics/efficient-sparse-attention) - Efficient sparse attention techniques dynamically select key token pairs, reducing compute, memory, ...

3. [Sparse Attention Mechanisms - ApX Machine Learning](https://apxml.com/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/sparse-attention-mechanisms) - Sparse attention mechanisms aim to alleviate this bottleneck by reducing the number of query-key pai...

4. [Survey Paper Sparsity in transformers: A systematic literature review](https://www.sciencedirect.com/science/article/abs/pii/S092523122400239X) - Transformers have become the state-of-the-art architectures for various tasks in Natural Language Pr...

5. [[PDF] Generating Long Sequences with Sparse Transformers - arXiv](https://arxiv.org/pdf/1904.10509.pdf)

6. [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) - por R Child · 2019 · Mencionado por 2973 — In this paper we introduce sparse factorizations of the a...

7. [Sparse Attention Mechanisms in Large Language Models](https://www.clausiuspress.com/assets/default/article/2024/11/12/article_1731408067.pdf)

8. [[2004.05150] Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - por I Beltagy · 2020 · Mencionado por 6923 — We introduce the Longformer with an attention mechanism...

9. [Longformer: The Long-Document Transformer | Paper Notes](https://rabzelj.com/blog/longformer-the-long-document-transformer-paper-notes) - Paper notes for Longformer: The Long-Document Transformer.

10. [Paper notes: Longformer, The Long-Document Transformer](https://www.kripner.com/2004.05150-Longformer/)

11. [[PDF] arXiv:2004.05150v2 [cs.CL] 2 Dec 2020](https://arxiv.org/pdf/2004.05150.pdf) - For. Longformer, the dilated sliding window attention computes only a fixed number of the diagonals ...

12. [Paper page - Big Bird: Transformers for Longer Sequences](https://huggingface.co/papers/2007.14062) - Join the discussion on this paper page

13. [Big bird | Proceedings of the 34th International Conference on Neural Information Processing Systems](https://dl.acm.org/doi/abs/10.5555/3495724.3497174)

14. [Big Bird: Transformers for Longer Sequences](https://papers.nips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html)

15. [Understanding BigBird's Block Sparse Attention - Hugging Face](https://huggingface.co/blog/big-bird) - BigBird relies on block sparse attention instead of normal attention (ie BERT's attention) and can h...

16. [Big Bird: Transformers for Longer Sequences | Papers | HyperAI](https://beta.hyper.ai/en/papers/2007.14062) - Build the Future of Artificial Intelligence

17. [FASA: Frequency-Aware Sparse Attention - arXiv.org](https://arxiv.org/html/2602.03152v3) - Our implementation of FASA is built upon the HuggingFace Transformers library (Wolf et al., 2020) . ...

18. [FASA: Frequency-aware Sparse Attention](https://arxiv.org/abs/2602.03152v2) - The deployment of Large Language Models (LLMs) faces a critical bottleneck when handling lengthy inp...

19. [[PDF] FASA: Frequency-aware Sparse Attention - arXiv.org](https://arxiv.org/pdf/2602.03152.pdf) - Implementation Details Our implementation of FASA is built upon the HuggingFace Transformers library...

20. [HW-Aligned Sparse Attention Architecture For Efficient Long-Context Modeling (DeepSeek et al.)](https://semiengineering.com/hw-aligned-sparse-attention-architecture-for-efficient-long-context-modeling-deepseek-et-al/) - A new technical paper titled “Native Sparse Attention: Hardware-Aligned and Natively Trainable Spars...

21. [Native Sparse Attention: Hardware-Aligned and Natively ...](https://arxiv.org/pdf/2502.11089v1.pdf)

22. [Hardware-Aligned and Natively Trainable Sparse Attention - arXiv](https://arxiv.org/abs/2502.11089) - We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovati...

23. [Efficient Sparse Attention for Long-Range Transformers - arXiv](https://arxiv.org/abs/2406.16747) - We introduce SPARSEK Attention, a novel sparse attention mechanism designed to overcome these comput...

24. [Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers](http://arxiv.org/abs/2406.16747) - Accommodating long sequences efficiently in autoregressive Transformers, especially within an extend...

25. [Efficient Sparse Attention for Long-Range Transformers](https://huggingface.co/papers/2406.16747) - Join the discussion on this paper page

26. [Sparser is Faster and Less is More: Efficient Sparse](https://arxiv.org/pdf/2406.16747v1.pdf)

27. [Accurate Sparse Attention Accelerating Any Model Inference](https://huggingface.co/papers/2502.18137) - Join the discussion on this paper page

28. [Accurate and Training-free Sparse Attention Accelerating Any Model ...](https://arxiv.org/abs/2502.18137) - In this paper, we propose SpargeAttn, a universal sparse and quantized attention for any model. Our ...

29. [Accurate Sparse Attention Accelerating Any Model Inference - arXiv](https://arxiv.org/html/2502.18137v1) - In this paper, we propose SpargeAttn, a universal sparse and quantized attention for any model. Our ...

30. [MiniMax Sparse Attention - arXiv](https://arxiv.org/abs/2606.13392) - Blockwise sparse attention built on GQA with a lightweight Index Branch that scores KV blocks and independently selects a Top-k subset per GQA group.

31. [SparDA: Sparse Decoupled Attention - arXiv](https://arxiv.org/abs/2606.04511) - Introduces a fourth per-layer projection (Forecast) alongside Q/K/V to predict KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch.

