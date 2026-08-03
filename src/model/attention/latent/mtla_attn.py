"""Multi-head Temporal Latent Attention (MTLA).

Implements Multi-head Temporal Latent Attention, arXiv:2505.13544
(Deng & Woodland, 2025). MTLA extends Multi-Head Latent Attention (MLA)
by additionally compressing the KV cache along the **temporal**
dimension. A small hyper-network dynamically merges temporally adjacent
KV cache vectors: at each stride boundary the cache vectors of
``mtla_merge_factor`` consecutive tokens are averaged into a single
representative vector, shrinking the temporal length by that factor.
Because the compressed cache is shorter than the input sequence, a
**stride-aware causal mask** is required to keep parallel training
consistent with the inference behaviour: each query position can only
attend to merged slots whose temporal support ends at or before the
query position.

The paper reports MTLA achieves competitive performance versus
standard MHA across speech translation, speech recognition, speech
understanding and text summarisation while delivering, on an English-
German speech translation task, a 5.3x decoding speedup and an 8.3x
reduction in GPU memory usage versus MHA, with no quality loss.

Reference:
    Deng, K., & Woodland, P. C. (2025). "Multi-head Temporal Latent
    Attention". arXiv:2505.13544.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class MTLAAttention(nn.Module):
    """Multi-head Temporal Latent Attention with hyper-network merging.

    Args:
        config: Configuration object. Relevant attributes: hidden_size,
            num_heads, dropout, use_bitnet, mode, and optional
            ``mtla_latent_rank`` (default ``hidden_size // 2``),
            ``mtla_merge_factor`` (default 2, must be >= 1) and
            ``mtla_stride`` (default equals ``mtla_merge_factor``).

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of attention heads.
        head_dim: Per-head dimensionality.
        latent_rank: Latent compression rank (along feature axis).
        merge_factor: Number of consecutive KV cache entries merged
            into a single temporal slot.
        stride: Stride between merged slots.
        dkv_proj: Latent down-projection.
        uk_proj, uv_proj: Key/value up-projections.
        q_proj: Query projection.
        merge_proj: Hyper-network gate producing per-slot merge weights.
        out_proj: Output projection.
        dropout, mode: As in the rest of the family.

    Raises:
        ValueError: If ``hidden_size`` not divisible by ``num_heads``;
            if ``mtla_merge_factor`` < 1.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for MTLAAttention")
        self.latent_rank = int(getattr(config, "mtla_latent_rank", max(1, self.hidden_size // 2)))
        self.merge_factor = int(getattr(config, "mtla_merge_factor", 2))
        if self.merge_factor < 1:
            raise ValueError("mtla_merge_factor must be >= 1")
        self.stride = int(getattr(config, "mtla_stride", self.merge_factor))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.dkv_proj = proj_cls(self.hidden_size, self.latent_rank, bias=False)
        self.uk_proj = proj_cls(self.latent_rank, self.num_heads * self.head_dim, bias=False)
        self.uv_proj = proj_cls(self.latent_rank, self.num_heads * self.head_dim, bias=False)
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.merge_proj = proj_cls(self.latent_rank, self.latent_rank, bias=True)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5

    def _merge_temporal(self, latents: torch.Tensor) -> torch.Tensor:
        """Merge temporally adjacent latent vectors via hyper-network gating.

        Args:
            latents: ``(B, S, latent_rank)``.

        Returns:
            ``(B, S_compressed, latent_rank)`` where
            ``S_compressed = ceil((S - merge_factor) / stride) + 1``.
        """
        bsz, seq_len, rank = latents.shape
        m, s = self.merge_factor, self.stride
        if m == 1:
            return latents
        num_slots = max(1, (seq_len - m) // s + 1)
        merged = []
        for i in range(num_slots):
            start = i * s
            end = min(start + m, seq_len)
            chunk = latents[:, start:end]
            gate = torch.sigmoid(self.merge_proj(chunk.mean(dim=1, keepdim=True)))
            merged.append((chunk * gate).mean(dim=1))
        return torch.stack(merged, dim=1)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute temporal-latent attention with stride-aware causal mask.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        c_kv = self.dkv_proj(x)
        c_kv_merged = self._merge_temporal(c_kv)
        kv_len = c_kv_merged.size(1)
        k = self.uk_proj(c_kv_merged).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.uv_proj(c_kv_merged).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            m, s = self.merge_factor, self.stride
            mask = torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool)
            for i in range(seq_len):
                last_visible_slot = max(0, (i - m) // s) if i >= m else 0
                if i >= m:
                    last_visible_slot = (i - m) // s + 1
                else:
                    last_visible_slot = 0
                mask[i, last_visible_slot + 1:] = False
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)