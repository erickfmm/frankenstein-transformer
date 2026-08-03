"""Multi-Head Low-Rank Attention (MLRA).

Implements Multi-Head Low-Rank Attention, arXiv:2603.02188 (Liu et al.
2026). MLRA extends Multi-Head Latent Attention (MLA) by splitting the
single shared latent head into ``mlra_num_latent_heads`` independent
latent sub-spaces, each of rank ``r = latent_rank / num_latent_heads``.
The key property is **partitionability**: each latent sub-head can be
assigned to a different tensor-parallel device, so each device loads
only ``1 / num_latent_heads`` of the KV cache instead of the whole
cache (which MLA forces). This enables efficient 4-way TP decoding and
delivers a 2.8x decoding speedup over MLA in the paper's experiments,
while reaching state-of-the-art perplexity and downstream task scores.

Formulation (per token ``x``): the latent cache is split into
``L = num_latent_heads`` disjoint sub-vectors ``c_1, ..., c_L`` (concatenated
into a single ``c_KV`` of rank ``r``); keys and values are reconstructed
per sub-head via per-block up-projections and concatenated into the full
``num_heads * head_dim`` dimension before standard softmax attention.

Reference:
    Liu, S., Peng, H., Zhang, Z., Chen, Z., & Guo, Y. (2026).
    "Multi-Head Low-Rank Attention". arXiv:2603.02188.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class MLRAAttention(nn.Module):
    """Multi-Head Low-Rank Attention with partitionable latent heads.

    Args:
        config: Configuration object. Relevant attributes: hidden_size,
            num_heads, dropout, use_bitnet, mode, and optional
            ``mlra_latent_rank`` (default ``hidden_size // 2``) and
            ``mlra_num_latent_heads`` (default 4, must divide
            ``latent_rank`` evenly).

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of query heads.
        head_dim: Per-head dimensionality.
        latent_rank: Total latent rank.
        num_latent_heads: Number of disjoint latent sub-spaces (``L``).
        sub_rank: ``latent_rank // num_latent_heads``.
        dkv_proj: Latent down-projection ``hidden_size -> r``.
        uk_projs, uv_projs: ModuleList of per-sub-head up-projections.
        q_proj: Query projection.
        out_proj: Output projection.
        dropout, mode: As in the rest of the family.

    Raises:
        ValueError: If ``hidden_size`` not divisible by ``num_heads``;
            if ``mlra_num_latent_heads`` does not divide ``latent_rank``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for MLRAAttention")
        self.latent_rank = int(getattr(config, "mlra_latent_rank", max(1, self.hidden_size // 2)))
        self.num_latent_heads = int(getattr(config, "mlra_num_latent_heads", 4))
        if self.latent_rank % self.num_latent_heads != 0:
            raise ValueError("mlra_num_latent_heads must divide mlra_latent_rank")
        self.sub_rank = self.latent_rank // self.num_latent_heads

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.dkv_proj = proj_cls(self.hidden_size, self.latent_rank, bias=False)
        uk_target = self.num_heads * self.head_dim
        if uk_target % self.num_latent_heads != 0:
            raise ValueError("num_heads*head_dim must be divisible by mlra_num_latent_heads")
        self.sub_out = uk_target // self.num_latent_heads
        self.uk_projs = nn.ModuleList(
            [proj_cls(self.sub_rank, self.sub_out, bias=False) for _ in range(self.num_latent_heads)]
        )
        self.uv_projs = nn.ModuleList(
            [proj_cls(self.sub_rank, self.sub_out, bias=False) for _ in range(self.num_latent_heads)]
        )
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute MLRA attention with partitioned latent sub-heads.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        c_kv = self.dkv_proj(x)
        c_chunks = c_kv.view(bsz, seq_len, self.num_latent_heads, self.sub_rank)
        k_parts = [
            uk(c_chunks[:, :, i]).view(bsz, seq_len, 1, -1)
            for i, uk in enumerate(self.uk_projs)
        ]
        v_parts = [
            uv(c_chunks[:, :, i]).view(bsz, seq_len, 1, -1)
            for i, uv in enumerate(self.uv_projs)
        ]
        k = torch.cat(k_parts, dim=2).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = torch.cat(v_parts, dim=2).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)