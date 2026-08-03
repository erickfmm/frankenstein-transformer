"""Grouped-Query Attention (GQA, Ainslie et al. 2023).

Projects queries to ``num_heads`` heads but keys and values to ``num_kv_heads``
heads.  Each key/value head is shared across ``num_heads / num_kv_heads``
query heads, interpolating between multi-head attention (``num_kv_heads ==
num_heads``) and multi-query attention (``num_kv_heads == 1``).

Reference:
    Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints", arXiv:2305.13245.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with configurable key-value heads.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``num_kv_heads``, ``dropout``, ``use_bitnet``, and
            optionally ``mode`` (``"encoder"`` or ``"decoder"``).

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        ValueError: If ``num_kv_heads`` is not in ``[1, num_heads]``.
        ValueError: If ``num_heads`` is not divisible by ``num_kv_heads``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = int(getattr(config, "num_kv_heads", 1))
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 1 <= int(self.num_kv_heads) <= self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be in [1, num_heads ({self.num_heads})]"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads "
                f"({self.num_kv_heads})"
            )
        self.num_groups = self.num_heads // self.num_kv_heads

        kv_hidden = self.num_kv_heads * self.head_dim
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, kv_hidden, bias=False)
        self.v_proj = proj_cls(self.hidden_size, kv_hidden, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

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

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
