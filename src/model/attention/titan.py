"""Titans memory-augmented attention.

Implements the core attention component of the Titans architecture, which
augments standard multi-head attention with a neural memory module that learns
to memorize at test time. This module provides the attention pathway that
interacts with the surprise-driven long-term memory. Uses HoPE or RoPE
positional encoding on query and key projections.

Reference:
    Behrouz et al. (2025), "Titans: Learning to Memorize at Test Time",
    arXiv:2501.00663.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear
from ..embeddings.hope import HoPE
from ..embeddings.rope import RoPE


class TitanAttention(nn.Module):
    """Multi-head attention with positional encoding for Titans architecture.

    Projects the input into query, key, and value tensors, applies HoPE or
    RoPE positional encoding to queries and keys, computes scaled dot-product
    attention with softmax, and aggregates values. Supports both encoder
    (bidirectional) and decoder (causal) modes. Designed to work alongside
    Titans' neural memory module for handling contexts beyond 2M tokens.

    Reference:
        Behrouz et al. (2025), "Titans: Learning to Memorize at Test Time",
        arXiv:2501.00663.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, ``positional_encoding``
            (``"hope"`` or ``"rope"``), ``hope_base``, ``hope_damping``,
            ``rope_base``, ``rope_scaling``, and optionally ``mode``
            (``"encoder"`` or ``"decoder"``).

    Attributes:
        hidden_size: Dimensionality of the input and output embeddings.
        num_heads: Number of parallel attention heads.
        head_dim: Dimensionality of each attention head
            (``hidden_size // num_heads``).
        scale: Scaling factor ``1 / sqrt(head_dim)`` applied to dot products.
        q_proj: Linear (or BitLinear) projection for queries.
        k_proj: Linear (or BitLinear) projection for keys.
        v_proj: Linear (or BitLinear) projection for values.
        out_proj: Linear (or BitLinear) output projection.
        pos_encoder: Positional encoding module (``HoPE`` or ``RoPE``).
        dropout: Dropout layer applied to attention weights.
        mode: ``"encoder"`` for bidirectional attention, ``"decoder"`` for
            causal (upper-triangular) masking.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``, or
            if ``positional_encoding`` is not one of ``{"hope", "rope"}``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for TitanAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        positional_encoding = getattr(config, "positional_encoding", None)
        if positional_encoding is None:
            positional_encoding = "hope" if bool(getattr(config, "use_hope", True)) else "rope"
        positional_encoding = str(positional_encoding).lower()

        if positional_encoding == "hope":
            self.pos_encoder = HoPE(self.head_dim, base=config.hope_base, damping=config.hope_damping)
        elif positional_encoding == "rope":
            self.pos_encoder = RoPE(
                self.head_dim,
                base=getattr(config, "rope_base", 10_000.0),
                scaling=getattr(config, "rope_scaling", 1.0),
            )
        else:
            raise ValueError(
                "positional_encoding must be one of {'hope', 'rope'} for TitanAttention"
            )

        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Titans multi-head attention with positional encoding.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoder for layer-dependent scaling. Defaults to ``0`` if
                ``None``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape
        logical_layer_idx = logical_layer_idx or 0

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.pos_encoder(q, logical_layer_idx=logical_layer_idx)
        k = self.pos_encoder(k, logical_layer_idx=logical_layer_idx)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
