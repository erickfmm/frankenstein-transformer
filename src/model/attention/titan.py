"""Titans memory-augmented attention.

Implements the core attention component of the Titans architecture, which
augments standard multi-head attention with a neural memory module that learns
to memorize at test time. This module provides the attention pathway that
interacts with the surprise-driven long-term memory. Uses a shared positional
encoding module (HoPE or RoPE) applied to query and key projections.

Reference:
    Behrouz et al. (2025), "Titans: Learning to Memorize at Test Time",
    arXiv:2501.00663.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


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
            (``"hope"`` or ``"rope"``), and optionally ``mode``
            (``"encoder"`` or ``"decoder"``).
        pos_encoder: Shared positional encoding module (``HoPE`` or
            ``RoPE``) to apply to queries and keys. If ``None``, no PE is
            applied.

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
        pos_encoder: Shared positional encoding module (``HoPE`` or
            ``RoPE``), or ``None``.
        pe_type: Positional encoding type string (lowercased).
        use_pe: Whether PE is enabled for this mixer.
        dropout: Dropout layer applied to attention weights.
        mode: ``"encoder"`` for bidirectional attention, ``"decoder"`` for
            causal (upper-triangular) masking.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
    """

    def __init__(self, config, pos_encoder=None):
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

        positional_encoding = str(getattr(config, "positional_encoding", "rope")).lower()
        if positional_encoding is None:
            positional_encoding = "hope" if bool(getattr(config, "use_hope", True)) else "rope"

        self.pos_encoder = pos_encoder
        self.pe_type = positional_encoding
        self.use_pe = bool(getattr(config, "titan_attn_use_pe", True))

        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None, pos_encoder=None) -> torch.Tensor:
        """Compute Titans multi-head attention with positional encoding.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoder for layer-dependent scaling. Defaults to ``0`` if
                ``None``.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this forward call.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape
        logical_layer_idx = logical_layer_idx or 0
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx, self.use_pe)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = apply_pe_to_scores(pe, self.pe_type, attn_scores, q, self.use_pe)
        if self.mode == "decoder":
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
