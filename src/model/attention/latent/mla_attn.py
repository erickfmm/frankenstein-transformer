"""Multi-Head Latent Attention (MLA + RoPE) for small models.

Implements the latent attention variant studied by Mehta et al. (2025),
arXiv:2506.09342, for small language models. MLA jointly compresses keys
and values into a single low-rank latent vector ``c_KV`` whose rank
``r_kv`` is strictly smaller than ``hidden_size``; only the latent is
cached and stored during inference, so the KV-cache footprint drops from
``2 * num_heads * head_dim`` to ``r_kv`` per token. The paper's Pareto-
optimal configuration is ``r_kv = hidden_size // 2`` plus rotary
positional embeddings (RoPE) applied to the *decompressed* queries and
keys, which recovers (and slightly exceeds) standard MHA quality while
halving the cache.

Formulation (per token ``x``):

    c_KV = W_DKV x                         # latent,  shape r_kv
    k    = W_UK c_KV                       # keys,    shape num_heads*head_dim
    v    = W_UV c_KV                       # values,  shape num_heads*head_dim
    q    = W_Q  x                          # queries, shape num_heads*head_dim
    [q, k] = RoPE([q, k])                  # rotary on decompressed Q/K
    out  = SDPA(q, k, v) * W_O             # standard softmax attention

Reference:
    Mehta, S., Dandekar, R., Dandekar, R., & Panat, S. (2025).
    "Latent Multi-Head Attention for Small Language Models",
    arXiv:2506.09342.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


class MLAAttention(nn.Module):
    """Multi-Head Latent Attention with RoPE (Mehta et al. 2025).

    Compresses the key-value cache into a low-rank latent vector via a
    shared down-projection ``W_DKV`` and two up-projections ``W_UK``,
    ``W_UV``. RoPE is applied to the decompressed query and key tensors,
    which the paper shows is essential to recover MHA quality on small
    models.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, ``mode`` and the
            optional ``mla_latent_rank`` (default ``hidden_size // 2``)
            and ``rope_base`` (default 10000.0, kept for backward compat).
        pos_encoder: Shared positional encoding module applied to the
            decompressed query and key tensors. If ``None``, no PE is
            applied.

    Attributes:
        hidden_size: Input embedding dimensionality.
        num_heads: Number of attention heads.
        head_dim: Dimensionality per head (``hidden_size // num_heads``).
        latent_rank: Rank ``r_kv`` of the joint key-value latent.
        rope_base: RoPE base frequency (kept for backward compat).
        pos_encoder: Shared positional encoding module (or ``None``).
        pe_type: Positional encoding type string (lowercased).
        use_pe: Whether PE is enabled for this mixer.
        dkv_proj: Down-projection ``hidden_size -> r_kv`` (latent).
        uk_proj: Key up-projection ``r_kv -> num_heads*head_dim``.
        uv_proj: Value up-projection ``r_kv -> num_heads*head_dim``.
        q_proj: Query projection ``hidden_size -> num_heads*head_dim``.
        out_proj: Output projection.
        dropout: Dropout layer.
        mode: ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
    """

    def __init__(self, config, pos_encoder=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for MLAAttention")
        self.latent_rank = int(getattr(config, "mla_latent_rank", max(1, self.hidden_size // 2)))
        self.rope_base = float(getattr(config, "rope_base", 10000.0))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.dkv_proj = proj_cls(self.hidden_size, self.latent_rank, bias=False)
        self.uk_proj = proj_cls(self.latent_rank, self.num_heads * self.head_dim, bias=False)
        self.uv_proj = proj_cls(self.latent_rank, self.num_heads * self.head_dim, bias=False)
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "mla_attn_use_pe", True))

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None, pos_encoder=None) -> torch.Tensor:
        """Compute latent attention with RoPE on decompressed Q/K.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoder. Defaults to ``0`` if ``None``.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this forward call.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        c_kv = self.dkv_proj(x)
        k = self.uk_proj(c_kv).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.uv_proj(c_kv).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = apply_pe_to_scores(pe, self.pe_type, attn_scores, q, self.use_pe)
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