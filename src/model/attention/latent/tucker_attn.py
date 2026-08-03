"""Tucker Attention: generalised low-rank attention.

Implements Tucker Attention, arXiv:2603.30033 (Klein et al. 2026).
Tucker Attention provides a unified low-rank view of Multi-Head
Attention (MHA), Grouped-Query Attention (GQA) and Multi-Head Latent
Attention (MLA): each is recovered as a special case of a Tucker-style
factorisation of the query, key and value weight tensors along the
``num_heads`` and ``hidden_size`` axes. Concretely, the canonical
``W_q, W_k, W_v`` of shape ``hidden_size -> num_heads * head_dim`` are
replaced by a small core tensor contracted against per-mode factor
matrices, exposing the *actual* ranks achieved by MHA, GQA and MLA.

Special cases:
    * MHA    : ``query_rank = key_rank = value_rank = hidden_size``.
    * GQA    : ``key_rank = value_rank < hidden_size`` (shared KV).
    * MLA    : ``key_rank = value_rank = latent_rank`` (joint KV latent).

The paper reports an order of magnitude fewer parameters than GQA and
MLA for comparable validation metrics on LLM and ViT test cases. Tucker
Attention is fully compatible with FlashAttention and RoPE.

Reference:
    Klein, T., Kusch, J., Sager, S., Schnake, S., & Schotthöfer, S.
    (2026). "Tucker Attention: A generalization of approximate attention
    mechanisms". arXiv:2603.30033.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class TuckerAttention(nn.Module):
    """Tucker-factorised low-rank attention (Klein et al. 2026).

    Args:
        config: Configuration object. Relevant attributes: hidden_size,
            num_heads, dropout, use_bitnet, mode, and optional
            ``tucker_query_rank``, ``tucker_key_rank``,
            ``tucker_value_rank`` (default ``hidden_size // 2`` for K
            and V, ``hidden_size`` for Q).

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of attention heads.
        head_dim: Per-head dimensionality.
        query_rank, key_rank, value_rank: Tucker ranks along the
            hidden axis for Q, K, V respectively.
        q_factor, k_factor, v_factor: Down-projections ``hidden_size ->
            rank``.
        q_core, k_core, v_core: Per-head core projections ``rank ->
            num_heads * head_dim``.
        out_proj: Output projection.
        dropout, mode: As in the rest of the family.

    Raises:
        ValueError: If ``hidden_size`` not divisible by ``num_heads``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for TuckerAttention")
        self.query_rank = int(getattr(config, "tucker_query_rank", self.hidden_size))
        self.key_rank = int(getattr(config, "tucker_key_rank", max(1, self.hidden_size // 2)))
        self.value_rank = int(getattr(config, "tucker_value_rank", max(1, self.hidden_size // 2)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_factor = proj_cls(self.hidden_size, self.query_rank, bias=False)
        self.k_factor = proj_cls(self.hidden_size, self.key_rank, bias=False)
        self.v_factor = proj_cls(self.hidden_size, self.value_rank, bias=False)
        self.q_core = proj_cls(self.query_rank, self.num_heads * self.head_dim, bias=False)
        self.k_core = proj_cls(self.key_rank, self.num_heads * self.head_dim, bias=False)
        self.v_core = proj_cls(self.value_rank, self.num_heads * self.head_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Tucker-factorised attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        q = self.q_core(self.q_factor(x)).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_core(self.k_factor(x)).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_core(self.v_factor(x)).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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