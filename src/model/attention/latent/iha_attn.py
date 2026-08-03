"""Interleaved Head Attention (IHA).

Implements Interleaved Head Attention, arXiv:2602.21371 (Duvvuri et al.
2026). Standard Multi-Head Attention (MHA) is limited by a fundamental
linear scaling constraint: ``H`` heads produce exactly ``H`` independent
attention matrices with no communication between heads *during* the
attention computation. This is problematic for multi-step reasoning,
where correct answers depend on aggregating evidence from multiple
parts of the context and composing latent token-to-token relations over
a chain of intermediate inferences.

IHA enables cross-head mixing by constructing ``P`` *pseudo-heads* per
head (typically ``P = H``), where each pseudo query/key/value is a
learned linear combination of all ``H`` original queries, keys and
values respectively. Interactions between pseudo-query and pseudo-key
heads induce up to ``P**2`` attention patterns per head with modest
parameter overhead ``O(H**2 * P)``. The paper proves IHA uses
``Theta(sqrt(k) * n**2)`` parameters vs. ``Theta(k * n**2)`` for MHA on
the synthetic Polynomial task, and ``ceil(sqrt(N_max))`` heads vs.
``N_max`` for MHA on the order-sensitive CPM-3 task. Empirically, IHA
improves Multi-Key retrieval on RULER by 10-20% (4k-16k context) and
improves GSM8K by 5.8% and MATH-500 by 2.8% (Majority Vote) over full
attention after reasoning fine-tuning.

Reference:
    Duvvuri, S. S., Ekbote, C., Bansal, R., Tiwari, R., Khatri, D.,
    Brandfonbrener, D., Liang, P., Dhillon, I., & Zaheer, M. (2026).
    "Interleaved Head Attention". arXiv:2602.21371.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class IHAAttention(nn.Module):
    """Interleaved Head Attention with learned cross-head mixing.

    Args:
        config: Configuration object. Relevant attributes: hidden_size,
            num_heads, dropout, use_bitnet, mode, and optional
            ``iha_num_pseudo_heads`` (default ``num_heads``).

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of original attention heads ``H``.
        head_dim: Per-head dimensionality.
        num_pseudo_heads: Number of pseudo-heads per head ``P``.
        q_proj, k_proj, v_proj: Projections producing the ``H`` original
            Q/K/V tensors.
        q_mix, k_mix, v_mix: Per-mode mixing matrices ``H*P x H``
            producing the pseudo-head Q/K/V.
        out_proj: Output projection (after averaging pseudo-head outputs).
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
            raise ValueError("hidden_size must be divisible by num_heads for IHAAttention")
        self.num_pseudo_heads = int(getattr(config, "iha_num_pseudo_heads", self.num_heads))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.q_mix = nn.Parameter(torch.zeros(self.num_heads * self.num_pseudo_heads, self.num_heads))
        self.k_mix = nn.Parameter(torch.zeros(self.num_heads * self.num_pseudo_heads, self.num_heads))
        self.v_mix = nn.Parameter(torch.zeros(self.num_heads * self.num_pseudo_heads, self.num_heads))
        nn.init.normal_(self.q_mix, std=0.02)
        nn.init.normal_(self.k_mix, std=0.02)
        nn.init.normal_(self.v_mix, std=0.02)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute interleaved-head attention with cross-head mixing.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        H, P, d = self.num_heads, self.num_pseudo_heads, self.head_dim
        q = self.q_proj(x).view(bsz, seq_len, H, d).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, H, d).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, H, d).transpose(1, 2)

        q_p = torch.einsum("ph,bhsd->bpsd", self.q_mix, q).reshape(bsz, H * P, seq_len, d)
        k_p = torch.einsum("ph,bhsd->bpsd", self.k_mix, k).reshape(bsz, H * P, seq_len, d)
        v_p = torch.einsum("ph,bhsd->bpsd", self.v_mix, v).reshape(bsz, H * P, seq_len, d)

        attn_scores = (q_p @ k_p.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out_p = attn_weights @ v_p
        out = out_p.view(bsz, H, P, seq_len, d).mean(dim=2).transpose(1, 2).contiguous().view(
            bsz, seq_len, H * d
        )
        return self.out_proj(out)