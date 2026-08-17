"""Grouped-head laTenT Attention (GTA).

Implements Grouped-Head laTenT Attention, arXiv:2506.17286 (Sun et al.
2025). GTA attacks the redundancy that attention maps across heads
exhibit high similarity (much of the per-head computation is
unnecessary) and that the value cache can be heavily compressed. It
combines two components:

1. **Shared attention map mechanism**: a single attention score tensor
   is computed per *group* of heads and reused across all heads in the
   group, shrinking the key cache.
2. **Nonlinear value decoder with learned projections**: the value cache
   is compressed into a low-rank latent space by a down-projection, and
   reconstructed by a non-linear (silu) decoder before the output
   projection, further cutting memory.

The paper reports GTA cuts attention FLOPs by up to 62.5% versus GQA
and shrinks the KV cache by up to 70%, while avoiding the extra
overhead of Multi-Head Latent Attention, achieving a 2x end-to-end
inference speedup.

Reference:
    Sun, L., Deng, C., Jiang, J., Wu, X., Zhang, H., Chen, L., Ni, L.,
    & Wang, J. (2025). "GTA: Grouped-head latenT Attention".
    arXiv:2506.17286.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GTAAttention(nn.Module):
    """Grouped-Head laTenT Attention with shared maps + latent values.

    Args:
        config: Configuration object. Relevant attributes: hidden_size,
            num_heads, dropout, use_bitnet, mode, and optional
            ``gta_num_shared_groups`` (default ``num_heads // 4``, must
            divide ``num_heads``) and ``gta_value_latent_rank`` (default
            ``hidden_size // 2``).
        pos_encoder: Shared positional encoding module applied to query
            and key tensors. If ``None``, no PE is applied.

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of query heads ``H``.
        head_dim: Per-head dimensionality.
        num_groups: Number of head groups sharing an attention map.
        group_size: ``num_heads // num_groups``.
        value_latent_rank: Latent rank of the value cache.
        q_proj, k_proj: Query/key projections.
        dv_proj: Value down-projection ``hidden_size -> value_latent_rank``.
        uv_proj: Non-linear value decoder ``value_latent_rank ->
            num_heads*head_dim`` (silu activated).
        out_proj: Output projection.
        dropout, mode: As in the rest of the family.
        pos_encoder: Shared positional encoding module (or ``None``).
        pe_type: Positional encoding type string (lowercased).
        use_pe: Whether PE is enabled for this mixer.

    Raises:
        ValueError: If ``hidden_size`` not divisible by ``num_heads`` or
            ``gta_num_shared_groups`` not dividing ``num_heads``.
    """

    def __init__(self, config, pos_encoder=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for GTAAttention")
        num_groups = int(getattr(config, "gta_num_shared_groups", max(1, self.num_heads // 4)))
        if self.num_heads % num_groups != 0:
            raise ValueError("gta_num_shared_groups must divide num_heads")
        self.num_groups = num_groups
        self.group_size = self.num_heads // num_groups
        self.value_latent_rank = int(
            getattr(config, "gta_value_latent_rank", max(1, self.hidden_size // 2))
        )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.dv_proj = proj_cls(self.hidden_size, self.value_latent_rank, bias=False)
        self.uv_proj = proj_cls(self.value_latent_rank, self.num_heads * self.head_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "gta_attn_use_pe", True))

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None, pos_encoder=None) -> torch.Tensor:
        """Compute GTA attention with shared group maps and latent values.

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
        G, S = self.num_groups, self.group_size
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_lat = self.dv_proj(x)
        v = F.silu(self.uv_proj(v_lat)).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        from ..common import apply_pe_to_qk, apply_pe_to_scores
        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        q_groups = q.view(bsz, G, S, seq_len, self.head_dim).mean(dim=2)
        k_groups = k.view(bsz, G, S, seq_len, self.head_dim).mean(dim=2)
        attn_scores = (q_groups @ k_groups.transpose(-2, -1)) * self.scale
        attn_scores = apply_pe_to_scores(pe, self.pe_type, attn_scores, q_groups, self.use_pe)
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.unsqueeze(2).expand(bsz, G, S, seq_len, seq_len).reshape(
            bsz, self.num_heads, seq_len, seq_len
        )
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)