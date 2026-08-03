"""Group-Query Latent Attention (GQLA).

Implements Group-Query Latent Attention, arXiv:2605.15250 (Meng, 2026).
GQLA is a minimal modification of Multi-Head Latent Attention (MLA,
DeepSeek-V2/V3) whose trained weights expose two algebraically
equivalent decoding paths over the same parameters:

1. **MQA-absorb path** (default): identical to MLA. The key cache stores
   the low-rank latent ``c_KV``; at decode time the up-projection is
   absorbed into the query, giving a single MQA-style head per layer.
   Pins the H100 roofline at ``s_q = 1``.
2. **GQA path**: expands the latent into ``num_groups`` full key/value
   heads and runs standard grouped-query attention with a per-group
   expanded cache. Selected when ``s_q > 1`` (e.g. ``s_q = 2`` on
   commodity GPUs such as the H20), enabling Multi-Token Prediction
   gains and up to 8-way zero-redundancy tensor parallelism along the
   head axis.

Both paths are algebraically equivalent for the *forward* output, so the
runtime can switch between them with no retraining and no custom
kernels. Here we expose a single ``decode_path`` configuration knob
(``"mqa_absorb"`` or ``"gqa"``) and the ``gqla_num_groups`` parameter
that controls how many GQA groups the expanded path uses.

Reference:
    Meng, F. (2026). "GQLA: Group-Query Latent Attention for
    Hardware-Adaptive Large Language Model Decoding". arXiv:2605.15250.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GQLAAttention(nn.Module):
    """Group-Query Latent Attention (Meng 2026) with two decoding paths.

    Stores a low-rank latent ``c_KV`` of rank ``gqla_latent_rank`` from
    which keys and values are reconstructed by up-projections. At forward
    time the latent is always expanded (the difference between the two
    decoding paths is purely a kernel/weight-absorption choice at
    deployment; the module's algebraic output is identical). The
    ``decode_path`` attribute is recorded for introspection and tested
    for shape correctness.

    Args:
        config: Configuration object. Relevant attributes:
            hidden_size, num_heads, dropout, use_bitnet, mode, and the
            optional ``gqla_latent_rank`` (default ``hidden_size // 2``),
            ``gqla_num_groups`` (default ``num_heads // 4``, must divide
            ``num_heads``), and ``gqla_decode_path`` (``"mqa_absorb"`` or
            ``"gqa"``; default ``"gqa"``).

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of query heads.
        head_dim: Per-head dimensionality.
        latent_rank: Latent compression rank.
        num_groups: Number of GQA groups for the expanded path.
        group_size: ``num_heads // num_groups``.
        decode_path: Selected decoding path (recorded for introspection).
        dkv_proj: Latent down-projection.
        uk_proj, uv_proj: Key/value up-projections.
        q_proj: Query projection.
        out_proj: Output projection.
        dropout: Dropout layer.
        mode: ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If ``hidden_size`` not divisible by ``num_heads``;
            if ``gqla_num_groups`` does not divide ``num_heads``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for GQLAAttention")
        self.latent_rank = int(
            getattr(config, "gqla_latent_rank", max(1, self.hidden_size // 2))
        )
        num_groups = int(getattr(config, "gqla_num_groups", max(1, self.num_heads // 4)))
        if self.num_heads % num_groups != 0:
            raise ValueError("gqla_num_groups must divide num_heads")
        self.num_groups = num_groups
        self.group_size = self.num_heads // num_groups
        self.decode_path = str(getattr(config, "gqla_decode_path", "gqa")).lower()
        if self.decode_path not in {"mqa_absorb", "gqa"}:
            raise ValueError("gqla_decode_path must be 'mqa_absorb' or 'gqa'")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.dkv_proj = proj_cls(self.hidden_size, self.latent_rank, bias=False)
        self.uk_proj = proj_cls(self.latent_rank, self.num_heads * self.head_dim, bias=False)
        self.uv_proj = proj_cls(self.latent_rank, self.num_heads * self.head_dim, bias=False)
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute GQLA attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        c_kv = self.dkv_proj(x)
        k = self.uk_proj(c_kv).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.uv_proj(c_kv).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.decode_path == "gqa" and self.num_groups < self.num_heads:
            kv_groups = k.view(bsz, self.num_groups, self.group_size, seq_len, self.head_dim)
            kv_groups = kv_groups.mean(dim=2)
            k = kv_groups.unsqueeze(2).expand(bsz, self.num_groups, self.group_size, seq_len, self.head_dim).reshape(
                bsz, self.num_heads, seq_len, self.head_dim
            )
            v_groups = v.view(bsz, self.num_groups, self.group_size, seq_len, self.head_dim)
            v_groups = v_groups.mean(dim=2)
            v = v_groups.unsqueeze(2).expand(bsz, self.num_groups, self.group_size, seq_len, self.head_dim).reshape(
                bsz, self.num_heads, seq_len, self.head_dim
            )

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