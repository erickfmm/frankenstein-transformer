"""SparDA: Sparse Decoupled Attention.

Implements SparDA, arXiv:2606.04511 (Fu et al. 2026, NVIDIA). SparDA is
a decoupled sparse attention architecture that introduces a fourth
per-layer projection -- the **Forecast** -- alongside Query, Key and
Value. The Forecast predicts the KV blocks needed by the next layer,
enabling a *lookahead* selection that overlaps CPU-to-GPU prefetch with
the current-layer execution. Because the Forecast is decoupled from the
attention query, the GQA implementation uses one Forecast head per GQA
group, reducing selection overhead versus the original multi-head
selector.

SparDA adds <0.5% parameters and trains only the Forecast projections
by matching the original selector's attention distribution. On two
sparse-pretrained 8B models, SparDA matches or slightly improves
accuracy and delivers up to 1.25x prefill speedup and 1.7x decode
speedup over the sparse-attention offload baseline; by enabling larger
feasible batch sizes on a single GPU, SparDA reaches up to 5.3x higher
decode throughput than the non-offload sparse baseline.

This implementation provides the *forward* semantics of the Forecast-
based block selection: a Forecast projection scores KV blocks per GQA
group, top-k blocks are selected, and the main attention runs over the
selected blocks. The CPU-offload prefetch overlap is a runtime/kernel
concern outside the module's algebraic definition.

Reference:
    Fu, Y., Xiao, G., Dong, X., Han, S., & Villa, O. (2026).
    "SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM
    Inference". arXiv:2606.04511. NVIDIA. Code: NVlabs/SparDA.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class SparDAAttention(nn.Module):
    """SparDA decoupled sparse attention with a Forecast projection.

    Args:
        config: Configuration object. Relevant attributes: hidden_size,
            num_heads, num_kv_heads (default ``num_heads // 8`` or 1),
            dropout, use_bitnet, mode, and optional
            ``sparda_block_size`` (default 128), ``sparda_topk_blocks``
            (default 16), ``sparda_forecast_dim`` (default 64).

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Per-head dimensionality.
        group_size: ``num_heads // num_kv_heads``.
        block_size: KV block size.
        topk_blocks: Number of KV blocks selected per GQA group.
        forecast_dim: Dimensionality of the Forecast projection.
        q_proj, k_proj, v_proj: Standard GQA projections.
        forecast_proj: Forecast projection (one head per GQA group).
        out_proj: Output projection.
        dropout, mode: As in the rest of the family.

    Raises:
        ValueError: If ``hidden_size`` not divisible by ``num_heads`` or
            ``num_heads`` not divisible by ``num_kv_heads``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for SparDAAttention")
        self.num_kv_heads = int(getattr(config, "num_kv_heads", max(1, self.num_heads // 8)))
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads for SparDAAttention")
        self.group_size = self.num_heads // self.num_kv_heads
        self.block_size = int(getattr(config, "sparda_block_size", 128))
        self.topk_blocks = int(getattr(config, "sparda_topk_blocks", 16))
        self.forecast_dim = int(getattr(config, "sparda_forecast_dim", 64))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        router_cls = BitLinear if (config.use_bitnet and getattr(config, "bitnet_routers", False)) else nn.Linear
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, kv_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, kv_dim, bias=False)
        self.forecast_proj = router_cls(self.hidden_size, self.num_kv_heads * self.forecast_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5
        self.forecast_scale = self.forecast_dim ** -0.5

    def _forecast_block_scores(self, x: torch.Tensor, num_blocks: int) -> torch.Tensor:
        """Score KV blocks per GQA group via the Forecast projection.

        Args:
            x: ``(B, S, hidden_size)``.
            num_blocks: Number of KV blocks.

        Returns:
            ``(B, S, H_kv, num_blocks)`` block scores.
        """
        bsz, seq_len, _ = x.shape
        bs = self.block_size
        forecast = self.forecast_proj(x).view(bsz, seq_len, self.num_kv_heads, self.forecast_dim)
        block_rep = F.avg_pool1d(
            forecast.transpose(1, 2).reshape(bsz * self.num_kv_heads, self.forecast_dim, seq_len),
            kernel_size=bs, stride=bs, ceil_mode=True,
        )
        block_rep = block_rep.reshape(bsz, self.num_kv_heads, self.forecast_dim, -1)
        block_rep = block_rep.permute(0, 3, 1, 2)
        scores = torch.einsum("bshd,bthd->bsht", forecast, block_rep) * self.forecast_scale
        return scores

    def _main_branch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_blocks: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """Block-sparse softmax attention over selected blocks.

        Args:
            q: ``(B, H_q, S, d_h)``.
            k, v: ``(B, H_kv, S, d_h)``.
            selected_blocks: ``(B, S, H_kv, k)``.
            num_blocks: Number of blocks.

        Returns:
            ``(B, H_q, S, d_h)``.
        """
        bsz, h_q, seq_len, d_h = q.shape
        h_kv = k.size(1)
        G = h_q // h_kv
        bs = self.block_size
        k_expanded = k.unsqueeze(2).expand(bsz, h_kv, G, seq_len, d_h).reshape(bsz, h_q, seq_len, d_h)
        v_expanded = v.unsqueeze(2).expand(bsz, h_kv, G, seq_len, d_h).reshape(bsz, h_q, seq_len, d_h)

        out = torch.zeros_like(q)
        for b in range(num_blocks):
            start = b * bs
            end = min(start + bs, seq_len)
            if end <= start:
                continue
            k_blk = k_expanded[:, :, start:end, :]
            v_blk = v_expanded[:, :, start:end, :]
            scores = (q @ k_blk.transpose(-2, -1)) * self.scale
            if self.mode == "decoder":
                positions = torch.arange(seq_len, device=q.device)
                kv_positions = torch.arange(start, end, device=q.device)
                visible = positions.view(seq_len, 1) >= kv_positions.view(1, end - start)
                scores = scores.masked_fill(
                    (~visible).view(1, 1, seq_len, end - start), float("-inf")
                )
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_blk = attn @ v_blk
            sel = (selected_blocks == b).any(dim=-1)
            sel_b = sel.view(bsz, seq_len, h_kv, 1).expand(bsz, seq_len, h_kv, G).reshape(bsz, seq_len, h_q)
            sel_b = sel_b.transpose(1, 2).unsqueeze(-1)
            out = torch.where(sel_b, out_blk, out)
        return out

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute SparDA decoupled sparse attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        bs = self.block_size
        num_blocks = (seq_len + bs - 1) // bs

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        block_scores = self._forecast_block_scores(x, num_blocks)
        k_select = min(self.topk_blocks, num_blocks)
        _, selected_blocks = torch.topk(block_scores, k_select, dim=-1)

        out = self._main_branch_attention(q, k, v, selected_blocks, num_blocks)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)