"""MiniMax Sparse Attention (MSA).

Implements MiniMax Sparse Attention, arXiv:2606.13392 (Lai et al. 2026,
MiniMax). MSA is a blockwise sparse attention built on top of Grouped
Query Attention (GQA). A lightweight **Index Branch** scores key-value
blocks and independently selects a Top-k subset of blocks for each GQA
group, enabling group-specific sparse retrieval while maintaining
efficient block-level execution; the **Main Branch** then performs exact
block-sparse softmax attention over only the selected blocks.

The paper deploys MSA on a 109B-parameter natively multimodal model
(MiniMax-M3), where it performs on par with full GQA while reducing
per-token attention compute by 28.4x at 1M context and delivering 14.2x
prefill / 7.6x decode wall-clock speedups on H800.

Key design principles:

* **Block partition**: tokens are partitioned into contiguous blocks of
  size ``msa_block_size`` (paper default 128).
* **Per-group Top-k selection**: ``msa_topk_blocks`` blocks are selected
  per GQA group; the block containing the query position is *always*
  forced into the selected set (one of the k slots is reserved for the
  local block).
* **Exp-free Top-k**: softmax is order-preserving, so the Index Branch
  bypasses ``exp``/``sum`` and feeds raw block scores directly into the
  Top-k selection.
* **Index projections are detached**: ``stopgrad(X)`` on the Index
  Branch input means the LM loss never trains the indexer directly; in
  the production setting a KL alignment loss updates the indexer (see
  the paper's training recipe). This module exposes the
  ``last_kl_alignment_loss`` attribute so a trainer can add it to the
  main loss.

Reference:
    Lai, X., Xu, W., Yang, Y., et al. (2026). "MiniMax Sparse
    Attention". arXiv:2606.13392.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class MSAAttention(nn.Module):
    """MiniMax Sparse Attention with per-group block-sparse selection.

    Args:
        config: Configuration object. Relevant attributes: hidden_size,
            num_heads, num_kv_heads (default ``num_heads // 8`` or 1),
            dropout, use_bitnet, mode, and optional ``msa_block_size``
            (default 128), ``msa_topk_blocks`` (default 16),
            ``msa_index_dim`` (default 64), ``msa_kl_loss_weight``
            (default 0.0).

    Attributes:
        hidden_size: Input dimensionality.
        num_heads: Number of query heads ``H_q``.
        num_kv_heads: Number of key-value heads ``H_kv``.
        head_dim: Per-head dimensionality.
        group_size: ``num_heads // num_kv_heads``.
        block_size: Block size ``B_k``.
        topk_blocks: ``k`` selected blocks per GQA group.
        index_dim: Index head dimension ``d_idx``.
        kl_loss_weight: Weight of the KL alignment loss (0 disables).
        q_proj, k_proj, v_proj: Main-branch GQA projections.
        q_idx_proj, k_idx_proj: Index-branch projections.
        out_proj: Output projection.
        last_kl_alignment_loss: Most recent KL alignment loss (or None).
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
            raise ValueError("hidden_size must be divisible by num_heads for MSAAttention")
        self.num_kv_heads = int(getattr(config, "num_kv_heads", max(1, self.num_heads // 8)))
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads for MSAAttention")
        self.group_size = self.num_heads // self.num_kv_heads
        self.block_size = int(getattr(config, "msa_block_size", 128))
        self.topk_blocks = int(getattr(config, "msa_topk_blocks", 16))
        self.index_dim = int(getattr(config, "msa_index_dim", 64))
        self.kl_loss_weight = float(getattr(config, "msa_kl_loss_weight", 0.0))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        router_cls = BitLinear if (config.use_bitnet and getattr(config, "bitnet_routers", False)) else nn.Linear
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, kv_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, kv_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_idx_proj = router_cls(self.hidden_size, self.num_kv_heads * self.index_dim, bias=False)
        self.k_idx_proj = router_cls(self.hidden_size, self.index_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.head_dim ** -0.5
        self.idx_scale = self.index_dim ** -0.5
        self.last_kl_alignment_loss: Optional[torch.Tensor] = None

    def _block_max_scores(self, q_idx: torch.Tensor, k_idx: torch.Tensor, num_blocks: int) -> torch.Tensor:
        """Compute per-block max-pooled index scores (causal).

        Args:
            q_idx: ``(B, S, H_kv, d_idx)`` index queries.
            k_idx: ``(B, S, 1, d_idx)`` index keys.
            num_blocks: Number of blocks ``B``.

        Returns:
            ``(B, S, H_kv, num_blocks)`` per-block scores.
        """
        bsz, seq_len, h_kv, d_idx = q_idx.shape
        bs = self.block_size
        # Token-level scores: (B, S, H_kv, S)
        token_scores = torch.einsum("bshd,btd->bsht", q_idx, k_idx.squeeze(2)) * self.idx_scale
        # Pad to a multiple of bs along the key axis so we can reshape into blocks
        pad = (bs - seq_len % bs) % bs
        if pad > 0:
            token_scores = F.pad(token_scores, (0, pad), value=float("-inf"))
        padded_len = seq_len + pad
        n_blocks = padded_len // bs
        # (B, S, H_kv, n_blocks, bs)
        block_scores = token_scores.view(bsz, seq_len, h_kv, n_blocks, bs)
        # Apply causal mask: query at position i can see key at position j iff j <= i
        positions = torch.arange(seq_len, device=q_idx.device)
        kv_positions = torch.arange(padded_len, device=q_idx.device)
        visible = positions.view(seq_len, 1) >= kv_positions.view(1, padded_len)
        visible = visible.view(1, seq_len, 1, n_blocks, bs)
        neg_inf = torch.finfo(token_scores.dtype).min
        block_scores = torch.where(visible, block_scores, torch.full_like(block_scores, neg_inf))
        return block_scores.max(dim=-1).values[:, :, :, :num_blocks]

    def _select_blocks(self, block_scores: torch.Tensor) -> torch.Tensor:
        """Top-k block selection per query per GQA group with forced local block.

        Args:
            block_scores: ``(B, S, H_kv, num_blocks)``.

        Returns:
            ``(B, S, H_kv, k)`` selected block indices.
        """
        bsz, seq_len, h_kv, num_blocks = block_scores.shape
        k = self.topk_blocks
        positions = torch.arange(seq_len, device=block_scores.device)
        local_block = (positions // self.block_size).view(1, seq_len, 1)
        bs_local = block_scores.clone()
        bs_local.scatter_(-1, local_block.unsqueeze(-1), float("inf"))
        topk_scores, topk_idx = torch.topk(bs_local, k, dim=-1)
        replace_idx = torch.full_like(topk_idx, 0)
        replace_idx[..., 0] = local_block.expand_as(topk_idx[..., 0])
        topk_idx = torch.where(
            torch.arange(k, device=block_scores.device).view(1, 1, 1, k) == 0,
            replace_idx,
            topk_idx,
        )
        return topk_idx

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
            sel = (selected_blocks == b).any(dim=-1)
            sel_q = sel.unsqueeze(1).unsqueeze(-1)
            if not sel_q.any():
                continue
            scores_visible = scores
            if self.mode == "decoder":
                positions = torch.arange(seq_len, device=q.device)
                kv_positions = torch.arange(start, end, device=q.device)
                visible = positions.view(seq_len, 1) >= kv_positions.view(1, end - start)
                causal_mask = ~visible
                scores_visible = scores_visible.masked_fill(
                    causal_mask.view(1, 1, seq_len, end - start), float("-inf")
                )
            attn = F.softmax(scores_visible, dim=-1)
            attn = self.dropout(attn)
            out_blk = attn @ v_blk
            sel_b = sel.view(bsz, seq_len, h_kv, 1).expand(bsz, seq_len, h_kv, G).reshape(bsz, seq_len, h_q)
            sel_b = sel_b.transpose(1, 2).unsqueeze(-1)
            out = torch.where(sel_b, out_blk, out)
        return out

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute MiniMax Sparse Attention.

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

        with torch.no_grad():
            x_detached = x.detach()
        q_idx = self.q_idx_proj(x_detached).view(bsz, seq_len, self.num_kv_heads, self.index_dim)
        k_idx = self.k_idx_proj(x_detached).view(bsz, seq_len, 1, self.index_dim)

        block_scores = self._block_max_scores(q_idx, k_idx, num_blocks)
        selected_blocks = self._select_blocks(block_scores)

        out = self._main_branch_attention(q, k, v, selected_blocks, num_blocks)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)