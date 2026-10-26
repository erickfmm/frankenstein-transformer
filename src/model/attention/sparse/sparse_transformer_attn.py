"""Sparse Transformer attention (Child et al., 2019; arXiv:1904.10509).

Implements factorized sparse attention patterns that reduce the quadratic O(n^2)
cost of dense self-attention to O(n * sqrt(n)). The factorized pattern splits
attention heads into two groups: half use a strided pattern (attending to every
``stride``-th position plus a local window) and half use a fixed pattern
(attending to a contiguous block plus summary columns). This preserves both local
and long-range information flow while drastically reducing memory and compute.

Reference:
    Child, R., Gray, S., Radford, A., & Sutskever, I. (2019).
    "Generating Long Sequences with Sparse Transformers."
    arXiv:1904.10509.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


class SparseTransformerAttention(nn.Module):
    """Factorized sparse attention with strided and fixed patterns.

    Splits attention heads into two groups. The first half uses a strided
    pattern that attends to a local window plus every ``stride``-th position.
    The second half uses a fixed pattern that attends to a contiguous block
    plus summary columns at the end of each stride block. When ``stride`` is
    not explicitly set, it defaults to ``sqrt(seq_len)``, yielding O(n * sqrt(n))
    complexity.

    In decoder mode, both patterns are additionally masked with a causal
    (lower-triangular) mask to prevent attending to future positions.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        stride (int): Stride length for sparse patterns. If 0, auto-resolved to
            ``sqrt(seq_len)`` at runtime.
        summary_cols (int): Number of summary columns per stride block in the
            fixed pattern.
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking on top of the sparse patterns.

    Reference:
        Child, R., Gray, S., Radford, A., & Sutskever, I. (2019).
        "Generating Long Sequences with Sparse Transformers."
        arXiv:1904.10509.
    """

    def __init__(self, config, pos_encoder=None):
        """Initialize SparseTransformerAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                sparse_transformer_stride (int, optional): Stride length.
                    Defaults to 0 (auto-resolve to sqrt(seq_len)).
                sparse_transformer_summary_cols (int, optional): Summary
                    columns per stride block. Defaults to 1.
                mode (str, optional): ``"encoder"`` or ``"decoder"``.
                    Defaults to ``"encoder"``.
            pos_encoder: Optional shared positional encoding module.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for SparseTransformerAttention")

        self.stride = int(getattr(config, "sparse_transformer_stride", 0))
        self.summary_cols = max(1, int(getattr(config, "sparse_transformer_summary_cols", 1)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "sparse_transformer_attn_use_pe", True))

    def _resolved_stride(self, seq_len: int) -> int:
        """Resolve the effective stride for the current sequence length.

        If ``self.stride`` was explicitly set to a positive value, it is used
        directly. Otherwise, the stride is computed as ``sqrt(seq_len)``,
        matching the O(n * sqrt(n)) complexity target of the original paper.

        Args:
            seq_len (int): Current sequence length.

        Returns:
            int: The resolved stride value, at least 1.
        """
        if self.stride > 0:
            return self.stride
        return max(1, int(math.sqrt(max(1, seq_len))))

    def _strided_mask(self, seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
        """Build the strided attention mask.

        For each query position ``i``, the strided pattern attends to:
        - A local window of the preceding ``stride - 1`` positions.
        - Every ``stride``-th position from ``i % stride`` up to ``i``.

        Args:
            seq_len (int): Sequence length.
            stride (int): Stride length for the pattern.
            device (torch.device): Device on which to create the mask tensor.

        Returns:
            torch.Tensor: Boolean mask of shape ``(seq_len, seq_len)`` where
                ``True`` indicates allowed attention.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            indices = list(range(max(0, i - stride + 1), i + 1))
            indices += list(range(i % stride, i + 1, stride))
            if indices:
                mask[i, torch.tensor(sorted(set(indices)), device=device)] = True
        return mask

    def _fixed_mask(self, seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
        """Build the fixed attention mask.

        For each query position ``i``, the fixed pattern attends to:
        - A contiguous block from ``block_start`` to ``i`` (inclusive), where
          ``block_start`` is the beginning of the stride block containing ``i``.
        - Summary columns: the last ``summary_cols`` positions of every stride
          block up to ``i``.

        Args:
            seq_len (int): Sequence length.
            stride (int): Stride length for block partitioning.
            device (torch.device): Device on which to create the mask tensor.

        Returns:
            torch.Tensor: Boolean mask of shape ``(seq_len, seq_len)`` where
                ``True`` indicates allowed attention.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            block_start = (i // stride) * stride
            block_end = min(i + 1, block_start + stride)
            indices = list(range(block_start, block_end))
            indices += [
                j
                for j in range(i + 1)
                if (j % stride) >= max(0, stride - self.summary_cols)
            ]
            if indices:
                mask[i, torch.tensor(sorted(set(indices)), device=device)] = True
        return mask

    def make_cache_slot(self, meta, cache):
        """Unbounded KV slot (strided/fixed patterns reach arbitrary past
        positions; pattern evaluated over ``positions``)."""
        from ...cache.slots import KVSlot

        kwargs = cache.codec_slot_kwargs()
        return KVSlot(meta, codec=kwargs["codec"], policy=None, num_sinks=kwargs["num_sinks"])

    def _pattern_strided(self, stride: int):
        def pattern(q_pos, kv_pos):
            local = (kv_pos > q_pos - stride) & (kv_pos <= q_pos)
            aligned = ((q_pos - kv_pos) % stride == 0) & (kv_pos <= q_pos)
            return local | aligned

        return pattern

    def _pattern_fixed(self, stride: int):
        def pattern(q_pos, kv_pos):
            block_start = (q_pos // stride) * stride
            in_block = (kv_pos >= block_start) & (kv_pos <= q_pos)
            summary = (kv_pos % stride) >= max(0, stride - self.summary_cols)
            summary = summary & (kv_pos <= q_pos) & (kv_pos >= 0)
            return in_block | summary

        return pattern

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None, pos_encoder=None, cache=None, cache_position: Optional[int] = None) -> torch.Tensor:
        """Compute factorized sparse attention.

        Projects queries, keys, and values, then applies strided and fixed
        sparse masks to the first and second halves of attention heads
        respectively. In decoder mode, both patterns are further constrained
        by a causal mask.

        Cached decoding evaluates both patterns as pure functions of the
        absolute positions over the cached ``positions`` metadata (B5 fix —
        summary columns change role as blocks complete, so the pattern must
        be position-based, never index-based). The auto stride uses the
        TOTAL accumulated length, matching the full forward's stride.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index. Not used.
            pos_encoder: Optional positional encoding module override.
            cache: Optional KV slot for incremental decoding (``None`` runs
                the legacy full-recompute path, D4).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            import torch.nn.functional as _F
            from ...cache.hooks import causal_mask_from_positions, kv_decode_append, pattern_mask, positions_for

            offset = cache_position
            positions = positions_for(offset, seq_len, x.device)
            q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe, position_offset=offset)
            K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)
            total = int(kv_positions.numel())
            stride = max(1, int(self.stride)) if self.stride > 0 else max(1, int(math.sqrt(max(1, total))))
            scores = (q @ K.transpose(-2, -1)) * self.scale

            if seq_len == 1:
                p = kv_positions.new_full((1,), int(kv_positions[-1]))
                allowed_s = self._pattern_strided(stride)(p, kv_positions)
                allowed_f = self._pattern_fixed(stride)(p, kv_positions)
                half = max(1, self.num_heads // 2)
                scores[:, :half] = scores[:, :half].masked_fill(
                    (~allowed_s).unsqueeze(0).unsqueeze(0), float("-inf")
                )
                scores[:, half:] = scores[:, half:].masked_fill(
                    (~allowed_f).unsqueeze(0).unsqueeze(0), float("-inf")
                )
            else:
                mask_s = pattern_mask(positions, kv_positions, self._pattern_strided(stride))
                mask_f = pattern_mask(positions, kv_positions, self._pattern_fixed(stride))
                half = max(1, self.num_heads // 2)
                scores[:, :half] = scores[:, :half].masked_fill(mask_s.unsqueeze(0).unsqueeze(0), float("-inf"))
                scores[:, half:] = scores[:, half:].masked_fill(mask_f.unsqueeze(0).unsqueeze(0), float("-inf"))

            scores = apply_pe_to_scores(
                pe, self.pe_type, scores, q, self.use_pe,
                kv_len=K.size(-2),
                q_offset=0 if offset is None else int(offset),
                kv_positions=kv_positions,
            )
            attn = _F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ V).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.out_proj(out)

        stride = self._resolved_stride(seq_len)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        strided_mask = self._strided_mask(seq_len, stride, x.device)
        fixed_mask = self._fixed_mask(seq_len, stride, x.device)
        if self.mode == "decoder":
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            strided_mask = strided_mask & causal
            fixed_mask = fixed_mask & causal

        half = max(1, self.num_heads // 2)
        scores[:, :half] = scores[:, :half].masked_fill(~strided_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        scores[:, half:] = scores[:, half:].masked_fill(~fixed_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        scores = apply_pe_to_scores(pe, self.pe_type, scores, q, self.use_pe)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
