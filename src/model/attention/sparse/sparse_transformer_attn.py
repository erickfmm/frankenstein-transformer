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

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


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

    def __init__(self, config):
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

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute factorized sparse attention.

        Projects queries, keys, and values, then applies strided and fixed
        sparse masks to the first and second halves of attention heads
        respectively. In decoder mode, both patterns are further constrained
        by a causal mask.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index for
                potential layer-specific behavior. Not used by this
                implementation.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape
        stride = self._resolved_stride(seq_len)

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
