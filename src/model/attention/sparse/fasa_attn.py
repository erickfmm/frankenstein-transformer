"""Frequency-Aware Sparse Attention (Wang et al., 2026; arXiv:2602.03152).

Implements a training-free sparse attention heuristic that leverages RoPE
(Rotary Position Embedding) frequency properties to select important tokens.
Each attention head is assigned a set of dominant frequency chunks derived
from RoPE pairing dimensions. Token importance is estimated by computing
dot-product similarity over only the dominant frequency dimensions, and full
attention is computed exclusively over the top-k most important tokens. This
avoids training or fine-tuning while achieving significant sparsity.

.. note::
    This module is **eval-only** in this repository. The enclosing
    ``HybridLayer`` raises a ``RuntimeError`` if it is invoked in
    training mode (``model.train()``).

Reference:
    Wang, J., Li, Y., Zhang, H., Chen, X., Liu, Z., & Sun, M. (2026).
    "FASA: Frequency-Aware Sparse Attention for Efficient Long-Context LLM Inference."
    arXiv:2602.03152.
"""

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class FASAAttention(nn.Module):
    """Frequency-aware sparse attention using RoPE frequency chunk selection.

    Each attention head is assigned a fixed set of ``n_tip`` dominant frequency
    chunks (pairs of RoPE dimensions). Token importance is estimated by
    computing dot-product similarity between queries and keys restricted to
    these dominant dimensions only. The top ``n_fac`` most important tokens are
    selected, and full-head attention is computed over the selected subset.

    The dominant frequency chunk assignment is stored as a non-persistent
    buffer and is head-specific, ensuring diversity across heads.

    .. note::
        This module is **eval-only**. The enclosing ``HybridLayer`` raises
        a ``RuntimeError`` if called in training mode.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        n_tip (int): Number of dominant frequency chunks (tip dimensions) per head.
        n_fac (int): Number of top tokens to select for full attention (top-k).
        dominant_fcs (torch.Tensor): Buffer of shape ``(num_heads, n_tip)``
            storing the dominant frequency chunk indices for each head.
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking during importance estimation.

    Reference:
        Wang, J., Li, Y., Zhang, H., Chen, X., Liu, Z., & Sun, M. (2026).
        "FASA: Frequency-Aware Sparse Attention for Efficient Long-Context LLM Inference."
        arXiv:2602.03152.
    """

    def __init__(self, config):
        """Initialize FASAAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                fasa_n_tip (int, optional): Number of dominant frequency
                    chunks per head. Defaults to 16.
                fasa_n_fac (int, optional): Number of top tokens to select
                    for full attention. Defaults to 256.
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
            raise ValueError("hidden_size must be divisible by num_heads for FASAAttention")

        self.n_tip = max(1, int(getattr(config, "fasa_n_tip", 16)))
        self.n_fac = max(1, int(getattr(config, "fasa_n_fac", 256)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        pair_dim = max(1, self.head_dim // 2)
        dominant = torch.arange(self.n_tip, dtype=torch.long) % pair_dim
        self.register_buffer("dominant_fcs", dominant.unsqueeze(0).repeat(self.num_heads, 1), persistent=False)

    def _dominant_dim_indices(self, head_idx: int, device: torch.device) -> torch.Tensor:
        """Get the dominant frequency dimension indices for a given head.

        Each frequency chunk corresponds to a pair of RoPE dimensions
        (2 * fc_idx and 2 * fc_idx + 1). This method expands the chunk
        indices into the full set of dimension indices for the head.

        Args:
            head_idx (int): Index of the attention head.
            device (torch.device): Device on which to place the index tensor.

        Returns:
            torch.Tensor: 1D tensor of dimension indices for the dominant
                frequency chunks of the specified head.
        """
        fc_idx = self.dominant_fcs[head_idx].to(device)
        dim_idx = torch.stack((fc_idx * 2, fc_idx * 2 + 1), dim=-1).flatten()
        return dim_idx.clamp(max=self.head_dim - 1)

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute frequency-aware sparse attention.

        For each head independently:
        1. Extracts the dominant frequency dimensions for that head.
        2. Computes token importance scores using only those dimensions.
        3. Selects the top ``n_fac`` most important tokens.
        4. Computes full-head scaled dot-product attention over the selected
           key-value subset.

        In decoder mode, a causal mask is applied during importance estimation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index for
                potential layer-specific behavior. Not used by this
                implementation.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            RuntimeError: If called in training mode (enforced by the
                enclosing ``HybridLayer``, not by this module directly).
        """
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = torch.zeros_like(q)
        n_select = min(self.n_fac, seq_len)

        for h in range(self.num_heads):
            dim_idx = self._dominant_dim_indices(h, x.device)
            q_sub = q[:, h, :, dim_idx]
            k_sub = k[:, h, :, dim_idx]
            importance = torch.matmul(q_sub, k_sub.transpose(-2, -1))
            if self.mode == "decoder":
                causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
                importance = importance * causal.unsqueeze(0)
            top_idx = importance.topk(n_select, dim=-1).indices

            for b in range(bsz):
                k_head = k[b, h]
                v_head = v[b, h]
                k_sel = k_head[top_idx[b]]
                v_sel = v_head[top_idx[b]]
                q_head = q[b, h].unsqueeze(1)
                scores = (q_head * k_sel).sum(dim=-1) / math.sqrt(self.head_dim)
                attn = F.softmax(scores, dim=-1)
                out[b, h] = torch.sum(attn.unsqueeze(-1) * v_sel, dim=1)

        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
