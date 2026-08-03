"""SpargeAttn (Zhang et al., 2025; arXiv:2502.18137).

Implements a training-free two-stage block-level sparse attention filter.
Stage 1 predicts low-value attention blocks using mean-pooled block
representations and a score threshold. Stage 2 applies a softmax-aware
pruning pass that further removes blocks whose post-softmax attention
maxima fall below the threshold. The method requires no training or
fine-tuning and can be applied directly to pre-trained dense models.

.. note::
    This module is **eval-only** in this repository. The enclosing
    ``HybridLayer`` raises a ``RuntimeError`` if it is invoked in
    training mode (``model.train()``).

Reference:
    Zhang, Z., Liu, Y., Peng, B., Gao, J., Li, Y., Wang, Z., & Sun, M. (2025).
    "SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference."
    arXiv:2502.18137.
"""

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class SpargeAttention(nn.Module):
    """Two-stage training-free block-level sparse attention filter.

    Operates in two stages:

    1. **Block score prediction**: Queries and keys are partitioned into
       blocks of size ``block_size``. Mean-pooled block representations
       are used to compute coarse block-level attention scores. Blocks
       with scores below ``threshold`` are pruned.
    2. **Softmax-aware pruning**: After computing full attention over the
       surviving blocks, the post-softmax attention weights are examined
       at block granularity. Blocks whose maximum attention weight falls
       below ``threshold`` are zeroed out.

    Diagonal (self-attention) entries are always preserved. In decoder mode,
    a causal mask is additionally applied.

    .. note::
        This module is **eval-only**. The enclosing ``HybridLayer`` raises
        a ``RuntimeError`` if called in training mode.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        block_size (int): Size of each block for coarse filtering.
        threshold (float): Score threshold for both stages of pruning.
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking.

    Reference:
        Zhang, Z., Liu, Y., Peng, B., Gao, J., Li, Y., Wang, Z., & Sun, M. (2025).
        "SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference."
        arXiv:2502.18137.
    """

    def __init__(self, config):
        """Initialize SpargeAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                sparge_block_size (int, optional): Block size for coarse
                    filtering. Defaults to 64.
                sparge_threshold (float, optional): Score threshold for both
                    pruning stages. Defaults to 0.01.
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
            raise ValueError("hidden_size must be divisible by num_heads for SpargeAttention")

        self.block_size = max(1, int(getattr(config, "sparge_block_size", 64)))
        self.threshold = float(getattr(config, "sparge_threshold", 0.01))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute two-stage block-level sparse attention.

        Stage 1: Partitions queries and keys into blocks, computes mean-pooled
        block scores, and prunes blocks below ``threshold``. Stage 2: Computes
        full attention over surviving blocks, then zeroes out blocks whose
        post-softmax maxima fall below ``threshold``. Diagonal entries are
        always preserved.

        In decoder mode, a causal mask is applied to the block-level mask.

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

        bs = self.block_size
        n_blocks = (seq_len + bs - 1) // bs
        padded_len = n_blocks * bs
        pad_len = padded_len - seq_len

        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        q_blocks = q.view(bsz, self.num_heads, n_blocks, bs, self.head_dim)
        k_blocks = k.view(bsz, self.num_heads, n_blocks, bs, self.head_dim)

        q_mean = q_blocks.mean(dim=3)
        k_mean = k_blocks.mean(dim=3)
        block_scores = torch.matmul(q_mean, k_mean.transpose(-2, -1)) / math.sqrt(self.head_dim)
        block_mask = block_scores > self.threshold

        full_mask = block_mask.unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, bs, -1, bs)
        full_mask = full_mask.reshape(bsz, self.num_heads, padded_len, padded_len)

        eye = torch.eye(padded_len, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        full_mask = full_mask | eye

        if self.mode == "decoder":
            causal = torch.tril(torch.ones(padded_len, padded_len, dtype=torch.bool, device=x.device))
            full_mask = full_mask & causal.unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~full_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_blocks = attn.view(bsz, self.num_heads, n_blocks, bs, n_blocks, bs)
        block_max = attn_blocks.amax(dim=(3, 5))
        softmax_mask = block_max > self.threshold
        softmax_full = softmax_mask.unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, bs, -1, bs)
        softmax_full = softmax_full.reshape(bsz, self.num_heads, padded_len, padded_len)
        softmax_full = softmax_full | eye

        attn = attn.masked_fill(~softmax_full, 0.0)

        out = torch.matmul(attn, v)
        out = out[:, :, :seq_len, :]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
