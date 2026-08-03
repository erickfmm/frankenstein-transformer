"""Standard scaled dot-product multi-head attention.

Implements the original Transformer attention mechanism with row-wise softmax
normalization. Complexity is O(n^2 * d) during training and O(n) per-step
inference with a KV cache. Provides full token-to-token routing with perfect
expressiveness.

Reference:
    Vaswani et al. (2017), "Attention Is All You Need", arXiv:1706.03762.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear


class StandardAttention(nn.Module):
    """Standard multi-head scaled dot-product attention (Vaswani et al. 2017).

    Projects the input into query, key, and value tensors, computes scaled
    dot-product attention scores, applies row-wise softmax, and aggregates
    values weighted by the resulting attention distribution. Supports both
    encoder (bidirectional) and decoder (causal) modes.

    Complexity:
        Training: O(n^2 * d). Inference: O(n) per step with KV cache.

    Reference:
        Vaswani et al. (2017), "Attention Is All You Need", arXiv:1706.03762.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, and optionally
            ``mode`` (``"encoder"`` or ``"decoder"``).

    Attributes:
        hidden_size: Dimensionality of the input and output embeddings.
        num_heads: Number of parallel attention heads.
        head_dim: Dimensionality of each attention head
            (``hidden_size // num_heads``).
        scale: Scaling factor ``1 / sqrt(head_dim)`` applied to dot products.
        q_proj: Linear (or BitLinear) projection for queries.
        k_proj: Linear (or BitLinear) projection for keys.
        v_proj: Linear (or BitLinear) projection for values.
        out_proj: Linear (or BitLinear) output projection.
        dropout: Dropout layer applied to attention weights.
        mode: ``"encoder"`` for bidirectional attention, ``"decoder"`` for
            causal (upper-triangular) masking.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for StandardAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute standard multi-head attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility with other attention modules).

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
