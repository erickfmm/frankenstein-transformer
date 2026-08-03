"""Factorized token embedding with optional Conv1d pre-projection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.common import BitConv1d, BitLinear

if TYPE_CHECKING:
    from ..tormented_bert_frankestein import UltraConfig


class FactorizedEmbedding(nn.Module):
    """Factorized token embedding with optional Conv1d pre-projection.

    Reduces the embedding lookup dimension to ``factorized_embedding_dim``,
    then projects up to ``hidden_size`` via a linear (or BitLinear) layer.
    Optionally applies a 1D convolution over the embedding stream for local
    context smoothing before projection.

    Attributes:
        low_dim: Reduced embedding dimension.
        use_conv: Whether the Conv1d pre-projection is active.
        embedding: Token embedding lookup table.
        conv: Optional Conv1d layer for local context smoothing.
        proj: Linear (or BitLinear) projection from ``low_dim`` to
            ``hidden_size``.
    """

    def __init__(self, config: "UltraConfig"):
        """Initialize factorized embedding from an UltraConfig.

        Args:
            config: Model configuration. Reads ``factorized_embedding_dim``,
                ``vocab_size``, ``use_embedding_conv``,
                ``embedding_conv_kernel``, ``use_bitnet``, and
                ``hidden_size``.
        """
        super().__init__()
        self.low_dim = config.factorized_embedding_dim
        self.use_conv = config.use_embedding_conv
        self.embedding = nn.Embedding(config.vocab_size, self.low_dim)
        kernel = max(int(config.embedding_conv_kernel), 1)
        padding = kernel // 2
        use_bitconv = bool(getattr(config, "use_bitnet_conv", False))
        conv_cls = BitConv1d if (config.use_bitnet and use_bitconv) else nn.Conv1d
        self.conv = (
            conv_cls(self.low_dim, self.low_dim, kernel_size=kernel, padding=padding)
            if self.use_conv
            else None
        )
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.proj = proj_cls(self.low_dim, config.hidden_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs and optionally apply Conv1d + projection.

        Sequence length is forced to match ``input_ids`` after convolution
        to keep MLM labels aligned (even-kernel symmetric padding can shift
        length by +1).

        Args:
            input_ids: Integer token indices of shape ``(B, S)``.

        Returns:
            Tensor of shape ``(B, S, hidden_size)``.
        """
        x = self.embedding(input_ids)
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            if x.size(1) != input_ids.size(1):
                target_len = input_ids.size(1)
                if x.size(1) > target_len:
                    x = x[:, :target_len, :]
                else:
                    pad_len = target_len - x.size(1)
                    x = F.pad(x, (0, 0, 0, pad_len))
        return self.proj(x)
