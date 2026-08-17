"""No Positional Encoding (NoPE).

Implements an identity positional encoder that leaves the input tensor
unchanged. Without an explicit position signal the model must learn position
implicitly from the causal mask (in decoder-only models) or from the attention
pattern alone. Empirically this enables strong length generalization when
training with sufficient causal masking.

Reference:
    Kazemnejad et al. (2023), "The Impact of Positional Encoding on Length
    Generalization in Transformers", arXiv:2305.19466.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NoPE(nn.Module):
    """No Positional Encoding (identity pass-through).

    The module performs no transformation on the input. Position information is
    expected to emerge implicitly through the causal mask applied to attention
    scores, as studied by Kazemnejad et al. (2023). The class exists primarily
    to keep the positional-encoding interface uniform across attention blocks.

    Reference:
        Kazemnejad et al. (2023), "The Impact of Positional Encoding on Length
        Generalization in Transformers", arXiv:2305.19466.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility with other positional encodings).

        Returns:
            The input tensor ``x`` unchanged.
        """
        return x