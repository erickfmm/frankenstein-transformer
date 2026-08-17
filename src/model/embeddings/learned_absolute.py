"""Learned absolute positional encoding.

Adds a learnable per-position embedding to the token embedding, as used in
BERT and the original Vision Transformer.

References:
    Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale", arXiv:2010.11929.
    Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding", arXiv:1810.04805.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedAbsolutePE(nn.Module):
    """Learnable absolute positional encoding.

    Stores a parameter tensor of shape ``(1, max_len, hidden_size)`` and adds a
    slice of it to the token embedding. The parameter is initialized with
    truncated normal noise.

    References:
        Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words:
        Transformers for Image Recognition at Scale", arXiv:2010.11929.
        Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional
        Transformers for Language Understanding", arXiv:1810.04805.

    Args:
        hidden_size: Hidden dimensionality of the model.
        max_len: Maximum sequence length covered by the learned table.
        init_std: Standard deviation of the truncated normal initializer.

    Attributes:
        pos_embed: Learnable positional embedding parameter of shape
            ``(1, max_len, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int = 512,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=init_std)

    def add(self, embedding: torch.Tensor) -> torch.Tensor:
        """Add the learned positional embedding to an embedding tensor.

        Args:
            embedding: Tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Tensor of same shape as ``embedding`` with the learned positional
            embedding added along the sequence dimension.
        """
        return embedding + self.pos_embed[:, : embedding.size(1)]

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