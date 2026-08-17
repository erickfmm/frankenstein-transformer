"""Parabolic Position Encoding (PaPE), naive formulation.

Augments query/key tensors with parabolic positional features derived from a
small set of learnable position projections. The resulting attention scores
include terms of the form ``a * p^2 + b * p`` (a parabola in the relative
position), which allows the model to express position-dependent attention
patterns that are not limited to linear or rotation-based decays.

Reference:
    Oehrstroem et al. (2026), "Parabolic Position Encoding for
    Transformers", arXiv:2602.01418.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaPE(nn.Module):
    """Parabolic Position Encoding (naive formulation).

    Computes per-head parabolic positional features from a learnable linear
    projection of the input positions and a learnable linear projection of the
    hidden state that produces the parabola coefficients ``a`` and ``b``. The
    query and key tensors are augmented along the last dimension with the
    parabolic features so that ``q^T k`` contains terms of the form
    ``a * p^2 + b * p``.

    Reference:
        Oehrstroem et al. (2026), "Parabolic Position Encoding for
        Transformers", arXiv:2602.01418.

    Args:
        hidden_size: Hidden dimensionality of the model.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_parabolas: Number of learnable parabolic basis functions per head.
        num_positions: Dimensionality of the input position tensor (``1`` for
            1D language).

    Attributes:
        hidden_size: Hidden dimensionality of the model.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_parabolas: Number of learnable parabolic basis functions per head.
        num_positions: Dimensionality of the input position tensor.
        num_pad: Number of padding elements added to align the augmented head
            size to a multiple of ``8`` (``0`` when already aligned).
        should_pad: Whether padding is required.
        position: Linear projection from positions to per-head parabolic
            features.
        ab: Linear projection from the hidden state to the parabolic
            coefficients ``a`` and ``b``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_parabolas: int = 4,
        num_positions: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_parabolas = num_parabolas
        self.num_positions = num_positions

        total_head_size = head_dim + 3 * num_parabolas + 2
        self.num_pad = 8 * math.ceil(total_head_size / 8) - total_head_size
        self.should_pad = self.num_pad != 0

        self.position = nn.Linear(num_positions, num_heads * num_parabolas, bias=False)
        self.ab = nn.Linear(hidden_size, 2 * num_heads * num_parabolas, bias=False)

    def _dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Contract the last dimension of ``x`` and ``y`` element-wise.

        Args:
            x: Tensor of shape ``(..., M)``.
            y: Tensor of shape ``(..., M)`` broadcastable with ``x``.

        Returns:
            Tensor of shape ``(..., 1)`` with the element-wise product summed
            over the last dimension.
        """
        return (x * y).sum(dim=-1, keepdim=True)

    def default_positions(
        self,
        batch_size: int,
        seq_len: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Build the default 1D position tensor.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target data type.

        Returns:
            Tensor of shape ``(batch_size, seq_len, 1)`` with positions
            ``0, 1, ..., seq_len - 1`` replicated across the batch.
        """
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        return pos.view(1, seq_len, 1).expand(batch_size, -1, -1)

    def encode_qk(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Augment query and key with parabolic positional features.

        Args:
            hidden_state: Hidden tensor of shape ``(B, S, hidden_size)``.
            query: Query tensor of shape ``(B, H, S, head_dim)``.
            key: Key tensor of shape ``(B, H, S, head_dim)``.
            positions: Position tensor of shape ``(B, S, num_positions)``.

        Returns:
            Tuple ``(query_aug, key_aug)`` of tensors with augmented last
            dimension padded to a multiple of ``8`` when required.
        """
        positions = positions.float()
        batch_size, seq_length, _ = positions.size()
        positions = self.position(positions)
        positions = positions.view(batch_size, seq_length, self.num_heads, self.num_parabolas)
        positions = positions.permute(0, 2, 1, 3)

        squared_positions = positions.pow(2)

        batch_size = key.size(0)
        seq_length = key.size(2)

        positions = positions.expand(batch_size, -1, -1, -1)
        squared_positions = squared_positions.expand(batch_size, -1, -1, -1)

        ab = self.ab(hidden_state)
        ab = ab.view(batch_size, seq_length, 2, self.num_heads, self.num_parabolas)
        ab = ab.permute(2, 0, 3, 1, 4)

        a = F.softplus(ab[0])
        b = ab[1]

        neg_squared_positions = -squared_positions

        ones = torch.ones(
            (batch_size, self.num_heads, seq_length, 1),
            device=positions.device,
            dtype=positions.dtype,
        )

        query = torch.cat(
            [
                query,
                self._dot(a, neg_squared_positions),
                a,
                a * 2 * positions,
                self._dot(b, positions),
                b,
            ],
            dim=-1,
        )
        key = torch.cat(
            [key, ones, neg_squared_positions, positions, ones, -positions],
            dim=-1,
        )

        if self.should_pad:
            query = F.pad(query, (0, self.num_pad), "constant", 0)
            key = F.pad(key, (0, self.num_pad), "constant", 0)
        return query, key

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