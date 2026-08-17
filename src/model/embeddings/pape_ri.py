"""Parabolic Position Encoding (PaPE), rotation-invariant variant.

A simplified PaPE formulation that augments query and key tensors with a single
learnable coefficient per head, producing a rotation-invariant parabolic
position bias. Only the ``a`` coefficient is learned from the hidden state
    while per-head position scales are drawn from a small uniform distribution
and stored in a non-persistent buffer.

Reference:
    Oehrstroem et al. (2026), "Parabolic Position Encoding for
    Transformers", arXiv:2602.01418. This is the rotation-invariant variant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaPERI(nn.Module):
    """Parabolic Position Encoding (rotation-invariant variant).

    Uses a single learnable ``a`` coefficient per head computed from the hidden
    state, combined with fixed per-head position scales. The augmented query
    and key contain parabolic features that yield a rotation-invariant
    positional bias inside the attention dot product.

    Reference:
        Oehrstroem et al. (2026), "Parabolic Position Encoding for
        Transformers", arXiv:2602.01418.

    Args:
        hidden_size: Hidden dimensionality of the model.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_positions: Dimensionality of the input position tensor. ``2`` for
            the default rotation-invariant formulation.

    Attributes:
        hidden_size: Hidden dimensionality of the model.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_positions: Dimensionality of the input position tensor.
        num_pad: Number of padding elements added to align the augmented head
            size to a multiple of ``8`` (``0`` when already aligned).
        should_pad: Whether padding is required.
        a: Linear projection from the hidden state to per-head ``a``
            coefficients.
        position_scales: Non-persistent buffer of shape
            ``(1, num_heads, 1, 1)`` with per-head position scales.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_positions: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_positions = num_positions

        total_head_size = head_dim + 2 * num_positions + 1
        self.num_pad = 8 * math.ceil(total_head_size / 8) - total_head_size
        self.should_pad = self.num_pad != 0

        self.a = nn.Linear(hidden_size, num_heads, bias=False)

        scale = math.sqrt(1.0 / num_positions)
        position_scales = torch.empty(1, num_heads, 1, 1).uniform_(-scale, scale)
        self.register_buffer("position_scales", position_scales, persistent=False)

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

        For ``num_positions == 1`` returns the standard 1D arange replicated
        across the batch. For higher-dimensional position tensors the caller is
        expected to provide explicit positions.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target data type.

        Returns:
            Tensor of shape ``(batch_size, seq_len, num_positions)`` when
            ``num_positions == 1``; otherwise raises ``ValueError``.
        """
        if self.num_positions != 1:
            raise ValueError(
                "default_positions is only defined for num_positions=1; pass "
                "explicit positions for higher-dimensional position tensors."
            )
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        return pos.view(1, seq_len, 1).expand(batch_size, -1, -1)

    def encode_qk(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Augment query and key with rotation-invariant parabolic features.

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
        batch_size, seq_length, num_positions = positions.size()
        num_heads = self.num_heads

        positions = positions.unsqueeze(1)
        position_scales = self.position_scales.to(device=positions.device, dtype=positions.dtype)
        positions = positions * position_scales
        squared_positions = positions.pow(2)

        a = F.softplus(self.a(hidden_state))
        a = a.permute(0, 2, 1)
        a = a.unsqueeze(-1).expand(batch_size, num_heads, seq_length, num_positions)

        neg_squared = -squared_positions

        ones = torch.ones(
            (batch_size, num_heads, seq_length, 1),
            device=positions.device,
            dtype=positions.dtype,
        )

        query = torch.cat(
            [query, self._dot(a, neg_squared), a, a * 2 * positions],
            dim=-1,
        )
        key = torch.cat(
            [key, ones, neg_squared, positions],
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