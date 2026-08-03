"""Rotary Position Embedding (RoPE).

Implements rotary position encoding by rotating query and key vectors by
position-dependent angles. Each consecutive pair of dimensions is treated as
a 2D plane and rotated by an angle proportional to the token position and
inversely proportional to a geometric progression of frequencies. This
encodes relative position information directly into the attention dot product
without requiring learned or absolute position embeddings.

Reference:
    Su et al. (2024), "RoFormer: Enhanced Transformer with Rotary Position
    Embedding", arXiv:2104.09864.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Position Embedding over consecutive dimension pairs.

    Applies a 2D rotation to each pair of adjacent dimensions in the input
    tensor. The rotation angle for pair ``i`` at position ``p`` is::

        theta_i(p) = p * scaling * base^{-i / (pair_dim - 1)}

    This enables the attention dot product ``q^T k`` to depend only on the
    relative position between tokens, as the rotation satisfies::

        R(p_q)^T R(p_k) = R(p_k - p_q)

    Reference:
        Su et al. (2024), "RoFormer: Enhanced Transformer with Rotary Position
        Embedding", arXiv:2104.09864.

    Args:
        head_dim: Dimensionality of each attention head. Must be even for
            proper pairing.
        base: Base frequency for the geometric progression of rotation
            frequencies. Defaults to ``10000.0``.
        scaling: Position scaling factor applied to token indices before
            computing angles. Defaults to ``1.0``.

    Attributes:
        head_dim: Total head dimensionality.
        pair_dim: Number of dimension pairs (``head_dim // 2``).
        base: Base frequency for inverse frequency computation.
        scaling: Position scaling factor.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0, scaling: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.pair_dim = head_dim // 2
        self.base = base
        self.scaling = scaling

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        """Apply rotary position encoding.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility with other positional encodings).

        Returns:
            Tensor of same shape as ``x`` with rotary position encoding
            applied. If ``pair_dim == 0``, returns ``x`` unchanged.
        """
        if self.pair_dim == 0:
            return x

        _, _, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        pos = torch.arange(seq_len, device=device, dtype=dtype) * self.scaling
        if self.pair_dim > 1:
            idx = torch.arange(self.pair_dim, device=device, dtype=dtype)
            inv_freq = self.base ** (-idx / (self.pair_dim - 1))
        else:
            inv_freq = torch.ones(1, device=device, dtype=dtype)

        angles = pos[:, None] * inv_freq[None, :]
        sin_term = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos_term = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x_even = x[..., : self.pair_dim * 2 : 2]
        x_odd = x[..., 1 : self.pair_dim * 2 : 2]

        y_even = x_even * cos_term - x_odd * sin_term
        y_odd = x_even * sin_term + x_odd * cos_term

        y = x.clone()
        y[..., : self.pair_dim * 2 : 2] = y_even
        y[..., 1 : self.pair_dim * 2 : 2] = y_odd
        return y
