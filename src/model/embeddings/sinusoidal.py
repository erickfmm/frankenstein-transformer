"""Sinusoidal positional encodings.

Provides two classical positional encodings based on the sin/cos formulation
of Vaswani et al. (2017): an additive absolute encoding applied to the token
embedding, and a rotary variant that rotates query/key vectors using
absolute-position sin/cos angles (as opposed to RoPE which uses relative
angles).

Reference:
    Vaswani et al. (2017), "Attention Is All You Need", arXiv:1706.03762.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalAbsolute(nn.Module):
    """Additive sinusoidal absolute positional encoding.

    Precomputes a fixed (non-learnable) sin/cos table of shape
    ``(1, max_len, hidden_size)`` and adds it to the token embedding before the
    transformer blocks. The angular frequency for dimension ``i`` is
    ``base^{-2i / hidden_size}``.

    Reference:
        Vaswani et al. (2017), "Attention Is All You Need",
        arXiv:1706.03762.

    Args:
        hidden_size: Hidden dimensionality of the model.
        max_len: Maximum sequence length for which the table is precomputed.
        base: Base of the geometric frequency progression.
        scale: Position scaling factor applied to token indices.

    Attributes:
        hidden_size: Hidden dimensionality of the model.
        max_len: Maximum sequence length covered by the precomputed table.
        base: Base of the geometric frequency progression.
        scale: Position scaling factor.
        pe: Precomputed sin/cos table of shape ``(1, max_len, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int = 512,
        base: float = 10_000.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.base = base
        self.scale = scale

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(base) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term * scale)
        pe[:, 1::2] = torch.cos(position * div_term * scale)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def add(self, embedding: torch.Tensor) -> torch.Tensor:
        """Add the precomputed positional encoding to an embedding tensor.

        Args:
            embedding: Tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Tensor of same shape as ``embedding`` with the sinusoidal
            positional encoding added along the sequence dimension.
        """
        return embedding + self.pe[:, : embedding.size(1)]

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


class SinusoidalRotary(nn.Module):
    """Rotary positional encoding with absolute-position sin/cos angles.

    Rotates each consecutive pair of dimensions in the input tensor by an angle
    proportional to the absolute token position. Unlike RoPE, which relies on
    relative-position rotation, this variant uses the absolute sin/cos angles
    from the original Vaswani et al. (2017) formulation.

    Reference:
        Vaswani et al. (2017), "Attention Is All You Need",
        arXiv:1706.03762.

    Args:
        head_dim: Dimensionality of each attention head. Must be even.
        max_len: Maximum sequence length for which the sin/cos table is
            precomputed.
        base: Base of the geometric frequency progression.
        scale: Position scaling factor applied to token indices.

    Attributes:
        head_dim: Total head dimensionality.
        pair_dim: Number of dimension pairs (``head_dim // 2``).
        max_len: Maximum sequence length covered by the precomputed table.
        base: Base of the geometric frequency progression.
        scale: Position scaling factor.
        sin: Precomputed sine table of shape ``(max_len, pair_dim)``.
        cos: Precomputed cosine table of shape ``(max_len, pair_dim)``.
    """

    def __init__(
        self,
        head_dim: int,
        max_len: int = 512,
        base: float = 10_000.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.pair_dim = head_dim // 2
        self.max_len = max_len
        self.base = base
        self.scale = scale

        pos = torch.arange(0, max_len).float() * scale
        if self.pair_dim > 1:
            idx = torch.arange(self.pair_dim).float()
            inv_freq = base ** (-idx / (self.pair_dim - 1))
        else:
            inv_freq = torch.ones(1)
        angles = pos[:, None] * inv_freq[None, :]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        """Apply rotary positional encoding using absolute-position angles.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility with other positional encodings).

        Returns:
            Tensor of same shape as ``x`` with rotary positional encoding
            applied. If ``pair_dim == 0``, returns ``x`` unchanged.
        """
        if self.pair_dim == 0:
            return x

        _, _, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if seq_len > self.max_len:
            pos = torch.arange(seq_len, device=device, dtype=dtype) * self.scale
            if self.pair_dim > 1:
                idx = torch.arange(self.pair_dim, device=device, dtype=dtype)
                inv_freq = self.base ** (-idx / (self.pair_dim - 1))
            else:
                inv_freq = torch.ones(1, device=device, dtype=dtype)
            angles = pos[:, None] * inv_freq[None, :]
            sin_term = torch.sin(angles).unsqueeze(0).unsqueeze(0)
            cos_term = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        else:
            sin_term = self.sin[:seq_len].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            cos_term = self.cos[:seq_len].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        x_even = x[..., : self.pair_dim * 2 : 2]
        x_odd = x[..., 1 : self.pair_dim * 2 : 2]

        y_even = x_even * cos_term - x_odd * sin_term
        y_odd = x_even * sin_term + x_odd * cos_term

        y = x.clone()
        y[..., : self.pair_dim * 2 : 2] = y_even
        y[..., 1 : self.pair_dim * 2 : 2] = y_odd
        return y