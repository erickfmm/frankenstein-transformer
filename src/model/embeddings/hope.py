"""Hybrid Positional Encoding (HoPE).

Implements hyperbolic positional encoding over consecutive dimension pairs
with monotonic exponential damping. Unlike RoPE which uses trigonometric
rotation (sin/cos), HoPE uses hyperbolic functions (sinh/cosh) combined with
an exponential damping factor that decays with position distance. A
layer-dependent scale factor modulates the angles, making deeper layers
sensitive to longer-range positions.

HoPE is used as the positional encoding for TitanAttention blocks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HoPE(nn.Module):
    """Hybrid Positional Encoding with hyperbolic rotation and exponential damping.

    Applies a hyperbolic transformation to each pair of adjacent dimensions.
    The angle for pair ``i`` at position ``p`` in layer ``l`` is::

        theta_i(p, l) = p * exp(-log(base) * i / (pair_dim - 1)) * (1 + 0.05 * l)

    The transformation uses ``cosh`` and ``sinh`` (instead of ``cos`` and
    ``sin``) multiplied by an exponential damping factor::

        damping(p, l) = exp(-damping * (1 + 0.05 * l) * p)

    This provides monotonic decay of positional influence with distance while
    the layer-dependent scaling allows deeper layers to attend over longer
    ranges.

    Args:
        head_dim: Dimensionality of each attention head. Must be even for
            proper pairing.
        base: Base for the geometric progression of inverse frequencies.
            Defaults to ``10000.0``.
        damping: Damping coefficient controlling the rate of exponential
            decay with position. Defaults to ``0.01``.

    Attributes:
        head_dim: Total head dimensionality.
        pair_dim: Number of dimension pairs (``head_dim // 2``).
        base: Base frequency for inverse frequency computation.
        damping: Damping coefficient for exponential position decay.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0, damping: float = 0.01):
        super().__init__()
        self.head_dim = head_dim
        self.pair_dim = head_dim // 2
        self.base = base
        self.damping = damping

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        """Apply hyperbolic positional encoding.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            logical_layer_idx: Logical layer index used to compute the
                layer-dependent angle scaling factor. Defaults to ``0``.

        Returns:
            Tensor of same shape as ``x`` with hyperbolic positional encoding
            applied. If ``pair_dim == 0``, returns ``x`` unchanged.
        """
        if self.pair_dim == 0:
            return x

        _, _, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        pos = torch.arange(seq_len, device=device, dtype=dtype)
        if self.pair_dim > 1:
            idx = torch.arange(self.pair_dim, device=device, dtype=dtype)
            inv_freq = torch.exp(-math.log(self.base) * idx / (self.pair_dim - 1))
        else:
            inv_freq = torch.ones(1, device=device, dtype=dtype)

        layer_scale = 1.0 + 0.05 * float(logical_layer_idx)
        angles = (pos[:, None] * inv_freq[None, :] * layer_scale).clamp(-12.0, 12.0)

        damping = torch.exp(-(self.damping * layer_scale) * pos).unsqueeze(-1)
        cosh_term = torch.cosh(angles) * damping
        sinh_term = torch.sinh(angles) * damping

        x_even = x[..., : self.pair_dim * 2 : 2]
        x_odd = x[..., 1 : self.pair_dim * 2 : 2]

        cosh_term = cosh_term.unsqueeze(0).unsqueeze(0)
        sinh_term = sinh_term.unsqueeze(0).unsqueeze(0)

        y_even = x_even * cosh_term + x_odd * sinh_term
        y_odd = x_even * sinh_term + x_odd * cosh_term

        y = x.clone()
        y[..., : self.pair_dim * 2 : 2] = y_even
        y[..., 1 : self.pair_dim * 2 : 2] = y_odd
        return y
