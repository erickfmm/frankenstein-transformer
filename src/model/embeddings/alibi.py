"""Attention with Linear Biases (ALiBi).

Implements an additive position bias applied directly to attention scores
instead of injecting positional information into the query/key
representations. The bias is a head-specific slope multiplied by the negative
absolute distance between query and key positions::

    bias[h, i, j] = slope_h * (-|i - j|)

This formulation allows input-length extrapolation at inference time without
retraining, since the bias only depends on relative distance and not on
absolute position. The ``forward`` method is an identity pass-through so the
module can be dropped into ``pos_encoder(q)`` / ``pos_encoder(k)`` call sites;
the additive bias is exposed through :meth:`bias` for use by attention layers.

Reference:
    Press & Smith (2021), "Train Short, Test Long: Attention with Linear
    Biases Enables Input Length Extrapolation", arXiv:2108.12409.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ALiBi(nn.Module):
    """Attention with Linear Biases.

    Stores per-head geometric slopes following Press's recipe. For a number of
    heads that is a power of two, slopes are a geometric progression starting
    at ``2 ** -(2 ** -(log2(n) - 3))``. For non-power-of-two head counts the
    closest lower power of two is used and the remaining slopes are taken from
    the interleaved slopes of ``2 * closest`` heads.

    Reference:
        Press & Smith (2021), "Train Short, Test Long: Attention with Linear
        Biases Enables Input Length Extrapolation", arXiv:2108.12409.

    Args:
        num_heads: Number of attention heads for which slopes are computed.

    Attributes:
        num_heads: Number of attention heads.
        slopes: Non-persistent buffer of shape ``(1, num_heads, 1, 1)``
            holding the per-head ALiBi slopes.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor(self._get_slopes(num_heads), dtype=torch.float32)
        slopes = slopes.view(1, num_heads, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _get_slopes(num_heads: int):
        """Return the ALiBi slopes for ``num_heads`` heads.

        Args:
            num_heads: Number of attention heads.

        Returns:
            List of length ``num_heads`` with per-head slopes.
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + ALiBi._get_slopes(2 * closest_power_of_2)[0::2][
                    : num_heads - closest_power_of_2
                ]
            )

    def bias(
        self,
        seq_len: int,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Compute the ALiBi additive attention bias.

        Args:
            seq_len: Sequence length for which to compute the bias.
            device: Device on which to materialize the bias tensor. Defaults
                to the slopes buffer device.
            dtype: Data type of the returned bias. Defaults to the slopes
                buffer dtype.

        Returns:
            Tensor of shape ``(1, num_heads, seq_len, seq_len)`` containing
            the additive bias ``slopes * (-|i - j|)``.
        """
        if device is None:
            device = self.slopes.device
        if dtype is None:
            dtype = self.slopes.dtype
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        distances = (pos[:, None] - pos[None, :]).abs()
        slopes = self.slopes.to(device=device, dtype=dtype)
        return slopes * (-distances)

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