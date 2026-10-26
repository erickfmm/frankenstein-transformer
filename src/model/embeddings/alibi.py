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
from typing import Optional

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
        q_len: Optional[int] = None,
        kv_len: Optional[int] = None,
        q_offset: int = 0,
        kv_positions: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the ALiBi additive attention bias.

        The full-square builder is the special case ``q_offset = 0`` and
        ``q_len == kv_len`` (bit-identical to the legacy
        ``bias(seq_len)`` signature, still accepted via the ``seq_len``
        keyword). Incremental decoding passes the cached key positions so
        eviction gaps produce correct relative distances (ALiBi carries no
        positional tensor in the cache — arXiv:2108.12409).

        Args:
            q_len: Number of query positions.
            kv_len: Number of key positions. Defaults to ``q_len``.
            q_offset: Absolute index of the FIRST query position. Default 0.
            kv_positions: Optional absolute int64 positions of the cached
                keys ``(kv_len,)``. Defaults to ``arange(kv_len)``.
            device: Device on which to materialize the bias tensor. Defaults
                to the slopes buffer device.
            dtype: Data type of the returned bias. Defaults to the slopes
                buffer dtype.
            seq_len: Legacy keyword alias for ``q_len`` (backward
                compatibility with ``bias(seq_len=...)`` call sites).

        Returns:
            Tensor of shape ``(1, num_heads, q_len, kv_len)`` containing
            the additive bias ``slopes * (-|q_pos - kv_pos|)``.
        """
        if q_len is None:
            if seq_len is None:
                raise ValueError("bias() requires q_len (or the legacy seq_len keyword)")
            q_len = int(seq_len)
        if device is None:
            device = self.slopes.device
        if dtype is None:
            dtype = self.slopes.dtype
        if kv_len is None:
            kv_len = q_len
        if kv_positions is None:
            kv_positions = torch.arange(kv_len, device=device, dtype=dtype)
        else:
            kv_positions = kv_positions.to(device=device, dtype=dtype)
        q_pos = torch.arange(q_len, device=device, dtype=dtype) + q_offset
        distances = (q_pos[:, None] - kv_positions[None, :]).abs()
        slopes = self.slopes.to(device=device, dtype=dtype)
        return slopes * (-distances)

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: int = 0,
        position_offset: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility with other positional encodings).

        Returns:
            The input tensor ``x`` unchanged.
        """
        return x