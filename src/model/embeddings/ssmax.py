"""Scalable Softmax (SSMax).

Implements a transversal logit rescale that applies to every positional
encoding. SSMax multiplies the pre-softmax attention logits by
``s * ln(n)`` where ``s`` is a learnable per-head scalar and ``n`` is the
sequence length. This counteracts "attention fading" --- the softmax
flattening that occurs as context grows --- and improves length
extrapolation across all positional encodings (most impactful when paired
with ``bam`` or ``alibi``).

The ``forward`` method is an identity pass-through so the module can be
attached to any positional-encoding module as a child submodule; the rescale
factor is exposed through :meth:`scale` for use by attention layers via
``apply_pe_to_scores``.

Reference:
    Bianchessi et al. (2025), "Bayesian Attention Mechanism: A Probabilistic
    Framework for Positional Encoding and Context Length Extrapolation",
    arXiv:2505.22842 (Section: Scalable Softmax).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SSMax(nn.Module):
    """Scalable Softmax per-head logit rescale.

    Holds a learnable per-head scalar ``s`` (init ``s_init``) and exposes a
    :meth:`scale` method that returns the multiplicative factor
    ``s * ln(seq_len)`` of shape ``(1, num_heads, 1, 1)`` for broadcasting
    against attention logits.

    Reference:
        Bianchessi et al. (2025), "Bayesian Attention Mechanism", arXiv:2505.22842.

    Args:
        num_heads: Number of attention heads.
        s_init: Initial value for the per-head scalar ``s``. The post-training
            distribution in the paper peaks around ``s ~ 1.5-3``; ``1.0`` is a
            neutral start. Default: 1.0.

    Attributes:
        num_heads: Number of attention heads.
        s: Per-head learnable scalar of shape ``(num_heads,)``.
    """

    def __init__(self, num_heads: int, s_init: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.s = nn.Parameter(torch.full((num_heads,), float(s_init)))

    def scale(
        self,
        seq_len: int,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Compute the SSMax multiplicative logit rescale factor.

        Args:
            seq_len: Sequence length for which to compute the scale.
            device: Device on which to materialise the tensor. Defaults to
                the ``s`` parameter device.
            dtype: Data type of the returned tensor. Defaults to the ``s``
                parameter dtype.

        Returns:
            Tensor of shape ``(1, num_heads, 1, 1)`` containing
            ``s[h] * ln(seq_len)``.
        """
        if device is None:
            device = self.s.device
        if dtype is None:
            dtype = self.s.dtype
        s = self.s.to(device=device, dtype=dtype)  # [H]
        factor = s * math.log(seq_len)  # [H]
        return factor.view(1, self.num_heads, 1, 1)

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility).

        Returns:
            The input tensor ``x`` unchanged.
        """
        return x