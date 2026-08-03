"""Root Mean Square Layer Normalization (RMSNorm).

Implements RMSNorm and its partial variant pRMSNorm, both introduced by
Zhang & Sennrich (2019), arXiv:1910.07467.

RMSNorm simplifies LayerNorm by removing the mean-subtraction (re-centering)
step and rescaling activations solely by their root mean square magnitude.
This preserves the re-scaling invariance property while reducing computation.

pRMSNorm further reduces overhead by estimating the RMS statistic from only
the first *p*% of the hidden dimensions, exploiting the assumption that
neurons within a layer are approximately i.i.d.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Applies RMS normalization to the last dimension of the input::

        y_i = g_i * x_i / RMS(x),   RMS(x) = sqrt( (1/n) * sum_j x_j^2 + eps )

    where ``g`` is a learnable per-dimension scale (initialized to 1) and no
    bias is used, following the original formulation.

    When ``partial_ratio`` is greater than 0, the RMS is estimated from only
    the first ``k = ceil(dim * partial_ratio)`` elements (pRMSNorm).  This is
    the partial variant from Section 5 of the paper; the recommended default
    ratio is 6.25%.

    Args:
        dim: Number of features in the input (normalized dimension).
        eps: Small constant for numerical stability. Defaults to ``1e-6``.
        partial_ratio: Fraction of dimensions used for RMS estimation.
            ``0.0`` (default) uses all dimensions (standard RMSNorm).
            Values in ``(0, 1]`` activate pRMSNorm. The number of dimensions
            used is ``max(1, ceil(dim * partial_ratio))``.

    Attributes:
        weight: Learnable per-dimension scale of shape ``(dim,)``,
            initialized to 1.
        eps: Epsilon value for numerical stability.
        partial_ratio: Fraction of dimensions used for RMS estimation.
        k: Number of dimensions used for RMS estimation (``dim`` when
            ``partial_ratio`` is 0, otherwise ``ceil(dim * partial_ratio)``).

    References:
        Zhang & Sennrich (2019). "Root Mean Square Layer Normalization."
        arXiv:1910.07467.
    """

    def __init__(self, dim: int, eps: float = 1e-6, partial_ratio: float = 0.0):
        super().__init__()
        self.eps = eps
        self.partial_ratio = float(partial_ratio)
        if self.partial_ratio > 0.0:
            self.k = max(1, math.ceil(dim * self.partial_ratio))
        else:
            self.k = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply (partial) RMS normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape as ``x``, normalized by RMS and scaled by
            the learnable ``weight``.
        """
        if self.partial_ratio > 0.0:
            ms = x[..., :self.k].pow(2).mean(dim=-1, keepdim=True)
        else:
            ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(ms + self.eps)
        return self.weight * x_normed
