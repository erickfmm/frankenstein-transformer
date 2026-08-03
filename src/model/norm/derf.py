"""Derf normalization layer.

Applies the error function (erf) as a smooth, saturating non-linearity with
learnable affine parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Derf(nn.Module):
    """Derf normalization layer.

    Applies the error function (erf) as a smooth, saturating non-linearity
    with learnable affine parameters. The formulation is::

        y = gamma * erf(alpha * x + s) + beta

    where ``alpha`` and ``s`` are scalar parameters controlling the slope and
    shift of the erf, and ``gamma``, ``beta`` are per-dimension scale and bias.

    Args:
        dim: Number of features in the input (normalized dimension).

    Attributes:
        alpha: Learnable scalar slope parameter, initialized to 1.0.
        s: Learnable scalar shift parameter, initialized to 0.0.
        gamma: Learnable per-dimension scale of shape ``(dim,)``, initialized
            to 1.
        beta: Learnable per-dimension bias of shape ``(dim,)``, initialized
            to 0.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.s = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Derf normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape as ``x``, transformed by the erf non-linearity
            with learnable affine parameters.
        """
        return self.gamma * torch.erf(self.alpha * x + self.s) + self.beta
