"""Dynamic Tanh normalization layer.

Applies standard normalization (subtract mean, divide by standard deviation)
followed by a learnable affine transformation through ``tanh``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicTanhNorm(nn.Module):
    """Dynamic Tanh normalization layer.

    Applies standard normalization (subtract mean, divide by standard deviation)
    followed by a learnable affine transformation through ``tanh``. The
    learnable parameters ``alpha`` and ``beta`` allow the layer to adapt the
    saturation point and slope of the tanh non-linearity per dimension.

    Args:
        dim: Number of features in the input (normalized dimension).
        eps: Small constant for numerical stability in standard deviation.
            Defaults to ``1e-6``.

    Attributes:
        alpha: Learnable scale parameter of shape ``(dim,)``, initialized to 1.
        beta: Learnable shift parameter of shape ``(dim,)``, initialized to 0.
        eps: Epsilon value for numerical stability.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """Apply dynamic tanh normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape as ``x``, normalized and passed through
            ``tanh`` with learnable affine parameters.
        """
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        x_norm = (x - mu) / (sigma + self.eps)
        return torch.tanh(x_norm * self.alpha + self.beta)
