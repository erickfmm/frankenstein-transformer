"""Common elementwise activation functions.

Groups the classical textbook activation functions: the sigmoid/tanh family,
the smooth softplus family, and the core rectifier / gating units used in the
feed-forward network. These are stateless (no learnable parameters).

Formulations follow Lederer (arXiv:2101.09957) and the survey by Dubey et al.
(arXiv:2109.14545).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sigmoid(nn.Module):
    """Logistic sigmoid: ``1 / (1 + e^{-x})``.

    Range ``(0, 1)``. Monotonic, smooth, bounded. Reference: Lederer §2.1.1.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the logistic sigmoid elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``1 / (1 + e^{-x})`` applied elementwise.
        """
        return torch.sigmoid(x)


class Tanh(nn.Module):
    """Hyperbolic tangent: ``(e^x - e^{-x}) / (e^x + e^{-x}) = 2 sigmoid(2x) - 1``.

    Range ``(-1, 1)``. Monotonic, smooth, bounded. Reference: Lederer §2.1.3.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the hyperbolic tangent elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``tanh`` applied elementwise.
        """
        return torch.tanh(x)


class Arctan(nn.Module):
    """Inverse tangent: ``arctan(x)``.

    Range ``(-pi/2, pi/2)``. Monotonic, smooth, bounded. Reference: Lederer
    §2.1.2.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``arctan`` elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``arctan(x)`` applied elementwise.
        """
        return torch.atan(x)


class Softsign(nn.Module):
    """Softsign (Elliott activation): ``x / (1 + |x|)``.

    Range ``(-1, 1)``. Once-differentiable at 0, smooth elsewhere. Cheaper
    than tanh. Reference: Lederer §2.1.4 (also called ``elliottsig``).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the softsign elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x / (1 + |x|)`` applied elementwise.
        """
        return F.softsign(x)


class Elliott(nn.Module):
    """Elliott activation (fast sigmoid approximation): ``x / (1 + |x|)``.

    Alias of :class:`Softsign`; the survey (arXiv:2109.14545) lists it under
    the name "Elliott". Range ``(0, 1)`` for the one-sided variant, but the
    implemented two-sided form matches the standard ``elliottsig``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Elliott activation elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x / (1 + |x|)`` applied elementwise.
        """
        return F.softsign(x)


class Identity(nn.Module):
    """Identity / linear activation: ``f(x) = x``.

    Range ``(-inf, inf)``. Useful as a no-op activation. Reference: Lederer
    §2.2.1.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            The same tensor ``x``.
        """
        return x


class Softplus(nn.Module):
    """Softplus: ``log(1 + e^x)``.

    Smooth approximation of ReLU; an antiderivative of the logistic sigmoid.
    Range ``[0, inf)``. Reference: Lederer §2.3.1, Glorot et al. (2011).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softplus elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``log(1 + e^x)`` applied elementwise.
        """
        return F.softplus(x)


class Mish(nn.Module):
    """Mish: ``x * tanh(softplus(x)) = x * tanh(log(1 + e^x))``.

    Non-monotonic, smooth, self-regularized. Reported to outperform Swish on
    several benchmarks. Reference: Misra (2019), arXiv:1908.08681; survey §7.1.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mish elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x * tanh(softplus(x))`` applied.
        """
        return F.mish(x)


class GELU(nn.Module):
    """Gaussian Error Linear Unit (exact): ``x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))``.

    Range ``(-0.17, inf)``. Smooth, probabilistic. Used in BERT, GPT-2/3.
    Reference: Hendrycks & Gimpel (2016), arXiv:1606.08415; survey §7.2.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the exact GELU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x * Phi(x)`` applied elementwise.
        """
        return F.gelu(x)


class GELUTanh(nn.Module):
    """GELU with the tanh approximation.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.

    Faster than exact GELU; the approximation used by the original GPT-2 and
    BERT implementations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tanh-approximated GELU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with the tanh-approx GELU applied.
        """
        return F.gelu(x, approximate="tanh")


class ReLU(nn.Module):
    """Rectified Linear Unit: ``max(0, x)``.

    Range ``[0, inf)``. Reference: Nair & Hinton (2010); Lederer §2.2.2.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``max(0, x)`` applied elementwise.
        """
        return F.relu(x)


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU / Swish with beta=1): ``x * sigmoid(x)``.

    Range ``(-0.278, inf)``. Smooth, non-monotonic. Default FFN activation in
    Llama, PaLM, and this codebase. Reference: Elfwing et al. (2018),
    Ramachandran et al. (2017) arXiv:1710.05941; survey §3.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiLU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x * sigmoid(x)`` applied.
        """
        return F.silu(x)


class Swish(nn.Module):
    r"""Parametric Swish: ``x * sigmoid(beta * x)`` with fixed ``beta``.

    Range ``(-c, inf)`` where ``c ~= 0.278 / beta``. Non-monotonic, smooth.
    When ``beta = 1`` this reduces to SiLU. Large ``beta`` approaches ReLU.
    Use :class:`~.learnable.SwishTrainable` for a learnable ``beta``.

    Args:
        beta: Fixed positive slope. Default: ``1.0``.

    Attributes:
        beta: The (fixed) sigmoid slope.

    Reference: Ramachandran et al. (2017), arXiv:1710.05941; survey §6.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the parametric Swish elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x * sigmoid(beta * x)`` applied.
        """
        return x * torch.sigmoid(self.beta * x)


__all__ = [
    "Sigmoid",
    "Tanh",
    "Arctan",
    "Softsign",
    "Elliott",
    "Identity",
    "Softplus",
    "Mish",
    "GELU",
    "GELUTanh",
    "ReLU",
    "SiLU",
    "Swish",
]
