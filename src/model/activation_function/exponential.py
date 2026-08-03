"""Exponential (ELU-based) activation functions.

Implements the exponential-linear family surveyed in arXiv:2109.14545 §5:
ELU and its scaled, parametric, deformable, and combined variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELU(nn.Module):
    """Exponential Linear Unit: ``x`` if ``x >= 0`` else ``alpha * (e^x - 1)``.

    Range ``(-alpha, inf)``. Smooth negative saturation; mean closer to zero
    than ReLU. Reference: Clevert et al. (2016), arXiv:1511.07289; Lederer
    §2.3.2; survey §5.

    Args:
        alpha: Saturation value for negative inputs. Default: ``1.0``.

    Attributes:
        alpha: The negative-saturation constant.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ELU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return F.elu(x, self.alpha)


class SELU(nn.Module):
    r"""Scaled Exponential Linear Unit.

    ``scale * x`` if ``x >= 0`` else ``scale * alpha * (e^x - 1)`` with fixed
    ``alpha ~= 1.6733`` and ``scale ~= 1.0507``. Induces self-normalizing
    activations that keep mean 0 and variance 1. Reference: Klambauer et al.
    (2017), arXiv:1706.02515; Lederer §2.3.2; survey §5.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SELU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return F.selu(x)


class CELU(nn.Module):
    r"""Continuously Differentiable ELU: ``x`` if ``x >= 0`` else ``alpha * (e^{x/alpha} - 1)``.

    Range ``(-alpha, inf)``. Unlike ELU, CELU is once-differentiable at 0 for
    any ``alpha``. Reference: Barron (2017), arXiv:1704.07483; survey §5.

    Args:
        alpha: Saturation value for negative inputs. Default: ``1.0``.

    Attributes:
        alpha: The negative-saturation constant.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CELU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return F.celu(x, self.alpha)


class PELU(nn.Module):
    r"""Parametric ELU: ``alpha/beta * x`` if ``x >= 0`` else ``alpha * (e^{x/beta} - 1)``.

    Both ``alpha`` and ``beta`` are learnable, generalizing ELU. Reference:
    Trottier et al. (2017), arXiv:1605.09332; survey §5.

    Args:
        dim: Number of features. ``alpha``/``beta`` have shape ``(dim,)``.
        alpha_init: Initial positive saturation. Default: ``1.0``.
        beta_init: Initial positive scale. Default: ``1.0``.

    Attributes:
        alpha: Learnable saturation parameter of shape ``(dim,)``.
        beta: Learnable scale parameter of shape ``(dim,)``.
    """

    def __init__(self, dim: int, alpha_init: float = 1.0, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))
        self.beta = nn.Parameter(torch.full((dim,), float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Parametric ELU elementwise (per-channel).

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape.
        """
        beta = self.beta.clamp(min=1e-5)
        pos = (self.alpha / beta) * x
        neg = self.alpha * (torch.exp(x / beta) - 1)
        return torch.where(x >= 0, pos, neg)


class MPELU(nn.Module):
    r"""Multi-Parametric ELU: ELU with learnable ``alpha`` and a PReLU-like slope ``beta``.

    ``max(0, x) + beta * min(0, alpha * (e^x - 1))``. Generalizes both ELU and
    PReLU. Reference: survey §5.

    Args:
        dim: Number of features.
        alpha_init: Initial ELU saturation. Default: ``1.0``.
        beta_init: Initial negative slope. Default: ``1.0``.

    Attributes:
        alpha: Learnable saturation of shape ``(dim,)``.
        beta: Learnable negative slope of shape ``(dim,)``.
    """

    def __init__(self, dim: int, alpha_init: float = 1.0, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))
        self.beta = nn.Parameter(torch.full((dim,), float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Multi-Parametric ELU elementwise (per-channel).

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape.
        """
        pos = F.relu(x)
        neg = self.beta * (self.alpha * (torch.exp(x) - 1))
        return pos + torch.where(x < 0, neg, torch.zeros_like(x))


class FELU(nn.Module):
    """Fast ELU: like ELU but ``alpha`` is learnable and clamped to ``[0, 1]``.

    A numerically cheaper variant that restricts the saturation constant.
    Reference: survey §5.

    Args:
        dim: Number of features.
        alpha_init: Initial saturation value. Default: ``1.0``.

    Attributes:
        alpha: Learnable saturation of shape ``(dim,)``.
    """

    def __init__(self, dim: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fast ELU elementwise (per-channel).

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape.
        """
        alpha = self.alpha.clamp(min=0.0, max=1.0)
        return torch.where(x >= 0, x, alpha * (torch.exp(x) - 1))


class EELU(nn.Module):
    r"""Elastic ELU: ELU with learnable ``alpha`` and a positive-region slope ``beta``.

    ``beta * max(0, x) + alpha * min(0, e^x - 1)``. Reference: survey §5.

    Args:
        dim: Number of features.
        alpha_init: Initial negative saturation. Default: ``1.0``.
        beta_init: Initial positive slope. Default: ``1.0``.

    Attributes:
        alpha: Learnable negative saturation of shape ``(dim,)``.
        beta: Learnable positive slope of shape ``(dim,)``.
    """

    def __init__(self, dim: int, alpha_init: float = 1.0, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))
        self.beta = nn.Parameter(torch.full((dim,), float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Elastic ELU elementwise (per-channel).

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape.
        """
        pos = self.beta * F.relu(x)
        neg = self.alpha * (torch.exp(x) - 1)
        return pos + torch.where(x < 0, neg, torch.zeros_like(x))


class PDELU(nn.Module):
    r"""Parametric Deformable ELU.

    ``x`` if ``x >= 0`` else ``alpha * (e^{x/alpha} - 1) + (1 - alpha) * x``.
    Learnable ``alpha`` interpolates between CELU and the identity on the
    negative branch. Reference: survey §5.

    Args:
        dim: Number of features.
        alpha_init: Initial deformation in ``(0, 1]``. Default: ``1.0``.

    Attributes:
        alpha: Learnable deformation parameter of shape ``(dim,)``.
    """

    def __init__(self, dim: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PDELU elementwise (per-channel).

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape.
        """
        alpha = self.alpha.clamp(min=1e-5)
        return torch.where(
            x >= 0,
            x,
            alpha * (torch.exp(x / alpha) - 1) + (1 - alpha) * x,
        )


class PREU(nn.Module):
    r"""Parametric Rectified Exponential Unit.

    ``max(0, x) + alpha * min(0, e^x - 1)`` with learnable ``alpha`` (negative
    branch) and ``beta`` (positive-branch slope). Range ``(-alpha, inf)``.
    Reference: survey §5.

    Args:
        dim: Number of features.
        alpha_init: Initial negative saturation. Default: ``1.0``.
        beta_init: Initial positive slope. Default: ``1.0``.

    Attributes:
        alpha: Learnable negative saturation of shape ``(dim,)``.
        beta: Learnable positive slope of shape ``(dim,)``.
    """

    def __init__(self, dim: int, alpha_init: float = 1.0, beta_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))
        self.beta = nn.Parameter(torch.full((dim,), float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PREU elementwise (per-channel).

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape.
        """
        pos = self.beta * F.relu(x)
        neg = self.alpha * (torch.exp(x) - 1).clamp(max=0)
        return pos + neg


class SoftExp(nn.Module):
    r"""Soft Exponential: interpolates between exponential, linear, and logarithmic.

    For ``alpha < 0``: ``-(log(1 - alpha*(x + alpha))) / alpha``;
    ``alpha == ``: ``x``;
    ``alpha > 0``: ``(exp(alpha * x) - 1) / alpha + alpha``.
    ``alpha`` is learnable. Reference: survey §5, Godfrey & Gashler (2015).

    Args:
        dim: Number of features.
        alpha_init: Initial interpolation parameter. Default: ``0.0``.

    Attributes:
        alpha: Learnable interpolation parameter of shape ``(dim,)``.
    """

    def __init__(self, dim: int, alpha_init: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Soft Exponential elementwise (per-channel).

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape.
        """
        a = self.alpha
        pos = a > 0
        neg = a < 0
        out = torch.zeros_like(x)
        # alpha > 0
        out = torch.where(
            pos & (a != 0),
            (torch.exp(a.clamp(min=0) * x) - 1) / a.clamp(min=1e-5) + a.clamp(min=0),
            out,
        )
        # alpha < 0
        safe_neg = a.clamp(max=-1e-5)
        arg = 1 - safe_neg * (x + safe_neg)
        out = torch.where(
            neg & (arg > 0),
            -torch.log(arg.clamp(min=1e-12)) / safe_neg,
            out,
        )
        # alpha == 0
        out = torch.where(~pos & ~neg, x, out)
        return out


class ELiSH(nn.Module):
    r"""Exponential Linear Sigmoid SquasHing.

    ``x * sigmoid(x)`` for ``x >= 0`` and ``(e^x - 1) / (1 + e^{-x})`` for
    ``x < 0`` (Swish positive branch, ELU-gated negative branch). Reference:
    Basirat & Roth (2019), survey §5.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ELiSH elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        sig = torch.sigmoid(x)
        return torch.where(x >= 0, x * sig, (torch.exp(x) - 1) * sig)


class HardELiSH(nn.Module):
    r"""Hard ELiSH: ELiSH with a hard-sigmoid approximation.

    ``x * relu6(x + 3) / 6`` for ``x >= 0`` and ``(e^x - 1) * relu6(x + 3) / 6``
    otherwise. Reference: Basirat & Roth (2019), survey §5.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply HardELiSH elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        hard_sig = F.relu6(x + 3) / 6
        return torch.where(x >= 0, x * hard_sig, (torch.exp(x) - 1) * hard_sig)


__all__ = [
    "ELU",
    "SELU",
    "CELU",
    "PELU",
    "MPELU",
    "FELU",
    "EELU",
    "PDELU",
    "PREU",
    "SoftExp",
    "ELiSH",
    "HardELiSH",
]
