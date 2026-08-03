"""ReLU-based activation functions.

Implements the rectifier family surveyed in arXiv:2109.14545 §4: leaky and
parametric variants, bounded/shifted ReLUs, and rectified-hyperbolic units.
Most are stateless; the parametric ones (PReLU) carry learnable parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyReLU(nn.Module):
    """Leaky ReLU: ``x`` if ``x >= 0`` else ``negative_slope * x``.

    Range ``(-inf, inf)``. Allows a small gradient when the unit is inactive.
    Reference: Maas et al. (2013); Lederer §2.2.3; survey §4.

    Args:
        negative_slope: Slope for negative inputs. Default: ``0.01``.

    Attributes:
        negative_slope: The negative-region slope.
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Leaky ReLU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with the leaky-relu transform applied.
        """
        return F.leaky_relu(x, self.negative_slope)


class ReLU6(nn.Module):
    """ReLU6: ``min(max(0, x), 6)``.

    Bounded ReLU clamped at 6, used in MobileNet for fixed-point friendliness.
    Reference: Howard et al. (2017).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU6 elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``min(max(0, x), 6)`` applied.
        """
        return F.relu6(x)


class HardSwish(nn.Module):
    """Hard-Swish: ``x * relu6(x + 3) / 6``.

    Cheap approximation of Swish used in MobileNetV3. Range ``(-1.67, inf)``.
    Reference: Howard et al. (2019), arXiv:1905.02244.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hard-Swish elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x * relu6(x + 3) / 6`` applied.
        """
        return F.hardswish(x)


class PReLU(nn.Module):
    """Parametric ReLU: ``max(0, x) + a * min(0, x)`` with learnable ``a``.

    Extends Leaky ReLU by making the negative slope a trainable parameter.
    With one parameter per channel (``num_parameters = dim``) it is
    per-channel; a single shared parameter gives the global variant.

    Args:
        dim: Number of features. The slope ``a`` has shape ``(dim,)``.
        init: Initial value of the negative slope. Default: ``0.25``.

    Attributes:
        weight: Learnable negative slope of shape ``(dim,)``.

    Reference: He et al. (2015), arXiv:1502.01852; survey §4.
    """

    def __init__(self, dim: int, init: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Parametric ReLU elementwise (per-channel slope).

        The slope is broadcast over the *last* dimension, matching the
        transformer convention ``(..., dim)``.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape with the parametric ReLU applied.
        """
        w = self.weight
        # Reshape weight to broadcast over the trailing feature dimension.
        view = [1] * (x.dim() - 1) + [w.numel()]
        return F.relu(x) + (x - F.relu(x)) * w.view(*view)


class AbsReLU(nn.Module):
    """Absolute-value ReLU (ABReLU / AB-ReLU): ``max(0, x) - a * max(0, -x)``.

    where ``a`` is the mean of the pre-activation over the batch/map (a
    data-dependent baseline subtracted before rectification). Implemented here
    as ``|x|`` style: ``max(0, x) - max(0, -x)`` = ``x`` rectified to the
    negative-symmetric form ``abs``-biased. Concretely we use the survey's
    definition ``max(0, x) - max(0, -x)`` giving a sign-preserving ramp.
    Reference: survey §4, Bjorck et al. (2017).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ABReLU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return x.clamp(min=0) - x.clamp(max=0).abs()


class NLReLU(nn.Module):
    """Natural-Logarithm ReLU: ``beta * log(1 + max(0, x))``.

    Range ``[0, inf)``; compresses large positive activations logarithmically.
    Reference: survey §4, Forest (2014).

    Args:
        beta: Positive scaling constant. Default: ``1.0``.

    Attributes:
        beta: The logarithmic scale.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply NLReLU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return self.beta * torch.log1p(F.relu(x))


class BReLU(nn.Module):
    """Bounded ReLU: ``min(max(0, x), t)``.

    Range ``[0, t]``. Clamps the positive region to a ceiling ``t``.

    Args:
        t: Upper bound for the positive region. Default: ``1.0``.

    Attributes:
        t: The ceiling value.
    """

    def __init__(self, t: float = 1.0):
        super().__init__()
        self.t = float(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Bounded ReLU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape clamped to ``[0, t]``.
        """
        return x.clamp(min=0.0, max=self.t)


class VReLU(nn.Module):
    """V-shaped ReLU: ``|x|``.

    Range ``[0, inf)``. Symmetric rectifier; equivalent to the absolute value.
    Reference: survey §4.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``|x|`` elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``|x|`` applied.
        """
        return torch.abs(x)


class Hexpo(nn.Module):
    """Hexpo: ``a * max(0, x) - c * max(0, -x)``.

    Asymmetric two-sided rectifier with separate positive/negative gains.
    Reference: survey §3.

    Args:
        a: Positive-region gain. Default: ``1.0``.
        c: Negative-region gain. Default: ``1.0``.

    Attributes:
        a, c: The two gains.
    """

    def __init__(self, a: float = 1.0, c: float = 1.0):
        super().__init__()
        self.a = float(a)
        self.c = float(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hexpo elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return self.a * x.clamp(min=0) - self.c * x.clamp(max=0)


class PenalizedTanh(nn.Module):
    r"""Penalized Tanh (pTanh): ``max(0, \tanh(x))``.

    A rectified, non-negative tanh with a soft gradient near zero. Range
    ``[0, 1)``. Reference: survey §3.

    Note:
        The general form is ``\tanh(x)`` for ``x > 0`` and ``a * \tanh(x)``
        with ``a in (0, 1)`` otherwise; the implemented form uses
        ``a -> 0`` (hard zeroing of negatives).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply penalized tanh elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``max(0, tanh(x))`` applied.
        """
        return torch.tanh(x).clamp(min=0)


class DisReLU(nn.Module):
    """Displaced ReLU: ``max(0, x - delta)``.

    Shifts the ReLU hinge to the right by ``delta``. Range ``[-delta, inf)``
    after the displacement. Reference: survey §4.

    Args:
        delta: Rightward shift of the rectifier hinge. Default: ``0.0``.

    Attributes:
        delta: The hinge displacement.
    """

    def __init__(self, delta: float = 0.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Displaced ReLU elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return (x - self.delta).clamp(min=0)


class LiSHT(nn.Module):
    """Linearly Scaled Hyperbolic Tangent: ``x * tanh(x)``.

    Non-monotonic, unbounded in magnitude. Range ``[0, inf)`` (by the survey's
    magnitude argument). Reference: survey §3, Roy et al. (2019).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LiSHT elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with ``x * tanh(x)`` applied.
        """
        return x * torch.tanh(x)


__all__ = [
    "LeakyReLU",
    "ReLU6",
    "HardSwish",
    "PReLU",
    "AbsReLU",
    "NLReLU",
    "BReLU",
    "VReLU",
    "Hexpo",
    "PenalizedTanh",
    "DisReLU",
    "LiSHT",
]
