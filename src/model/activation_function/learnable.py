"""Learnable / adaptive activation functions.

Implements:

* :class:`RationalActivation` — the **Rational Activation Function (RAF)** from
  *Transformers with Learnable Activation Functions* (Fang et al., EACL 2023,
  arXiv:2208.14111). A RAF is a learnable ratio of two low-degree polynomials
  (a Padé approximant), with five "version" variants (A/B/C/D/N) controlling
  the denominator form. The default configuration matches the paper: degree
  ``(5, 4)``, version ``"A"`` (the "safe" per-term absolute-value denominator),
  initialized by a least-squares fit to GELU on ``[-3, 3]``.

  Note:
      Despite being requested as "Rectified Activation Function", the paper
      defines **RAF = Rational Activation Function**. This implementation
      follows the paper.

* :class:`SwishTrainable` — Swish with a learnable ``beta`` parameter.
* :class:`Maxout` — the Goodfellow et al. (2013) maxout unit.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Initialization target functions and rational fitting
# --------------------------------------------------------------------------- #
def _approx_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def _approx_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def _approx_leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, 0.01)


def _approx_leaky_relu_01(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, 0.1)


def _approx_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _approx_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def _approx_swish(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def _approx_identity(x: torch.Tensor) -> torch.Tensor:
    return x


_APPROX_FUNCTIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": _approx_gelu,
    "relu": _approx_relu,
    "leaky_relu": _approx_leaky_relu,
    "leaky_relu_0.1": _approx_leaky_relu_01,
    "sigmoid": _approx_sigmoid,
    "tanh": _approx_tanh,
    "swish": _approx_swish,
    "silu": _approx_swish,
    "identity": _approx_identity,
}


# Exact (5, 4) least-squares presets from the rational_activations library
# (fitted on [-3, 3]).
_RATIONAL_PRESETS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "gelu": (
        (
            -0.0012423594497499122,
            0.5080497063245629,
            0.41586363182937475,
            0.13022718688035761,
            0.024355900098993424,
            0.00290283948155535,
        ),
        (
            -0.06675015696494944,
            0.17927646217001553,
            0.03746682605496631,
            1.6561610853276082e-10,
        ),
    ),
    "relu": (
        (
            0.029963801610813613,
            0.6168978366891341,
            2.37534759733888,
            3.0659900472408443,
            1.5246831881677423,
            0.2528070864040542,
        ),
        (
            -1.191550121923625,
            4.4080487697236626,
            0.9110357113686055,
            0.34884977946384615,
        ),
    ),
    "identity": (
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ),
}


def _fit_rational(
    degrees: Tuple[int, int],
    approx_func: str,
    num_samples: int = 2001,
    lo: float = -3.0,
    hi: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit of a Padé ``P(x)/Q(x)`` to ``approx_func`` on ``[lo, hi]``.

    Returns ``(numerator, denominator)`` where ``numerator`` has length
    ``m + 1`` and ``denominator`` has length ``n`` (the constant term of the
    denominator is the fixed ``1``). Falls back to a torch least-squares solve
    when no preset is available.

    Args:
        degrees: ``(m, n)`` numerator and denominator degrees.
        approx_func: Name of the target function (see ``_APPROX_FUNCTIONS``).
        num_samples: Number of sample points. Default: ``2001``.
        lo: Lower fit bound. Default: ``-3.0``.
        hi: Upper fit bound. Default: ``3.0``.

    Returns:
        Tuple of tensors ``(numerator[m+1], denominator[n])``.
    """
    m, n = degrees
    if approx_func in _RATIONAL_PRESETS:
        num_preset, den_preset = _RATIONAL_PRESETS[approx_func]
        # Truncate/extend to match requested degrees.
        numerator = list(num_preset) + [0.0] * (m + 1 - len(num_preset))
        denominator = list(den_preset) + [0.0] * (n - len(den_preset))
        return torch.tensor(numerator[: m + 1]), torch.tensor(denominator[:n])

    target_fn = _APPROX_FUNCTIONS.get(approx_func, _approx_gelu)
    x = torch.linspace(lo, hi, num_samples, dtype=torch.float64)
    y = target_fn(x)

    # Build Vandermonde matrix for numerator (powers 0..m) and denominator
    # contributions (powers 1..n). Solve P(x) = y * (1 + sum_k b_k x^{k+1}).
    num_cols = [x.pow(k) for k in range(m + 1)]
    den_cols = [-(y) * x.pow(k + 1) for k in range(n)]
    A = torch.stack(num_cols + den_cols, dim=1)
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    numerator = sol[: m + 1].float()
    denominator = sol[m + 1 :].float()
    return numerator, denominator


class RationalActivation(nn.Module):
    r"""Rational Activation Function (RAF): a learnable Padé ``P(x)/Q(x)``.

    .. math::
        F(x) = \frac{P(x)}{Q(x)} = \frac{\sum_{j=0}^{m} a_j x^{j}}{1 + R(x)}

    where the denominator term ``R(x)`` depends on ``version``:

    * ``"A"`` (default, "safe"): ``R(x) = \sum_{k} |b_k x^{k+1}|``  (abs per term)
    * ``"B"``: ``R(x) = |\sum_{k} b_k x^{k+1}|``  (abs of the whole sum)
    * ``"C"``: ``R(x) = |b_0 + b_1 x + \dots + b_n x^{n}|`` with floor ``0.1``
    * ``"D"``: like ``"B"`` but with uniform multiplicative noise on the
      denominator weights during training only.
    * ``"N"``: ``R(x) = \sum_{k} b_k x^{k+1}``  (no abs; can have poles)

    Versions A/B/D keep ``Q(x) >= 1`` (no division by zero); C keeps it
    ``>= 0.1``; N is unsafe. The paper uses version ``"A"``.

    Args:
        degrees: ``(m, n)`` numerator and denominator polynomial degrees.
            Default: ``(5, 4)`` (the paper default).
        version: Denominator form. One of ``"A"`` (default), ``"B"``, ``"C"``,
            ``"D"``, ``"N"``.
        approx_func: Initialization target. One of ``"gelu"`` (default),
            ``"relu"``, ``"leaky_relu"``, ``"leaky_relu_0.1"``, ``"sigmoid"``,
            ``"tanh"``, ``"swish"``, ``"silu"``, ``"identity"``.
        trainable: If ``False``, freeze the rational parameters. Default:
            ``True``.
        input_scaling: If ``True``, apply per-token min-max scaling of the
            input to ``[-3, 3]`` before the rational (the RAFT preprocessing
            that keeps inputs inside the fitted range). Default: ``False``.
        noise_eps: Std of the version-D multiplicative noise. Default: ``0.1``.

    Attributes:
        numerator: Learnable parameter of shape ``(m + 1,)``.
        denominator: Learnable parameter of shape ``(n,)``.

    Reference:
        Fang et al. (2023). *Transformers with Learnable Activation Functions.*
        arXiv:2208.14111.
    """

    def __init__(
        self,
        degrees: Tuple[int, int] = (5, 4),
        version: str = "A",
        approx_func: str = "gelu",
        trainable: bool = True,
        input_scaling: bool = False,
        noise_eps: float = 0.1,
    ):
        super().__init__()
        if version not in {"A", "B", "C", "D", "N"}:
            raise ValueError(
                f"version must be one of {{A, B, C, D, N}}, got {version!r}"
            )
        m, n = degrees
        if not (isinstance(m, int) and isinstance(n, int) and m >= 1 and n >= 1):
            raise ValueError(f"degrees must be (m, n) with m>=1, n>=1, got {degrees}")
        if approx_func not in _APPROX_FUNCTIONS:
            raise ValueError(
                f"approx_func must be one of {sorted(_APPROX_FUNCTIONS)}, "
                f"got {approx_func!r}"
            )
        self.version = version
        self.degrees = (m, n)
        self.input_scaling = bool(input_scaling)
        self.noise_eps = float(noise_eps)

        numerator, denominator = _fit_rational((m, n), approx_func)
        self.numerator = nn.Parameter(numerator, requires_grad=trainable)
        self.denominator = nn.Parameter(denominator, requires_grad=trainable)

    @staticmethod
    def _scale_to_range(x: torch.Tensor, lo: float = -3.0, hi: float = 3.0) -> torch.Tensor:
        """Per-token min-max scaling to ``[lo, hi]`` (RAFT preprocessing).

        Args:
            x: Input tensor of shape ``(..., dim)``; scaling runs over ``dim``.
            lo: Target lower bound. Default: ``-3.0``.
            hi: Target upper bound. Default: ``3.0``.

        Returns:
            Tensor of same shape rescaled into ``[lo, hi]``.
        """
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
        span = (x_max - x_min).clamp(min=1e-8)
        return (x - x_min) / span * (hi - lo) + lo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rational activation elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        if self.input_scaling:
            x = self._scale_to_range(x)
        shape = x.shape
        z = x.reshape(-1)

        m, n = self.degrees
        max_len = max(m + 1, n + 1)
        # Monomials x^0, x^1, ..., x^{max_len-1}.
        xps = [torch.ones_like(z)]
        for _ in range(max_len - 1):
            xps.append(xps[-1] * z)
        xps_t = torch.stack(xps, dim=1)  # (N, max_len)

        numerator = (xps_t[:, : m + 1] * self.numerator).sum(dim=1)

        if self.version == "A":
            # 1 + sum_k |b_k x^{k+1}|  (per-term abs)
            denom_weights = torch.cat(
                [
                    torch.tensor([1.0], device=x.device, dtype=x.dtype),
                    self.denominator,
                ]
            )
            terms = xps_t[:, 1 : n + 1] * self.denominator.unsqueeze(0)
            denominator = 1.0 + terms.abs().sum(dim=1)
        elif self.version == "B":
            # 1 + |sum_k b_k x^{k+1}|
            terms = xps_t[:, 1 : n + 1] * self.denominator.unsqueeze(0)
            denominator = 1.0 + terms.sum(dim=1).abs()
        elif self.version == "C":
            # 0.1 + |b_0 + b_1 x + ...|
            terms = xps_t[:, :n] * self.denominator.unsqueeze(0)
            denominator = 0.1 + terms.sum(dim=1).abs()
        elif self.version == "D":
            terms = xps_t[:, 1 : n + 1] * self.denominator.unsqueeze(0)
            if self.training:
                noise = torch.empty_like(terms).uniform_(
                    1.0 - self.noise_eps, 1.0 + self.noise_eps
                )
                terms = terms * noise
            denominator = 1.0 + terms.sum(dim=1).abs()
        else:  # "N"
            terms = xps_t[:, 1 : n + 1] * self.denominator.unsqueeze(0)
            denominator = 1.0 + terms.sum(dim=1)

        return (numerator / denominator.clamp(min=1e-12)).reshape(shape)


class SwishTrainable(nn.Module):
    r"""Swish with a learnable ``beta``: ``x * sigmoid(beta * x)``.

    When ``beta -> 0`` the function approaches the linear/2 map; large
    ``beta`` recovers ReLU. Shared-scalar ``beta`` by default (per-channel via
    ``dim``).

    Args:
        dim: If given, ``beta`` is per-channel of shape ``(dim,)``. If ``None``,
            ``beta`` is a single shared scalar. Default: ``None``.
        beta_init: Initial value of ``beta``. Default: ``1.0`` (== SiLU).

    Attributes:
        beta: Learnable Swish slope (scalar or ``(dim,)``).

    Reference: Ramachandran et al. (2017), arXiv:1710.05941; survey §6.
    """

    def __init__(self, dim: int | None = None, beta_init: float = 1.0):
        super().__init__()
        if dim is None:
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.beta = nn.Parameter(torch.full((dim,), float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply trainable Swish elementwise.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape.
        """
        return x * torch.sigmoid(self.beta * x)


class Maxout(nn.Module):
    r"""Maxout unit: ``max_{i=0..k-1} (W_i x + b_i)``.

    Computes ``k`` linear projections of the input and takes the elementwise
    maximum. A universal approximator of any continuous function given enough
    pieces. Increases parameter count by a factor of ``k``.

    Args:
        dim: Input/output feature dimension.
        num_pieces: Number of linear pieces ``k`` to maximize over. Default: ``2``.

    Attributes:
        proj: Linear layer producing ``dim * num_pieces`` outputs.

    Reference: Goodfellow et al. (2013), arXiv:1302.4389; Lederer §2.3.4.
    """

    def __init__(self, dim: int, num_pieces: int = 2):
        super().__init__()
        if num_pieces < 1:
            raise ValueError(f"num_pieces must be >= 1, got {num_pieces}")
        self.dim = int(dim)
        self.num_pieces = int(num_pieces)
        self.proj = nn.Linear(self.dim, self.dim * self.num_pieces)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Maxout.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of shape ``(..., dim)``.
        """
        lead_shape = x.shape[:-1]
        out = self.proj(x)
        out = out.view(*lead_shape, self.num_pieces, self.dim)
        return out.max(dim=-2).values


__all__ = [
    "RationalActivation",
    "SwishTrainable",
    "Maxout",
]
