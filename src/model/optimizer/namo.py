"""NAMO optimizer: norm-based adaptive moment estimation with orthogonalized momentum.

NAMO (Zhang, Liu & Schaeffer 2026, arXiv:2602.17080) augments Muon's
orthogonalized update with an Adam-type, norm-based noise adaptation.  Muon
updates hidden-layer matrices with the orthogonal factor of the momentum
matrix, but matrix orthogonalization is an unbounded operation that can
amplify stochastic noise, so Muon's effective stepsize never shrinks when
gradients become noisy.  NAMO tracks a *single scalar* second-moment EMA of
the squared Frobenius norm of the gradient and rescales the whole
orthogonalized update by the bias-corrected norm ratio

    alpha_t = sqrt(1 - mu_2^t) / (1 - mu_1^t) * ||M_t||_F / (sqrt(v_t) + eps),

an aggregated signal-to-noise statistic analogous to Adam's coordinate-wise
ratio.  Because the scaling is a single scalar the update direction stays
exactly orthogonal, the optimizer state grows by only one scalar per matrix
over Muon (zero additional memory), and the per-step overhead is O(mn).

The optimizer follows Algorithm 1 of the paper verbatim.  In particular the
momentum recursion keeps the paper's ``(1 - mu_1)`` factor: unlike ``muon``
and ``nor_muon`` the constant scale does not cancel here, because ``alpha_t``
reads ``||M_t||_F`` and the bias corrections directly.  The reference
implementation released with the paper additionally exposes Nesterov
momentum, a clamp on the effective stepsize and a ``0.2 * sqrt(max(m, n))``
shape scaling; Algorithm 1 defines none of them, so this module follows the
paper.

Following the Muon hybrid recipe, only hidden-layer 2-D matrices take the
NAMO path; embedding tables, learned positional encodings, output heads and
every non-matrix parameter (biases, norm scales, convolution kernels) receive
decoupled AdamW updates configured by the shared per-group ``betas``/``eps``
suffixes.

Reference:
    Zhang, M., Liu, Y., & Schaeffer, H. (2026). Adam Improves Muon: Adaptive
    Moment Estimation with Orthogonalized Momentum. arXiv:2602.17080.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update

_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients shared with Muon."""


def _newton_schulz(
    matrix: torch.Tensor,
    steps: int,
    eps: float,
    coefficients: tuple[float, float, float] = _COEFFICIENTS,
) -> torch.Tensor:
    """Approximate the orthogonal polar factor of a matrix.

    Implements Algorithm 1, line 6 of the NAMO paper (the same
    Newton-Schulz routine used by Muon): the input is normalized by its
    Frobenius norm, transposed so that ``rows <= cols`` when necessary, and
    refined with ``steps`` quintic iterations before being transposed back.
    Iterations run in bfloat16, matching the other Muon-family modules in this
    package.

    Args:
        matrix: A 2-D tensor whose orthogonal factor is sought.
        steps: Number of Newton-Schulz iterations (the paper uses five).
        eps: Numerical-stability term of the Frobenius normalization.
        coefficients: Quintic polynomial ``(a, b, c)``; defaults to the Muon
            triple.

    Returns:
        The orthogonalized matrix in bfloat16, in the input orientation.
    """
    x = matrix.bfloat16()
    x = x / (x.norm() + eps)

    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True

    a, b, c = coefficients
    for _ in range(int(steps)):
        gram = x @ x.T
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x

    if transposed:
        x = x.T
    return x


class NAMO(Optimizer):
    r"""NAMO optimizer with norm-based adaptive orthogonal updates.

    Accumulates momentum on matrix-shaped gradients, orthogonalizes it with
    Newton-Schulz iterations, and rescales the whole orthogonalized direction
    by a single Adam-type adaptive scalar built from a second-moment EMA of
    the squared Frobenius norm of the gradient:

    .. math::

        M_t &= \mu_1 M_{t-1} + (1-\mu_1) G_t \\
        v_t &= \mu_2 v_{t-1} + (1-\mu_2) \lVert G_t \rVert_F^2 \\
        O_t &= \mathrm{NS}_K(M_t) \\
        \alpha_t &= \frac{\sqrt{1-\mu_2^t}}{1-\mu_1^t}
                    \frac{\lVert M_t \rVert_F}{\sqrt{v_t} + \epsilon} \\
        \Theta_t &= \Theta_{t-1} - \eta\,\alpha_t\,(O_t + \lambda \Theta_{t-1})

    Hidden-layer 2-D parameters take this path; parameters in the embedding,
    positional-encoding and head groups, as well as any parameter that is not a
    2-D matrix, take a decoupled AdamW update using the group's ``betas`` and
    ``eps``.

    Reference:
        Zhang, M., Liu, Y., & Schaeffer, H. (2026). Adam Improves Muon:
        Adaptive Moment Estimation with Orthogonalized Momentum.
        arXiv:2602.17080.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).  NAMO is not RMS-aligned, so the
            paper's optimal learning rates for hidden matrices are roughly an
            order of magnitude above AdamW's.
        betas: ``(mu_1, mu_2)`` momenta: the decay of the momentum buffer and
            of the scalar norm EMA (default: ``(0.95, 0.99)``, the paper's
            GPT-2 values).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        ns_eps: Epsilon added during the Frobenius normalization before
            Newton-Schulz (default: 1e-7).
        weight_decay: Decoupled weight decay coefficient (default: 0.0),
            scaled by the adaptive stepsize exactly as in the paper's update.

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer, scalar
            second moment and step counter for matrix parameters; first and
            second moments for the AdamW fallback).
        betas (tuple): The ``(mu_1, mu_2)`` momenta.  Stored as an instance
            attribute rather than in ``defaults`` because the factory injects
            per-group ``betas`` keys for the AdamW fallback, which would
            otherwise shadow the global momenta.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.99),
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        weight_decay: float = 0.0,
    ) -> None:
        """Initializes the NAMO optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            betas: ``(mu_1, mu_2)`` momentum decays (default: ``(0.95, 0.99)``).
            ns_steps: Newton-Schulz iteration count (default: 5).
            ns_eps: Frobenius-normalization epsilon (default: 1e-7).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).

        Raises:
            ValueError: If ``lr``, ``betas[i]``, ``ns_steps``, ``ns_eps`` or
                ``weight_decay`` is out of its valid range.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 1 <= ns_steps:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if not 0.0 <= ns_eps:
            raise ValueError(f"Invalid ns_eps value: {ns_eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.betas = (float(betas[0]), float(betas[1]))
        defaults = dict(
            lr=lr,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss.  Optional for most use cases.

        Returns:
            The loss value if ``closure`` is provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mu1, mu2 = self.betas
        for group in self.param_groups:
            use_adamw = str(group.get("name", "other")) in ADAMW_FALLBACK_GROUPS
            eps = float(group.get("eps", 1e-8))
            lr = float(group["lr"])
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Paper hybrid recipe: embeddings, positional encodings,
                # output heads and every non-matrix tensor take AdamW; only
                # hidden-layer 2-D matrices are orthogonalized.
                if use_adamw or p.ndim != 2:
                    adamw_fallback_update(
                        p,
                        p.grad,
                        self.state[p],
                        lr=lr,
                        betas=group.get("betas", (0.9, 0.95)),
                        eps=eps,
                        weight_decay=group["weight_decay"],
                    )
                    continue

                if p.numel() == 0:
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    state["norm_second_moment"] = torch.zeros(
                        (), dtype=torch.float32, device=grad.device
                    )
                    state["step"] = 0
                state["step"] += 1
                t = state["step"]

                # Algorithm 1, line 4: M_t = mu_1 * M_{t-1} + (1 - mu_1) * G_t.
                # The (1 - mu_1) factor is kept: alpha_t reads ||M_t||_F, so
                # unlike Muon the constant scale does not cancel.
                buffer = state["momentum_buffer"]
                buffer.mul_(mu1).add_(grad, alpha=1.0 - mu1)

                # Algorithm 1, line 5: v_t = mu_2 * v_{t-1}
                #   + (1 - mu_2) * ||G_t||_F^2.
                second = state["norm_second_moment"]
                grad_sq = grad.square().sum()
                second.mul_(mu2).add_(grad_sq.to(second.dtype), alpha=1.0 - mu2)

                # Algorithm 1, line 6: O_t = Orth(M_t) ~ NS_K(M_t).
                orth = _newton_schulz(
                    buffer, int(group["ns_steps"]), float(group["ns_eps"])
                )

                # Algorithm 1, line 7:
                #   alpha_t = sqrt(1 - mu_2^t) / (1 - mu_1^t)
                #             * ||M_t||_F / (sqrt(v_t) + eps).
                bias_correction1 = 1.0 - mu1**t
                bias_correction2 = 1.0 - mu2**t
                alpha = (
                    math.sqrt(bias_correction2)
                    / bias_correction1
                    * float(buffer.norm())
                    / (float(second.sqrt()) + eps)
                )

                # Paper section 4.1: the adaptive scaling also multiplies the
                # decoupled weight decay:
                #   Theta_t = Theta_{t-1} - eta * alpha_t * (O_t + lambda * Theta_{t-1}).
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - lr * alpha * float(group["weight_decay"]))

                # Algorithm 1, line 8: Theta_t = Theta_{t-1} - eta * alpha_t * O_t.
                p.add_(orth.to(p.dtype), alpha=-lr * alpha)

        return loss
