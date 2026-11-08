"""NorMuon optimizer with neuron-wise normalized orthogonal updates.

NorMuon (Li et al. 2025, arXiv:2510.05491) augments Muon with a neuron-wise
(row-level) adaptive learning rate applied *after* the Newton-Schulz
orthogonalization.  Muon's orthogonalization equalizes the singular values of
the update but leaves the L2 norm of individual neuron rows highly non-uniform,
so a few neurons can dominate the optimization.  NorMuon tracks a per-row
second-moment EMA of the orthogonalized update and divides each neuron row by
its RMS before rescaling the whole update to Adam's empirical scale, so the
final direction keeps Muon's low condition number while making the per-neuron
contributions balanced.

Following the Muon hybrid recipe, only hidden-layer 2-D matrices take the
NorMuon path; embedding tables, learned positional encodings, output heads and
every non-matrix parameter (biases, norm scales, convolution kernels) receive
decoupled AdamW updates instead of being skipped.

Momentum parameterization note: Algorithm 1 of the paper writes
``M_t = beta_1 M_{t-1} + (1 - beta_1) G_t`` while this module (like ``muon``
and ``muon2``) accumulates ``M_t = beta_1 M_{t-1} + G_t``.  For ``M_0 = 0`` the
two recursions differ by the constant factor ``(1 - beta_1)``, and the
Frobenius normalization at the start of Newton-Schulz removes any constant
scale, so the orthogonalized directions -- and therefore the row statistics
and the final update -- are identical.

Reference:
    Li, Z., Liu, L., Liang, C., Chen, W., & Zhao, T. (2025). NorMuon: Making
    Muon more efficient and scalable. arXiv:2510.05491.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update

_RMS_TARGET = 0.2
"""Target RMS of the update (Algorithm 1, line 10): Adam's empirical scale."""

_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients shared with Muon."""


def _newton_schulz(
    matrix: torch.Tensor,
    steps: int,
    eps: float,
    coefficients: tuple[float, float, float] = _COEFFICIENTS,
) -> torch.Tensor:
    """Approximate the orthogonal polar factor of a matrix.

    Implements Algorithm 1, line 6 of the NorMuon paper (the same Newton-Schulz
    routine used by Muon): the input is normalized by its Frobenius norm,
    transposed so that ``rows <= cols`` when necessary, and refined with
    ``steps`` quintic iterations before being transposed back.  Iterations run
    in bfloat16, matching the other Muon-family modules in this package.

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


class NorMuon(Optimizer):
    r"""NorMuon optimizer with neuron-wise normalized orthogonal updates.

    Accumulates momentum on matrix-shaped gradients, orthogonalizes it with
    Newton-Schulz iterations, and tracks a per-row (per-neuron) second-moment
    EMA ``v_t \in \mathbb{R}^m`` of the orthogonalized direction.  Each neuron
    row is divided by ``\sqrt{v_i} + \epsilon`` and the whole update is rescaled
    by ``\hat\eta = 0.2\,\eta\,\sqrt{mn} / \lVert \hat O_t \rVert_F`` so the final
    step has RMS ``0.2 * \eta`` (Adam's empirical scale).

    Hidden-layer 2-D parameters take this path; parameters in the embedding,
    positional-encoding and head groups, as well as any parameter that is not a
    2-D matrix, take a decoupled AdamW update using the group's ``betas`` and
    ``eps``.

    Reference:
        Li, Z., Liu, L., Liang, C., Chen, W., & Zhao, T. (2025). NorMuon:
        Making Muon more efficient and scalable. arXiv:2510.05491.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).  The RMS alignment keeps the update
            at Adam's scale, so Adam-style schedules apply unchanged.
        momentum: Momentum coefficient of the first-moment buffer
            (default: 0.95, the paper's ``beta_1``).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        ns_eps: Epsilon added during the Frobenius normalization before
            Newton-Schulz, and the floor of the RMS-alignment denominator
            (default: 1e-7).
        betas: ``(beta1, beta2)`` coefficients.  Index 1 is the decay of the
            neuron-wise second moment (default: ``(0.9, 0.95)``, the paper's
            ``beta_2 = 0.95``); both indices also drive the AdamW fallback.
        eps: Numerical-stability term of the row normalization and the AdamW
            fallback (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0);
            applied as ``W <- (1 - lr * wd) * W``, algebraically identical to
            the paper's ``-eta * lambda * W`` term.

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer and
            per-row second moment for matrix parameters; first/second moments
            for the AdamW fallback).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Initializes the NorMuon optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            momentum: First-moment coefficient (default: 0.95).
            ns_steps: Newton-Schulz iteration count (default: 5).
            ns_eps: Frobenius-normalization / RMS-alignment epsilon
                (default: 1e-7).
            betas: ``(beta1, beta2)`` moment coefficients; index 1 is the
                neuron-wise second-moment decay (default: ``(0.9, 0.95)``).
            eps: Row-normalization / fallback epsilon (default: 1e-8).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).

        Raises:
            ValueError: If ``lr``, ``momentum``, ``ns_steps``, ``ns_eps``,
                ``eps``, ``betas`` or ``weight_decay`` is out of its valid
                range.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 1 <= ns_steps:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if not 0.0 <= ns_eps:
            raise ValueError(f"Invalid ns_eps value: {ns_eps}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            betas=betas,
            eps=eps,
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

        for group in self.param_groups:
            use_adamw = str(group.get("name", "other")) in ADAMW_FALLBACK_GROUPS
            beta2 = float(group["betas"][1])
            eps = float(group["eps"])
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
                        lr=group["lr"],
                        betas=group.get("betas", (0.9, 0.95)),
                        eps=group.get("eps", 1e-8),
                        weight_decay=group["weight_decay"],
                    )
                    continue

                if p.numel() == 0:
                    continue

                grad = p.grad
                rows, cols = p.shape
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    state["row_second_moment"] = torch.zeros(
                        rows, dtype=torch.float32, device=grad.device
                    )

                # Algorithm 1, line 5: M_t = beta_1 * M_{t-1} + G_t.
                # (See the module docstring for the equivalence with the
                # paper's (1 - beta_1) scaling.)
                buffer = state["momentum_buffer"]
                buffer.mul_(float(group["momentum"])).add_(grad)

                # Algorithm 1, line 6: O_t = NewtonSchulz_5(M_t).
                orth = _newton_schulz(buffer, int(group["ns_steps"]), float(group["ns_eps"]))
                orth = orth.to(torch.float32)

                # Algorithm 1, line 7: v_t = beta_2 * v_{t-1}
                #   + (1 - beta_2) * mean_cols(O_t * O_t).
                row_moment = orth.square().mean(dim=1)
                second = state["row_second_moment"]
                second.mul_(beta2).add_(row_moment, alpha=1.0 - beta2)

                # Algorithm 1, lines 8-9: O_t <- O_t / (sqrt(V_t) + eps),
                # where V_t broadcasts the per-row statistic over columns.
                normalized = orth / (second.sqrt().unsqueeze(1) + eps)

                # Algorithm 1, line 11: decoupled decay -eta * lambda * W_t.
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))

                # Algorithm 1, line 10: eta_hat =
                #   0.2 * eta * sqrt(m n) / ||O_t||_F.
                denominator = normalized.norm().clamp_min(float(group["ns_eps"]))
                eta_hat = (
                    _RMS_TARGET
                    * float(group["lr"])
                    * math.sqrt(rows * cols)
                    / denominator
                )

                # Algorithm 1, line 11: W_{t+1} = W_t - eta_hat * O_t.
                p.add_(normalized.to(p.dtype), alpha=-float(eta_hat))

        return loss
