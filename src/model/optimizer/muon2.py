"""Muon^2 optimizer with adaptive second-moment preconditioning.

Muon^2 (Liu et al. 2026, arXiv:2604.09967) extends Muon by preconditioning the
momentum matrix with an Adam-style second-moment EMA *before* the
Newton-Schulz orthogonalization.  The second moment tightens the spectrum of
the momentum entering the iteration, so each Newton-Schulz step aligns better
with the true orthogonalized update and fewer iterations are needed (the
paper reports a 40% reduction, N_s=3 matching Muon at N_s=5).

Following the Muon hybrid recipe, only hidden-layer 2-D matrices take the
Muon^2 path; embedding tables, learned positional encodings, output heads and
every non-matrix parameter (biases, norm scales, convolution kernels) receive
decoupled AdamW updates instead of being skipped.  The ``factored`` flag
switches the dense second moment for Adafactor-style row/column statistics
(Muon^2-F), which keeps memory close to Muon.

Reference:
    Liu, Z., Zhang, R., Wang, Z., Zhao, Y., Su, Y., Yang, Z., & Zhang, Z.
    (2026). Muon^2: Boosting Muon via Adaptive Second-Moment
    Preconditioning. arXiv:2604.09967.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update


class Muon2(Optimizer):
    r"""Muon^2 optimizer with second-moment preconditioned orthogonal updates.

    Accumulates a momentum EMA ``M_t`` (Eq. 2) and an Adam-style second
    moment ``V_t`` (Eq. 3), precondition the momentum with
    ``M_t / (sqrt(V_t) + eps)`` (Eq. 4), orthogonalizes the result with
    Newton-Schulz iterations (Eq. 5) and applies the ``eta * sqrt(m/n)``
    scaled update (Eq. 6).  Hidden-layer 2-D parameters take this path;
    parameters in the embedding, positional-encoding and head groups, as
    well as any parameter that is not a 2-D matrix, take a decoupled AdamW
    update using the group's ``betas`` and ``eps``.

    The ``factored`` flag replaces the dense ``V_t`` with Adafactor-style
    row/column statistics (Algorithm 2, Muon^2-F): only a first-moment buffer
    and two vectors are stored per matrix parameter.

    Reference:
        Liu, Z., Zhang, R., Wang, Z., Zhao, Y., Su, Y., Yang, Z., & Zhang, Z.
        (2026). Muon^2: Boosting Muon via Adaptive Second-Moment
        Preconditioning. arXiv:2604.09967.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 0.02).
        betas: ``(beta1, beta2)`` coefficients of the momentum and
            second-moment EMAs (default: ``(0.9, 0.95)``).  They also drive
            the AdamW fallback.
        eps: Numerical-stability term of the preconditioner and the AdamW
            fallback (default: 1e-8).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        ns_eps: Epsilon added during the Frobenius normalization before
            Newton-Schulz (default: 1e-7).
        factored: Use the Adafactor-style row/column second moment
            (Muon^2-F) instead of a dense second moment (default: False).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (first/second moments for
            matrix parameters; row/column statistics in the factored variant;
            first/second moments for the AdamW fallback).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        factored: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        """Initializes the Muon^2 optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 0.02).
            betas: ``(beta1, beta2)`` moment coefficients
                (default: ``(0.9, 0.95)``).
            eps: Preconditioner / fallback epsilon (default: 1e-8).
            ns_steps: Newton-Schulz iteration count (default: 5).
            ns_eps: Frobenius-normalization epsilon (default: 1e-7).
            factored: Enable the Muon^2-F row/column second moment
                (default: False).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).

        Raises:
            ValueError: If ``lr``, ``eps``, ``betas``, ``ns_steps`` or
                ``ns_eps`` are out of their valid ranges.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 1 <= ns_steps:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if not 0.0 <= ns_eps:
            raise ValueError(f"Invalid ns_eps value: {ns_eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            factored=bool(factored),
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
            beta1, beta2 = group["betas"]
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

                grad = p.grad
                rows, cols = p.shape
                state = self.state[p]

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    if group["factored"]:
                        state["row_stats"] = torch.zeros(
                            rows, dtype=grad.dtype, device=grad.device
                        )
                        state["col_stats"] = torch.zeros(
                            cols, dtype=grad.dtype, device=grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                # Eq. 2: M_t = beta1 * M_{t-1} + (1 - beta1) * G_t
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Eq. 3 (Algorithm 1) or the factored variant (Algorithm 2).
                if group["factored"]:
                    row_stats = state["row_stats"]
                    col_stats = state["col_stats"]
                    grad_sq = grad * grad
                    row_stats.mul_(beta2).add_(
                        grad_sq.sum(dim=1), alpha=1.0 - beta2
                    )
                    col_stats.mul_(beta2).add_(
                        grad_sq.sum(dim=0), alpha=1.0 - beta2
                    )
                    row_sum = row_stats.sum().clamp_min(group["eps"])
                    v_hat = row_stats.unsqueeze(1) * col_stats.unsqueeze(0)
                    v_hat = v_hat / row_sum
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2
                    )
                    v_hat = exp_avg_sq

                # Eq. 4: M_tilde_t = M_t / (sqrt(V_t) + eps)
                m_hat = exp_avg / (v_hat.sqrt() + group["eps"])

                # Decoupled weight decay (repo Muon-family convention).
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Eq. 5: O_t = NewtonSchulz(M_tilde_t, K)
                x = m_hat.bfloat16()
                x = x / (x.norm() + group["ns_eps"])

                transposed = False
                if x.size(0) > x.size(1):
                    x = x.T
                    transposed = True

                a, b, c = 3.4445, -4.7750, 2.0315
                for _ in range(int(group["ns_steps"])):
                    a_mat = x @ x.T
                    b_mat = b * a_mat + c * (a_mat @ a_mat)
                    x = a * x + b_mat @ x

                if transposed:
                    x = x.T

                # Eq. 6: W_{t+1} = W_t - eta * sqrt(m / n) * O_t
                p.add_(
                    x.to(p.dtype),
                    alpha=-group["lr"] * math.sqrt(cols / rows),
                )

        return loss
