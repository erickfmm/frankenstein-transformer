"""AdaMuon optimizer with sign-stabilized orthogonal updates.

AdaMuon (Si et al. 2025, arXiv:2507.11005) extends Muon by combining
element-wise adaptivity with orthogonal updates.  The momentum buffer is
passed through a sign transform before Newton-Schulz orthogonalization
(the paper's Theorem 1 shows that ``sign`` is the unique admissible
element-wise transformation), the squared orthogonalized direction is
tracked with a second momentum buffer, and the variance-normalized update
is rescaled to an RMS of ``0.2`` so Adam learning-rate schedules can be
reused directly.  Only parameters with ``ndim >= 2`` are processed.

Reference:
    Si, C., Zhang, D., & Shen, W. (2025). AdaMuon: Adaptive Muon Optimizer.
    arXiv:2507.11005.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

_RMS_TARGET = 0.2
"""Target RMS of the update (Eq. 8): Adam's empirical update scale."""


class AdaMuon(Optimizer):
    r"""AdaMuon optimizer with sign-stabilized orthogonal updates.

    Accumulates momentum on matrix-shaped gradients, applies a sign
    transform followed by Newton-Schulz iterations to obtain an orthogonal
    direction ``O_t``, tracks the element-wise variance of that direction in
    a second momentum buffer ``V_t``, and applies the variance-normalized
    direction rescaled by ``gamma_t = 0.2 * sqrt(mn) / ||O_hat_t||_F``.  Only
    parameters with ``ndim >= 2`` are processed; others are left unchanged.

    Reference:
        Si, C., Zhang, D., & Shen, W. (2025). AdaMuon: Adaptive Muon
        Optimizer. arXiv:2507.11005.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).  The RMS alignment keeps the
            update at Adam's scale, so Adam-style schedules apply unchanged.
        momentum: Momentum coefficient shared by the first- and second-
            moment buffers (default: 0.95).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        ns_eps: Epsilon for numerical stability in the Newton-Schulz
            normalization and the RMS-alignment denominator (default: 1e-7).
        weight_decay: Decoupled weight decay coefficient (default: 0.0);
            applied as ``W <- (1 - lr * wd) * W``, which is algebraically
            identical to the paper's ``-eta * lambda * W`` term.

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (first- and second-
            moment buffers).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.95,
        ns_steps=5,
        ns_eps=1e-7,
        weight_decay=0.0,
    ):
        """Initializes the AdaMuon optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            momentum: Shared first/second-moment coefficient (default: 0.95).
            ns_steps: Newton-Schulz iteration count (default: 5).
            ns_eps: Numerical stability epsilon (default: 1e-7).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).

        Raises:
            ValueError: If ``lr``, ``momentum``, ``ns_steps`` or ``ns_eps``
                is out of its valid range.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 1 <= ns_steps:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if not 0.0 <= ns_eps:
            raise ValueError(f"Invalid ns_eps value: {ns_eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
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

        for group in self.param_groups:
            momentum = group["momentum"]
            eps = group.get("eps", 1e-8)
            for p in group["params"]:
                if p.grad is None or p.ndim < 2:
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                # Eq. (Alg. 1): M_t = beta * M_{t-1} + G_t
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)

                # Eq. (Alg. 1): O_t = NewtonSchulz(sign(M_t), T)
                x = torch.sign(buffer).bfloat16()
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

                direction = x.to(p.dtype)

                # Eq. (5): V_t = beta * V_{t-1} + (1 - beta) * O_t * O_t
                second = state["exp_avg_sq"]
                second.mul_(momentum).addcmul_(direction, direction, value=1.0 - momentum)

                # Eq. (6): O_hat_t = O_t / (sqrt(V_t) + eps)
                normalized = direction / (second.sqrt() + eps)

                # Eq. (8): gamma_t = 0.2 * sqrt(mn) / ||O_hat_t||_F
                denominator = normalized.norm().clamp_min(group["ns_eps"])
                gamma = _RMS_TARGET * math.sqrt(p.size(0) * p.size(1)) / denominator

                # Eq. (9): W <- W - eta * (gamma * O_hat + lambda * W)
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(normalized, alpha=-group["lr"] * gamma)

        return loss
