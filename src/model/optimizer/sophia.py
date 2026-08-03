"""Sophia optimizer with diagonal Hessian estimation.

Sophia (Liu et al. 2023, arXiv:2305.14342) is a second-order optimizer that
estimates the diagonal of the Hessian via a Hutchinson-style stochastic
estimator and uses it to clip gradient updates element-wise.  This provides
stronger curvature-aware preconditioning than Adam while maintaining
comparable memory overhead (three state buffers per parameter).

Reference:
    Liu, H., Li, Z., Hall, D., Liang, P., & Ma, T. (2023). Sophia: A
    Scalable Stochastic Second-order Optimizer for Language Model Pre-training.
    arXiv:2305.14342.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class Sophia(Optimizer):
    """Sophia optimizer with diagonal Hessian estimation.

    Maintains an exponential moving average of gradients and a diagonal
    Hessian estimate.  The Hessian estimate is updated periodically (every
    ``update_k`` steps) from an externally supplied Hutchinson estimator.
    Updates are clipped element-wise by ``rho * hessian`` before being
    applied.

    Reference:
        Liu, H., Li, Z., Hall, D., Liang, P., & Ma, T. (2023). Sophia: A
        Scalable Stochastic Second-order Optimizer for Language Model
        Pre-training. arXiv:2305.14342.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for first-moment and Hessian moving averages
            (default: ``(0.965, 0.99)``).
        rho: Clipping strength for the Hessian-based update bound
            (default: 0.04).
        weight_decay: Decoupled weight decay coefficient (default: 1e-1).
        update_k: Frequency (in steps) for updating the Hessian estimate
            (default: 10).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (step counter, first
            moment buffer, Hessian estimate buffer).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        update_k=10,
    ):
        """Initializes the Sophia optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            betas: Momentum and Hessian decay coefficients
                (default: ``(0.965, 0.99)``).
            rho: Hessian clipping strength (default: 0.04).
            weight_decay: Decoupled weight decay coefficient (default: 1e-1).
            update_k: Hessian update frequency in steps (default: 10).
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            update_k=update_k,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, hessian_estimate=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss.  Optional for most use cases.
            hessian_estimate: A dict mapping parameters to their diagonal
                Hessian estimates (from a Hutchinson estimator).  If
                ``None``, the Hessian estimate is not updated this step.

        Returns:
            The loss value if ``closure`` is provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if hessian_estimate is not None and step % int(group["update_k"]) == 1:
                    est = hessian_estimate.get(p) if isinstance(hessian_estimate, dict) else None
                    if est is not None:
                        hessian.mul_(beta2).add_(est, alpha=1 - beta2)

                p.mul_(1 - group["lr"] * group["weight_decay"])

                h_max = torch.max(group["rho"] * hessian, torch.full_like(p, 1e-15))
                update = exp_avg / h_max
                update.clamp_(-1.0, 1.0)
                p.add_(update, alpha=-group["lr"])

        return loss
