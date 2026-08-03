"""Lion optimizer with sign-based updates.

Lion (EvoLved Sign Momentum; Chen et al. 2023, arXiv:2302.06675) is a
discovered optimizer that uses only sign operations for the update direction,
requiring a single momentum buffer per parameter.  It combines an
exponentially weighted moving average of past gradients with the current
gradient via a sign-based update rule, achieving strong performance with
minimal memory overhead.

Reference:
    Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X.,
    Luong, T., Hsieh, C.-J., Lu, Y., & Le, Q. V. (2023). Symbolic Discovery
    of Optimization Algorithms. arXiv:2302.06675.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer with sign-based updates.

    Maintains a single exponential moving average of gradients and uses the
    sign of a weighted combination of the current gradient and the momentum
    buffer to determine the update direction.  Decoupled weight decay is
    applied before the update.

    Reference:
        Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong,
        X., Luong, T., Hsieh, C.-J., Lu, Y., & Le, Q. V. (2023). Symbolic
        Discovery of Optimization Algorithms. arXiv:2302.06675.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-4).
        betas: Coefficients for the momentum combination and the exponential
            moving average decay (default: ``(0.9, 0.99)``).
        weight_decay: Decoupled weight decay coefficient (default: 0.1).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1):
        """Initializes the Lion optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-4).
            betas: Momentum coefficients ``(beta1, beta2)``
                (default: ``(0.9, 0.99)``).
            weight_decay: Decoupled weight decay coefficient (default: 0.1).
        """
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                if wd > 0:
                    p.mul_(1 - lr * wd)

                c = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1)
                p.add_(torch.sign(c), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
