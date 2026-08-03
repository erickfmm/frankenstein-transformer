"""Turbo-Muon optimizer with AOL-preconditioned orthogonalization.

Turbo-Muon (Boissin et al. 2025, arXiv:2512.04632) extends Muon by applying
an Approximate Orthogonalization via Lowdin (AOL) preconditioner before the
Newton-Schulz iterations.  The AOL step scales rows by the inverse square
root of the row-wise absolute sum of the Gram matrix, accelerating convergence
and reducing the number of Newton-Schulz iterations from 5 to 4.

Reference:
    Boissin, T., Fournier, L., & Gidel, G. (2025). Turbo-Muon: Accelerating
    Muon with AOL Preconditioning. arXiv:2512.04632.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class TurboMuon(Optimizer):
    """Turbo-Muon optimizer with AOL-preconditioned orthogonalization.

    Applies momentum to matrix-shaped gradients, preconditions with AOL
    (row-wise absolute-sum scaling of the Gram matrix), then orthogonalizes
    via Newton-Schulz iterations.  Only parameters with ``ndim >= 2`` are
    processed.

    Reference:
        Boissin, T., Fournier, L., & Gidel, G. (2025). Turbo-Muon:
        Accelerating Muon with AOL Preconditioning. arXiv:2512.04632.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient for the gradient buffer
            (default: 0.95).
        nesterov: Whether to use Nesterov-style momentum (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 4).
        ns_eps: Epsilon for numerical stability in AOL scaling and
            normalization (default: 1e-7).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer).
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=4,
        ns_eps=1e-7,
        weight_decay=0.0,
    ):
        """Initializes the Turbo-Muon optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 0.02).
            momentum: Momentum coefficient (default: 0.95).
            nesterov: Use Nesterov momentum (default: True).
            ns_steps: Newton-Schulz iteration count (default: 4).
            ns_eps: Numerical stability epsilon (default: 1e-7).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
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
            for p in group["params"]:
                if p.grad is None or p.ndim < 2:
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)
                g = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                a, b, c = 3.4445, -4.7750, 2.0315
                x = g.bfloat16()

                transposed = False
                if x.size(0) > x.size(1):
                    x = x.T
                    transposed = True

                a_mat = x @ x.T
                s = torch.sum(torch.abs(a_mat), dim=1, keepdim=True).clamp(min=group["ns_eps"]).pow(-0.5)
                x = x * s
                a_mat = a_mat * s * s.T

                for i in range(int(group["ns_steps"])):
                    b_mat = b * a_mat + c * (a_mat @ a_mat)
                    x = a * x + b_mat @ x
                    if i < int(group["ns_steps"]) - 1:
                        a_mat = x @ x.T

                if transposed:
                    x = x.T

                p.add_(x.to(p.dtype), alpha=-group["lr"])

        return loss
