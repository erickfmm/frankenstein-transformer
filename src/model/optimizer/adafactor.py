"""Adafactor optimizer with factorized second-moment estimation.

Adafactor (Shazeer & Stern 2018, arXiv:1804.04235) reduces memory consumption
by factorizing the second-moment accumulator for matrix-shaped parameters into
row and column running averages, achieving sublinear (O(n + m)) memory instead
of the O(nm) required by Adam.  Scalar and vector parameters fall back to a
full second-moment buffer.  A relative-step clipping mechanism stabilises
updates without per-parameter learning rates.

Reference:
    Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive Learning Rates with
    Sublinear Memory Cost. arXiv:1804.04235.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    """Adafactor optimizer with factorized second-moment estimation.

    Factorizes the exponentially weighted moving average of squared gradients
    into row-wise and column-wise components for matrix parameters, reducing
    memory from O(nm) to O(n + m).  Scalar and vector parameters use a
    standard full second-moment buffer.  Relative-step clipping caps the
    root-mean-square of the update to ``clip_threshold``.

    Reference:
        Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive Learning Rates
        with Sublinear Memory Cost. arXiv:1804.04235.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).
        beta2_decay: Exponent controlling the decay schedule for the
            second-moment running average.  The effective beta2 at step t is
            ``1.0 - t ** (-beta2_decay)`` (default: 0.8).
        eps: Tuple of ``(eps1, eps2)``.  ``eps1`` is added inside the square
            root for numerical stability; ``eps2`` is added outside
            (default: ``(1e-30, 1e-3)``).
        clip_threshold: Root-mean-square clipping threshold.  When the RMS of
            the update exceeds this value, the update is scaled down
            proportionally (default: 1.0).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (step counter, row/column
            running averages, or full second-moment buffer).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        beta2_decay=0.8,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
    ):
        """Initializes the Adafactor optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            beta2_decay: Exponent for the beta2 decay schedule (default: 0.8).
            eps: Tuple of ``(eps1, eps2)`` for numerical stability
                (default: ``(1e-30, 1e-3)``).
            clip_threshold: RMS clipping threshold (default: 1.0).
        """
        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps=eps,
            clip_threshold=clip_threshold,
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
            lr = float(group["lr"])
            eps1, eps2 = group["eps"]
            clip_threshold = float(group["clip_threshold"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if grad.dim() >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad.shape[:-1], dtype=grad.dtype, device=grad.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad.shape[:-2] + (grad.shape[-1],), dtype=grad.dtype, device=grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                t = state["step"]
                beta2 = 1.0 - (t ** -float(group["beta2_decay"]))

                grad_sq = grad.pow(2).add(eps1)

                if grad.dim() >= 2:
                    row_sums = torch.mean(grad_sq, dim=-1)
                    col_sums = torch.mean(grad_sq, dim=-2)

                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2).add_(row_sums, alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(col_sums, alpha=1.0 - beta2)

                    row_avg = exp_avg_sq_row / (1.0 - beta2 ** t)
                    col_avg = exp_avg_sq_col / (1.0 - beta2 ** t)

                    row_avg_mean = torch.mean(row_avg, dim=-1, keepdim=True)
                    v = (row_avg.unsqueeze(-1) * col_avg.unsqueeze(-2)) / (row_avg_mean + eps1)
                    update = grad / torch.sqrt(v + eps2)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)
                    v = exp_avg_sq / (1.0 - beta2 ** t)
                    update = grad / torch.sqrt(v + eps2)

                rms = torch.norm(update) / (update.numel() ** 0.5)
                divisor = max(1.0, float(rms) / clip_threshold)
                update.div_(divisor)

                p.add_(update, alpha=-lr)

        return loss
