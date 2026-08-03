"""GaLore AdamW optimizer with low-rank gradient projection.

GaLore (Zhao et al. 2024, arXiv:2403.03507) reduces memory consumption during
training by projecting 2D gradients into a low-rank subspace via periodic SVD.
AdamW-style momentum and adaptive learning rates are applied in the compressed
space, and the resulting update is projected back to the original parameter
space.  This achieves O(nr) memory for the optimizer state instead of O(nm).

Reference:
    Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y.
    (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
    Projection. arXiv:2403.03507.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class GaLoreAdamW(Optimizer):
    """GaLore AdamW optimizer with low-rank gradient projection.

    Projects 2D gradients into a low-rank subspace via periodic truncated SVD,
    applies AdamW updates in the compressed space, and projects the result
    back to the original parameter dimensions.  Non-matrix parameters are
    updated with standard AdamW.

    Reference:
        Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y.
        (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
        Projection. arXiv:2403.03507.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).
        rank: Rank of the low-rank projection subspace (default: 128).
        update_proj_gap: Number of steps between SVD-based projector
            recomputations (default: 200).
        betas: Coefficients for first and second moment running averages
            (default: ``(0.9, 0.999)``).
        eps: Term added for numerical stability (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 1e-2).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (step counter, projector
            matrix, low-rank moment buffers).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        rank=128,
        update_proj_gap=200,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    ):
        """Initializes the GaLore AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            rank: Low-rank projection rank (default: 128).
            update_proj_gap: Steps between projector recomputations
                (default: 200).
            betas: Momentum decay coefficients (default: ``(0.9, 0.999)``).
            eps: Numerical stability term (default: 1e-8).
            weight_decay: Weight decay coefficient (default: 1e-2).
        """
        defaults = dict(
            lr=lr,
            rank=rank,
            update_proj_gap=update_proj_gap,
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
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                g_low = grad
                if grad.dim() == 2:
                    if "projector" not in state or step % int(group["update_proj_gap"]) == 1:
                        u, _, vh = torch.linalg.svd(grad.float(), full_matrices=False)
                        rank = min(int(group["rank"]), min(grad.shape))
                        if grad.shape[0] >= grad.shape[1]:
                            state["projector"] = u[:, :rank].to(grad.dtype)
                            state["proj_type"] = "left"
                        else:
                            state["projector"] = vh[:rank, :].to(grad.dtype)
                            state["proj_type"] = "right"

                    pmat = state["projector"]
                    if state["proj_type"] == "left":
                        g_low = pmat.T @ grad
                    else:
                        g_low = grad @ pmat.T

                if "exp_avg" not in state or state["exp_avg"].shape != g_low.shape:
                    state["exp_avg"] = torch.zeros_like(g_low)
                    state["exp_avg_sq"] = torch.zeros_like(g_low)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g_low, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_low, g_low, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                update_low = exp_avg / denom

                if grad.dim() == 2:
                    pmat = state["projector"]
                    if state["proj_type"] == "left":
                        update = pmat @ update_low
                    else:
                        update = update_low @ pmat
                else:
                    update = update_low

                p.add_(update, alpha=-step_size)

        return loss
