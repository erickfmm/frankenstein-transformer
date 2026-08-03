"""LAMB: Layer-wise Adaptive Moments for Batch training.

LAMB extends AdamW with a layer-wise trust ratio that scales the
update step proportionally to the ratio of parameter norm to update
norm. This normalization enables stable training with very large batch
sizes (e.g. 32K+ for BERT pretraining) without catastrophic divergence.

Reference:
    You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S.,
    Song, X., Demmel, J., Keutzer, K., & Hsieh, C.-J. (2020). Large
    Batch Optimization for Deep Learning: Training BERT in 76 minutes.
    *arXiv:1904.00962*.
    https://arxiv.org/abs/1904.00962
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class LAMB(Optimizer):
    """LAMB optimizer with layer-wise adaptive trust ratio scaling.

    Implements the full LAMB algorithm from scratch (not wrapping
    AdamW) to provide the trust-ratio normalization step. Each
    parameter's update is scaled by
    :math:`\\phi(\\|w\\|) / \\|u\\|` where :math:`\\phi` is a
    minimum-clamped identity function.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Total: 2 buffers, O(2n) memory.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default ``1e-3``).
        betas: Coefficients for first and second moment estimates
            (default ``(0.9, 0.999)``).
        eps: Numerical stability term for denominator (default ``1e-6``).
        weight_decay: Decoupled weight decay coefficient (default ``0.0``).

    Reference:
        You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S.,
        Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., &
        Hsieh, C.-J. (2020). Large Batch Optimization for Deep
        Learning: Training BERT in 76 minutes. *arXiv:1904.00962*.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single LAMB optimization step.

        Computes bias-corrected Adam-style moment estimates, applies
        decoupled weight decay, then scales the update by the
        layer-wise trust ratio :math:`\\|w\\| / \\|u\\|`.

        Args:
            closure: Optional callable that reevaluates the model and
                returns the loss.

        Returns:
            The loss value returned by `closure` (if provided), or
            ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** state["step"]
                bias_c2 = 1 - beta2 ** state["step"]
                update = (exp_avg / bias_c1) / ((exp_avg_sq / bias_c2).sqrt() + eps)
                if weight_decay != 0:
                    update = update + weight_decay * p

                w_norm = torch.norm(p).clamp(min=1e-12)
                u_norm = torch.norm(update).clamp(min=1e-12)
                trust_ratio = (w_norm / u_norm).item()
                p.add_(update, alpha=-lr * trust_ratio)

        return loss
