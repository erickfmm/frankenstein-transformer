"""Conda optimizer with column-normalized projected Adam updates.

Conda (Wang et al. 2025, arXiv:2509.24218) keeps Adam's coordinate-wise
adaptivity inside the singular subspace of the first moment.  The momentum
buffer ``M_t`` is decomposed with a *lazy* SVD (refreshed every
``update_proj_gap`` steps), the second moment is an EMA of the squared
projected gradients ``(P^T G_t)^2``, and the bias-corrected, epsilon-stabilized
Adam update is applied in the projected basis before being mapped back to the
parameter space.  This yields a column-wise adaptive normalizer that is milder
than Muon's uniform singular-value normalization and preserves the relative
order of the singular values.

Only 2-D parameters take the Conda path; every non-matrix parameter (biases,
normalization scales, convolution kernels) receives a decoupled AdamW update,
mirroring the released reference implementation.

Reference:
    Wang, J., Zhou, P., Dong, Y., Li, H., Li, J., Zhou, X., Lao, Q., Fang, C.,
    & Lin, Z. (2025). Conda: Column-Normalized Adam for Training Large Language
    Models Faster. arXiv:2509.24218.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from .base import adamw_fallback_update


class Conda(Optimizer):
    r"""Conda: Column-Normalized Adam.

    Accumulates the first moment of the gradient, periodically orthogonalizes
    it with an SVD, and normalizes the projected first moment element-wise by
    the square root of an EMA of the squared *projected* gradients.  The
    resulting update lives in the same subspace as the momentum and is mapped
    back through the projection basis before being applied.

    For a 2-D parameter :math:`W \in \mathbb{R}^{m \times n}` the update is

    .. math::

        \begin{aligned}
        M_t &= \beta_1 M_{t-1} + (1-\beta_1) G_t, \\
        P_t &= \text{SVD basis of } M_t \text{ (refreshed every } T \text{ steps)}, \\
        N_t &= \beta_2 N_{t-1} + (1-\beta_2)(P_t^\top G_t)^2, \\
        W_t &= W_{t-1} - \eta\, P_t \frac{M_t'/(1-\beta_1^t)}
                   {\sqrt{N_t/(1-\beta_2^t)} + \epsilon},
        \end{aligned}

    with :math:`M_t' = P_t^\top M_t`.  When :math:`m > n` the right singular
    basis :math:`V_t` is used instead (``M_t' = M_t V_t`` and the update is
    ``(M_t'/\sqrt{N_t}) V_t^\top``), following the paper's Appendix A.4.

    Reference:
        Wang, J., Zhou, P., Dong, Y., Li, H., Li, J., Zhou, X., Lao, Q., Fang,
        C., & Lin, Z. (2025). Conda: Column-Normalized Adam for Training Large
        Language Models Faster. arXiv:2509.24218.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for the first and second moment running averages
            (default: ``(0.9, 0.99)``).
        eps: Term added to the projected second-moment denominator for
            numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        correct_bias: Whether to apply the :math:`1-\beta^t` bias corrections
            of Algorithm 1 (default: ``True``).
        update_proj_gap: Number of steps between SVD refreshes of the
            projection basis (default: 2000).
        scale: Multiplicative scale applied to the update before the step;
            the paper's pretraining runs use ``0.25`` and its LoRA runs ``1.0``
            (default: 1.0).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (first moment, projected
            second moment, projection basis and its orientation, step counter).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        correct_bias=True,
        update_proj_gap=2000,
        scale=1.0,
    ):
        """Initializes the Conda optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            betas: First/second-moment decay coefficients
                (default: ``(0.9, 0.99)``).
            eps: Numerical stability term for the projected denominator
                (default: 1e-8).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).
            correct_bias: Enable the bias corrections (default: ``True``).
            update_proj_gap: Steps between SVD refreshes (default: 2000).
            scale: Update scale factor (default: 1.0).

        Raises:
            ValueError: If any hyper-parameter is outside its valid range.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 1 <= update_proj_gap:
            raise ValueError(f"Invalid update_proj_gap value: {update_proj_gap}")
        if not 0.0 <= scale:
            raise ValueError(f"Invalid scale value: {scale}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            update_proj_gap=update_proj_gap,
            scale=scale,
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
            lr = group["lr"]
            eps = group.get("eps", 1e-8)
            weight_decay = group["weight_decay"]
            update_proj_gap = int(group["update_proj_gap"])
            scale = group["scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Non-matrix parameters take the decoupled AdamW fallback,
                # exactly as in the released reference implementation where
                # only ``grad.ndim == 2`` tensors reach the Conda path.
                if p.ndim != 2:
                    adamw_fallback_update(
                        p,
                        grad,
                        self.state[p],
                        lr=lr,
                        betas=(beta1, beta2),
                        eps=eps,
                        weight_decay=weight_decay,
                    )
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq_proj"] = None
                    state["proj_basis"] = None
                    state["proj_type"] = None

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]

                # Eq. (5): M_t = beta1 * M_{t-1} + (1 - beta1) * G_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Algorithm 1, lines 6-10: lazy SVD of the first moment.
                if step % update_proj_gap == 0 or state["proj_basis"] is None:
                    u, _, vh = torch.linalg.svd(exp_avg, full_matrices=False)
                    if grad.shape[0] <= grad.shape[1]:
                        state["proj_basis"] = u
                        state["proj_type"] = "left"
                    else:
                        state["proj_basis"] = vh
                        state["proj_type"] = "right"

                basis = state["proj_basis"]

                # Eq. (5): M'_t = P^T M_t and the projected gradient P^T G_t.
                if state["proj_type"] == "left":
                    grad_proj = basis.T @ grad
                    mom_proj = basis.T @ exp_avg
                else:
                    grad_proj = grad @ basis.T
                    mom_proj = exp_avg @ basis.T

                if (
                    state["exp_avg_sq_proj"] is None
                    or state["exp_avg_sq_proj"].shape != grad_proj.shape
                ):
                    state["exp_avg_sq_proj"] = torch.zeros_like(grad_proj)
                exp_avg_sq_proj = state["exp_avg_sq_proj"]

                # Eq. (5): N_t = beta2 * N_{t-1} + (1 - beta2) * (P^T G_t)^2
                exp_avg_sq_proj.mul_(beta2).addcmul_(grad_proj, grad_proj, value=1.0 - beta2)

                # Algorithm 1, line 13: bias correction of both moments.
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                else:
                    bias_correction1 = 1.0
                    bias_correction2 = 1.0
                mom_proj = mom_proj / bias_correction1
                denom = (exp_avg_sq_proj / bias_correction2).sqrt().add_(eps)

                # Eq. (5): W_t = W_{t-1} - eta * P_t (M'_t / sqrt(N_t)).
                if state["proj_type"] == "left":
                    update = basis @ (mom_proj / denom)
                else:
                    update = (mom_proj / denom) @ basis

                update.mul_(scale)

                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss
