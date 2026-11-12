r"""LaProp: decoupled momentum and adaptivity in Adam-style optimization.

LaProp (Ziyin, Wang & Ueda, 2020, arXiv:2002.04839) identifies an unnecessary
coupling in Adam: Adam's parameter-space momentum is reweighted on every step
by the current preconditioner through the factor :math:`H_t^{-1}H_{t-1}`, so a
single pathologically large gradient (noise, out-of-distribution batch, or an
explosion) can erase the accumulated momentum.  LaProp sets :math:`K_t = I`
in the generalized adaptive framework, i.e. it takes the exponential moving
average of the *preconditioned* gradient rather than of the raw gradient, and
leaves the parameter-space momentum additive exactly like SGD:

.. math::

    n_t &= \nu n_{t-1} + (1-\nu) g_t^2 \\
    m_t &= \mu m_{t-1} + (1-\mu) \frac{g_t}{\sqrt{\hat{n}_t} + \epsilon} \\
    \theta_{t+1} &= \theta_t - \lambda \hat{m}_t

with :math:`\hat{n}_t = n_t / (1-\nu^t)` and
:math:`\hat{m}_t = m_t / (1-\mu^t)` the usual bias corrections.  The method
has exactly the same state (two buffers), per-step cost and hyperparameter
count as Adam; :math:`\nu \to 0` recovers signed-gradient methods and
:math:`\mu = 0` recovers RMSProp with bias correction (the paper calls LaProp
"RMSProp with momentum" plus both corrections).

Reference:
    Ziyin, L., Wang, Z. T., & Ueda, M. (2021). LaProp: Separating Momentum
    and Adaptivity in Adam. *arXiv:2002.04839*.
    https://arxiv.org/abs/2002.04839
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class LaProp(Optimizer):
    r"""LaProp optimizer (Algorithm 1 of arXiv:2002.04839).

    The first moment is an EMA of the *normalized* gradient
    :math:`g_t / (\sqrt{\hat{n}_t} + \epsilon)`, so the momentum accumulates
    preconditioned directions instead of raw gradients.  Only the second
    moment needs bias correction inside its own recursion; the first moment is
    corrected once, in the step size.

    State buffers (per parameter):
        ``step``: Update counter :math:`t` (scalar).
        ``exp_avg``: First moment :math:`m_t` of the preconditioned gradient
            (1 buffer).
        ``exp_avg_sq``: Second moment :math:`n_t` of the raw gradient
            (1 buffer).
        Total: 2 tensors, :math:`\mathcal{O}(2n)` memory — identical to Adam.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.  Groups may carry ``lr``, ``betas``, ``eps`` and
            ``weight_decay``.
        lr: Learning rate :math:`\lambda` (default ``1e-3``).
        betas: Coefficients :math:`(\mu, \nu)` for the first and second moment
            running averages (default ``(0.9, 0.999)``, Adam's settings — the
            paper states Adam's hyperparameters carry over).
        eps: Numerical-stability term :math:`\epsilon` added to the
            bias-corrected second-moment denominator (default ``1e-8``).
        weight_decay: L2 weight-decay coefficient, coupled into the gradient
            (default ``0.0``; matches the reference LaProp implementation).

    Raises:
        ValueError: If ``lr``, ``eps`` or either beta is out of range.

    Reference:
        Ziyin, L., Wang, Z. T., & Ueda, M. (2021). LaProp: Separating Momentum
        and Adaptivity in Adam. *arXiv:2002.04839*.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Initializes the LaProp optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default ``1e-3``).
            betas: Coefficients for the first and second moment running
                averages (default ``(0.9, 0.999)``).
            eps: Term added to the denominator for numerical stability
                (default ``1e-8``).
            weight_decay: L2 weight-decay coefficient coupled into the
                gradient (default ``0.0``).

        Raises:
            ValueError: If ``lr``, ``eps`` or either beta is out of range.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (Algorithm 1 of the paper).

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss.  Optional for most use cases.

        Returns:
            The loss value if ``closure`` is provided, otherwise ``None``.

        Raises:
            RuntimeError: If a parameter has a sparse gradient.
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
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # n_t = nu * n_{t-1} + (1 - nu) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction2 = 1.0 - beta2**step
                # m_t = mu * m_{t-1} + (1 - mu) * g_t / (sqrt(n_t / (1 - nu^t)) + eps)
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + eps
                exp_avg.mul_(beta1).addcdiv_(grad, denom, value=1.0 - beta1)

                bias_correction1 = 1.0 - beta1**step
                # theta_{t+1} = theta_t - lambda * m_t / (1 - mu^t)
                p.add_(exp_avg, alpha=-lr / bias_correction1)

        return loss
