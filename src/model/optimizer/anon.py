"""Anon optimizer with continuously tunable adaptivity.

Anon (Anon et al. 2026, arXiv:2605.02317) introduces a continuously tunable
adaptivity exponent ``gamma`` that interpolates between SGD-like (gamma=0)
and Adam-like (gamma=1) behaviour.  It uses an Incremental Delay Update (IDU)
mechanism that accumulates second-moment information at exponentially spaced
intervals, enabling stable convergence across all adaptivity values.

Reference:
    Anon, A., et al. (2026). Anon: Extrapolating Adaptivity Beyond SGD and
    Adam. arXiv:2605.02317.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer

version_higher = torch.__version__ >= "1.5.0"


class Anon(Optimizer):
    r"""Adaptivity Non-restricted Optimizer with Novel convergence technique.

    Implements the Anon optimizer from "Anon: Extrapolating Adaptivity Beyond
    SGD and Adam" (arXiv:2605.02317).  Anon introduces continuously tunable
    adaptivity via ``gamma`` and a novel Incremental Delay Update (IDU)
    mechanism for stable convergence across all adaptivity values.

    Args:
        params:  iterable of parameters to optimise or dicts defining groups.
        lr:       learning rate (default: 1e-3).
        betas:    coefficients for first/second moment estimates (default: (0.9, 0.999)).
        eps:      term added for numerical stability (default: 1e-16).
        gamma:    adaptivity exponent; γ=0 → SGD-like, γ=1 → Adam-like (default: 0.).
        weight_decay:       weight decay coefficient (default: 0).
        weight_decouple:    use decoupled weight decay (default: False).
        fixed_decay:        use a fixed decay schedule (default: False).
        rectify:            enable RAdam-style rectification (default: False).
        degenerated_to_sgd: fallback to SGD behaviour (default: False).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (step counter, first and
            second moment buffers, fixed learning-rate buffer, IDU counter).
        degenerated_to_sgd (bool): Whether to fall back to SGD behaviour.
        weight_decouple (bool): Whether to use decoupled weight decay.
        rectify (bool): Whether RAdam-style rectification is enabled.
        fixed_decay (bool): Whether a fixed decay schedule is used.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        gamma: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        rectify: bool = False,
        degenerated_to_sgd: bool = False,
    ) -> None:
        """Initializes the Anon optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            betas: Coefficients for first and second moment running averages
                (default: ``(0.9, 0.999)``).
            eps: Term added for numerical stability (default: 1e-16).
            gamma: Adaptivity exponent; 0 → SGD-like, 1 → Adam-like
                (default: 0.0).
            weight_decay: Weight decay coefficient (default: 0.0).
            weight_decouple: Use decoupled weight decay (default: False).
            fixed_decay: Use a fixed decay schedule (default: False).
            rectify: Enable RAdam-style rectification (default: False).
            degenerated_to_sgd: Fall back to SGD behaviour (default: False).

        Raises:
            ValueError: If ``lr``, ``eps``, or ``betas`` are out of valid
                range.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            gamma=gamma,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super().__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay

    def __setstate__(self, state: dict) -> None:
        """Restores optimizer state from a serialized dict.

        Args:
            state: Serialized optimizer state dict.
        """
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Applies the Incremental Delay Update (IDU) mechanism: the fixed
        learning-rate buffer is updated at exponentially spaced steps
        (step == 2^t) using the accumulated second-moment information.

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss.  Optional for most use cases.

        Returns:
            The loss value if ``closure`` is provided, otherwise ``None``.

        Raises:
            RuntimeError: If any parameter has sparse gradients.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Anon does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['fixed_lr'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['t'] = 0

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                fixed_lr = state['fixed_lr']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']

                t = state['t']

                if state['step'] == 2 ** t:
                    if t == 0:
                        state['t'] += 1
                        bias_correction2 = 1 - beta2
                        fixed_lr.add_(
                            exp_avg_sq.div_(bias_correction2).add_(group['eps']).pow_(group['gamma']).rsqrt_()
                        )
                        exp_avg_sq.zero_()
                    else:
                        state['t'] += 1
                        bias_correction2 = 1 - beta2 ** (state['step'] / 2)
                        fixed_lr.pow_(2).reciprocal_().add_(
                            exp_avg_sq.div_(bias_correction2).add_(group['eps']).pow_(group['gamma'])
                        ).div_(2).rsqrt_()
                        exp_avg_sq.zero_()

                step_size = group['lr'] / bias_correction1
                p.data.addcmul_(exp_avg, fixed_lr, value=-step_size)

        return loss
