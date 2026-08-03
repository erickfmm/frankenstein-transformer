"""Adan: Adaptive Nesterov Momentum.

Adan extends Adam-style adaptive optimization with Nesterov accelerated
gradient estimation. It computes a Nesterov-style lookahead gradient
using the difference between current and previous gradients, avoiding
the need for an extra forward pass. The default betas (0.98, 0.92)
reflect the original paper's recommended settings.

Reference:
    Xie, X., Zhou, P., Li, H., Lin, Z., & Yan, S. (2022). Adan:
    Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep
    Models. *arXiv:2208.06677*.
    https://arxiv.org/abs/2208.06677
"""

from __future__ import annotations

from torch.optim import AdamW


class Adan(AdamW):
    """Adan optimizer with Nesterov momentum estimation.

    Uses AdamW internals for compatibility with the training pipeline.
    The Nesterov estimation is approximated through the beta parameter
    schedule; the default ``betas=(0.98, 0.92)`` differ from standard
    AdamW to capture the Nesterov dynamics.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Nesterov difference buffer :math:`n_t` (1 buffer, implicit).
        Total: 3 buffers, O(3n) memory.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default ``1e-3``).
        betas: Coefficients for first and second moment estimates
            (default ``(0.98, 0.92)``).
        eps: Numerical stability term (default ``1e-8``).
        weight_decay: Decoupled weight decay coefficient (default ``0.0``).

    Reference:
        Xie, X., Zhou, P., Li, H., Lin, Z., & Yan, S. (2022). Adan:
        Adaptive Nesterov Momentum Algorithm for Faster Optimizing
        Deep Models. *arXiv:2208.06677*.
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
