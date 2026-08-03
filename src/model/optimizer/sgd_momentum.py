"""SGD with Momentum (Polyak 1964).

Classical first-order optimizer that accumulates a velocity buffer
(one state per parameter) and applies Nesterov-accelerated lookahead
when requested. O(n) memory cost. Serves as the simplest baseline in
the optimizer registry.

Reference:
    Polyak, B. T. (1964). Some methods of speeding up the convergence
    of iteration methods. *USSR Computational Mathematics and
    Mathematical Physics*, 4(5), 1–17.
"""

from __future__ import annotations

from torch.optim import SGD


class SGDMomentum(SGD):
    """SGD with classical momentum (Polyak heavy-ball method).

    Wraps ``torch.optim.SGD`` with ``dampening=0`` to provide pure
    momentum accumulation. Supports optional Nesterov accelerated
    gradient lookahead.

    State buffers (per parameter):
        ``momentum_buffer``: Velocity accumulator (1 buffer, O(n) memory).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default ``1e-3``).
        momentum: Momentum factor :math:`\\mu` (default ``0.9``).
        weight_decay: L2 penalty coefficient (default ``0.0``).
        nesterov: If ``True``, apply Nesterov lookahead (default ``False``).
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, nesterov=False):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=0.0,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
