"""RMSProp (Tieleman & Hinton 2012).

RMSProp is the mini-batch-safe version of rprop: keep a moving average of the
squared gradient for each weight and divide the gradient by its square root,

    MeanSquare(w, t) = 0.9 * MeanSquare(w, t-1) + 0.1 * (dE/dw(t))^2
    w_{t+1} = w_t - eta * g_t / sqrt(MeanSquare(w, t)),

so the divisor stays nearly identical for adjacent mini-batches ("Dividing the
gradient by sqrt(MeanSquare(w, t)) makes the learning work much better",
Tijmen Tieleman, unpublished). One state buffer, no bias correction, no first
moment — the momentum/centered variants are deliberate non-goals of this
implementation.

Reference:
    Tieleman, T., & Hinton, G. E. (2012). RMSProp: Divide the Gradient by a
    Running Average of Its Recent Magnitude. CSC321 course slides, Lecture 6e,
    University of Toronto.
    https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
"""

from __future__ import annotations

from torch.optim import RMSprop


class RMSProp(RMSprop):
    """RMSProp with the paper's 0.9/0.1 running-average split.

    Wraps ``torch.optim.RMSprop`` with ``momentum=0`` and ``centered=False``.
    The squared-gradient average ``square_avg`` is the only state buffer, and
    the update is ``theta -= lr * g / (sqrt(square_avg) + eps)`` — the slides'
    equation with ``alpha = 0.9`` (torch's own default is ``0.99``).

    State buffers (per parameter):
        ``square_avg``: Exponential moving average of the squared gradient
        (1 buffer, O(n) memory).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default ``1e-3``; torch's own default is ``1e-2``).
        alpha: EMA decay of the squared-gradient running average; the slides'
            0.9/0.1 split (default ``0.9``).
        eps: Denominator epsilon added after the square root (default ``1e-8``;
            a numerical guard not present in the slides).
        weight_decay: L2 penalty coefficient folded into the gradient
            (default ``0.0``; the CSC321 deck defines no weight decay).

    Raises:
        ValueError: If ``lr``, ``alpha``, ``eps`` or ``weight_decay`` are out
            of their valid ranges.
    """

    def __init__(self, params, lr=1e-3, alpha=0.9, eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=0.0,
            centered=False,
        )
