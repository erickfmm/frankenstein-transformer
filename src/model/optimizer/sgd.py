"""Vanilla mini-batch stochastic gradient descent (Hinton et al. 2012).

Plain mini-batch SGD as defined in the CSC321 lecture notes ("Overview of
mini-batch gradient descent"): on redundant datasets, update the weights from
the gradient of a mini-batch instead of the full batch,

    theta_{t+1} = theta_t - eta * g_t,

with no momentum and no optimizer state. Serves as the zero-state baseline of
the optimizer registry; :class:`~src.model.optimizer.sgd_momentum.SGDMomentum`
adds the classical velocity buffer on top.

Reference:
    Hinton, G. E., Srivastava, N., & Swersky, K. (2012). Neural Networks for
    Machine Learning, Lecture 6: Overview of Mini-Batch Gradient Descent.
    CSC321 course slides, University of Toronto.
    https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
"""

from __future__ import annotations

from torch.optim import SGD


class SGDOptimizer(SGD):
    """Mini-batch stochastic gradient descent without momentum.

    Wraps ``torch.optim.SGD`` with ``momentum=0``, ``dampening=0`` and
    ``nesterov=False`` so the update is exactly the scaled gradient of the
    current mini-batch.

    State buffers (per parameter):
        none — this is the zero-state baseline (no velocity, no moments).

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default ``1e-3``).
        weight_decay: L2 penalty coefficient folded into the gradient
            (default ``0.0``; the CSC321 deck defines no weight decay).
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        super().__init__(
            params,
            lr=lr,
            momentum=0.0,
            dampening=0.0,
            weight_decay=weight_decay,
            nesterov=False,
        )
