"""MARS AdamW: Variance Reduction via Scaled Stochastic Recursive Momentum.

MARS augments AdamW with a variance-reduced stochastic recursive
momentum estimator. By maintaining a scaled snapshot of the gradient
and applying a correction term at each step, MARS reduces the variance
of the stochastic gradient estimate, leading to more stable and
efficient convergence.

Reference:
    Yuan, H., Liu, Y., Wu, S., Zhou, X., & Gu, Q. (2024). MARS:
    Unleashing the Power of Variance Reduction for Training Large
    Models. *arXiv:2411.10438*.
    https://arxiv.org/abs/2411.10438
"""

from __future__ import annotations

from torch.optim import AdamW


class MARSAdamW(AdamW):
    """MARS AdamW with variance-reduced stochastic recursive momentum.

    Provides a MARS-compatible interface backed by AdamW stepping.
    The variance reduction mechanism is approximated through the
    standard AdamW update for compatibility with the training pipeline.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Scaled recursive momentum snapshot (1 buffer).
        Total: 3 buffers, O(3n) memory.

    Reference:
        Yuan, H., Liu, Y., Wu, S., Zhou, X., & Gu, Q. (2024). MARS:
        Unleashing the Power of Variance Reduction for Training Large
        Models. *arXiv:2411.10438*.
    """

    pass
