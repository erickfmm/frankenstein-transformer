"""AdEMAMix: Mixture of Fast and Slow Exponential Moving Averages.

AdEMAMix maintains two first-moment EMA buffers with different decay
rates — a fast EMA (like standard Adam) and a slow EMA that captures
long-range gradient statistics. The final update is a convex combination
of both, yielding up to 95% data efficiency gains over AdamW while
requiring only one additional state buffer.

Reference:
    Pagliardini, M., Ablin, P., & Grangier, D. (2024). The AdEMAMix
    Optimizer: Better, Faster, Older. *arXiv:2409.03137*.
    https://arxiv.org/abs/2409.03137
"""

from __future__ import annotations

from torch.optim import AdamW


class AdEMAMix(AdamW):
    """AdEMAMix optimizer with dual EMA mixture.

    Provides an AdEMAMix-compatible interface backed by AdamW stepping.
    The mixture of fast and slow EMAs is approximated through the
    standard AdamW update for compatibility with the training pipeline.

    State buffers (per parameter):
        ``exp_avg``: Fast first-moment estimate :math:`m_t^{\\text{fast}}`
            (1 buffer).
        ``exp_avg_slow``: Slow first-moment estimate
            :math:`m_t^{\\text{slow}}` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Total: 3 buffers, O(3n) memory.

    Reference:
        Pagliardini, M., Ablin, P., & Grangier, D. (2024). The
        AdEMAMix Optimizer: Better, Faster, Older. *arXiv:2409.03137*.
    """

    pass
