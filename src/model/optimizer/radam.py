"""RAdam: Rectified Adam.

Applies variance rectification to the adaptive learning rate of Adam,
automatically switching from a warmup-phase SMA (simple moving average)
regime to the standard Adam update once the exponential moving average
has accumulated sufficient samples. This removes the need for manual
learning-rate warmup heuristics.

Reference:
    Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J.
    (2020). On the Variance of the Adaptive Learning Rate and Beyond.
    *arXiv:1908.03265*.
    https://arxiv.org/abs/1908.03265
"""

from __future__ import annotations

from torch.optim import RAdam as TorchRAdam


class RAdamOptimizer(TorchRAdam):
    """Rectified Adam with automatic variance-based warmup.

    Thin wrapper around ``torch.optim.RAdam``. The rectification
    term :math:`r_t` gates the adaptive learning rate, suppressing
    updates when the EMA variance is unreliable during early steps.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Total: 2 buffers, O(2n) memory.

    Reference:
        Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., &
        Han, J. (2020). On the Variance of the Adaptive Learning Rate
        and Beyond. *arXiv:1908.03265*.
    """

    pass
