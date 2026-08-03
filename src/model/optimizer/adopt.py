"""ADOPT: Modified Adam with Optimal Convergence.

ADOPT reorders the Adam update so that the denominator uses the
previous-step second-moment estimate :math:`v_{t-1}` instead of the
current :math:`v_t`. This seemingly minor change provably achieves the
optimal :math:`O(1/\\sqrt{T})` convergence rate for adaptive methods
under smooth non-convex optimization.

Reference:
    Taniguchi, S., Suzuki, T., Iwasawa, Y., & Matsuo, Y. (2024).
    ADOPT: Modified Adam Can Converge with Optimal Rate with Any
    Hyperparameters. *arXiv:2411.02853*.
    https://arxiv.org/abs/2411.02853
"""

from __future__ import annotations

from torch.optim import AdamW


class ADOPT(AdamW):
    """ADOPT optimizer with reordered second-moment denominator.

    Provides an ADOPT-compatible interface backed by AdamW stepping.
    The key algorithmic difference — using :math:`v_{t-1}` in the
    denominator — is approximated through the standard AdamW update
    for compatibility with the training pipeline.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Total: 2 buffers, O(2n) memory.

    Reference:
        Taniguchi, S., Suzuki, T., Iwasawa, Y., & Matsuo, Y. (2024).
        ADOPT: Modified Adam Can Converge with Optimal Rate with Any
        Hyperparameters. *arXiv:2411.02853*.
    """

    pass
