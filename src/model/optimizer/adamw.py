"""AdamW: Adam with Decoupled Weight Decay.

Implements the AdamW variant where weight decay is applied directly to
the parameters rather than being mixed into the adaptive gradient
update. This decoupling improves generalization and simplifies
hyperparameter tuning compared to L2-regularized Adam.

Reference:
    Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay
    Regularization. *arXiv:1711.05101*.
    https://arxiv.org/abs/1711.05101
"""

from __future__ import annotations

from torch.optim import AdamW


class AdamWOptimizer(AdamW):
    """AdamW optimizer with decoupled weight decay.

    Thin wrapper around ``torch.optim.AdamW`` that serves as the
    canonical entry in the optimizer registry. All hyperparameters
    (lr, betas, eps, weight_decay) are routed per parameter group
    by the factory.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Total: 2 buffers, O(2n) memory.

    Reference:
        Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay
        Regularization. *arXiv:1711.05101*.
    """

    pass
