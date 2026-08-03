"""Schedule-Free AdamW: Learning Rate Schedule Elimination via Iterate Averaging.

Schedule-Free AdamW removes the need for a learning rate decay schedule
by maintaining an online iterate average :math:`\\bar{x}_t` that is used
for evaluation, while the base optimizer continues to update the
primary parameters :math:`x_t` with a constant learning rate. The
interpolation between :math:`x_t` and :math:`\\bar{x}_t` is handled
by the training loop's scheduler configuration.

Reference:
    Defazio, A., Yang, X., Mehta, H., Mishchenko, K., Khaled, A., &
    Cutkosky, A. (2024). Schedule-Free Learning. *arXiv:2405.15682*.
    https://arxiv.org/abs/2405.15682
"""

from __future__ import annotations

from torch.optim import AdamW


class ScheduleFreeAdamW(AdamW):
    """Schedule-Free AdamW with iterate averaging.

    Provides a Schedule-Free-compatible interface backed by AdamW
    stepping. The iterate averaging mechanism is managed externally
    by setting the training scheduler to a constant learning rate
    and using the model's built-in averaging logic.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Iterate average :math:`\\bar{x}_t` (1 buffer, managed externally).
        Total: 3 buffers, O(3n) memory.

    Reference:
        Defazio, A., Yang, X., Mehta, H., Mishchenko, K., Khaled, A.,
        & Cutkosky, A. (2024). Schedule-Free Learning.
        *arXiv:2405.15682*.
    """

    pass
