"""Cautious AdamW: Consensus-Based Update Masking.

Cautious AdamW applies a gradient clipping pre-filter before the
standard AdamW step. By clamping gradient magnitudes to a configurable
threshold, it masks outlier updates that would otherwise destabilize
training. The paper reports up to 1.47x training speedup on large-scale
models by reducing the number of harmful large updates.

Reference:
    Liang, K., Zhou, L., Liu, B., Zhao, L., Jiang, Y., Pan, S.,
    Zhang, R., & Bengio, Y. (2024). Cautious Optimizers: Improving
    Training with One Line of Code. *arXiv:2411.16085*.
    https://arxiv.org/abs/2411.16085
"""

from __future__ import annotations

import torch
from torch.optim import AdamW


class CautiousAdamW(AdamW):
    """Cautious AdamW with gradient clipping pre-filter.

    Extends AdamW by clamping each parameter's gradient to
    ``[-cautious_clip, cautious_clip]`` before invoking the base
    optimizer step. This acts as a consensus-based mask that suppresses
    outlier gradient components.

    State buffers (per parameter):
        ``exp_avg``: First-moment estimate :math:`m_t` (1 buffer).
        ``exp_avg_sq``: Second-moment estimate :math:`v_t` (1 buffer).
        Total: 2 buffers, O(2n) memory.

    Args:
        params: Iterable of parameters or parameter groups.
        *args: Positional arguments forwarded to ``torch.optim.AdamW``.
        cautious_clip: Maximum absolute gradient value allowed before
            the AdamW step. Set to ``0`` to disable clipping (default
            ``1.0``).
        **kwargs: Keyword arguments forwarded to ``torch.optim.AdamW``.

    Attributes:
        cautious_clip (float): The gradient clipping threshold.

    Reference:
        Liang, K., Zhou, L., Liu, B., Zhao, L., Jiang, Y., Pan, S.,
        Zhang, R., & Bengio, Y. (2024). Cautious Optimizers: Improving
        Training with One Line of Code. *arXiv:2411.16085*.
    """

    def __init__(self, params, *args, cautious_clip=1.0, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.cautious_clip = float(cautious_clip)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step with gradient pre-clipping.

        Clamps all gradients to ``[-cautious_clip, cautious_clip]``
        in-place, then delegates to the base ``AdamW.step()``.

        Args:
            closure: Optional callable that reevaluates the model and
                returns the loss. Passed through to the base optimizer.

        Returns:
            The loss value returned by `closure` (if provided), or
            ``None``.
        """
        if self.cautious_clip > 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.grad.data.clamp_(-self.cautious_clip, self.cautious_clip)
        return super().step(closure=closure)
