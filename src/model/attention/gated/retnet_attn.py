"""RetNet Attention (gated-package alias).

Provides a thin wrapper around MultiScaleRetention to expose the
``retnet_attn`` block name within the gated-attention package for
naming consistency. RetNet (Retentive Network) uses multi-scale
exponential decay retention as a drop-in replacement for standard
softmax attention, achieving O(N) inference complexity with
comparable training parallelism.

Reference:
    Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J.,
    & Wei, F. (2023). Retentive Network: A Successor to Transformer
    for Large Language Models. arXiv:2307.08621.
"""

from typing import Optional

import torch
import torch.nn as nn

from ..retnet import MultiScaleRetention


class RetNetAttention(nn.Module):
    """RetNet retention attention alias.

    Wraps the existing MultiScaleRetention implementation to provide
    an explicit ``retnet_attn`` block name in the gated-attention
    package. All computation is delegated to the inner
    MultiScaleRetention module.

    Args:
        config: Model configuration object forwarded to
            MultiScaleRetention. Must include hidden_size, num_heads,
            dropout, and optionally use_bitnet and mode.

    Attributes:
        inner (MultiScaleRetention): The wrapped multi-scale retention
            module that performs the actual computation.
        mode (str): ``"encoder"`` or ``"decoder"``, forwarded from
            config.

    Reference:
        Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J.,
        & Wei, F. (2023). Retentive Network: A Successor to Transformer
        for Large Language Models. arXiv:2307.08621.
    """

    def __init__(self, config):
        """Initialize RetNetAttention.

        Args:
            config: Model configuration object. See class docstring for
                required attributes.
        """
        super().__init__()
        self.inner = MultiScaleRetention(config)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute multi-scale retention over the input sequence.

        Delegates to the inner MultiScaleRetention module.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface
                compatibility with other attention mixers.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        return self.inner(x)
