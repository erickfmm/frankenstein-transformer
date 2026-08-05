"""Standard fixed-weight residual connection.

Implements the classical skip connection from He et al. (2015):

    h_l = h_{l-1} + f_l(h_{l-1})

with coefficient ``1`` for both the residual and the layer output. This
is the original Transformer-style residual, kept here as the default
backwards-compatible option.
"""

from __future__ import annotations

import torch

from .base import ResidualBase


class StandardResidual(ResidualBase):
    """Fixed-unit-coefficient residual ``h_l = h_{l-1} + f_l``.

    Stateless: the residual is just the running sum of all previous
    layer outputs plus the embedding (held by the caller in ``x``).
    This class keeps no buffers and adds no learnable parameters.

    The forward pass is intentionally a no-op on the layer output: the
    residual sum is performed by :class:`HybridLayer` before this
    module is called, which matches the original implementation and
    minimises refactor risk.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialise the standard residual module.

        Args:
            hidden_size: C-dim width of the residual stream (unused,
                kept for API symmetry with AttnRes variants).
        """
        super().__init__(hidden_size=hidden_size)

    def forward(
        self,
        layer_idx: int,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return the layer output unchanged.

        The residual merge is applied by :class:`HybridLayer` itself
        for the standard residual to preserve backwards-compatible
        shape / dtype handling.

        Args:
            layer_idx: Zero-based logical layer index (ignored).
            layer_output: Layer function output of shape
                ``(B, S, hidden_size)``.

        Returns:
            ``layer_output`` unchanged.
        """
        return layer_output


__all__ = ["StandardResidual"]
