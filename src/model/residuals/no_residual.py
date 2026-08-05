"""No-residual baseline (experimental).

Drops the identity mapping entirely:

    h_l = f_l(h_{l-1})

This is mathematically a composition of layer functions with no skip
path. It is included as an experimental ablation point — paper §2.1
notes that the identity term in the gradient ``I + higher-order``
terms is critical for deep networks; the no-residual variant therefore
exists mostly as a diagnostic to quantify how much of AttnRes's gain
comes from the per-layer attention mechanism vs. the simple fact of
having a residual at all.

Because dropping the residual makes deep stacks numerically unstable,
this module is tagged experimental and only useful for shallow toy
models or as a probe in controlled ablations. Production runs should
keep the standard residual enabled.
"""

from __future__ import annotations

import torch

from .base import ResidualBase


class NoResidual(ResidualBase):
    """Drop the skip connection: ``h_l = f_l``.

    Stateless: this class never modifies the layer output. The caller
    (:class:`HybridLayer`) is responsible for skipping the standard
    ``residual + layer_output`` step when the selected residual type is
    ``none``. This matches the pattern used by :class:`StandardResidual`
    and keeps the per-layer forward pass in a single place.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialise the no-residual module.

        Args:
            hidden_size: C-dim width of the residual stream (unused,
                kept for API symmetry).
        """
        super().__init__(hidden_size=hidden_size)
        self.is_no_residual = True

    def forward(
        self,
        layer_idx: int,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return the layer output unchanged.

        The caller handles the actual residual merge (or lack thereof)
        based on the residual type.

        Args:
            layer_idx: Zero-based logical layer index (ignored).
            layer_output: Layer function output of shape
                ``(B, S, hidden_size)``.

        Returns:
            ``layer_output`` unchanged.
        """
        return layer_output


__all__ = ["NoResidual"]
