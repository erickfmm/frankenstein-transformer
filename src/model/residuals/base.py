"""Abstract base class for residual-connection modules.

A residual module owns the cross-layer state required by
attention-residual variants (a per-layer output buffer, block
accumulators, …) and exposes a uniform call signature so
:class:`HybridLayer` can dispatch transparently regardless of the
selected strategy.

Lifecycle::

    residual = build_residual(config)
    ...
    for layer_idx in range(num_layers * num_loops):
        layer_output = layer(x, ...)
        x = residual(layer_idx, layer_output)
    x = residual.finalize(x)

``StandardResidual`` and ``NoResidual`` are stateless; they keep no
buffers and ignore ``reset_state`` calls. ``FullAttentionResidual`` and
``BlockAttentionResidual`` store per-layer (or per-block) outputs on
``register_state`` and reset them between forward passes via
:meth:`reset_state`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ResidualBase(nn.Module):
    """Base class for cross-layer residual-connection modules.

    Subclasses must implement :meth:`forward` to combine the new layer
    output with whatever cross-layer state they keep. The state is
    optionally managed via :meth:`reset_state` (called once per forward
    pass) and :meth:`register_state` (called once before the looped
    depth forward so the module knows the total logical depth).

    All tensor shapes follow the convention ``(B, S, D)`` for the
    standard residual stream and ``(B, S, n, D)`` for the
    manifold-constrained ``n``-stream residual (mHC). The AttnRes
    variants add a single optional keyword argument to switch behaviour
    for the n-stream case.

    Attributes:
        hidden_size: Dimensionality of the C-dim stream.
        n_streams: Number of residual streams (``1`` for the standard
            residual, ``n`` when mHC is active). Set by
            :meth:`set_streams` once at construction time.
        num_layers: Total logical depth (``num_layers * num_loops``).
            ``0`` until :meth:`register_state` is called.
        is_attn_res: True if this residual needs depth-wise attention
            over previous layer outputs (Full/Block AttnRes).
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialise the residual module.

        Args:
            hidden_size: C-dimensional width of the residual stream.
        """
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_streams: int = 1
        self.num_layers: int = 0
        self.is_attn_res: bool = False

    def set_streams(self, n_streams: int) -> None:
        """Configure the number of parallel residual streams (mHC).

        Called once after construction when ``use_mhc`` is enabled.
        Stateless residuals ignore the value; AttnRes variants use it to
        decide whether to attend per-stream (``independent``) or jointly
        over the flattened stream (``joint``).

        Args:
            n_streams: Number of residual streams (``1`` for standard,
                ``>1`` for mHC).
        """
        self.n_streams = max(1, int(n_streams))

    def register_state(self, num_layers: int) -> None:
        """Pre-allocate the buffers needed for the upcoming forward pass.

        Stateless variants (Standard / No) accept the call but do
        nothing. AttnRes variants use it to size their layer-output
        buffer and reset their block accumulator index.

        Args:
            num_layers: Total logical depth (``num_layers * num_loops``).
        """
        self.num_layers = int(num_layers)

    def reset_state(self) -> None:
        """Discard the cross-layer state accumulated so far.

        Called at the start of each forward pass. The allocation
        tracked in :attr:`num_layers` (set by :meth:`register_state`)
        is preserved — only the per-pass buffer / accumulators are
        cleared. Subclasses that also need to reset the allocation
        should override this and call ``super().reset_state()``.
        """
        pass

    def forward(
        self,
        layer_idx: int,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Combine the current layer output with the residual stream.

        Subclasses implement the actual combination (sum, attention,
        etc.). Stateless variants may simply return ``layer_output``.

        Args:
            layer_idx: Zero-based global logical index of the layer
                whose output is being added (already accounts for
                ``num_loops``).
            layer_output: Tensor of shape ``(B, S, hidden_size)`` (or
                ``(B, S, n, hidden_size)`` for mHC).

        Returns:
            The new residual stream with the same shape as
            ``layer_output``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(layer_idx, layer_output)."
        )

    def finalize(self, residual_stream: torch.Tensor) -> torch.Tensor:
        """Apply any post-loop transformation to the residual stream.

        For Standard / No / Full AttnRes this is the identity. Block
        AttnRes optionally folds the last partial-block accumulation
        back into the main stream so callers always see a regular
        C-dim tensor.

        Args:
            residual_stream: The stream after the loop has finished.

        Returns:
            The (possibly transformed) residual stream.
        """
        return residual_stream

    def extra_state(self) -> dict:
        """Return non-parameter state useful for debugging / logging.

        Returns:
            A dict of named state entries. Subclasses should override
            this to expose relevant diagnostic information.
        """
        return {
            "type": type(self).__name__,
            "n_streams": self.n_streams,
            "num_layers": self.num_layers,
        }


__all__ = ["ResidualBase"]
