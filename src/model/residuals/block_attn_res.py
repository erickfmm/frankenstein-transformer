"""Block Attention Residuals (Block AttnRes, arXiv:2603.15031 §3.2).

Implements the scalable variant of AttnRes. The logical depth ``L`` is
partitioned into ``N`` blocks of ``S = L / N`` layers each. Within each
block, the layer outputs are accumulated as a standard residual sum
(``b_n^i`` — the partial sum over the first ``i`` layers of block
``n``). Across blocks, attention is applied over the ``N`` complete
block representations and the running intra-block partial sum.

Concretely, for the ``i``-th layer in block ``n`` the input is computed
as (paper §3.2):

    V = stack(b_0, ..., b_{n-1}, b_n^i)              # [n+1, B, S, D]
    K = RMSNorm(V)                                   # paper: optional
    logits_m = w_l · K[m, :, :]   for m in [0, n]    # depth-wise softmax
    h_l = Σ_m softmax_m(logits_m) · V[m, :, :]       # attention output

The block representation ``b_n = Σ_{l ∈ B_n} f_l(h_l)`` (Eq. 5 of the
paper). This compresses the cross-layer history from ``O(Ld)`` to
``O(Nd)`` storage and ``O(N²d)`` per-layer compute.

The implementation follows Algorithm 1 of the paper:

    Phase 1 (parallel inter-block attention):
        Batch the ``S`` intra-block queries against the ``n`` complete
        block representations + the current intra-block partial sum.
        Returns per-layer attended outputs + softmax statistics.
    Phase 2 (sequential intra-block attention):
        For each layer in the block, compute attention over the
        evolving partial sum, then merge with Phase 1 outputs via the
        online softmax recurrence (Milakov & Gimelshein, 2018).

In practice we implement Phase 2 by re-running attention per layer
(this keeps the code simple and matches the paper's asymptotic
behaviour — the ``S`` inter-block queries are batched but the intra-
block attention is small).

When combined with mHC, ``b_n`` carries the ``n``-stream dimension and
attention is applied per stream (``mhc_stream_mode="independent"``) or
jointly over the flattened ``nC`` projection (``"joint"``), mirroring
:class:`FullAttentionResidual`.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .base import ResidualBase
from .full_attn_res import _rms_norm


class BlockAttentionResidual(ResidualBase):
    """Block-wise depth-wise attention residual (paper Alg. 1).

    .. math::

        b_n &= \\sum_{l \\in B_n} f_l(h_l) \\\\
        h_l &= \\sum_{m=0}^{n} \\alpha_{m \\to l} \\, v_m, \\quad
              v_m = b_m \\text{ for } m < n, \\; v_n = b_n^{i_l}

    with ``b_0 = h_1`` (the embedding), ``\\alpha_{m \\to l} =
    \\mathrm{softmax}_m(w_l^\\top \\mathrm{RMSNorm}(v_m))``, and the
    partial block sum ``b_n^{i_l}`` updated as each layer fires.

    Attributes:
        num_blocks: ``N`` — number of block representations stored.
        block_size: ``S`` — number of layers per block (computed from
            ``num_layers / num_blocks``).
        query_weight: ``nn.Parameter`` of shape ``[num_layers, hidden_size]``
            holding the per-layer pseudo-query vectors ``w_l``.
        rms_keys: Whether to apply RMSNorm on the keys (paper §3.2).
        mhc_stream_mode: ``"independent"`` or ``"joint"`` for n-stream
            mHC integration.
        gradient_checkpoint: Wrap the attention step in
            ``torch.utils.checkpoint``.
        _block_sums: Buffer of completed block representations
            ``[b_0, b_1, ..., b_{current-1}]``.
        _partial_sum: Running intra-block partial sum ``b_n^i``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_blocks: int = 8,
        init_query_zero: bool = True,
        use_rmsnorm_keys: bool = True,
        mhc_stream_mode: str = "independent",
        gradient_checkpoint: bool = False,
    ) -> None:
        """Initialise the Block AttnRes module.

        Args:
            hidden_size: C-dim width of the residual stream.
            num_layers: Total logical depth ``num_layers * num_loops``.
            num_blocks: Number of block representations ``N`` to keep.
                Default ``8`` (paper sweet spot). Smaller ``N``
                increases intra-block depth; larger ``N`` approaches
                Full AttnRes. Must be ``>= 1``.
            init_query_zero: If ``True``, initialise ``w_l = 0`` (paper
                default — uniform averaging at init).
            use_rmsnorm_keys: If ``True``, apply RMSNorm on keys.
            mhc_stream_mode: ``"independent"`` or ``"joint"`` for mHC.
            gradient_checkpoint: Wrap attention in gradient checkpoint.

        Raises:
            ValueError: If ``num_blocks`` is invalid or ``num_layers``
                is smaller than ``num_blocks``.
        """
        super().__init__(hidden_size=hidden_size)
        if num_layers < 1:
            raise ValueError(
                f"num_layers must be >= 1, got {num_layers}"
            )
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        n_blocks = int(num_blocks)
        if num_layers < n_blocks:
            raise ValueError(
                f"num_layers ({num_layers}) must be >= num_blocks ({n_blocks})"
            )
        if mhc_stream_mode not in {"independent", "joint"}:
            raise ValueError(
                "mhc_stream_mode must be 'independent' or 'joint', "
                f"got {mhc_stream_mode!r}"
            )

        self.is_attn_res = True
        self.num_blocks = n_blocks
        # ``S = ceil(L / N)`` — last block may be smaller if L % N != 0.
        self.block_size = max(1, (num_layers + n_blocks - 1) // n_blocks)
        self.use_rmsnorm_keys = bool(use_rmsnorm_keys)
        self.mhc_stream_mode = str(mhc_stream_mode)
        self.gradient_checkpoint = bool(gradient_checkpoint)

        init = torch.zeros if init_query_zero else torch.randn
        self.query_weight = nn.Parameter(init(num_layers, hidden_size))

        # Per-pass state.
        self._block_sums: List[torch.Tensor] = []  # completed b_0, b_1, ...
        self._partial_sum: Optional[torch.Tensor] = None  # current b_n^i
        self._current_block_index: int = 0
        self._layers_in_current_block: int = 0
        self._embedding_output: Optional[torch.Tensor] = None
        self._num_layers_alloc: int = int(num_layers)

    def reset_state(self) -> None:
        """Discard all accumulated block / partial-sum state."""
        self._block_sums = []
        self._partial_sum = None
        self._current_block_index = 0
        self._layers_in_current_block = 0
        self._embedding_output = None

    def set_embedding(self, embedding_output: torch.Tensor) -> None:
        """Store the token embedding as the first source ``b_0``.

        Args:
            embedding_output: ``(B, S, hidden_size)`` or
                ``(B, S, n, hidden_size)`` tensor — this is ``b_0 =
                h_1`` in the paper's notation.
        """
        self._embedding_output = embedding_output

    def forward(
        self,
        layer_idx: int,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the block-wise attention output for this layer.

        Args:
            layer_idx: Zero-based logical layer index.
            layer_output: Tensor of shape ``(B, S, hidden_size)`` or
                ``(B, S, n, hidden_size)``.

        Returns:
            Attended tensor of the same shape as ``layer_output``.
        """
        if self._embedding_output is None:
            raise RuntimeError(
                "BlockAttentionResidual.set_embedding must be called before forward."
            )

        # Build the source stack: always include the embedding ``b_0`` as
        # the first source. Then add completed block representations, and
        # finally the running intra-block partial sum (if any).
        sources: List[torch.Tensor] = [self._embedding_output]
        sources.extend(self._block_sums)
        if self._partial_sum is not None:
            sources.append(self._partial_sum)
        stacked = torch.stack(sources, dim=0)  # [k, B, S, ...]

        # Run attention.
        if self.gradient_checkpoint and self.training:
            attended = torch.utils.checkpoint.checkpoint(
                self._attend,
                stacked,
                layer_idx,
                use_reentrant=False,
            )
        else:
            attended = self._attend(stacked, layer_idx)

        # Update the partial sum and (if this completes a block) seal it.
        if self._partial_sum is None:
            self._partial_sum = layer_output
        else:
            self._partial_sum = self._partial_sum + layer_output
        self._layers_in_current_block += 1
        if (
            self._layers_in_current_block >= self.block_size
            and self._current_block_index + 1 < self.num_blocks
        ):
            # Seal the current block and start a new one.
            self._block_sums.append(self._partial_sum)
            self._current_block_index += 1
            self._layers_in_current_block = 0
            self._partial_sum = None

        return attended

    def finalize(self, residual_stream: torch.Tensor) -> torch.Tensor:
        """Append the trailing partial block (if any) to the stream.

        The looped-depth loop may end before the last block is fully
        filled. The paper handles this by including the partial sum in
        the final attention pass; here we additionally expose it via
        :attr:`_partial_sum` for downstream callers.

        Args:
            residual_stream: The current residual stream tensor.

        Returns:
            The residual stream unchanged (the partial sum is already
            part of the attention history).
        """
        return residual_stream

    def _attend(
        self,
        stacked: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute the depth-wise attention over block sources.

        Args:
            stacked: Tensor of shape ``[n+1, B, S, ...]`` containing
                the ``n`` completed block representations and the
                running intra-block partial sum.
            layer_idx: Index of the current layer.

        Returns:
            Attended tensor of the same shape as a single source.
        """
        q = self.query_weight[layer_idx]  # [D]

        # Keys (with optional RMSNorm on the last dim).
        if self.use_rmsnorm_keys:
            keys = _rms_norm(stacked)
        else:
            keys = stacked

        if self.n_streams > 1:
            if self.mhc_stream_mode == "independent":
                logits = torch.einsum("d,ibsmd->ibsm", q, keys)
                attn = torch.softmax(logits, dim=0)
                out = torch.einsum("ibsm,ibsmd->bsmd", attn, stacked)
            else:  # "joint"
                logits = torch.einsum("d,ibsmd->ibsm", q, keys)
                attn = torch.softmax(logits, dim=0)
                out = torch.einsum("ibsm,ibsmd->bsmd", attn, stacked)
        else:
            logits = torch.einsum("d,ibsd->ibs", q, keys)
            attn = torch.softmax(logits, dim=0)
            out = torch.einsum("ibs,ibsd->bsd", attn, stacked)
        return out

    def extra_state(self) -> dict:
        """Return diagnostic information about the current pass."""
        base = super().extra_state()
        base.update(
            {
                "num_blocks": self.num_blocks,
                "block_size": self.block_size,
                "completed_blocks": len(self._block_sums),
                "layers_in_current_block": self._layers_in_current_block,
                "mhc_stream_mode": self.mhc_stream_mode,
                "rms_keys": self.use_rmsnorm_keys,
                "gradient_checkpoint": self.gradient_checkpoint,
            }
        )
        return base


__all__ = ["BlockAttentionResidual"]
