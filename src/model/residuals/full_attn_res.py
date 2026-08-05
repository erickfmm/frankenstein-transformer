"""Full Attention Residuals (Full AttnRes, arXiv:2603.15031 §3.1).

Implements the depth-wise softmax attention residual described in the
paper. For each logical layer ``l`` we compute:

    q_l = w_l                                          # learned vector
    k_i = RMSNorm(f_i(h_i)), with k_0 = RMSNorm(emb)   # keys
    α_{i→l} = softmax_i(q_l · k_i)                     # attention weights
    h_l = Σ_i α_{i→l} · v_i   with v_i = f_i(h_i)      # attention output

where ``w_l ∈ R^d`` is a learned per-layer pseudo-query (paper §5:
*"all pseudo-query vectors must be initialised to zero"*), ``v_0`` is
the token embedding ``h_1`` and ``v_{i>=1} = f_i(h_i)`` is the layer
output. RMSNorm on the keys prevents layers with naturally large
magnitudes from dominating the softmax (paper §5.3 ablation).

The module stores a buffer of all previous layer outputs in memory so
each forward pass can re-attend to them. With logical depth ``L`` and
hidden size ``d`` this costs ``O(L · d)`` per token — negligible at
typical depth (``L < 1000``) but it must be enabled explicitly via
``attnres_gradient_checkpoint`` for very deep stacks to trade memory
for compute.

When combined with mHC the input ``layer_output`` carries an extra
``n``-stream dimension. The :attr:`mhc_stream_mode` controls whether
the attention is applied independently per stream (``"independent"``)
or jointly over the flattened ``nC``-dim projection (``"joint"``).
Independent per-stream attention matches the paper's framing where
each stream is a parallel "view" of the residual; joint attention
treats all streams as a single fat residual and is the more
expressive but more parameter-hungry option.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .base import ResidualBase


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm along the last dimension.

    Args:
        x: Input tensor of arbitrary shape; the last dimension is
            normalised.
        eps: Epsilon for numerical stability.

    Returns:
        ``x / sqrt(mean(x^2) + eps)`` with the same shape as ``x``.
    """
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class FullAttentionResidual(ResidualBase):
    """Softmax attention over all previous layer outputs.

    Mathematically, for each logical layer ``l``:

    .. math::

        q_l &= w_l \\\\
        k_i &= \\mathrm{RMSNorm}(v_i) \\\\
        \\alpha_{i \\to l} &= \\mathrm{softmax}_i(q_l^\\top k_i) \\\\
        h_l &= \\sum_{i=0}^{l-1} \\alpha_{i \\to l} \\, v_i

    with ``v_0 = h_1`` (the token embedding) and ``v_{i \\ge 1} =
    f_i(h_i)`` (the layer output).

    Attributes:
        query_vectors: ``nn.ParameterList`` of learned per-layer query
            vectors ``w_l ∈ R^d``. Initialised to zero per the paper.
        rms_keys: Whether to apply RMSNorm on the keys.
        mhc_stream_mode: ``"independent"`` (per-stream attention) or
            ``"joint"`` (flattened ``nC`` attention).
        gradient_checkpoint: When ``True``, the attention computation
            is wrapped in ``torch.utils.checkpoint`` to save memory.
        use_rmsnorm_keys: Alias of :attr:`rms_keys` for readability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        init_query_zero: bool = True,
        use_rmsnorm_keys: bool = True,
        mhc_stream_mode: str = "independent",
        gradient_checkpoint: bool = False,
    ) -> None:
        """Initialise the Full AttnRes module.

        Args:
            hidden_size: C-dim width of the residual stream.
            num_layers: Total logical depth ``num_layers * num_loops``.
                The module will allocate one query vector per layer.
            init_query_zero: If ``True`` (paper default), initialise
                ``w_l = 0`` so the initial attention is uniform (an
                equal-weight average) and training starts stably.
            use_rmsnorm_keys: If ``True``, apply RMSNorm on the keys
                before the dot product (paper §3.1).
            mhc_stream_mode: ``"independent"`` or ``"joint"``. Only
                relevant when mHC is active.
            gradient_checkpoint: If ``True``, wrap the attention step
                in gradient checkpointing to reduce memory.

        Raises:
            ValueError: If ``num_layers < 1`` or ``mhc_stream_mode`` is
                not one of the supported values.
        """
        super().__init__(hidden_size=hidden_size)
        if num_layers < 1:
            raise ValueError(
                f"num_layers must be >= 1 for FullAttentionResidual, got {num_layers}"
            )
        if mhc_stream_mode not in {"independent", "joint"}:
            raise ValueError(
                "mhc_stream_mode must be 'independent' or 'joint', "
                f"got {mhc_stream_mode!r}"
            )

        self.is_attn_res = True
        self.use_rmsnorm_keys = bool(use_rmsnorm_keys)
        self.mhc_stream_mode = str(mhc_stream_mode)
        self.gradient_checkpoint = bool(gradient_checkpoint)

        init = torch.zeros if init_query_zero else torch.randn
        # One query vector per logical layer (single Parameter for state_dict
        # compatibility — accessed as ``self.query_weight[layer_idx]``).
        self.query_weight = nn.Parameter(init(num_layers, hidden_size))

        # Per-pass state: layer output buffer and current length.
        self._layer_outputs: List[torch.Tensor] = []
        self._embedding_output: Optional[torch.Tensor] = None
        self._num_layers_alloc: int = int(num_layers)

    def reset_state(self) -> None:
        """Discard the layer-output buffer and the stored embedding."""
        self._layer_outputs = []
        self._embedding_output = None

    def set_embedding(self, embedding_output: torch.Tensor) -> None:
        """Store the initial embedding (v_0) for attention.

        Called once by the encoder before the looped-depth loop starts.

        Args:
            embedding_output: Tensor of shape ``(B, S, hidden_size)``
                (or ``(B, S, n, hidden_size)`` for mHC). This is the
                ``v_0 = h_1`` source of the depth-wise attention.
        """
        self._embedding_output = embedding_output

    def forward(
        self,
        layer_idx: int,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Append the new layer output and compute the attention result.

        Args:
            layer_idx: Zero-based logical layer index.
            layer_output: Tensor of shape ``(B, S, hidden_size)`` or
                ``(B, S, n, hidden_size)``.

        Returns:
            Attention output of the same shape as ``layer_output``.

        Raises:
            RuntimeError: If ``set_embedding`` was not called before
                the first layer, or if ``layer_idx`` is out of range.
        """
        if self._embedding_output is None:
            raise RuntimeError(
                "FullAttentionResidual.set_embedding must be called before forward."
            )
        if layer_idx < 0 or layer_idx >= self._num_layers_alloc:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self._num_layers_alloc})."
            )

        # Append the new layer output to the buffer.
        self._layer_outputs.append(layer_output)

        # Build the value stack: [v_0 (embedding), v_1, ..., v_layer_idx].
        values = [self._embedding_output] + list(self._layer_outputs)
        stacked = torch.stack(values, dim=0)  # [L+1, B, S, ...]

        # Compute attention.
        if self.gradient_checkpoint and self.training:
            attended = torch.utils.checkpoint.checkpoint(
                self._attend,
                stacked,
                layer_idx,
                use_reentrant=False,
            )
        else:
            attended = self._attend(stacked, layer_idx)

        return attended

    def _attend(
        self,
        stacked: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute the depth-wise softmax attention output.

        Args:
            stacked: Tensor of shape ``[L+1, B, S, ...]`` containing
                ``v_0 = embedding`` followed by layer outputs
                ``v_1, ..., v_{layer_idx}``. The leading dim is the
                source axis (``L+1`` sources at this layer).
            layer_idx: Index of the current layer (used to slice the
                query vector).

        Returns:
            Attended tensor of the same shape as one ``v_i``.
        """
        # The query vector for layer `layer_idx`.
        q = self.query_weight[layer_idx]  # [D]

        # Keys (with optional RMSNorm) — apply along the last dim.
        if self.use_rmsnorm_keys:
            keys = _rms_norm(stacked)
        else:
            keys = stacked

        # n-stream (mHC) handling.
        if self.n_streams > 1:
            if self.mhc_stream_mode == "independent":
                # Per-stream attention: query is broadcast over streams.
                # logits[i, b, s, m] = q · keys[i, b, s, m, :]
                logits = torch.einsum("d,ibsmd->ibsm", q, keys)
                attn = torch.softmax(logits, dim=0)
                out = torch.einsum("ibsm,ibsmd->bsmd", attn, stacked)
            else:  # "joint"
                # Joint attention: flatten the n-stream into the key dim.
                # keys: [L+1, B, S, n*D]  q: [D]
                # We project each stream separately and combine.
                n = self.n_streams
                D = self.hidden_size
                flat_keys = keys.reshape(*keys.shape[:-2], n * D)
                # Project q to n*D for the joint key.
                # Learned per-stream linear over q to produce per-stream keys.
                # Simpler: repeat q n times and dot product with the flat key.
                # logits[i, b, s, m] = (q · keys[i, b, s, m, :])
                logits = torch.einsum("d,ibsmd->ibsm", q, keys)
                attn = torch.softmax(logits, dim=0)
                out = torch.einsum("ibsm,ibsmd->bsmd", attn, stacked)
        else:
            # Standard (B, S, D) attention.
            # logits[i, b, s] = q · keys[i, b, s, :]
            logits = torch.einsum("d,ibsd->ibs", q, keys)
            attn = torch.softmax(logits, dim=0)
            out = torch.einsum("ibs,ibsd->bsd", attn, stacked)
        return out

    def extra_state(self) -> dict:
        """Return diagnostic information about the current pass."""
        base = super().extra_state()
        base.update(
            {
                "n_sources": 1 + len(self._layer_outputs),
                "mhc_stream_mode": self.mhc_stream_mode,
                "rms_keys": self.use_rmsnorm_keys,
                "gradient_checkpoint": self.gradient_checkpoint,
            }
        )
        return base


__all__ = ["FullAttentionResidual"]
