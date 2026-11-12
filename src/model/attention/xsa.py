"""Exclusive Self Attention (XSA).

Implements the *exclusive self attention* mechanism from Zhai (2026),
arXiv:2603.09078. Standard softmax attention aggregates context into
``y_i``, but ``y_i`` tends to keep a large component along the token's own
value vector ``v_i`` (the *attention similarity bias*), duplicating work
that the residual path and the following FFN already do. XSA removes that
component with a single, parameter-free projection step:

    v̂_i = v_i / ‖v_i‖₂
    z_i = y_i − (y_i · v̂_i) v̂_i

where ``y_i`` is the standard (causal) softmax attention output of head
``h`` at position ``i``. The layer therefore carries only contextual
information orthogonal to the self value direction, and adds no learned
parameters beyond the usual ``W_q``, ``W_k``, ``W_v``, ``W_o``.

Complexity:
    Training: ``O(n² · d)`` (softmax attention) + ``O(n · d)`` exclusion.
    Inference: ``O(t · d)`` per decode step with a KV cache.

Reference:
    Zhai, S. (2026). Exclusive Self Attention. arXiv:2603.09078.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..numerics import module_softmax
from ..norm import attach_qk_norm, record_qk_max_logits

from .common import BitLinear, apply_pe_to_qk, apply_pe_to_scores
from .params import mixer_params, use_pe_for


class ExclusiveSelfAttention(nn.Module):
    """Exclusive Self Attention: softmax attention minus the self-value direction.

    Computes standard scaled dot-product attention (bidirectional in encoder
    mode, causal in decoder mode) and then orthogonalizes each head's output
    against that token's own normalized value vector, following Eq. 2 of
    arXiv:2603.09078:

    .. math::
        z_i = y_i - (y_i^T \\hat{v}_i)\\,\\hat{v}_i,
        \\qquad \\hat{v}_i = v_i / \\|v_i\\|_2

    The step is parameter-free and row-local: it depends only on the current
    token's value, so a plain KV cache supports exact incremental decoding.

    Args:
        config: Model configuration object with the following relevant
            attributes:
            hidden_size (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads. Must divide
                ``hidden_size`` evenly.
            dropout (float): Dropout probability applied to attention
                weights.
            bitnet.enabled (bool): If True, uses ``BitLinear`` for the
                Q/K/V/O projections instead of ``nn.Linear``.
            mode (str, optional): ``"encoder"`` or ``"decoder"``. Defaults to
                ``"encoder"``. In decoder mode, a causal mask is applied.
        pos_encoder: Optional shared positional encoding module.
        params: Optional pre-bound component parameters (none are used by
            this mixer; accepted for the registry builder contract).

    Attributes:
        hidden_size (int): Input embedding dimensionality.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality per head (``hidden_size // num_heads``).
        total_dim (int): Total Q/K/V dimensionality (``head_dim * num_heads``).
        scale (float): Scaling factor for dot-product attention
            (``head_dim ** -0.5``).
        q_proj (nn.Module): Query projection.
        k_proj (nn.Module): Key projection.
        v_proj (nn.Module): Value projection.
        out_proj (nn.Module): Output projection.
        dropout (nn.Dropout): Dropout layer applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.

    Reference:
        Zhai, S. (2026). Exclusive Self Attention. arXiv:2603.09078.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        """Initialize ExclusiveSelfAttention.

        Args:
            config: Model configuration object. See class docstring for
                required attributes.
            pos_encoder: Optional shared positional encoding module.
            params: Optional pre-bound component parameters (unused).

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        """
        params = mixer_params(config, "xsa", params)
        super().__init__()
        self.hidden_size = config.dims.hidden_size
        self.num_heads = config.dims.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for ExclusiveSelfAttention"
            )

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dims.dropout)
        self.mode = getattr(config.dims, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "xsa", True))
        attach_qk_norm(self, config, self.head_dim)

    @staticmethod
    def _exclude_self(value: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        """Remove the projection of the attention output on the self value.

        Args:
            value: Per-head value tensor of shape
                ``(batch, heads, seq_len, head_dim)``.
            attended: Per-head attention output ``y`` with the same shape.

        Returns:
            ``z = y - (y · v̂) v̂`` with ``v̂ = v / ‖v‖₂``, same shape as
            ``attended`` (Eq. 2 of arXiv:2603.09078).
        """
        unit = F.normalize(value, dim=-1)
        projection = (attended * unit).sum(dim=-1, keepdim=True)
        return attended - projection * unit

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute exclusive self attention over the input sequence.

        Standard scaled dot-product attention followed by the parameter-free
        exclusion of each token's own value direction. In decoder mode a
        causal mask is applied before softmax.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoding; accepted for interface compatibility.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.
            cache: Optional KV slot for incremental decoding. The exclusion
                depends only on the current token's value, so nothing extra
                is cached. ``None`` runs the full-recompute path.
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        idx = logical_layer_idx or 0

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ..cache.hooks import cached_qk_attention

            attended, _, _ = cached_qk_attention(
                slot,
                q,
                k,
                v,
                cache_position=cache_position,
                scale=self.scale,
                pe=pe,
                pe_type=self.pe_type,
                use_pe=self.use_pe,
                x_for_pe=x,
                logical_layer_idx=idx,
                normalize=None,
                dropout=self.dropout,
                causal=True,
                use_fixed_softmax=self.use_fixed_softmax,
            )
            z = self._exclude_self(v, attended)
            return self.out_proj(z.transpose(1, 2).contiguous().view(bsz, seq_len, hidden))

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, idx, self.use_pe)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = apply_pe_to_scores(pe, self.pe_type, attn_scores, q, self.use_pe)
        record_qk_max_logits(self, attn_scores)
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = module_softmax(self, attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = attn_weights @ v
        z = self._exclude_self(v, attended)
        out = z.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
