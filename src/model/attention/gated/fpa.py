"""First Passage Attention (FPA).

Implements First Passage Attention from Liu (2026), arXiv:2610.22714.
FPA augments softmax attention with a *first-passage* logit bias: instead of
summing decay contributions along the key-query path (as FoX does), each head
computes a scalar field ``phi`` at every position and a scalar level ``psi``
per key, and a key's logit is lowered by the depth of the *deepest dip* of the
field below the key's level over the span after the key:

    O       = sum_{s <= t} w_ts v_s,   w_ts ∝ exp(l_ts)
    l_ts    = q_t^T k_s / sqrt(d) + b_ts
    b_ts    = beta * min(0, min_{s<u<=t} phi_u - psi_s),   b_tt = 0
    W_t     = sum_{u<=t} delta_u                (learned walk)
    phi_u   = W_u + g_u                         (field)
    psi_s   = W_s + h_s                         (level of key s)
    beta > 0                                    (learned strength per head)

One linear map per layer produces ``(delta, g, h)`` for every head from the
same normalised input as Q/K/V, and ``beta`` is a learned per-head scalar, so
the mixer adds ``3*H*(D+1) + H`` parameters to a layer of width ``D`` with
``H`` heads (paper Table 14: 27,696 at D=768, H=12).

Special cases (paper Prop. 2.2):

* (a) ``g = h = 0`` — stack form: ``b_ts = -beta * L_ts`` with ``L_ts`` the
  Lindley/Skorokhod queue regulator.
* (b) additionally ``delta_u <= 0`` — exactly FoX: ``b_ts = sum log f_u`` with
  ``f_u = exp(beta*delta_u)`` in ``(0, 1]``.
* (c) ``delta = 0, h = 0, g <= 0, beta = 1`` — bottleneck bias
  ``min_{s<u<=t} g_u``.
* (d) ``delta_u = -1, g = 0, psi_s = W_s - e_s`` — Expire-Span
  ``beta*min(0, e_s - (t - s))``.

Training uses the paper's straight-through estimator: the forward pass takes
the hard range minimum and the backward pass takes the gradient of the soft
minimum ``-tau * log sum exp(-phi_u / tau)`` with ``tau = 1``; the outer
``min(0, .)`` always acts on the hard value.

This implementation is a CPU-safe reference (the paper's fused Triton kernel is
out of scope): the bias is the exact O(T^2) element-wise matrix, built with two
flips and a cumulative min, exactly like FoX's cumulative-bias matrix.

Decoding is exact (paper Prop. 2.4b): the bias needs only the running range
minimum ``phi_bar_{t,s} = min(phi_bar_{t-1,s}, phi_t)`` and ``psi_s`` per
cached key, i.e. two scalars per key and head, stored as slot extras.

Reference:
    Liu, Y. (2026). First Passage Attention. arXiv:2610.22714.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...numerics import module_softmax
from ...norm import attach_qk_norm, record_qk_max_logits

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


class FirstPassageAttention(nn.Module):
    """First Passage Attention with a learned walk, field and key levels.

    Computes scaled dot-product attention with the first-passage bias
    ``b_ts = beta * min(0, min_{s<u<=t} phi_u - psi_s)`` added in logit space.
    The field and the levels share a learned walk (a prefix sum of per-token
    increments), so a token can erase newer keys while preserving older ones.
    In decoder mode an additional causal mask is applied; in encoder mode the
    causal bias is extended symmetrically over the open span between two
    positions (a repo-side generalisation: the paper only defines the causal
    head).

    Args:
        config: Model configuration object with the following relevant
            attributes:
            hidden_size (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads. Must divide
                ``hidden_size`` evenly.
            dropout (float): Dropout probability applied to attention weights.
            use_bitnet (bool): If True, uses BitLinear for the projections
                instead of ``nn.Linear``.
            mode (str, optional): ``"encoder"`` or ``"decoder"``. Defaults to
                ``"encoder"``. In decoder mode, a causal mask is applied in
                addition to the first-passage bias.
            fpa_beta_init (float, optional): Initial value of the learned
                per-head strength ``beta``. Defaults to 1.0 (paper Table 16).
            fpa_tau (float, optional): Temperature of the soft-minimum
                straight-through gradient. Defaults to 1.0 (paper §2).
            fpa_use_pe (bool, optional): Whether this mixer consumes the
                model-wide positional encoding. Defaults to ``False`` (the
                paper trains FPA without positional information).
        pos_encoder: Optional shared positional encoding module.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.

    Reference:
        Liu, Y. (2026). First Passage Attention. arXiv:2610.22714.
    """

    def __init__(self, config, pos_encoder=None):
        """Initialize FirstPassageAttention.

        Args:
            config: Model configuration object. See class docstring for
                required attributes.
            pos_encoder: Optional shared positional encoding module.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for FirstPassageAttention"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        # Field projection: (delta, g, h) for every head, with bias (paper
        # Table 15: "linear layers without biases, except the field projection
        # of FPA").
        self.field_proj = proj_cls(self.hidden_size, 3 * self.num_heads, bias=True)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)

        # Learned positive strength per head (paper §2 / Table 16: beta = 1).
        beta_init = float(getattr(config, "fpa_beta_init", 1.0))
        self.beta = nn.Parameter(torch.full((self.num_heads,), beta_init))
        # Temperature of the straight-through soft minimum (paper §2: tau = 1).
        self.tau = float(getattr(config, "fpa_tau", 1.0))

        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "fpa_use_pe", False))
        attach_qk_norm(self, config, self.head_dim)

    # ------------------------------------------------------------------
    # Field / level / bias construction
    # ------------------------------------------------------------------
    def _field_and_level(self, x: torch.Tensor):
        """Compute the per-head scalar field ``phi`` and key levels ``psi``.

        One linear map gives the per-token increment ``delta`` and the two
        offsets ``g`` and ``h`` for every head; the walk is their prefix sum
        and the field/level are ``W + g`` and ``W + h`` (paper eq. 2). The
        projection runs in single precision (paper Table 16 keeps the FPA
        field in fp32 under autocast).

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Tuple ``(phi, psi)`` of float tensors, each ``(B, nH, S)``.
        """
        bsz, seq_len, _ = x.shape
        xd = x.float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            dgh = self.field_proj(xd)
        dgh = dgh.view(bsz, seq_len, 3, self.num_heads).permute(2, 0, 3, 1)
        delta, g, h = dgh[0], dgh[1], dgh[2]
        walk = torch.cumsum(delta, dim=-1)
        return walk + g, walk + h

    @staticmethod
    def _finite(x: torch.Tensor) -> torch.Tensor:
        """Replace non-finite range minima (empty spans) by a large sentinel.

        An empty span has no field value, so its bias must be exactly zero
        (``b_tt = 0``). Substituting a large finite number keeps the STE
        combination ``hard + (soft - soft.detach())`` NaN-free while the
        outer ``min(0, .)`` maps the sentinel to zero.

        Args:
            x: Range-minimum tensor that may contain ``+inf``.

        Returns:
            Tensor with ``+inf`` entries replaced by ``finfo.max / 4``.
        """
        big = torch.finfo(x.dtype).max / 4
        return torch.where(torch.isfinite(x), x, torch.full_like(x, big))

    @classmethod
    def _hard_range_min(cls, phi: torch.Tensor) -> torch.Tensor:
        """Exact causal range minimum ``G[t, s] = min_{s<u<=t} phi_u``.

        Args:
            phi: Field values ``(B, nH, T)``.

        Returns:
            Range minima ``(B, nH, T, T)``; empty spans hold the finite
            sentinel so the bias is exactly zero there.
        """
        bsz, heads, seq_len = phi.shape
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=phi.device)
        )
        # M[t, u] = phi_u if u <= t else +inf.
        masked = phi.unsqueeze(-2).expand(bsz, heads, seq_len, seq_len).masked_fill(
            ~causal, float("inf")
        )
        # Suffix minimum: S[t, u] = min_{u <= v <= t} phi_v.
        suffix = torch.flip(
            torch.cummin(torch.flip(masked, dims=(-1,)), dim=-1).values, dims=(-1,)
        )
        suffix = cls._finite(suffix)
        # Shift one column: G[t, s] = S[t, s + 1] = min_{s < u <= t} phi_u.
        out = torch.full_like(suffix, torch.finfo(suffix.dtype).max / 4)
        out[..., :-1] = suffix[..., 1:]
        return out

    @classmethod
    def _soft_range_min(cls, phi: torch.Tensor, tau: float) -> torch.Tensor:
        """Soft causal range minimum for the straight-through gradient.

        ``G_soft[t, s] = -tau * log sum_{s<u<=t} exp(-phi_u / tau)`` — the
        paper's differentiable surrogate for the inner minimum (paper §2).
        The forward value is discarded by the STE; only its gradient is used.

        Args:
            phi: Field values ``(B, nH, T)``.
            tau: Soft-minimum temperature (positive).

        Returns:
            Soft range minima ``(B, nH, T, T)`` with finite sentinels for
            empty spans.
        """
        bsz, heads, seq_len = phi.shape
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=phi.device)
        )
        neg = (-phi / tau).unsqueeze(-2).expand(bsz, heads, seq_len, seq_len)
        masked = neg.masked_fill(~causal, float("-inf"))
        # Suffix log-sum-exp: L[t, u] = log sum_{u <= v <= t} exp(-phi_v / tau).
        lse = torch.flip(
            torch.logcumsumexp(torch.flip(masked, dims=(-1,)), dim=-1), dims=(-1,)
        )
        suffix = cls._finite(-tau * lse)
        out = torch.full_like(suffix, torch.finfo(suffix.dtype).max / 4)
        out[..., :-1] = suffix[..., 1:]
        return out

    def _bias_matrix(self, phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """First-passage bias matrix ``(B, nH, S, S)`` (causal or symmetric).

        Args:
            phi: Field values ``(B, nH, S)``.
            psi: Key levels ``(B, nH, S)``.

        Returns:
            Bias logits ``(B, nH, S, S)`` to add to the scaled scores.
        """
        hard = self._hard_range_min(phi)
        if self.training:
            # Straight-through: forward hard minimum, backward soft minimum.
            soft = self._soft_range_min(phi, self.tau)
            span = hard.detach() + (soft - soft.detach())
        else:
            span = hard
        if self.mode != "decoder":
            # Repo-side bidirectional extension: the open-span minimum in
            # either direction (the paper defines only the causal head).
            span = torch.minimum(span, span.transpose(-1, -2))
        return self.beta.view(1, -1, 1, 1) * torch.clamp(
            span - psi.unsqueeze(-2), max=0.0
        )

    @classmethod
    def _cached_range_min(
        cls,
        phi_c: torch.Tensor,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Range minima for incremental decoding against the cached field.

        ``G[t, s] = min`` of the cached field at absolute positions
        ``kv_positions`` strictly after key ``s`` and at most the query's
        absolute position (paper Prop. 2.4b). Cache positions are contiguous
        (no eviction yet), so the suffix-minimum shift-by-one is exact.

        Args:
            phi_c: Cached per-token field ``(B, nH, T)``.
            q_positions: Absolute query positions ``(S,)``.
            kv_positions: Absolute cached positions ``(T,)``.

        Returns:
            Range minima ``(B, nH, S, T)`` with finite sentinels for empty
            spans.
        """
        bsz, heads, kv_len = phi_c.shape
        q_len = q_positions.numel()
        visible = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        masked = phi_c.unsqueeze(-2).expand(bsz, heads, q_len, kv_len).masked_fill(
            ~visible, float("inf")
        )
        suffix = torch.flip(
            torch.cummin(torch.flip(masked, dims=(-1,)), dim=-1).values, dims=(-1,)
        )
        suffix = cls._finite(suffix)
        out = torch.full_like(suffix, torch.finfo(suffix.dtype).max / 4)
        out[..., :-1] = suffix[..., 1:]
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute First Passage Attention over the input sequence.

        Builds the per-head field, level and range-minimum bias, adds it to the
        scaled dot-product logits (plus the causal mask in decoder mode), and
        applies softmax. With a KV slot, decoding appends the per-token field
        and level as extras and evaluates the bias row over the cached prefix
        exactly (paper Prop. 2.4b).

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index forwarded to the
                positional encoder.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.
            cache: Optional KV slot for incremental decoding (``None`` runs
                the legacy full-recompute path, D5).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        phi, psi = self._field_and_level(x)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ...cache.hooks import (
                causal_mask_from_positions,
                kv_decode_append,
                positions_for,
            )

            offset = cache_position
            positions = positions_for(offset, seq_len, x.device)
            q, k = apply_pe_to_qk(
                pe,
                self.pe_type,
                q,
                k,
                x,
                logical_layer_idx or 0,
                self.use_pe,
                position_offset=offset,
            )
            K, V, kv_positions, extras = kv_decode_append(
                slot,
                k,
                v,
                positions,
                extras={"field_phi": phi, "key_level_psi": psi},
            )
            phi_c = extras["field_phi"]
            psi_c = extras["key_level_psi"]
            span = self._cached_range_min(phi_c, positions, kv_positions)
            bias = self.beta.view(1, -1, 1, 1) * torch.clamp(
                span - psi_c.unsqueeze(-2), max=0.0
            )
            attn = (q @ K.transpose(-2, -1)) * self.scale + bias.to(q.dtype)
            attn = apply_pe_to_scores(
                pe,
                self.pe_type,
                attn,
                q,
                self.use_pe,
                kv_len=K.size(-2),
                q_offset=0 if offset is None else int(offset),
                kv_positions=kv_positions,
            )
            mask = causal_mask_from_positions(positions, kv_positions)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            record_qk_max_logits(self, attn)
            attn = module_softmax(self, attn, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ V).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.total_dim)
            return self.out_proj(out)

        q, k = apply_pe_to_qk(
            pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe
        )

        bias = self._bias_matrix(phi, psi)
        attn = (q @ k.transpose(-2, -1)) * self.scale + bias.to(q.dtype)
        attn = apply_pe_to_scores(pe, self.pe_type, attn, q, self.use_pe)
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        record_qk_max_logits(self, attn)
        attn = module_softmax(self, attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.total_dim)
        return self.out_proj(out)
