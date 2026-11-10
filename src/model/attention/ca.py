"""Continuum Attention (CA).

Implements Continuum Attention, arXiv:2610.20586 (A. Khadangi, 2026).
Conventional causal attention aggregates values only at discrete token
positions. CA instead interpolates adjacent key and value vectors into a
continuous, piecewise-linear memory trajectory and lets each query retrieve
information by normalised integration of an exponential relevance density
over its visible prefix, with constant half-token extensions at both ends.
Because the score is affine inside each segment, the integrals have closed
forms in exponential moments, so the operator is exactly computable without
numerical quadrature and remains a standard Transformer attention block.

Forward equations (paper numbering):

    q_i = W_Q x_i,  k_i = W_K x_i,  v_i = W_V x_i                     (1)
    k_j(u) = (1-u) k_{j-1} + u k_j,  v_j(u) = (1-u) v_{j-1} + u v_j  (2)-(3)
    s_i(t) = gamma q_i^T k(t) / sqrt(d_h)                             (4)
    a_ij = s_i(j-1),  b_ij = s_i(j),  Delta_ij = b_ij - a_ij          (5)
    s_ij(u) = a_ij + u Delta_ij                                       (6)
    p_i(t) = exp(s_i(t)) / Z_i                                        (7)
    o_i = integral p_i(t) v(t) dt                                     (8)
    M_ij = exp(a_ij) J_0(Delta_ij)                                    (14)
    U_ij = exp(a_ij) [(J_0 - J_1) v_{j-1} + J_1 v_j]                  (15)
    mu(Delta_ij) = J_1(Delta_ij) / J_0(Delta_ij)                      (16)
    Z_i = 1/2 e^{s_i(0)} + 1/2 e^{s_i(i)} + sum_j M_ij                (19)+(21)
    o_i = (U_i^cap + sum_j U_ij) / (Z_i^cap + sum_j M_ij)             (23)
    o_i = sum_j w_ij v_j,  w_ij >= 0,  sum_j w_ij = 1                 (24)

with the exponential moments

    J_n(Delta) = integral_0^1 u^n exp(Delta u) du,  n in {0, 1, 2}    (10)
    J_0(Delta) = (e^Delta - 1) / Delta                                (11)
    J_1(Delta) = ((Delta - 1) e^Delta + 1) / Delta^2                  (12)
    J_2(Delta) = ((Delta^2 - 2 Delta + 2) e^Delta - 2) / Delta^3      (13)

and the series J_n(Delta) = sum_r Delta^r / (r! (n + r + 1)) used near
Delta = 0 to avoid catastrophic cancellation (Eq. 34).

The implementation computes the *scaled* moments exp(a - m) J_0 and
exp(a - m) J_1 relative to the row maximum m (paper section 2.6), so that
no exponent overflows: the segment mass is (e^{b-m} - e^{a-m}) / Delta and
the first moment is (Delta e^{b-m} - e^{b-m} + e^{a-m}) / Delta^2, both
bounded by construction. The final output is exactly Eq. (23)/(24), i.e. a
normalised nonnegative weighted sum of the original value vectors.

The encoder (bidirectional) mode is the natural symmetric extension of the
causal-only paper: every query integrates over the full trajectory
[-1/2, n - 1/2], with segments j = 1..n-1 and caps at positions 0 and
n - 1. Decoder mode is paper-exact (visible prefix j <= i).

Reference:
    Khadangi, A. (2026). "Continuum Attention". arXiv:2610.20586.
"""

from __future__ import annotations

from .params import mixer_params, use_pe_for

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import BitLinear
from ..norm import attach_qk_norm, record_qk_max_logits


def _exp_moments(delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the exponential moments ``J_0, J_1, J_2`` of Eq. (10).

    Uses the closed forms (Eqs. 11-13) away from zero and the truncated
    series (Eq. 34, four terms) for ``|delta| < 1e-2`` to avoid the
    catastrophic cancellation of ``exp(Delta) - 1``.

    Args:
        delta: Segment score differences ``Delta_ij`` (any shape).

    Returns:
        Tuple ``(J_0, J_1, J_2)`` in float32 with the shape of ``delta``.
    """
    delta = delta.to(torch.float32)
    small = delta.abs() < 1e-2
    safe = torch.where(small, torch.ones_like(delta), delta)
    exp_d = torch.exp(safe)
    j0 = torch.expm1(safe) / safe
    j1 = ((safe - 1.0) * exp_d + 1.0) / (safe * safe)
    j2 = ((safe * safe - 2.0 * safe + 2.0) * exp_d - 2.0) / (safe * safe * safe)
    ds = torch.where(small, delta, torch.zeros_like(delta))
    ds2 = ds * ds
    ds3 = ds2 * ds
    j0 = torch.where(small, 1.0 + ds / 2.0 + ds2 / 6.0 + ds3 / 24.0, j0)
    j1 = torch.where(small, 0.5 + ds / 3.0 + ds2 / 8.0 + ds3 / 30.0, j1)
    j2 = torch.where(small, 1.0 / 3.0 + ds / 4.0 + ds2 / 10.0 + ds3 / 36.0, j2)
    return j0, j1, j2


class ContinuumAttention(nn.Module):
    """Continuum Attention: exact integration over a piecewise-linear memory.

    Projects the input into queries, keys, and values, computes the endpoint
    score matrix ``s_i(j) = gamma q_i^T k_j / sqrt(d_h)``, and replaces the
    softmax with the closed-form segment integration of Eqs. (14)-(23). The
    result is a normalised nonnegative weighted sum of the original value
    vectors (Eq. 24), so the mixer is a drop-in attention operator with the
    standard Q/K/V/O projections.

    With ``ca_learned_gain`` each head learns a positive score gain
    ``gamma = exp(rho)`` (one scalar per head, the paper's 101M flagship);
    otherwise the gain stays fixed at ``ca_gain_init`` and the mixer adds no
    parameters beyond the projections (the paper's CA-P configuration).

    Complexity:
        Training: O(n^2 * d). Inference: O(n) per step with KV cache.

    Reference:
        Khadangi (2026), "Continuum Attention", arXiv:2610.20586.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, ``ca_learned_gain``,
            ``ca_gain_init``, ``ca_use_pe``, and optionally ``mode``
            (``"encoder"`` or ``"decoder"``).
        pos_encoder: Optional shared positional encoding module.

    Attributes:
        hidden_size: Dimensionality of the input and output embeddings.
        num_heads: Number of parallel attention heads.
        head_dim: Dimensionality of each attention head.
        scale: Scaling factor ``1 / sqrt(head_dim)`` applied to dot products.
        log_gain: Per-head log score gain ``rho`` (learned variant only).
        gain_init: Fixed score gain (unlearned variant only).
        q_proj: Linear (or BitLinear) projection for queries.
        k_proj: Linear (or BitLinear) projection for keys.
        v_proj: Linear (or BitLinear) projection for values.
        out_proj: Linear (or BitLinear) output projection.
        dropout: Dropout layer applied to the effective weights.
        mode: ``"encoder"`` for bidirectional attention, ``"decoder"`` for
            causal (upper-triangular) masking.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        params = mixer_params(config, "ca", params)
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for ContinuumAttention"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        # Score gain gamma = 1 / tau (Eq. 4). Learned: one scalar per head
        # (paper section 2.7 / 3.1, 128 log-gains for 8 layers x 16 heads);
        # fixed: no parameters beyond the projections (CA-P).
        self.ca_learned_gain = bool(params.value("learned_gain", True))
        self.gain_init = float(params.value("gain_init", 1.0))
        if self.gain_init <= 0.0:
            raise ValueError(f"ca_gain_init must be > 0, got {self.gain_init}")
        if self.ca_learned_gain:
            self.log_gain = nn.Parameter(
                torch.full((self.num_heads,), math.log(self.gain_init))
            )
        else:
            self.register_parameter("log_gain", None)

        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "ca", True))
        attach_qk_norm(self, config, self.head_dim)

    # ------------------------------------------------------------------
    # Core operator
    # ------------------------------------------------------------------
    def _gain(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return the per-head score gain ``gamma`` as a broadcastable tensor."""
        if self.ca_learned_gain:
            return torch.exp(self.log_gain).to(dtype=dtype, device=device).view(
                1, self.num_heads, 1, 1
            )
        return torch.tensor(self.gain_init, dtype=dtype, device=device)

    def _effective_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """Map endpoint scores to CA effective weights (Eqs. 19-24).

        Args:
            scores: Endpoint score matrix ``(B, H, S, L)``. Entries that are
                not visible to their query row must be ``-inf`` (causal
                masking); equal-score rows reduce exactly to uniform causal
                attention, and a single visible position yields weight 1
                (``o_0 = v_0``).

        Returns:
            Effective weights ``(B, H, S, L)`` with ``w >= 0`` and
            ``sum_j w_ij = 1`` over the visible prefix of each row.
        """
        dtype = scores.dtype
        vis = torch.isfinite(scores)
        s = scores.to(torch.float32)

        # Row maximum over the visible prefix (paper section 2.6). Every row
        # has at least one visible entry (the query's own position).
        neg_inf = torch.finfo(torch.float32).min
        s_masked = torch.where(vis, s, torch.full_like(s, neg_inf))
        m = s_masked.max(dim=-1, keepdim=True).values
        # Replace non-visible entries with a finite sentinel (the row max) so
        # that Delta = b - a never forms -inf - -inf = NaN; their weights are
        # zeroed explicitly below.
        s_safe = torch.where(vis, s, m.expand_as(s))

        # Segment endpoints (Eq. 5): a = s(j-1), b = s(j), Delta = b - a.
        a = s_safe[..., :-1]
        b = s_safe[..., 1:]
        delta = b - a
        e_a = torch.exp(a - m)
        e_b = torch.exp(b - m)

        # Scaled moments e^{a-m} J_0 and e^{a-m} J_1, written in their
        # overflow-free forms (see module docstring).
        small = delta.abs() < 1e-2
        d = torch.where(small, torch.ones_like(delta), delta)
        mass = (e_b - e_a) / d
        first = (d * e_b - e_b + e_a) / (d * d)
        # Series expansion near Delta = 0 (Eq. 34).
        ds = torch.where(small, delta, torch.zeros_like(delta))
        ds2 = ds * ds
        ds3 = ds2 * ds
        mass_series = e_a * (1.0 + ds / 2.0 + ds2 / 6.0 + ds3 / 24.0)
        first_series = e_a * (0.5 + ds / 3.0 + ds2 / 8.0 + ds3 / 30.0)
        mass = torch.where(small, mass_series, mass)
        first = torch.where(small, first_series, first)

        # A segment contributes iff its right endpoint is visible (the
        # visible set is a prefix).
        seg_vis = vis[..., 1:]
        zero = torch.zeros_like(mass)
        mass = torch.where(seg_vis, mass, zero)
        first = torch.where(seg_vis, first, zero)

        # Half-token boundary caps (Eqs. 19-20): first and last visible
        # memory positions.
        cap_first = 0.5 * torch.exp(s_safe[..., 0] - m[..., 0])
        own = vis.to(torch.int64).sum(dim=-1) - 1
        own = own.clamp_min(0)
        own_score = torch.gather(s_safe, -1, own.unsqueeze(-1)).squeeze(-1)
        cap_last = 0.5 * torch.exp(own_score - m[..., 0])

        # Assemble the effective weights: left endpoint of segment j gets
        # (M_ij - M1_ij), right endpoint gets M1_ij (Eq. 15 rewritten with
        # the scaled moments).
        w = torch.zeros_like(s)
        w[..., :-1] += mass - first
        w[..., 1:] += first
        w[..., 0] += cap_first
        w.scatter_add_(-1, own.unsqueeze(-1), cap_last.unsqueeze(-1))

        z = cap_first + cap_last + mass.sum(dim=-1)
        z = z.clamp_min(torch.finfo(torch.float32).tiny)
        w = w / z.unsqueeze(-1)
        w = torch.where(vis, w, torch.zeros_like(w))
        return w.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Continuum Attention over the input sequence.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index (for layer-dependent PEs).
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this forward call.
            cache: Optional :class:`~src.model.cache.slots.KVSlot` for
                incremental decoding. ``None`` runs the legacy full-recompute
                path (D4).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        gamma = self._gain(x.dtype, x.device)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ..cache.hooks import cached_qk_attention

            # Fold gamma into q before the PE rotation: rotation is linear, so
            # gamma * R(q) == R(gamma * q), and the cached branch computes
            # (q K^T) * scale internally.
            q = q * gamma
            out, _, _ = cached_qk_attention(
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
                logical_layer_idx=logical_layer_idx or 0,
                normalize=self._effective_weights,
                dropout=self.dropout,
                causal=True,
                use_fixed_softmax=False,
            )
            return self.out_proj(out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden))

        from .common import apply_pe_to_qk, apply_pe_to_scores

        idx = logical_layer_idx or 0
        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, idx, self.use_pe)

        scores = (q @ k.transpose(-2, -1)) * self.scale * gamma
        scores = apply_pe_to_scores(pe, self.pe_type, scores, q, self.use_pe)
        record_qk_max_logits(self, scores)

        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = self._effective_weights(scores)
        weights = self.dropout(weights)

        out = (weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
