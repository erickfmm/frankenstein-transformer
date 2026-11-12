"""Kronecker Product Attention (KPA), arXiv:2610.12657.

KPA factorises each per-token key and value matrix as a normalized sum of
Kronecker products of two compact, *contextual* matrix factors:

.. math::

    K_t = \\frac{1}{R_K}\\sum_{r=1}^{R_K} A^K_r(x_t) \\otimes B^K_r(x_t),
    \\qquad A^K_r \\in \\mathbb{R}^{h_1\\times d_1},\\;
    B^K_r \\in \\mathbb{R}^{h_2\\times d_2},

with :math:`h = h_1 h_2` heads and :math:`d_h = d_1 d_2` head width, and the
same construction for queries (rank :math:`R_Q`) and values (rank
:math:`R_V`).  Both factors are produced by independent bias-free linear
projections of the hidden state (eqs. 2-5 of the paper).  Because the
Kronecker product obeys the mixed-product identity
:math:`(M\\otimes N)(P\\otimes Q) = (MP)\\otimes(NQ)`, the attention score of
head :math:`(i_1,i_2)` factors into a head-level and a feature-level inner
product (Proposition 2.2), and the per-token KV cache only needs
:math:`(R_K+R_V)(h_1d_1+h_2d_2)` floats instead of
:math:`2h_1h_2d_1d_2` for MHA (eq. 10).

Kronecker-Separable RoPE (Definition 2.3 / Theorem 2.4) rotates only the
:math:`B` factor with frequencies calibrated to :math:`d_2`, which preserves
the relative-position property without MLA's decoupled RoPE branch.  This
module follows the repo convention of applying the shared positional
encoder to the materialized per-head Q/K (the ``tucker`` precedent), which
coincides with the paper's operator when the rotary base is calibrated to
``d_2``.

Special case: ``h1 = num_heads, d1 = 1, h2 = 1, d2 = head_dim`` makes every
factor a vector and recovers Tensor Product Attention (TPA).

Reference:
    Nikpour, B., Rajabzadeh, H., Dialameh, M., Partovi Nia, V., & Omidi, P.
    (2026). "Kronecker Product Attention: Memory-Efficient KV Compression
    via Structured Factorization". arXiv:2610.12657.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...numerics import module_softmax
from ..common import BitLinear
from ..params import mixer_params, use_pe_for


def _auto_head_split(num_heads: int) -> int:
    """Return the default ``h1`` for the Kronecker head split.

    The paper's best configurations use an asymmetric split with the larger
    factor in ``h1`` (``8x2`` for 16 heads, ``10x2`` for 20).  Pick the
    largest proper divisor of ``num_heads`` (at most ``num_heads // 2``) so
    a non-trivial two-level structure exists; fall back to ``1`` for prime
    or single-head counts.

    Args:
        num_heads: Total number of attention heads.

    Returns:
        A positive divisor ``h1`` of ``num_heads``.
    """
    best = 1
    for candidate in range(1, num_heads // 2 + 1):
        if num_heads % candidate == 0:
            best = candidate
    return best


def _auto_feature_split(head_dim: int, preferred: int = 8) -> int:
    """Return the default ``d1`` for the Kronecker feature split.

    The paper fixes ``d1 = d2 = 8`` (Appendix C).  Use the largest divisor of
    ``head_dim`` that is at most ``preferred`` so ``d1 * d2 == head_dim``
    holds exactly for smaller head widths.

    Args:
        head_dim: Per-head dimensionality.
        preferred: Preferred factor width (default 8, the paper's value).

    Returns:
        A positive divisor ``d1`` of ``head_dim``.
    """
    best = 1
    for candidate in range(1, min(head_dim, preferred) + 1):
        if head_dim % candidate == 0:
            best = candidate
    return best


class KPAAttention(nn.Module):
    """Kronecker Product Attention (Nikpour et al. 2026).

    Args:
        config: Configuration object. Relevant attributes: ``dims.hidden_size``,
            ``dims.num_heads``, ``dims.dropout``, ``dims.mode``,
            ``bitnet.enabled`` and the optional ``model.attention.kpa`` knobs
            ``h1``, ``h2``, ``d1``, ``d2``, ``query_rank``, ``key_rank``,
            ``value_rank``.
        pos_encoder: Shared positional encoding module applied to the
            materialized query/key tensors. If ``None``, no PE is applied.
        params: Already-bound :class:`~src.registry.binding.ComponentParams`
            namespace; when omitted the knobs are read from the config.

    Attributes:
        hidden_size: Input dimensionality ``H``.
        num_heads: Total number of heads ``h = h1 * h2``.
        head_dim: Per-head dimensionality ``d_h = d1 * d2``.
        h1, h2: Head-partition and within-head factor counts.
        d1, d2: Head-partition and within-head factor widths.
        query_rank, key_rank, value_rank: Kronecker ranks ``R_Q``, ``R_K``,
            ``R_V``.
        w_aq, w_bq: Query factor projections ``H -> R_Q*h1*d1`` and
            ``H -> R_Q*h2*d2``.
        w_ak, w_bk: Key factor projections.
        w_av, w_bv: Value factor projections.
        out_proj: Output projection ``H -> H``.
        dropout, mode: As in the rest of the family.
        pos_encoder: Shared positional encoding module (or ``None``).
        pe_type: Positional encoding type string (lowercased).
        use_pe: Whether PE is enabled for this mixer.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``, if
            ``h1 * h2 != num_heads``, if ``d1 * d2 != head_dim`` or if a rank
            is not positive.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        params = mixer_params(config, "kpa", params)
        super().__init__()
        self.hidden_size = config.dims.hidden_size
        self.num_heads = config.dims.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for KPAAttention")
        self.head_dim = self.hidden_size // self.num_heads

        h1 = params.value("h1", None)
        h2 = params.value("h2", None)
        if h1 is None:
            h1 = _auto_head_split(self.num_heads)
        h1 = int(h1)
        if h2 is None:
            h2 = self.num_heads // h1 if h1 > 0 else 1
        h2 = int(h2)
        if h1 <= 0 or h2 <= 0 or h1 * h2 != self.num_heads:
            raise ValueError(
                f"kpa head split h1*h2 must equal num_heads ({self.num_heads}); "
                f"got h1={h1}, h2={h2}"
            )

        d1 = params.value("d1", None)
        d2 = params.value("d2", None)
        if d1 is None:
            d1 = _auto_feature_split(self.head_dim)
        d1 = int(d1)
        if d1 <= 0 or self.head_dim % d1 != 0:
            # Documented fallback: the default ``d1 = 8`` need not divide
            # small head widths (e.g. ``head_dim = 12``); use the largest
            # divisor not exceeding the preferred width instead.
            d1 = _auto_feature_split(self.head_dim)
        if d2 is None:
            d2 = self.head_dim // d1
        d2 = int(d2)
        if d1 <= 0 or d2 <= 0 or d1 * d2 != self.head_dim:
            raise ValueError(
                f"kpa feature split d1*d2 must equal head_dim ({self.head_dim}); "
                f"got d1={d1}, d2={d2}"
            )

        self.h1, self.h2, self.d1, self.d2 = h1, h2, d1, d2
        self.query_rank = int(params.value("query_rank", 6))
        self.key_rank = int(params.value("key_rank", 2))
        self.value_rank = int(params.value("value_rank", 2))
        for name, rank in (
            ("query_rank", self.query_rank),
            ("key_rank", self.key_rank),
            ("value_rank", self.value_rank),
        ):
            if rank <= 0:
                raise ValueError(f"kpa {name} must be positive; got {rank}")

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        self.w_aq = proj_cls(self.hidden_size, self.query_rank * h1 * d1, bias=False)
        self.w_bq = proj_cls(self.hidden_size, self.query_rank * h2 * d2, bias=False)
        self.w_ak = proj_cls(self.hidden_size, self.key_rank * h1 * d1, bias=False)
        self.w_bk = proj_cls(self.hidden_size, self.key_rank * h2 * d2, bias=False)
        self.w_av = proj_cls(self.hidden_size, self.value_rank * h1 * d1, bias=False)
        self.w_bv = proj_cls(self.hidden_size, self.value_rank * h2 * d2, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dims.dropout)
        self.mode = getattr(config.dims, "mode", "encoder")
        self.scale = self.head_dim ** -0.5
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "kpa", True))

    def _factor_flat(self, x: torch.Tensor, proj: nn.Module, rank: int, width: int) -> torch.Tensor:
        """Project ``x`` into a flattened per-token factor (eq. 5).

        Args:
            x: Hidden state ``(B, T, H)``.
            proj: Aggregate projection ``H -> rank * width``.
            rank: Kronecker rank ``R``.
            width: Factor width ``h_i * d_i``.

        Returns:
            Flattened factor ``(B, T, R * width)``.
        """
        return proj(x)

    def _materialize(
        self,
        a_flat: torch.Tensor,
        b_flat: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        """Materialize ``Q``/``K``/``V`` from a pair of factors (eqs. 2-4).

        Applies the normalized Kronecker sum over the rank axis and reshapes
        the two factor axes into ``(num_heads, head_dim)`` with ``i1`` major
        and ``i2`` minor (matching the paper's head indexing
        ``(i1, i2) -> i1 * h2 + i2``).

        Args:
            a_flat: Head-partition factor ``(B, T, R*h1*d1)``.
            b_flat: Within-head factor ``(B, T, R*h2*d2)``.
            rank: Kronecker rank ``R`` (normalizer ``1/R``).

        Returns:
            Tensor ``(B, num_heads, T, head_dim)``.
        """
        bsz, seq_len, _ = a_flat.shape
        a = a_flat.view(bsz, seq_len, rank, self.h1, self.d1)
        b = b_flat.view(bsz, seq_len, rank, self.h2, self.d2)
        # (1/R) sum_r A_r ⊗ B_r, axes -> (h1, h2, d1, d2)
        merged = torch.einsum("btrid,btrjf->btijdf", a, b) / float(rank)
        merged = merged.reshape(bsz, seq_len, self.h1 * self.h2, self.d1 * self.d2)
        return merged.transpose(1, 2)

    def _new_factors(self, x: torch.Tensor) -> dict:
        """Compute the four flattened K/V factors for the incoming tokens.

        Args:
            x: Hidden state ``(B, T, H)``.

        Returns:
            Mapping with ``a_k``, ``b_k``, ``a_v`` and ``b_v`` flattened
            per-token factors (``(B, T, R_i * h_i * d_i)``).
        """
        return {
            "a_k": self._factor_flat(x, self.w_ak, self.key_rank, self.h1 * self.d1),
            "b_k": self._factor_flat(x, self.w_bk, self.key_rank, self.h2 * self.d2),
            "a_v": self._factor_flat(x, self.w_av, self.value_rank, self.h1 * self.d1),
            "b_v": self._factor_flat(x, self.w_bv, self.value_rank, self.h2 * self.d2),
        }

    def make_cache_slot(self, meta, cache):
        """Create the per-token Kronecker factor slot.

        The four cached streams hold ``A^K``, ``B^K`` (pre-rotated by the
        shared PE on expansion), ``A^V`` and ``B^V``; each is a per-token
        latent of ``R_i * h_i * d_i`` floats.
        """
        from ...cache.slots import LatentKVSlot

        return LatentKVSlot(
            meta,
            axes={"a_k": "token", "b_k": "token", "a_v": "token", "b_v": "token"},
            **cache.codec_slot_kwargs(),
        )

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Kronecker Product Attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoder. Defaults to ``0`` if ``None``.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this forward call.
            cache: Optional latent slot caching the Kronecker K/V factors
                (``None`` runs the uncached path, D5).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        slot = cache if getattr(cache, "kind", "") == "latent_kv" else None
        if slot is not None:
            from ...cache.hooks import cached_latent_attention

            q = self._materialize(
                self._factor_flat(x, self.w_aq, self.query_rank, self.h1 * self.d1),
                self._factor_flat(x, self.w_bq, self.query_rank, self.h2 * self.d2),
                self.query_rank,
            )

            def _expand(c_all):
                k = self._materialize(c_all["a_k"], c_all["b_k"], self.key_rank)
                v = self._materialize(c_all["a_v"], c_all["b_v"], self.value_rank)
                return k, v

            attended = cached_latent_attention(
                slot,
                q,
                self._new_factors(x),
                cache_position=cache_position,
                scale=self.scale,
                expand_fn=_expand,
                pe=pe,
                pe_type=self.pe_type,
                use_pe=self.use_pe,
                x_for_pe=x,
                logical_layer_idx=logical_layer_idx or 0,
                dropout=self.dropout,
                causal=True,
                use_fixed_softmax=self.use_fixed_softmax,
            )
            out = attended.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            return self.out_proj(out)

        q = self._materialize(
            self._factor_flat(x, self.w_aq, self.query_rank, self.h1 * self.d1),
            self._factor_flat(x, self.w_bq, self.query_rank, self.h2 * self.d2),
            self.query_rank,
        )
        k = self._materialize(
            self._factor_flat(x, self.w_ak, self.key_rank, self.h1 * self.d1),
            self._factor_flat(x, self.w_bk, self.key_rank, self.h2 * self.d2),
            self.key_rank,
        )
        v = self._materialize(
            self._factor_flat(x, self.w_av, self.value_rank, self.h1 * self.d1),
            self._factor_flat(x, self.w_bv, self.value_rank, self.h2 * self.d2),
            self.value_rank,
        )

        from ..common import apply_pe_to_qk, apply_pe_to_scores

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = apply_pe_to_scores(pe, self.pe_type, attn_scores, q, self.use_pe)
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = module_softmax(self, attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)
