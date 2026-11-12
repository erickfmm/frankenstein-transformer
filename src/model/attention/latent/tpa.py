"""Tensor Product Attention (TPA), arXiv:2501.06425.

TPA factorises each per-token query, key and value slice into a normalized
sum of rank-one outer products of two *contextual vector* factors:

.. math::

    Q_t = \\frac{1}{R_Q}\\sum_{r=1}^{R_Q} a^Q_r(x_t)\\otimes b^Q_r(x_t),
    \\qquad a^Q_r \\in \\mathbb{R}^{h},\\; b^Q_r \\in \\mathbb{R}^{d_h},

and likewise for keys (rank :math:`R_K`) and values (rank :math:`R_V`)
(eq. 3.1 of the paper).  The factors are produced by bias-free linear
projections of the hidden state, merged along the rank axis
(:math:`W^{aQ}: d_{model}\\to R_Q h`, :math:`W^{bQ}: d_{model}\\to R_Q d_h`);
the per-token slice is then
:math:`Q_t = (1/R_Q) A_Q(x_t)^{\\top} B_Q(x_t) \\in \\mathbb{R}^{h\\times d_h}`.
Attention itself is standard scaled dot-product attention (eqs. 3.2-3.3).

Because RoPE acts on the head-width axis only, it commutes with the
factorisation (Theorem 3.1): the paper pre-rotates the :math:`B_K` factors so
cached keys are already rotated (eq. 3.5).  This module follows the repo
convention of applying the shared positional encoder to the materialized
per-head Q/K (the ``tucker``/``kpa`` precedent); the two are algebraically
equivalent and pre-rotation is a decode-time optimisation.

The factorized KV cache stores ``{A^K, B^K, A^V, B^V}`` per token, i.e.
:math:`(R_K+R_V)(h+d_h)` floats instead of :math:`2 h d_h` for MHA (eq. 10).

Relation to the rest of the family: TPA is the vector (CP) special case of
Kronecker Product Attention (``kpa``, arXiv:2610.12657) with
``h1 = num_heads, d1 = 1, h2 = 1, d2 = head_dim``, and the activation-space
counterpart of Tucker Attention's weight factorisation (arXiv:2603.30033).

Reference:
    Zhang, Y., Liu, Y., Yuan, H., Qin, Z., Yuan, Y., Gu, Q., & Yao, A. C.-C.
    (2025). "Tensor Product Attention Is All You Need". arXiv:2501.06425.
    NeurIPS 2025 (Spotlight).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...numerics import module_softmax
from ..common import BitLinear
from ..params import mixer_params, use_pe_for


class TPAAttention(nn.Module):
    """Tensor Product Attention (Zhang et al. 2025).

    Args:
        config: Configuration object. Relevant attributes: ``dims.hidden_size``,
            ``dims.num_heads``, ``dims.dropout``, ``dims.mode``,
            ``bitnet.enabled`` and the optional ``model.attention.tpa`` knobs
            ``query_rank``, ``key_rank``, ``value_rank``.
        pos_encoder: Shared positional encoding module applied to the
            materialized query/key tensors. If ``None``, no PE is applied.
        params: Already-bound :class:`~src.registry.binding.ComponentParams`
            namespace; when omitted the knobs are read from the config.

    Attributes:
        hidden_size: Input dimensionality ``H``.
        num_heads: Number of attention heads ``h``.
        head_dim: Per-head dimensionality ``d_h = H / h``.
        query_rank, key_rank, value_rank: CP ranks ``R_Q``, ``R_K``, ``R_V``.
        w_aq, w_bq: Query factor projections ``H -> R_Q*h`` and
            ``H -> R_Q*d_h``.
        w_ak, w_bk: Key factor projections.
        w_av, w_bv: Value factor projections.
        out_proj: Output projection ``H -> H``.
        dropout, mode: As in the rest of the family.
        pos_encoder: Shared positional encoding module (or ``None``).
        pe_type: Positional encoding type string (lowercased).
        use_pe: Whether PE is enabled for this mixer.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads`` or if
            a rank is not positive.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        params = mixer_params(config, "tpa", params)
        super().__init__()
        self.hidden_size = config.dims.hidden_size
        self.num_heads = config.dims.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for TPAAttention")
        self.head_dim = self.hidden_size // self.num_heads

        self.query_rank = int(params.value("query_rank", 8))
        self.key_rank = int(params.value("key_rank", 2))
        self.value_rank = int(params.value("value_rank", 2))
        for name, rank in (
            ("query_rank", self.query_rank),
            ("key_rank", self.key_rank),
            ("value_rank", self.value_rank),
        ):
            if rank <= 0:
                raise ValueError(f"tpa {name} must be positive; got {rank}")

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        self.w_aq = proj_cls(self.hidden_size, self.query_rank * self.num_heads, bias=False)
        self.w_bq = proj_cls(self.hidden_size, self.query_rank * self.head_dim, bias=False)
        self.w_ak = proj_cls(self.hidden_size, self.key_rank * self.num_heads, bias=False)
        self.w_bk = proj_cls(self.hidden_size, self.key_rank * self.head_dim, bias=False)
        self.w_av = proj_cls(self.hidden_size, self.value_rank * self.num_heads, bias=False)
        self.w_bv = proj_cls(self.hidden_size, self.value_rank * self.head_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dims.dropout)
        self.mode = getattr(config.dims, "mode", "encoder")
        self.scale = self.head_dim ** -0.5
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "tpa", True))

    def _materialize(
        self,
        a_flat: torch.Tensor,
        b_flat: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        """Materialize ``Q``/``K``/``V`` from a pair of vector factors.

        Implements :math:`(1/R)\\sum_r a_r \\otimes b_r` in merged form,
        :math:`(1/R) A(x)^{\\top} B(x)` (eq. 3.1).

        Args:
            a_flat: Head-axis factor ``(B, T, R*h)``.
            b_flat: Feature-axis factor ``(B, T, R*d_h)``.
            rank: CP rank ``R`` (normalizer ``1/R``).

        Returns:
            Tensor ``(B, num_heads, T, head_dim)``.
        """
        bsz, seq_len, _ = a_flat.shape
        a = a_flat.view(bsz, seq_len, rank, self.num_heads)
        b = b_flat.view(bsz, seq_len, rank, self.head_dim)
        merged = torch.einsum("btrh,btrd->bthd", a, b) / float(rank)
        return merged.transpose(1, 2)

    def _new_factors(self, x: torch.Tensor) -> dict:
        """Compute the four flattened K/V factors for the incoming tokens.

        Args:
            x: Hidden state ``(B, T, H)``.

        Returns:
            Mapping with ``a_k``, ``b_k``, ``a_v`` and ``b_v`` flattened
            per-token factors (``(B, T, R_i * h)`` / ``(B, T, R_i * d_h)``).
        """
        return {
            "a_k": self.w_ak(x),
            "b_k": self.w_bk(x),
            "a_v": self.w_av(x),
            "b_v": self.w_bv(x),
        }

    def make_cache_slot(self, meta, cache):
        """Create the per-token vector-factor slot.

        The four cached streams hold ``A^K``, ``B^K``, ``A^V`` and ``B^V``;
        each is a per-token latent of ``R_i * h`` / ``R_i * d_h`` floats.
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
        """Compute Tensor Product Attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoder. Defaults to ``0`` if ``None``.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this forward call.
            cache: Optional latent slot caching the CP K/V factors
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

            q = self._materialize(self.w_aq(x), self.w_bq(x), self.query_rank)

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

        q = self._materialize(self.w_aq(x), self.w_bq(x), self.query_rank)
        k = self._materialize(self.w_ak(x), self.w_bk(x), self.key_rank)
        v = self._materialize(self.w_av(x), self.w_bv(x), self.value_rank)

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
