"""Multi-matrix Factorization Attention (MFA), arXiv:2412.19255.

MFA factorises the query-key circuit of multi-head attention as

.. math::

    W_c \\approx S_q Q_c S_k^{\\top}, \\qquad
    U_c \\approx S_v O_c^{\\top},

with shared projections :math:`S_q, S_k, S_v \\in \\mathbb{R}^{H \\times C}`
and head-specific factors :math:`Q_c, O_c \\in \\mathbb{R}^{C \\times C}`
(eqs. 6-7 of the paper).  Keys and values are *single-head*,
:math:`k_j = x_j S_k \\in \\mathbb{R}^{C}` and
:math:`v_j = x_j S_v \\in \\mathbb{R}^{C}`, so the KV cache per token is
:math:`2C` floats **independent of the head count** :math:`m`, while each
head keeps a rank :math:`C > d` interaction.  Attention itself is ordinary
scaled dot-product attention (eq. 7)::

    q_c = x S_q Q_c,   A_c = softmax(q_c k^T / sqrt(C)),
    y_c = A_c v,       O   = sum_c y_c O_c^T.

The value projection can be re-parameterised from the key projection
(MFA-Key-Reuse, eqs. 8-9):

.. math::

    S_v = (I + \\mathrm{diag}(\\alpha) N) S_k,

with :math:`N \\in \\mathbb{R}^{C \\times C}` and
:math:`\\alpha \\in \\mathbb{R}^{C}` initialised to **zero** (the paper's
stability recipe), so :math:`S_v = S_k` at step 0 and decode caches only
the key stream (``C`` floats/token).

Relation to the family: MFA is the capacity-oriented counterpart of MLA
(``mla``) - MLA compresses a full multi-head K/V pair into one latent,
MFA keeps many high-rank heads over one shared K/V pair - and the direct
generalisation of MQA (``num_kv_heads = 1``) with a factorised, expanded
QK circuit.  RoPE applies row-wise to the :math:`C`-dim queries and keys,
so the mixer is PE-compatible without MLA's decoupled branch.

Reference:
    Hu, J., Li, H., Zhang, Y., Wang, Z., Zhou, S., Zhang, X., Shum, H.-Y.,
    & Jiang, D. (2025). "Multi-matrix Factorization Attention".
    arXiv:2412.19255.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...numerics import module_softmax
from ...norm import attach_qk_norm, record_qk_max_logits
from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores
from ..params import mixer_params, use_pe_for


class MFAAttention(nn.Module):
    """Multi-matrix Factorization Attention (Hu et al. 2025).

    Args:
        config: Configuration object. Relevant attributes:
            ``dims.hidden_size``, ``dims.num_heads``, ``dims.dropout``,
            ``dims.mode``, ``bitnet.enabled`` and the optional
            ``model.attention.mfa`` knobs ``factor_rank`` and
            ``key_reuse``.
        pos_encoder: Shared positional encoding module applied to the
            per-head queries and the shared key. If ``None``, no PE is
            applied.
        params: Already-bound :class:`~src.registry.binding.ComponentParams`
            namespace; when omitted the knobs are read from the config.

    Attributes:
        hidden_size: Input dimensionality ``H``.
        num_heads: Number of attention heads ``m``.
        factor_rank: Low-rank factorization dimension ``C`` (per-head
            width, shared K/V width and KV-cache width).
        head_dim: Equals ``factor_rank`` (kept for the cache/PE plumbing).
        num_kv_heads: Always ``1`` (single shared key/value head).
        key_reuse: Whether the MFA-KR value re-parameterisation is active.
        w_q_shared: Shared query down-projection ``H -> C``.
        w_k_shared: Shared key down-projection ``H -> C`` (the only key
            projection).
        w_v_shared: Shared value down-projection ``H -> C`` (``None``
            under ``key_reuse``).
        key_reuse_mix: ``N`` of eq. 8 (``C x C``), ``None`` otherwise.
        key_reuse_gate: Zero-initialised ``alpha`` of eq. 8, ``None``
            otherwise.
        w_q_heads: Per-head query factors ``(m, C, C)``.
        w_o_heads: Per-head output projections ``(m, C, H)``.
        dropout, mode: As in the rest of the family.
        pos_encoder: Shared positional encoding module (or ``None``).
        pe_type: Positional encoding type string (lowercased).
        use_pe: Whether PE is enabled for this mixer.

    Raises:
        ValueError: If ``factor_rank`` is not positive.

    Note:
        Unlike MHA, ``hidden_size`` need not be divisible by ``num_heads``:
        every head is ``factor_rank`` wide and the shared K/V head is
        ``factor_rank`` wide too, which is what lets MFA scale the head
        count independently of the model width (paper Table 7: 18 heads of
        width 256 at ``hidden_size = 2048``).
    """

    def __init__(self, config, pos_encoder=None, params=None):
        params = mixer_params(config, "mfa", params)
        super().__init__()
        self.hidden_size = int(config.dims.hidden_size)
        self.num_heads = int(config.dims.num_heads)
        self.factor_rank = int(params.value("factor_rank", 128))
        if self.factor_rank <= 0:
            raise ValueError(f"mfa factor_rank must be positive; got {self.factor_rank}")
        # The MFA head width *is* the factorization rank; the KV cache is a
        # single shared head of that width (eq. 6).
        self.head_dim = self.factor_rank
        self.num_kv_heads = 1
        self.key_reuse = bool(params.value("key_reuse", False))

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        self.w_q_shared = proj_cls(self.hidden_size, self.factor_rank, bias=False)
        self.w_k_shared = proj_cls(self.hidden_size, self.factor_rank, bias=False)
        if self.key_reuse:
            self.w_v_shared = None
            self.key_reuse_mix = nn.Parameter(
                torch.randn(self.factor_rank, self.factor_rank) * 0.02
            )
            self.key_reuse_gate = nn.Parameter(torch.zeros(self.factor_rank))
        else:
            self.w_v_shared = proj_cls(self.hidden_size, self.factor_rank, bias=False)
            self.register_parameter("key_reuse_mix", None)
            self.register_parameter("key_reuse_gate", None)

        self.w_q_heads = nn.Parameter(
            torch.empty(self.num_heads, self.factor_rank, self.factor_rank)
        )
        self.w_o_heads = nn.Parameter(
            torch.empty(self.num_heads, self.factor_rank, self.hidden_size)
        )
        for tensor in (self.w_q_heads, self.w_o_heads):
            nn.init.normal_(tensor, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(config.dims.dropout)
        self.mode = getattr(config.dims, "mode", "encoder")
        self.scale = self.factor_rank ** -0.5
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "mfa", True))
        attach_qk_norm(self, config, self.head_dim)

    def _values_from_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """Reconstruct values from cached keys (MFA-KR, eqs. 8-9).

        Implements :math:`v = k\\,(I + \\mathrm{diag}(\\alpha) N)`.

        Args:
            keys: Shared key stream ``(B, T, C)``.

        Returns:
            Value stream ``(B, T, C)``.
        """
        return keys + (self.key_reuse_gate * keys) @ self.key_reuse_mix

    def _project_heads(self, attended: torch.Tensor) -> torch.Tensor:
        """Apply the per-head output projections and sum the heads.

        Implements :math:`O = \\sum_c y_c O_c^{\\top}` with
        :math:`O_c \\in \\mathbb{R}^{C \\times H}`.

        Args:
            attended: Attended values ``(B, m, T, C)``.

        Returns:
            Output tensor ``(B, T, H)``.
        """
        return torch.einsum("bmtd,mch->bth", attended, self.w_o_heads)

    def make_cache_slot(self, meta, cache):
        """Create the single-shared-head K/V slot.

        Plain MFA caches the shared ``k`` and ``v`` streams; MFA-KR caches
        only ``k`` (the value is reconstructed by eq. 9), which is the
        paper's additional 50% cache reduction.
        """
        from ...cache.slots import LatentKVSlot

        axes = {"k": "token"} if self.key_reuse else {"k": "token", "v": "token"}
        return LatentKVSlot(meta, axes=axes, **cache.codec_slot_kwargs())

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Multi-matrix Factorization Attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoder. Defaults to ``0`` if ``None``.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this forward call.
            cache: Optional latent slot caching the shared K/V stream(s)
                (``None`` runs the uncached path, D5).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        idx = logical_layer_idx or 0

        q_shared = self.w_q_shared(x)
        q = torch.einsum("btc,mcd->bmtd", q_shared, self.w_q_heads)
        if self.qk_norm is not None:
            q = self.qk_norm.normalize_q(q)

        slot = cache if getattr(cache, "kind", "") == "latent_kv" else None
        if slot is not None:
            from ...cache.hooks import cached_latent_attention

            latents = {"k": self.w_k_shared(x)}
            if not self.key_reuse:
                latents["v"] = self.w_v_shared(x)

            def _expand(c_all):
                keys = c_all["k"].unsqueeze(1)
                if self.qk_norm is not None:
                    keys = self.qk_norm.normalize_k(keys)
                if self.key_reuse:
                    values = self._values_from_keys(c_all["k"]).unsqueeze(1)
                else:
                    values = c_all["v"].unsqueeze(1)
                return keys, values

            attended = cached_latent_attention(
                slot,
                q,
                latents,
                cache_position=cache_position,
                scale=self.scale,
                expand_fn=_expand,
                pe=pe,
                pe_type=self.pe_type,
                use_pe=self.use_pe,
                x_for_pe=x,
                logical_layer_idx=idx,
                dropout=self.dropout,
                causal=True,
                use_fixed_softmax=bool(getattr(self, "use_fixed_softmax", False)),
            )
            return self._project_heads(attended)

        keys = self.w_k_shared(x).view(bsz, seq_len, 1, self.factor_rank).transpose(1, 2)
        if self.qk_norm is not None:
            keys = self.qk_norm.normalize_k(keys)
        if self.key_reuse:
            values = self._values_from_keys(self.w_k_shared(x)).view(
                bsz, seq_len, 1, self.factor_rank
            ).transpose(1, 2)
        else:
            values = self.w_v_shared(x).view(
                bsz, seq_len, 1, self.factor_rank
            ).transpose(1, 2)

        q, keys = apply_pe_to_qk(pe, self.pe_type, q, keys, x, idx, self.use_pe)

        attn_scores = (q @ keys.transpose(-2, -1)) * self.scale
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
        attended = attn_weights @ values
        return self._project_heads(attended)
