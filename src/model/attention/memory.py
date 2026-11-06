"""Memory Attention (MA).

Memory Attention replaces the dedicated value projection with a layer-specific,
token-ID-indexed memory table added to the contextual keys. For hidden states
``X`` and token IDs ``s`` the layer computes ``Q = X W_Q`` and ``K = X W_K``
(no ``W_V``), retrieves ``M = Norm(E[s])`` from a learnable table
``E ∈ R^{N × d_v}`` (RMSNorm per key/value head), and constructs values as
``V = K + M`` from the *unrotated* keys. Rotary positional encodings apply to
the scoring pair only, ``Q_R = RoPE(Q)`` and ``K_R = RoPE(K)``, leaving the
standard softmax weighting ``softmax(Q_R K_R^T / sqrt(d_k) + C)`` unchanged.

Reference:
    Kang (2026), "Memory Attention", arXiv:2609.28399.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from ..numerics import module_softmax
from ..norm import attach_qk_norm, record_qk_max_logits

from .common import BitLinear


class MemoryAttention(nn.Module):
    """Memory Attention with a token-ID memory table in place of ``W_V``.

    Each layer owns one memory table indexed by token IDs. Values are the
    contextual keys plus the per-KV-head RMS-normalized memory row (paper
    eqs. 7-9); attention weighting and aggregation are the standard softmax
    operator. The mixer supports grouped-query head sharing: keys are projected
    to ``num_kv_heads`` heads and values (key + memory) are expanded across
    query groups, so query heads sharing a key/value head also share its
    memory contribution (paper Appendix A).

    Complexity:
        Training: O(n^2 * d) attention with O(n * d_v) value construction.
        Inference: O(n) per step with a conventional KV cache.

    Reference:
        Kang (2026), "Memory Attention", arXiv:2609.28399.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``num_kv_heads``, ``vocab_size``, ``dropout``,
            ``use_bitnet``, and optionally ``mode`` (``"encoder"`` or
            ``"decoder"``).

    Attributes:
        hidden_size: Dimensionality of the input and output embeddings.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (``d_v = num_kv_heads *
            head_dim``); also the memory-table width.
        head_dim: Per-head dimensionality (``hidden_size // num_heads``).
        q_proj: Query projection.
        k_proj: Key projection (shared scoring/value basis; no value
            projection exists).
        out_proj: Output projection.
        memory: Token-ID memory table ``E`` of shape ``(vocab_size, d_v)``.
        memory_norm_weight: Per-key/value-head RMSNorm gains of shape
            ``(d_v,)``.
        dropout: Dropout applied to the attention weights.
        mode: ``"encoder"`` for bidirectional attention, ``"decoder"`` for
            causal masking.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        ValueError: If ``num_kv_heads`` is not in ``[1, num_heads]``.
        ValueError: If ``num_heads`` is not divisible by ``num_kv_heads``.
    """

    def __init__(self, config, pos_encoder=None) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = int(getattr(config, "num_kv_heads", 1))
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        # Paper Appendix A: RMSNorm epsilon 1e-6 (fixed, not a knob).
        self.memory_eps = 1e-6

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for MemoryAttention")
        if not 1 <= int(self.num_kv_heads) <= self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be in [1, num_heads ({self.num_heads})]"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads "
                f"({self.num_kv_heads})"
            )
        self.num_groups = self.num_heads // self.num_kv_heads
        kv_hidden = self.num_kv_heads * self.head_dim

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, kv_hidden, bias=False)
        # No value projection: V = K + Norm(E[s]) (paper eq. 8/10).
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        # Layer-specific token memory E in R^{N x d_v} (paper eq. 7). Stay a
        # plain lookup table under BitNet: BitNet quantizes Linear layers, and
        # the per-head RMSNorm below bounds each retrieved row.
        self.memory = nn.Embedding(int(config.vocab_size), kv_hidden)
        self.memory_norm_weight = nn.Parameter(torch.ones(kv_hidden))

        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "memory_attn_use_pe", True))
        attach_qk_norm(self, config, self.head_dim)

    def _memory_lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve and normalize ``M = Norm(E[s])`` (paper eq. 7).

        RMSNorm is applied independently within each token vector and each
        key/value head, with a learnable per-element gain.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)``.

        Returns:
            Memory vectors of shape ``(batch, seq_len, num_kv_heads * head_dim)``.
        """
        rows = self.memory(input_ids)
        bsz, seq_len, width = rows.shape
        x = rows.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.memory_eps)
        x = x / rms
        weight = self.memory_norm_weight.view(self.num_kv_heads, self.head_dim)
        return (x * weight).view(bsz, seq_len, width)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Memory Attention (optionally over a KV cache).

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            input_ids: Token IDs of shape ``(batch_size, seq_len)``; required
                because the memory table is indexed by token ID.
            logical_layer_idx: Logical layer index forwarded to the positional
                encoder.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this call.
            cache: Optional :class:`~src.model.cache.slots.KVSlot` for
                incremental decoding. ``None`` runs the legacy full-recompute
                path (D5).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``input_ids`` is ``None`` (text-token mixer).
        """
        if input_ids is None:
            raise ValueError(
                "memory_attn requires input_ids: its value memory is indexed by token ID"
            )
        bsz, seq_len, hidden = x.shape
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q = self.qk_norm.normalize_q(q)
            k = self.qk_norm.normalize_k(k)

        # Eq. (8): values are the (still unrotated) keys plus the normalized
        # token memory. RoPE below touches the scoring pair only (eq. 9).
        m = self._memory_lookup(input_ids).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = k + m

        idx = logical_layer_idx or 0
        from .common import apply_pe_to_qk, apply_pe_to_scores

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ..cache.hooks import causal_mask_from_positions, kv_decode_append, positions_for

            offset = cache_position
            positions = positions_for(offset, seq_len, x.device)
            # Keys are cached POST-RoPE; values are the constructed V, so the
            # slot is byte-identical to a standard KV cache (memory is read
            # only at append time).
            q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, idx, self.use_pe, position_offset=offset)
            K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)

            attn_scores = (q @ K.transpose(-2, -1)) * self.scale
            attn_scores = apply_pe_to_scores(
                pe,
                self.pe_type,
                attn_scores,
                q,
                self.use_pe,
                kv_len=K.size(-2),
                q_offset=0 if offset is None else int(offset),
                kv_positions=kv_positions,
            )
            record_qk_max_logits(self, attn_scores)
            mask = causal_mask_from_positions(positions, kv_positions)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = module_softmax(self, attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = (attn_weights @ V).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.out_proj(out)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, idx, self.use_pe)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = apply_pe_to_scores(pe, self.pe_type, attn_scores, q, self.use_pe)
        record_qk_max_logits(self, attn_scores)
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = module_softmax(self, attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
