"""ProbSparse self-attention (Wu, 2022; arXiv:2209.00260).

Implements the *ProbSparse* self-attention of the Deep Sparse Conformer
(InterSpeech 2022), adapted from Informer (arXiv:2012.07436). Instead of
computing the full ``L x L`` score matrix, the layer measures how "dominant"
each query is under a max-mean approximation of the KL divergence between its
attention distribution and the uniform distribution, keeps only the top-``u``
queries (``u = c2 * ceil(ln L)``) and scores those against all keys. The probe
that ranks queries only touches a random subset of ``L'_K = c1 * ceil(ln L)``
keys, so the whole layer runs in ``O(L log L)`` time and memory.

Equations (paper Eq. 5--10 and Algorithm 1)::

    A(Q, K, V) = sum_j [ k(Q, k_j) / sum_l k(Q, k_l) ] v_j
               = E_{p(k_j | Q)}[v_j]
               ~= Softmax(Q_bar K^T / sqrt(d)) V                     (7)

    M_bar(q_i, K) = max_j { q_i k_j^T / sqrt(d) }
                    - (1/L) sum_j q_i k_j^T / sqrt(d)                (8)

    MHSA-Sparse(X) = Concat(head_1, ..., head_h) W_O                  (9)
    head_i         = A(X W_Q^i, X W_K^i, X W_V^i)                    (10)

    S = (Q_bar_U1 K^T + Q_bar_U2 P^T) / sqrt(d_q),
        Q_U1 = Q + U_1,  Q_U2 = Q + U_2,  P = X_P W_P               (Alg. 1)

``U_1``/``U_2`` are Transformer-XL-style per-head bias vectors and ``W_P``
projects a sinusoidal relative-position encoding (arXiv:1901.02860), exactly
as in the paper's Algorithm 1. Query rows outside the top-``u`` set keep the
head-mean of ``V`` (the Informer reference convention), so the layer never
materialises their score rows.

Mode semantics
--------------
Encoder mode (the paper's setting) uses the full ProbSparse selection. Decoder
mode computes all query rows: the top-``u`` rule is a *block-global* decision,
and filling the discarded rows with a bidirectional statistic would leak
future tokens, so causal decoding attends densely (with the same ``U_1``/
``U_2``/``W_P`` terms) and caches K/V exactly like standard attention.

Reference:
    Wu, X. (2022). "Deep Sparse Conformer for Speech Recognition."
    Proc. InterSpeech 2022. arXiv:2209.00260.
    Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W.
    (2021). "Informer: Beyond Efficient Transformer for Long Sequence
    Time-Series Forecasting." AAAI 2021. arXiv:2012.07436.
"""

from __future__ import annotations

from ..params import mixer_params, use_pe_for

from typing import Optional

import math

import torch
import torch.nn as nn
from ...numerics import module_softmax
from ...norm import attach_qk_norm, record_qk_max_logits

from ..common import BitLinear, apply_pe_to_qk


def _sinusoid(distances: torch.Tensor, dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sinusoidal relative-position encoding (Transformer-XL style).

    Args:
        distances (torch.Tensor): Relative distances of arbitrary shape.
        dim (int): Encoding width (one model hidden size ``H``).
        dtype (torch.dtype): Output dtype (defaults to ``float32``).

    Returns:
        torch.Tensor: ``(*distances.shape, dim)`` sinusoidal features; even
        indices use ``sin`` and odd indices ``cos``, truncated/padded to
        ``dim``.
    """
    half = (int(dim) + 1) // 2
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=distances.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = distances.to(torch.float32).unsqueeze(-1) * freq
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb[..., : int(dim)].to(dtype)


class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention with query-side top-u selection.

    Each head ranks its queries with the max-mean sparsity measure of Eq. 8,
    keeps the ``u = c2 * ceil(ln L)`` most dominant ones, and scores only those
    against every key. The ranking probe itself only reads a random subset of
    ``c1 * ceil(ln L)`` keys, giving ``O(L log L)`` cost.

    Attributes:
        hidden_size (int): Model hidden dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension (``hidden_size / num_heads``).
        scale (float): Score scaling ``1 / sqrt(head_dim)``.
        c1 (float): Key-sampling factor for the sparsity probe (paper: 5.0).
        c2 (float): Query-selection factor for the dominant-query count
            (paper: 5.0).
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        p_proj (nn.Module): Relative-position projection ``W_P``.
        u1 (nn.Parameter): Per-head query bias ``U_1``, shape
            ``(1, num_heads, 1, head_dim)``.
        u2 (nn.Parameter): Per-head query bias ``U_2``, same shape.
        dropout (nn.Dropout): Dropout on the attention weights.
        mode (str): ``"encoder"`` (ProbSparse selection) or ``"decoder"``
            (dense causal attention with the same relative-position terms).

    Reference:
        Wu, X. (2022). "Deep Sparse Conformer for Speech Recognition."
        arXiv:2209.00260.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        """Initialize ProbSparseAttention.

        Args:
            config: Nested model configuration. Reads ``dims.hidden_size``,
                ``dims.num_heads``, ``dims.dropout``, ``dims.mode``,
                ``bitnet.enabled``, ``positional_encoding`` and the
                ``attention.probsparse`` knobs (``c1``, ``c2``).
            pos_encoder: Optional shared positional-encoding module.
            params: Pre-bound knob namespace; derived from the config when
                omitted.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        """
        params = mixer_params(config, "probsparse", params)
        super().__init__()
        self.hidden_size = config.dims.hidden_size
        self.num_heads = config.dims.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for ProbSparseAttention"
            )

        self.c1 = max(0.1, float(params.value("c1", 5.0)))
        self.c2 = max(0.1, float(params.value("c2", 5.0)))

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        # W_P in Algorithm 1: projects the relative-position encoding.
        self.p_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.u1 = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))
        self.u2 = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))

        self.dropout = nn.Dropout(config.dims.dropout)
        self.mode = getattr(config.dims, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "probsparse", True))
        attach_qk_norm(self, config, self.head_dim)

    # ------------------------------------------------------------------ #
    # Relative position (Algorithm 1: P = X_P W_P)
    # ------------------------------------------------------------------ #
    def _relative_scores(
        self,
        q_u2: torch.Tensor,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the relative-position score term ``Q_U2 P^T``.

        Builds the sinusoidal encoding of every relative distance occurring
        between the query and key positions, projects it through ``W_P`` and
        contracts it with the ``U_2``-shifted queries. Only the distance range
        actually present is materialised (``2S - 1`` values for a causal
        chunk, ``2L - 1`` for a full square), never the ``(L, L, d)`` tensor.

        Args:
            q_u2 (torch.Tensor): ``U_2``-shifted queries ``(B, H, S, D)``.
            q_positions (torch.Tensor): Absolute query positions, either
                shared ``(S,)`` or per-head ``(B, H, S)``.
            kv_positions (torch.Tensor): Absolute key positions ``(T,)``.

        Returns:
            torch.Tensor: Score term ``(B, H, S, T)`` (unscaled).
        """
        if q_positions.dim() == 1:
            q_positions = q_positions.view(1, 1, -1)
        d_min = int(q_positions.min().item()) - int(kv_positions.max().item())
        d_max = int(q_positions.max().item()) - int(kv_positions.min().item())
        offsets = torch.arange(d_min, d_max + 1, device=q_u2.device)
        rel = _sinusoid(offsets, self.hidden_size, self.p_proj.weight.dtype)
        projected = self.p_proj(rel)
        projected = projected.view(-1, self.num_heads, self.head_dim)
        projected = projected.permute(1, 0, 2)  # (H, R, D)

        scores = torch.einsum("bhsd,hrd->bhsr", q_u2, projected)
        index = (
            q_positions.unsqueeze(-1) - kv_positions.view(1, 1, 1, -1)
        ) - d_min
        index = index.expand(scores.size(0), self.num_heads, -1, -1)
        return scores.gather(-1, index)

    def _score_pe(
        self,
        scores: torch.Tensor,
        pe,
        seq_len: int,
        kv_len: int,
        selected_idx: Optional[torch.Tensor] = None,
        kv_positions: Optional[torch.Tensor] = None,
        q_offset: int = 0,
    ) -> torch.Tensor:
        """Apply additive PE bias / SSMax to (possibly row-selected) scores.

        Mirrors :func:`src.model.attention.common.apply_pe_to_scores` but can
        gather the rows of a row-subset (ProbSparse keeps only ``u`` queries).

        Args:
            scores (torch.Tensor): ``(B, H, S, kv_len)`` score tensor.
            pe: Positional-encoding module.
            seq_len (int): Number of modelled query positions.
            kv_len (int): Number of keys.
            selected_idx (Optional[torch.Tensor]): ``(B, H, S)`` indices of
                the kept queries when ``scores`` holds a subset of rows.
            kv_positions (Optional[torch.Tensor]): Cached absolute key
                positions (incremental decoding).
            q_offset (int): Absolute index of the first query.

        Returns:
            torch.Tensor: Scores with the PE terms applied.
        """
        if not self.use_pe or pe is None:
            return scores
        if self.pe_type in ("alibi", "bam"):
            bias = pe.bias(
                seq_len,
                kv_len,
                q_offset,
                kv_positions=kv_positions,
                device=scores.device,
                dtype=scores.dtype,
            )
            if selected_idx is not None:
                bias = bias.expand(scores.size(0), -1, -1, -1)
                gather_idx = selected_idx.unsqueeze(-1).expand(-1, -1, -1, kv_len)
                bias = bias.gather(2, gather_idx)
            scores = scores + bias
        ssmax = getattr(pe, "ssmax", None)
        if ssmax is not None:
            scores = scores * ssmax.scale(
                kv_len, device=scores.device, dtype=scores.dtype
            )
        return scores

    # ------------------------------------------------------------------ #
    # ProbSparse selection
    # ------------------------------------------------------------------ #
    def _sparsity_probe(
        self,
        q_u1: torch.Tensor,
        k: torch.Tensor,
        logical_layer_idx: Optional[int],
    ) -> torch.Tensor:
        """Rank queries with the max-mean measure of Eq. 8.

        Args:
            q_u1 (torch.Tensor): ``U_1``-shifted queries ``(B, H, S, D)``.
            k (torch.Tensor): Keys ``(B, H, T, D)``.
            logical_layer_idx (Optional[int]): Layer index used to seed the
                key subsample, keeping training and evaluation deterministic.

        Returns:
            torch.Tensor: Sparsity scores ``(B, H, S)`` (unscaled, as in the
            paper's Algorithm 1 which omits the ``sqrt(d)`` factor).
        """
        num_keys = k.size(-2)
        probe_len = min(
            num_keys, max(1, int(round(self.c1 * math.ceil(math.log(max(num_keys, 2))))))
        )
        if probe_len >= num_keys:
            k_part = k
        else:
            generator = torch.Generator(device=k.device)
            generator.manual_seed(int(logical_layer_idx or 0) * 100003 + num_keys)
            idx = torch.randperm(num_keys, device=k.device, generator=generator)[
                :probe_len
            ].sort().values
            k_part = k.index_select(-2, idx)
        sampled = q_u1 @ k_part.transpose(-2, -1)
        # Eq. 8: max_j(scores) - mean_j(scores), with the paper's 1/L_K
        # normaliser (L_K is the FULL key count, not the probe size).
        return sampled.max(dim=-1).values - sampled.sum(dim=-1) / float(num_keys)

    def _gather_rows(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Gather query rows ``index`` ``(B, H, U)`` from ``tensor``."""
        expanded = index.unsqueeze(-1).expand(-1, -1, -1, tensor.size(-1))
        return tensor.gather(2, expanded)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute ProbSparse attention.

        Encoder mode selects the dominant queries (paper Algorithm 1); decoder
        mode computes every row with a causal mask so incremental decoding is
        exact. When a KV slot is supplied, keys/values are appended and the
        whole cached prefix is attended (single-query decode makes the
        selection degenerate, so both modes reduce to dense-over-cache).

        Args:
            x (torch.Tensor): Input ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index (PE and
                probe seeding).
            pos_encoder: Optional positional-encoding module overriding
                ``self.pos_encoder``.
            cache: Optional KV slot for incremental decoding.
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            torch.Tensor: Output ``(batch, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            return self._forward_cached(
                x, q, k, v, slot, pe, logical_layer_idx, cache_position
            )

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        q_u1 = q + self.u1
        q_u2 = q + self.u2
        positions = torch.arange(seq_len, device=x.device)

        if self.mode == "decoder":
            scores = (q_u1 @ k.transpose(-2, -1)) * self.scale
            scores = scores + self._relative_scores(q_u2, positions, positions) * self.scale
            mask = positions[None, :] > positions[:, None]
            scores = scores.masked_fill(mask.view(1, 1, seq_len, seq_len), float("-inf"))
            scores = self._score_pe(scores, pe, seq_len, seq_len)
            record_qk_max_logits(self, scores)
            attn = module_softmax(self, scores, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.out_proj(out)

        # ---- Encoder: ProbSparse dominant-query selection (Algorithm 1) ----
        sparsity = self._sparsity_probe(q_u1, k, logical_layer_idx)
        num_queries = min(
            seq_len,
            max(1, int(round(self.c2 * math.ceil(math.log(max(seq_len, 2)))))),
        )
        _, top_idx = sparsity.topk(num_queries, dim=-1)  # (B, H, U)

        q_u1_sel = self._gather_rows(q_u1, top_idx)
        q_u2_sel = self._gather_rows(q_u2, top_idx)

        scores = (q_u1_sel @ k.transpose(-2, -1)) * self.scale
        scores = scores + self._relative_scores(q_u2_sel, positions[top_idx], positions) * self.scale
        scores = self._score_pe(
            scores, pe, seq_len, seq_len, selected_idx=top_idx
        )
        record_qk_max_logits(self, scores)
        attn = module_softmax(self, scores, dim=-1)
        attn = self.dropout(attn)
        selected = (attn @ v).transpose(1, 2)  # (B, U, H, D)

        # Informer convention: rows outside the top-u set keep the head-mean
        # of V (Algorithm 1 line 21 replaces only the selected rows).
        context = v.mean(dim=-2, keepdim=True).transpose(1, 2)
        context = context.expand(bsz, seq_len, self.num_heads, self.head_dim).clone()
        index = top_idx.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        context.scatter_(1, index, selected)
        return self.out_proj(context.contiguous().view(bsz, seq_len, hidden))

    def _forward_cached(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slot,
        pe,
        logical_layer_idx: Optional[int],
        cache_position: Optional[int],
    ) -> torch.Tensor:
        """Incremental (KV-cached) attention branch.

        The cache is a plain per-token K/V slot: single-query decode makes the
        dominant-query selection degenerate, so attending the full cached
        prefix is exact (class (i) of the cache compatibility catalogue).

        Args:
            x (torch.Tensor): Current hidden states ``(B, S, H)``.
            q (torch.Tensor): Queries ``(B, H_n, S, D)``.
            k (torch.Tensor): Keys ``(B, H_n, S, D)``.
            v (torch.Tensor): Values ``(B, H_n, S, D)``.
            slot: ``KVSlot``/``RingSlot`` returned by the cache manager.
            pe: Positional-encoding module.
            logical_layer_idx (Optional[int]): Logical layer index.
            cache_position (Optional[int]): Absolute index of the first token.

        Returns:
            torch.Tensor: Output ``(B, S, H)``.
        """
        from ...cache.hooks import (
            causal_mask_from_positions,
            kv_decode_append,
            positions_for,
        )

        bsz, seq_len, hidden = x.shape
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
        K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)

        q_u1 = q + self.u1
        q_u2 = q + self.u2
        scores = (q_u1 @ K.transpose(-2, -1)) * self.scale
        scores = scores + self._relative_scores(q_u2, positions, kv_positions) * self.scale
        scores = self._score_pe(
            scores,
            pe,
            seq_len,
            K.size(-2),
            kv_positions=kv_positions,
            q_offset=0 if offset is None else int(offset),
        )
        mask = causal_mask_from_positions(positions, kv_positions)
        scores = scores.masked_fill(mask.view(1, 1, seq_len, K.size(-2)), float("-inf"))

        record_qk_max_logits(self, scores)
        attn = module_softmax(self, scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
