"""Multi-Token Attention (MTA).

Implements Multi-Token Attention, arXiv:2504.00927 (Golovneva, Wang,
Weston, Sukhbaatar; COLM 2025). Standard soft attention computes every
weight from the dot product of a *single* query and a *single* key vector,
which forces all the information needed to locate a relevant context region
into one vector. MTA keeps the quadratic attention skeleton but lets
neighbouring queries, keys and heads interact through small learned
convolutions applied to the attention score map.

Forward equations (paper numbering):

    K = H W_k,  V = H W_v,  Q = H W_q                               (1)
    A_hat = Q K^T / sqrt(d),  A = Softmax(Mask_{-inf}(A_hat))       (1)
    A = Softmax(Conv2d_theta(A_hat))                               (3)
    a_ij = Softmax( sum_{i'=0}^{cq-1} sum_{j'=-floor(ck/2)}^{ceil(ck/2)-1}
                    1_{i >= j - j'} theta_{i',j'} q_{i-i'} k_{j-j'}^T / sqrt(d) )  (4)
    A = Softmax(Mask_{-inf}(Conv2d_theta(Mask_0(A_hat))))          (5)  [default]
    A = Mask_0(Conv2d_theta(Softmax(Mask_{-inf}(A_hat))))          (6)  [post-softmax]
    A^1_new = w_11 A^1 + w_12 A^2,  A^2_new = w_21 A^1 + w_22 A^2  (7)  [head mixing]
    o = gated_group_norm(A V),  scale_h = sigmoid(g_h)             (3.3)
    out = concat_h(o) W_o

The default configuration applies the key-query convolution both before and
after the softmax and the head-mixing convolution after the softmax, as in
the paper's 880M reference run. Kernels are initialised to the identity
(paper section 4.6: identity init -- a plain Transformer -- converges best).
``ch`` is clamped to the largest divisor of ``num_heads`` not exceeding the
requested group size, so tiny presets never fail to build; ``ch == 1``
disables head mixing.

Decoder mode uses the paper's default double causal masking (Eq. 5): the
future is zeroed before the convolution (``Mask_0``) and set to -inf after it
(``Mask_{-inf}``). The encoder (bidirectional) mode simply drops both masks.

The gated group normalization normalises the attention output across each
group of heads (``num_heads / ch`` groups) and applies a per-head sigmoid
gate before the output projection, following the placement of Diff
Transformer's normalisation that MTA cites (section 3.3); the paper's Figure 1
draws the same block between the last MTA convolution and the value matmul.

Incremental decoding keeps the classic K/V cache plus a FIFO tail of the last
``cq - 1`` query vectors (stored as per-token slot ``extras`` under ``"q"``,
token axis last). Each decode step rebuilds the ``cq``-row score window
against the cached keys, so the double-masked convolution of Eq. 5 is
reproduced exactly -- full-forward and prefill+decode outputs agree.

Complexity:
    Training: O(n^2 * d * nH + n^2 * cq * ck * nH).
    Inference: O(n * d * nH) per step plus an O(cq * ck * nH) conv correction.

Reference:
    Golovneva, O., Wang, T., Weston, J., & Sukhbaatar, S. (2025).
    "Multi-Token Attention". COLM 2025. arXiv:2504.00927.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear, apply_pe_to_qk, apply_pe_to_scores
from .params import mixer_params, use_pe_for
from ..numerics import module_softmax
from ..norm import attach_qk_norm, record_qk_max_logits


def _largest_divisor(n: int, limit: int) -> int:
    """Largest divisor of ``n`` that does not exceed ``limit``."""
    limit = max(1, min(int(limit), int(n)))
    for candidate in range(limit, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


class MultiTokenAttention(nn.Module):
    """Multi-Token Attention (MTA, arXiv:2504.00927).

    Projects the input into queries, keys and values, then applies a causal
    key-query convolution over the attention score map (Eq. 5 pre-softmax and
    Eq. 6 post-softmax) and a head-mixing convolution over groups of heads
    (Eq. 7, post-softmax). The resulting weights are multiplied by the values
    and passed through a grouped normalisation with per-head sigmoid gates
    before the output projection.

    The mixer is a drop-in replacement for standard soft attention: with
    identity kernels (the default initialisation) and ``gated_group_norm``
    disabled it reproduces standard multi-head attention exactly.

    Args:
        config: Model configuration with ``dims`` (``hidden_size``,
            ``num_heads``, ``dropout``, optional ``mode``), ``bitnet`` and the
            nested ``attention`` knobs. Optional ``positional_encoding`` /
            ``positional_encoding_parameters`` decide the ``use_pe`` default.
        pos_encoder: Optional shared positional-encoding module.
        params: Optional pre-bound :class:`ComponentParams` for ``mta``.

    Attributes:
        hidden_size: Dimensionality of the input and output embeddings.
        num_heads: Number of parallel attention heads.
        head_dim: Dimensionality of each attention head.
        cq: Query-axis kernel size (paper default 6).
        ck: Key-axis kernel size, forced odd (paper default 11).
        ch: Effective head-mixing group size (divisor of ``num_heads``).
        head_groups: ``num_heads // ch`` mixing groups.
        qk_pre: Pre-softmax key-query kernels ``(nH, cq, ck)``.
        qk_post: Post-softmax key-query kernels ``(nH, cq, ck)``.
        head_kernel: Head-mixing kernels ``(head_groups, ch, ch)``.
        norm: Grouped output normalisation (``None`` when disabled).
        gate: Per-head sigmoid-gate logits ``(nH,)``.
        mode: ``"encoder"`` (bidirectional) or ``"decoder"`` (causal).

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        params = mixer_params(config, "mta", params)
        super().__init__()
        dims = config.dims
        self.hidden_size = int(dims.hidden_size)
        self.num_heads = int(dims.num_heads)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for MultiTokenAttention")
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.cq = max(1, int(params.value("cq", 6)))
        ck = max(1, int(params.value("ck", 11)))
        self.ck = ck if ck % 2 == 1 else ck + 1
        self.ch = _largest_divisor(self.num_heads, int(params.value("ch", 16)))
        self.head_groups = self.num_heads // self.ch
        self.pre_softmax = bool(params.value("pre_softmax", True))
        self.post_softmax = bool(params.value("post_softmax", True))
        self.gated_group_norm = bool(params.value("gated_group_norm", True))

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        # Eq. (5)/(6): one cq x ck kernel per head, identity-initialised so the
        # mixer starts as a plain Transformer (paper section 4.6).
        self.qk_pre = nn.Parameter(torch.zeros(self.num_heads, self.cq, self.ck))
        self.qk_post = nn.Parameter(torch.zeros(self.num_heads, self.cq, self.ck))
        with torch.no_grad():
            self.qk_pre[:, -1, self.ck // 2] = 1.0
            self.qk_post[:, -1, self.ck // 2] = 1.0

        # Eq. (7): non-overlapping head groups, identity-initialised.
        if self.head_groups > 1:
            self.head_kernel = nn.Parameter(
                torch.eye(self.ch).unsqueeze(0).repeat(self.head_groups, 1, 1)
            )
        else:
            self.head_kernel = None

        if self.gated_group_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.hidden_size))
            self.norm_bias = nn.Parameter(torch.zeros(self.hidden_size))
            self.gate = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.norm_weight = None
            self.norm_bias = None
            self.gate = None
        self.norm_eps = 1e-5

        self.dropout = nn.Dropout(dims.dropout)
        self.mode = getattr(dims, "mode", "encoder")
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "mta", True))
        self.use_fixed_softmax = False
        attach_qk_norm(self, config, self.head_dim)

    # ------------------------------------------------------------------ #
    # MTA convolutions
    # ------------------------------------------------------------------ #
    def _qk_conv(self, scores: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """2-D key-query convolution over ``(query, key)`` per head (Eq. 3-6).

        Args:
            scores: Attention scores or weights ``(B, nH, S, T)``.
            weight: Per-head kernels ``(nH, cq, ck)``.

        Returns:
            Convolved tensor with the same shape (shifted by ``cq-1`` in the
            query axis and ``ck//2`` in the key axis, i.e. a causal window).
        """
        heads = scores.size(1)
        w = weight.unsqueeze(1)
        padded = F.pad(scores, (self.ck // 2, self.ck // 2, self.cq - 1, 0))
        y = F.conv2d(padded, w, groups=heads)
        return y

    def _head_conv(self, weights: torch.Tensor) -> torch.Tensor:
        """Non-overlapping head-mixing convolution (Eq. 7).

        Args:
            weights: Attention weights ``(B, nH, S, T)``.

        Returns:
            Mixed weights with the same shape.
        """
        if self.head_kernel is None:
            return weights
        bsz, _, seq_len, kv_len = weights.shape
        grouped = weights.reshape(bsz, self.head_groups, self.ch, seq_len, kv_len)
        mixed = torch.einsum("gco,bgcst->bgost", self.head_kernel, grouped)
        return mixed.reshape(bsz, self.num_heads, seq_len, kv_len)

    def _gated_norm(self, out: torch.Tensor) -> torch.Tensor:
        """Group-normalise the per-head output and apply sigmoid gates.

        Args:
            out: Attended values ``(B, nH, S, head_dim)``.

        Returns:
            Gated tensor of the same shape.
        """
        if not self.gated_group_norm:
            return out
        bsz, _, seq_len, _ = out.shape
        grouped = out.permute(0, 1, 3, 2).reshape(
            bsz, self.head_groups, self.ch * self.head_dim, seq_len
        )
        mean = grouped.mean(dim=2, keepdim=True)
        var = grouped.var(dim=2, unbiased=False, keepdim=True)
        grouped = (grouped - mean) * torch.rsqrt(var + self.norm_eps)
        grouped = grouped.reshape(bsz, self.num_heads, self.head_dim, seq_len)
        grouped = grouped * self.norm_weight.view(self.num_heads, self.head_dim, 1)
        grouped = grouped + self.norm_bias.view(self.num_heads, self.head_dim, 1)
        grouped = grouped * torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        return grouped.permute(0, 1, 3, 2)

    def _mix(
        self,
        scores: torch.Tensor,
        future_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the MTA stack to one score block, returning the weights.

        Args:
            scores: Raw scores ``(B, nH, S, T)`` (PE bias already applied).
            future_mask: Boolean ``(S, T)`` mask, ``True`` = not visible, or
                ``None`` for bidirectional (encoder) semantics.

        Returns:
            Mixed attention weights ``(B, nH, S, T)``.
        """
        if future_mask is not None and self.pre_softmax:
            scores = scores.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), 0.0)
        if self.pre_softmax:
            scores = self._qk_conv(scores, self.qk_pre)
        if future_mask is not None:
            scores = scores.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = module_softmax(self, scores, dim=-1)
        # Zero-padded query rows (start of a decode window) are fully masked
        # and would otherwise produce NaN rows that the post-softmax conv
        # spreads into real rows.
        weights = torch.nan_to_num(weights, nan=0.0)
        if self.head_kernel is not None:
            weights = self._head_conv(weights)
        if self.post_softmax:
            weights = self._qk_conv(weights, self.qk_post)
            if future_mask is not None:
                weights = weights.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), 0.0)
        weights = self.dropout(weights)
        return weights

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
        """Compute Multi-Token Attention over the input sequence.

        Args:
            x: Input tensor ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index (layer-dependent PEs).
            pos_encoder: Optional PE module overriding ``self.pos_encoder``.
            cache: Optional ``KVSlot``/``RingSlot`` for incremental decoding;
                also carries the query tail under ``extras["q"]``.
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        idx = logical_layer_idx or 0

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ..cache.hooks import (
                causal_mask_from_positions,
                kv_decode_append,
                positions_for,
            )

            offset = int(cache_position or 0)
            history = self.cq - 1
            positions = positions_for(offset, seq_len, x.device)
            q, k = apply_pe_to_qk(
                pe, self.pe_type, q, k, x, idx, self.use_pe, position_offset=offset
            )
            # The query tail travels as a per-token extra (token axis last).
            K, V, kv_positions, extras = kv_decode_append(
                slot, k, v, positions, extras={"q": q.permute(0, 1, 3, 2)}
            )
            q_hist = extras.get("q")
            if q_hist is None:
                q_hist = q.permute(0, 1, 3, 2)
            total = q_hist.size(-1)
            start = max(0, total - seq_len - history)
            window = q_hist[..., start : total - seq_len].permute(0, 1, 3, 2)
            if window.size(2) < history:
                pad = window.new_zeros((bsz, self.num_heads, history - window.size(2), self.head_dim))
                window = torch.cat([pad, window], dim=2)
            q_window = torch.cat([window, q], dim=2)
            row_positions = torch.arange(
                offset + seq_len - q_window.size(2),
                offset + seq_len,
                device=x.device,
                dtype=torch.long,
            )
            scores = (q_window @ K.transpose(-2, -1)) * self.scale
            scores = apply_pe_to_scores(
                pe,
                self.pe_type,
                scores,
                q_window,
                self.use_pe,
                kv_len=K.size(-2),
                q_offset=int(row_positions[0].item()),
                kv_positions=kv_positions,
            )
            record_qk_max_logits(self, scores)
            future_mask = causal_mask_from_positions(row_positions, kv_positions)
            weights = self._mix(scores, future_mask)
            out = weights @ V
            out = self._gated_norm(out[:, :, -seq_len:, :])
            out = out.transpose(1, 2).reshape(bsz, seq_len, hidden)
            return self.out_proj(out)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, idx, self.use_pe)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = apply_pe_to_scores(pe, self.pe_type, scores, q, self.use_pe)
        record_qk_max_logits(self, scores)
        future_mask = None
        if self.mode == "decoder":
            future_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        weights = self._mix(scores, future_mask)
        out = weights @ v
        out = self._gated_norm(out)
        out = out.transpose(1, 2).reshape(bsz, seq_len, hidden)
        return self.out_proj(out)
