"""Fast Conformer limited-context attention (Rekesh et al., 2023; arXiv:2305.05084).

Implements the long-form attention mechanism of Fast Conformer: every query
attends to a fixed-size local window around it, augmented with a small number
of **global attention tokens pinned at the head of the sequence**. The global
tokens attend to every position and every position attends to them, so they
broadcast long-range context at ``O(n * (w + g))`` cost. The paper's §2.2
variant gives the global token its own query/key/value projections
(Longformer-style); the §3.4 long-form model, which the paper evaluates,
shares the local projections. This module implements both, selected by
``fastconformer.global_attn_separate``.

The mechanism follows the paper's prose and the overlapping-chunk global
attention it inherits from Longformer (arXiv:2004.05150); the reference
implementation here masks a dense score matrix (mathematically identical to
the paper's chunked kernel, which is an optimization and is out of scope).
The paper gives no explicit equations (ICASSP short paper); the formulation is
the exact encoding of its §2.2/§3.4 description and of the authors' NeMo
implementation (``multi_head_attention.py``, "sliding window local+global
attention from Longformer").

Reference:
    Rekesh, D., Koluguri, N. R., Kriman, S., Majumdar, S., Noroozi, V.,
    Huang, H., Hrinchuk, O., Puvvada, K., Kumar, A., Balam, J., & Ginsburg, B.
    (2023). "Fast Conformer with Linearly Scalable Attention for Efficient
    Speech Recognition." arXiv:2305.05084.

    Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The
    Long-Document Transformer." arXiv:2004.05150.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from ...numerics import module_softmax

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores
from ...norm import attach_qk_norm, record_qk_max_logits


class FastConformerAttention(nn.Module):
    """Limited-context attention with a head-pinned global token.

    Each query attends to a symmetric local window of ``window_size`` tokens
    (``window_size // 2`` on each side) plus the first ``global_tokens``
    positions of the sequence, spaced ``global_tokens_spacing`` apart. Those
    global positions attend to all positions and are attended by all
    positions. With ``global_attn_separate`` the global token uses its own
    query/key/value projections (paper §2.2, Longformer-style, ``+3 H^2``
    parameters); otherwise it shares the local projections (paper §3.4, the
    evaluated long-form model).

    Global positions are the **input tokens at the head of the sequence**
    (the paper's "addition of a global token" is the designation of those
    positions, not an extra learned embedding), so the mixer adds no
    parameters beyond the projections.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1/sqrt(d)).
        window_size (int): Total local span ``w``; per-side context is
            ``window_size // 2``. Paper §3.4 uses 128 per side (``w = 256``).
        global_tokens (int): Number of leading global attention tokens ``g``.
        global_tokens_spacing (int): Stride ``s`` between global positions
            ``{0, s, 2s, ..., (g-1)s}``.
        global_attn_separate (bool): Whether the global token uses separate
            query/key/value projections (paper §2.2).
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        global_q_proj (nn.Module | None): Global query projection (§2.2 only).
        global_k_proj (nn.Module | None): Global key projection (§2.2 only).
        global_v_proj (nn.Module | None): Global value projection (§2.2 only).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode intersects the
            window/global pattern with a causal mask.

    Reference:
        Rekesh, D., et al. (2023). "Fast Conformer with Linearly Scalable
        Attention for Efficient Speech Recognition." arXiv:2305.05084.
        Beltagy, I., et al. (2020). "Longformer: The Long-Document
        Transformer." arXiv:2004.05150.
    """

    def __init__(self, config, pos_encoder=None):
        """Initialize FastConformerAttention.

        Args:
            config: Configuration object with the following expected
                attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                fastconformer_window_size (int, optional): Total local span
                    ``w``. Defaults to 256 (the paper's 128 per side).
                fastconformer_global_tokens (int, optional): Number of leading
                    global attention tokens ``g``. Defaults to 1 (the paper's
                    "single global attention token").
                fastconformer_global_tokens_spacing (int, optional): Stride
                    ``s`` between global positions. Defaults to 1.
                fastconformer_global_attn_separate (bool, optional): Use
                    separate global q/k/v projections. Defaults to False (the
                    paper's §3.4 shared-projection model).
                mode (str, optional): ``"encoder"`` or ``"decoder"``.
                    Defaults to ``"encoder"``.
            pos_encoder: Optional shared positional encoding module.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``,
                or if a knob is out of range.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for FastConformerAttention"
            )

        self.window_size = int(getattr(config, "fastconformer_window_size", 256))
        self.global_tokens = int(getattr(config, "fastconformer_global_tokens", 1))
        self.global_tokens_spacing = int(
            getattr(config, "fastconformer_global_tokens_spacing", 1)
        )
        self.global_attn_separate = bool(
            getattr(config, "fastconformer_global_attn_separate", False)
        )
        if self.window_size < 2:
            raise ValueError(
                f"fastconformer_window_size must be >= 2, got {self.window_size}"
            )
        if self.global_tokens < 0:
            raise ValueError(
                f"fastconformer_global_tokens must be >= 0, got {self.global_tokens}"
            )
        if self.global_tokens_spacing < 1:
            raise ValueError(
                "fastconformer_global_tokens_spacing must be >= 1, got "
                f"{self.global_tokens_spacing}"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        if self.global_attn_separate:
            self.global_q_proj = proj_cls(
                self.hidden_size, self.hidden_size, bias=False
            )
            self.global_k_proj = proj_cls(
                self.hidden_size, self.hidden_size, bias=False
            )
            self.global_v_proj = proj_cls(
                self.hidden_size, self.hidden_size, bias=False
            )
        else:
            self.global_q_proj = None
            self.global_k_proj = None
            self.global_v_proj = None

        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "fastconformer_attn_use_pe", True))
        attach_qk_norm(self, config, self.head_dim)

    # ------------------------------------------------------------------
    # Pattern helpers
    # ------------------------------------------------------------------
    def _global_positions(self, total: int, device: torch.device) -> torch.Tensor:
        """Absolute indices of the global attention tokens within ``total``.

        The global positions are ``{0, s, 2s, ..., (g-1)s}`` clipped to the
        sequence length (NeMo: ``is_index_global_attn[:, :g*s:s]``).

        Args:
            total: Sequence length (or accumulated cache length).
            device: Device for the index tensor.

        Returns:
            int64 tensor of global positions (possibly empty).
        """
        if self.global_tokens <= 0 or total <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        idx = torch.arange(self.global_tokens, device=device) * self.global_tokens_spacing
        return idx[idx < total]

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the window + head-global attention mask.

        A pair ``(i, j)`` is allowed when ``j`` is inside the symmetric local
        window ``|i - j| <= window_size // 2``, when ``j`` is a global
        position (every query attends to the global tokens), or when ``i`` is
        a global position (the global token attends to every position). In
        decoder mode the pattern is intersected with the causal mask.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device for the mask tensor.

        Returns:
            Boolean mask ``(seq_len, seq_len)`` where ``True`` = allowed.
        """
        idx = torch.arange(seq_len, device=device)
        row = idx[:, None]
        col = idx[None, :]
        half = self.window_size // 2

        window = (row - col).abs() <= half
        globals_idx = self._global_positions(seq_len, device)
        if globals_idx.numel():
            is_global_col = (col == globals_idx.view(-1, 1)).any(dim=0)
            is_global_row = (row == globals_idx.view(1, -1)).any(dim=1)
            mask = window | is_global_col.unsqueeze(0) | is_global_row.unsqueeze(1)
        else:
            mask = window

        if self.mode == "decoder":
            mask = mask & (row >= col)
        return mask

    def _global_membership(
        self, positions: torch.Tensor, total: int
    ) -> torch.Tensor:
        """Boolean ``(T,)`` marker: is each absolute position a global token?"""
        globals_idx = self._global_positions(total, positions.device)
        if globals_idx.numel() == 0:
            return torch.zeros(positions.numel(), dtype=torch.bool, device=positions.device)
        return (positions.view(1, -1) == globals_idx.view(-1, 1)).any(dim=0)

    @staticmethod
    def _swap_global_rows(
        base: torch.Tensor, glob: torch.Tensor, is_global: torch.Tensor
    ) -> torch.Tensor:
        """Return ``base`` with the global positions replaced by ``glob``.

        Args:
            base: Shared-projection tensor ``(B, H, T, D)``.
            glob: Global-projection tensor ``(B, H, T, D)``.
            is_global: Boolean ``(T,)`` marker of global positions.

        Returns:
            Effective tensor with global rows taken from ``glob``.
        """
        if not bool(is_global.any()):
            return base
        out = base.clone()
        out[:, :, is_global, :] = glob[:, :, is_global, :]
        return out

    def make_cache_slot(self, meta, cache):
        """Bounded ring slot: pinned head region covering the global tokens.

        The ring retains the first ``(g-1)*s + 1`` positions (a superset of
        the global positions) plus the last ``window_size // 2 + 1`` tokens.
        Because the pinned prefix may include non-global positions when
        ``global_tokens_spacing > 1``, the cached branch always evaluates the
        exact pattern mask over absolute ``positions`` (like
        ``longformer_attn``), so retention only has to be a superset of the
        pattern.
        """
        from ...cache.slots import RingSlot

        num_sinks = 0
        if self.global_tokens > 0:
            num_sinks = (self.global_tokens - 1) * self.global_tokens_spacing + 1
        capacity = num_sinks + self.window_size // 2 + 1
        kwargs = cache.codec_slot_kwargs()
        return RingSlot(
            meta,
            capacity=capacity,
            num_sinks=num_sinks,
            **{k: v for k, v in kwargs.items() if k in ("codec", "policy")},
        )

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
        """Compute limited-context attention with a head-pinned global token.

        Projects queries, keys and values (plus the separate global
        projections when configured), applies the window + global-token
        pattern, and normalizes. In decoder mode the pattern is causal.

        Cached decoding stores post-RoPE K/V in a bounded ring whose pinned
        head region covers the global positions; the exact pattern is
        evaluated against the cached absolute ``positions`` (B5), so decode
        never attends outside the window/global set. Global **query** rows
        only occur in the first prefill chunk (global positions are at the
        head); with separate projections their row is computed from that
        chunk, which is exact for a from-position-0 prefill — the only case
        ``generate()`` produces. Chunked encoder-mode prefill with separate
        projections should use the legacy uncached path.

        Args:
            x (torch.Tensor): Input of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index for
                layer-dependent positional encoding.
            pos_encoder: Optional positional encoding module override.
            cache: Optional ring/KV slot for incremental decoding (``None``
                runs the legacy full-recompute path).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            torch.Tensor: Output of shape ``(batch, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, hidden = x.shape
        half = self.window_size // 2

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ...cache.hooks import kv_decode_append, pattern_mask, positions_for

            offset = cache_position
            positions = positions_for(offset, seq_len, x.device)
            q, k = apply_pe_to_qk(
                pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe,
                position_offset=offset,
            )

            qg = kg = vg = None
            if self.global_attn_separate:
                qg = self.global_q_proj(x).view(
                    bsz, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                kg = self.global_k_proj(x).view(
                    bsz, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                vg = self.global_v_proj(x).view(
                    bsz, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                if self.qk_norm is not None:
                    qg, kg = self.qk_norm(qg, kg)
                qg, kg = apply_pe_to_qk(
                    pe, self.pe_type, qg, kg, x, logical_layer_idx or 0, self.use_pe,
                    position_offset=offset,
                )

            total = int(positions[-1].item()) + 1 if positions.numel() else seq_len
            is_global = self._global_membership(positions, total)
            if self.global_attn_separate:
                k_eff = self._swap_global_rows(k, kg, is_global)
                v_eff = self._swap_global_rows(v, vg, is_global)
            else:
                k_eff, v_eff = k, v

            K, V, kv_positions, _ = kv_decode_append(slot, k_eff, v_eff, positions)
            scores = (q @ K.transpose(-2, -1)) * self.scale

            # Global query rows: the global token scores all positions with
            # its own (global) key projection (NeMo `_compute_out_global_to_all`).
            global_rows = None
            chunk_is_cache = seq_len == kv_positions.numel()
            if self.global_attn_separate and qg is not None and bool(is_global.any()):
                if chunk_is_cache:
                    global_rows = torch.nonzero(is_global, as_tuple=False).flatten()
                    grow = (qg[:, :, global_rows, :] @ kg.transpose(-2, -1)) * self.scale
                    scores = scores.clone()
                    scores[:, :, global_rows, :] = grow

            def _pattern(q_pos, kv_pos):
                window = (q_pos - kv_pos).abs() <= half
                gidx = self._global_positions(total, q_pos.device)
                if gidx.numel():
                    is_col = (kv_pos.view(1, -1) == gidx.view(-1, 1)).any(dim=0)
                    is_row = (q_pos.view(-1, 1) == gidx.view(1, -1)).any(dim=1)
                    return window | is_col.unsqueeze(0) | is_row.unsqueeze(1)
                return window

            mask = pattern_mask(positions, kv_positions, _pattern)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            scores = apply_pe_to_scores(
                pe, self.pe_type, scores, q, self.use_pe,
                kv_len=K.size(-2),
                q_offset=0 if offset is None else int(offset),
                kv_positions=kv_positions,
            )
            record_qk_max_logits(self, scores)
            attn = module_softmax(self, scores, dim=-1)
            attn = self.dropout(attn)
            out = attn @ V
            if global_rows is not None:
                out = out.clone()
                out[:, :, global_rows, :] = attn[:, :, global_rows, :] @ vg
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.out_proj(out)

        # ------------------------------------------------------------------
        # Legacy full-recompute path
        # ------------------------------------------------------------------
        q, k = apply_pe_to_qk(
            pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe
        )

        qg = kg = vg = None
        if self.global_attn_separate:
            qg = self.global_q_proj(x).view(
                bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            kg = self.global_k_proj(x).view(
                bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            vg = self.global_v_proj(x).view(
                bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            if self.qk_norm is not None:
                qg, kg = self.qk_norm(qg, kg)
            qg, kg = apply_pe_to_qk(
                pe, self.pe_type, qg, kg, x, logical_layer_idx or 0, self.use_pe
            )

        scores = (q @ k.transpose(-2, -1)) * self.scale
        globals_idx = self._global_positions(seq_len, x.device)

        if self.global_attn_separate:
            is_global = self._global_membership(
                torch.arange(seq_len, device=x.device), seq_len
            )
            k_eff = self._swap_global_rows(k, kg, is_global)
            v_eff = self._swap_global_rows(v, vg, is_global)
            scores = (q @ k_eff.transpose(-2, -1)) * self.scale
            global_rows = None
            if globals_idx.numel():
                global_rows = globals_idx
                grow = (qg[:, :, global_rows, :] @ kg.transpose(-2, -1)) * self.scale
                scores = scores.clone()
                scores[:, :, global_rows, :] = grow
        else:
            v_eff = v
            global_rows = None

        mask = self._build_mask(seq_len, x.device)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        scores = apply_pe_to_scores(pe, self.pe_type, scores, q, self.use_pe)
        record_qk_max_logits(self, scores)
        attn = module_softmax(self, scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v_eff
        if global_rows is not None:
            out = out.clone()
            out[:, :, global_rows, :] = attn[:, :, global_rows, :] @ vg
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
