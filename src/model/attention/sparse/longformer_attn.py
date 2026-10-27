"""Longformer sparse attention (Beltagy et al., 2020; arXiv:2004.05150).

Implements a sliding-window attention mechanism with optional global tokens
that scales linearly with sequence length. Each token attends to a fixed-size
local window of neighboring tokens, while designated global tokens attend to
and are attended by all positions. This yields O(n * w) complexity where w is
the window size, making it practical for documents with thousands of tokens.

Reference:
    Beltagy, I., Peters, M. E., & Cohan, A. (2020).
    "Longformer: The Long-Document Transformer."
    arXiv:2004.05150.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...numerics import module_softmax

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


class LongformerAttention(nn.Module):
    """Sliding-window attention with global tokens for linear-complexity long sequences.

    Each query token attends to a symmetric local window of ``window_size``
    tokens centered on its position. Additionally, a set of global tokens
    (specified by index) attend to all positions and are attended by all
    positions, enabling task-specific information routing (e.g., CLS token
    for classification). In decoder mode, the window is further restricted
    to past positions only via a causal mask.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        window_size (int): Size of the local attention window (total span).
        global_tokens (list): Indices of tokens that attend globally.
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking on top of the window pattern.

    Reference:
        Beltagy, I., Peters, M. E., & Cohan, A. (2020).
        "Longformer: The Long-Document Transformer."
        arXiv:2004.05150.
    """

    def __init__(self, config, pos_encoder=None):
        """Initialize LongformerAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                longformer_window_size (int, optional): Total span of the
                    local attention window. Defaults to 256.
                longformer_global_tokens (list, optional): Indices of global
                    tokens. Defaults to ``[0]`` (e.g., CLS token).
                mode (str, optional): ``"encoder"`` or ``"decoder"``.
                    Defaults to ``"encoder"``.
            pos_encoder: Optional shared positional encoding module.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for LongformerAttention")

        self.window_size = max(1, int(getattr(config, "longformer_window_size", 256)))
        global_tokens = getattr(config, "longformer_global_tokens", [0])
        self.global_tokens = global_tokens if isinstance(global_tokens, list) else [0]

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "longformer_attn_use_pe", True))

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the sliding-window plus global-token attention mask.

        For each query position ``i``, the mask allows attention to a symmetric
        window of ``window_size // 2`` positions on each side. Global tokens
        additionally attend to all positions and are attended by all positions.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device on which to create the mask tensor.

        Returns:
            torch.Tensor: Boolean mask of shape ``(seq_len, seq_len)`` where
                ``True`` indicates allowed attention.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        half_w = self.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            mask[i, start:end] = True

        for g in self.global_tokens:
            if 0 <= int(g) < seq_len:
                mask[int(g), :] = True
                mask[:, int(g)] = True

        return mask

    def make_cache_slot(self, meta, cache):
        """Unbounded KV slot: global tokens may sit anywhere, so eviction
        cannot be applied safely (pattern-mask over ``positions`` instead)."""
        from ...cache.slots import KVSlot

        kwargs = cache.codec_slot_kwargs()
        return KVSlot(meta, codec=kwargs["codec"], policy=None, num_sinks=kwargs["num_sinks"])

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute sliding-window attention with global tokens.

        Projects queries, keys, and values, then applies the combined
        window + global-token mask. In decoder mode, the mask is further
        intersected with a causal (lower-triangular) mask.

        Cached decoding evaluates the window/globals pattern against the
        cached absolute ``positions`` (B5 fix), which stays exact under
        any position set.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index. Not used.
            pos_encoder: Optional positional encoding module override.
            cache: Optional KV slot for incremental decoding (``None`` runs
                the legacy full-recompute path, D4).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            torch.Tensor: Output tensor of shape
                ``(batch, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            import torch.nn.functional as _F
            from ...cache.hooks import causal_mask_from_positions, kv_decode_append, pattern_mask, positions_for

            offset = cache_position
            positions = positions_for(offset, seq_len, x.device)
            q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe, position_offset=offset)
            K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)
            scores = (q @ K.transpose(-2, -1)) * self.scale
            half_w = self.window_size // 2
            globals_t = torch.tensor(
                [g for g in self.global_tokens if g >= 0], dtype=torch.long, device=x.device
            )

            def _pattern(q_pos, kv_pos):
                q_col = q_pos.reshape(-1, 1)
                kv_row = kv_pos.reshape(1, -1)
                window = (q_col - kv_row).abs() <= half_w
                is_global_kv = (
                    (kv_row == globals_t.reshape(1, -1)).any(dim=1)
                    if globals_t.numel()
                    else torch.zeros_like(kv_row[0])
                )
                is_global_q = (
                    (q_col == globals_t.reshape(1, -1)).any(dim=1)
                    if globals_t.numel()
                    else torch.zeros_like(q_col[0])
                )
                return window | is_global_kv.unsqueeze(0) | is_global_q.unsqueeze(1)

            if seq_len == 1:
                # Decode: cached keys are all causal; allowed = window ∪
                # globals ∪ (query-is-global ⇒ everything causal).
                p = kv_positions.new_full((1,), int(kv_positions[-1]))
                allowed = _pattern(p, kv_positions)
                scores = scores.masked_fill((~allowed).unsqueeze(0).unsqueeze(0), float("-inf"))
            else:
                mask = pattern_mask(positions, kv_positions, _pattern)
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            scores = apply_pe_to_scores(
                pe, self.pe_type, scores, q, self.use_pe,
                kv_len=K.size(-2),
                q_offset=0 if offset is None else int(offset),
                kv_positions=kv_positions,
            )
            attn = module_softmax(self, scores, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ V).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.out_proj(out)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._build_mask(seq_len, x.device)
        if self.mode == "decoder":
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = mask & causal
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        scores = apply_pe_to_scores(pe, self.pe_type, scores, q, self.use_pe)
        attn = module_softmax(self, scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
