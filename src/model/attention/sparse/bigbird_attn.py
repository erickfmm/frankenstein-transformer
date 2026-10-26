"""BigBird sparse attention (Zaheer et al., 2020; arXiv:2007.14062).

Implements a sparse attention mechanism that combines three complementary
patterns: local sliding-window attention, random attention links, and global
tokens. Together these patterns satisfy all theoretical properties needed for
Turing completeness of the transformer while achieving O(n) linear complexity.
The combination of local, random, and global attention ensures that information
can flow between any pair of tokens in O(log n) steps.

Reference:
    Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S.,
    Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020).
    "Big Bird: Transformers for Longer Sequences."
    arXiv:2007.14062.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


class BigBirdAttention(nn.Module):
    """Sparse attention with local window, random links, and global tokens.

    Each query token attends to three sets of keys:
    1. **Local window**: A symmetric window of ``window_size`` tokens centered
       on the query position.
    2. **Random links**: A fixed number of randomly selected tokens per query,
       providing long-range connectivity.
    3. **Global tokens**: The first ``num_global`` tokens attend to and are
       attended by all positions, serving as information hubs.

    The combined pattern achieves O(n) complexity while provably preserving
    the expressiveness of dense attention. In decoder mode, all patterns are
    intersected with a causal mask.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        window_size (int): Size of the local attention window (total span).
        num_random (int): Number of random attention links per query.
        num_global (int): Number of global tokens (first ``num_global`` positions).
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking on top of the sparse patterns.

    Reference:
        Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S.,
        Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020).
        "Big Bird: Transformers for Longer Sequences."
        arXiv:2007.14062.
    """

    def __init__(self, config, pos_encoder=None):
        """Initialize BigBirdAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                bigbird_window_size (int, optional): Total span of the local
                    attention window. Defaults to 128.
                bigbird_num_random (int, optional): Number of random attention
                    links per query. Defaults to 64.
                bigbird_num_global (int, optional): Number of global tokens.
                    Defaults to 2.
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
            raise ValueError("hidden_size must be divisible by num_heads for BigBirdAttention")

        self.window_size = max(1, int(getattr(config, "bigbird_window_size", 128)))
        self.num_random = max(1, int(getattr(config, "bigbird_num_random", 64)))
        self.num_global = max(1, int(getattr(config, "bigbird_num_global", 2)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "bigbird_attn_use_pe", True))
        self._bb_seed = 0

    def make_cache_slot(self, meta, cache):
        """Unbounded KV slot: random links may point anywhere causal, so no
        eviction is safe; the pattern is evaluated over ``positions``."""
        from ...cache.slots import KVSlot

        kwargs = cache.codec_slot_kwargs()
        return KVSlot(meta, codec=kwargs["codec"], policy=None, num_sinks=kwargs["num_sinks"])

    def _random_links(self, q_pos: int, limit: int, device: torch.device) -> torch.Tensor:
        """Deterministic per-position random links (B5/R2 fix).

        Links for query at ABSOLUTE position ``q_pos`` are drawn from
        ``[0, limit)`` with a generator seeded from ``(layer seed, q_pos)``
        so the full forward and every incremental decode step draw the
        SAME links for the same position (else cached/full parity fails).
        In decoder mode ``limit = q_pos + 1``: links beyond the query are
        causal-masked anyway, so the effective mask is unchanged while
        becoming reproducible.
        """
        gen = torch.Generator(device="cpu")
        seed = ((int(self._bb_seed) + 0x9E3779B9) * (int(q_pos) + 1)) & 0x7FFFFFFF
        gen.manual_seed(seed)
        return torch.randint(0, max(1, limit), (self.num_random,), generator=gen).to(device)

    def _build_mask(self, seq_len: int, device: torch.device, seed: int = 0, offset: int = 0) -> torch.Tensor:
        """Build the combined BigBird attention mask.

        Constructs a boolean mask that encodes all three BigBird attention
        patterns for the given sequence length:

        1. **Local window**: Each position ``i`` attends to a symmetric window
           of ``window_size // 2`` positions on each side.
        2. **Random links**: Each position ``i`` attends to ``num_random``
           randomly sampled positions — deterministic per (seed, absolute
           position) so incremental decoding reproduces them exactly (B5).
        3. **Global tokens**: The first ``num_global`` positions attend to all
           positions and are attended by all positions.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device on which to create the mask tensor.
            seed (int): Layer seed for the random links (logical layer idx).
            offset (int): Absolute index of the first row (cached chunks).

        Returns:
            torch.Tensor: Boolean mask of shape ``(seq_len, seq_len)`` where
                ``True`` indicates allowed attention.
        """
        self._bb_seed = seed
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        half_w = self.window_size // 2

        for i in range(seq_len):
            abs_i = offset + i
            start = max(0, abs_i - half_w)
            end = min(offset + seq_len, abs_i + half_w + 1)
            mask[i, max(0, start - offset): max(0, end - offset)] = True

        for i in range(seq_len):
            abs_i = offset + i
            limit = abs_i + 1 if self.mode == "decoder" else offset + seq_len
            rand_idx = self._random_links(abs_i, limit, device) - offset
            rand_idx = rand_idx[(rand_idx >= 0) & (rand_idx < seq_len)]
            if rand_idx.numel():
                mask[i, rand_idx] = True

        for g in range(min(self.num_global, offset + seq_len)):
            if g >= offset:
                mask[g - offset, :] = True
            if 0 <= g - offset < seq_len:
                mask[:, g - offset] = True

        return mask

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None, pos_encoder=None, cache=None, cache_position: Optional[int] = None) -> torch.Tensor:
        """Compute BigBird sparse attention.

        Projects queries, keys, and values, then applies the combined
        window + random + global mask. In decoder mode, the mask is further
        intersected with a causal (lower-triangular) mask.

        Random links are deterministic per (logical layer, absolute query
        position) (R2 mitigation), so cached decoding reproduces the full
        path's links exactly by evaluating the pattern over the cached
        ``positions`` metadata.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index (seeds the
                random links).
            pos_encoder: Optional positional encoding module override.
            cache: Optional KV slot for incremental decoding (``None`` runs
                the legacy full-recompute path, D4).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        seed = int(logical_layer_idx or 0)

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
            self._bb_seed = seed
            q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, seed, self.use_pe, position_offset=offset)
            K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)
            scores = (q @ K.transpose(-2, -1)) * self.scale
            half_w = self.window_size // 2

            def _pattern(q_pos, kv_pos):
                q_col = q_pos.reshape(-1, 1)
                kv_row = kv_pos.reshape(1, -1)
                window = (q_col - kv_row).abs() <= half_w
                is_global_kv = kv_row < self.num_global
                is_global_q = q_col < self.num_global
                random_ok = torch.zeros_like(window)
                for idx in range(q_col.shape[0]):
                    links = self._random_links(int(q_col[idx]), int(q_col[idx]) + 1, kv_row.device)
                    random_ok[idx] = (kv_row[0].unsqueeze(1) == links.unsqueeze(0)).any(dim=1)
                return window | is_global_kv | is_global_q | random_ok

            if seq_len == 1:
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
            attn = _F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ V).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.out_proj(out)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, seed, self.use_pe)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._build_mask(seq_len, x.device, seed=seed, offset=0)
        if self.mode == "decoder":
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = mask & causal
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        scores = apply_pe_to_scores(pe, self.pe_type, scores, q, self.use_pe)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
