"""Forgetting Transformer (FoX) Attention.

Implements the Forgetting Transformer (FoX) attention mechanism from
Lin et al. (2025), arXiv:2503.02130 (ICLR 2025). FoX injects a learned
forget gate into the softmax logit space via cumulative log-bias terms,
controlling recency while preserving full softmax attention expressiveness.
The attention computation is:

    O = softmax(Q K^T + D) V

where D_ij = Σ_{l=j}^{i} log f_l is a cumulative log-forget bias matrix
constructed from per-head forget gates f_t ∈ (0, 1). This formulation is
FlashAttention-compatible, enabling efficient hardware-aware
implementations.

Reference:
    Lin, Z., Gou, M., Gong, Y., Liu, X., Shen, Y., Xu, R., Lin, C.,
    Yang, Y., Jiao, J., Duan, N., & Chen, W. (2025).
    Forgetting Transformer: Softmax Attention with a Forget Gate.
    arXiv:2503.02130. ICLR 2025.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...numerics import module_softmax

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


class ForgettingAttention(nn.Module):
    """Forgetting Transformer attention with log-space forget gates.

    Computes standard scaled dot-product attention with an additive
    forget bias D in the logit space. The bias is constructed from
    per-head forget gates f_t via cumulative log-sums, creating a
    lower-triangular matrix that progressively discounts past tokens.
    In decoder mode, an additional causal mask is applied.

    Args:
        config: Model configuration object with the following relevant
            attributes:
            hidden_size (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads. Must divide
                hidden_size evenly.
            dropout (float): Dropout probability applied to attention
                weights.
            use_bitnet (bool): If True, uses BitLinear for Q/K/V/O
                projections instead of nn.Linear.
            mode (str, optional): ``"encoder"`` or ``"decoder"``.
                Defaults to ``"encoder"``. In decoder mode, a causal
                mask is applied in addition to the forget bias.

    Attributes:
        hidden_size (int): Input embedding dimensionality.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality per head (hidden_size // num_heads).
        total_dim (int): Total Q/K/V dimensionality (head_dim * num_heads).
        scale (float): Scaling factor for dot-product attention
            (head_dim ** -0.5).
        q_proj (nn.Module): Query projection.
        k_proj (nn.Module): Key projection.
        v_proj (nn.Module): Value projection.
        f_proj (nn.Linear): Per-head forget gate projection.
        out_proj (nn.Module): Output projection.
        dropout (nn.Dropout): Dropout layer applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Lin, Z., Gou, M., Gong, Y., Liu, X., Shen, Y., Xu, R., Lin, C.,
        Yang, Y., Jiao, J., Duan, N., & Chen, W. (2025).
        Forgetting Transformer: Softmax Attention with a Forget Gate.
        arXiv:2503.02130. ICLR 2025.
    """

    def __init__(self, config, pos_encoder=None):
        """Initialize ForgettingAttention.

        Args:
            config: Model configuration object. See class docstring for
                required attributes.
            pos_encoder: Optional shared positional encoding module.

        Raises:
            ValueError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for ForgettingAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.f_proj = proj_cls(self.hidden_size, self.num_heads, bias=True)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "fox_attn_use_pe", False))

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Forgetting Transformer attention over the input sequence.

        Constructs a cumulative log-forget bias matrix D from per-head
        forget gates f_t, adds it to the scaled dot-product attention
        logits, and applies softmax. In decoder mode, a causal mask is
        applied on top of the forget bias.

        Cached decoding stores the per-token cumulative log-forget sum
        ``c_j`` as slot extras (plan §6.2): the bias ``D_ij = c_i - c_j``
        is pairwise-relative, so decoding is a row of ``D`` against the
        cached ``c_j`` column, with the running sum continued from the
        last cached token.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface
                compatibility with other attention mixers.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.
            cache: Optional KV slot for incremental decoding (``None`` runs
                the legacy full-recompute path, D4).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        f = torch.sigmoid(self.f_proj(x)).permute(0, 2, 1)
        log_f = torch.log(f + 1e-6)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ...cache.hooks import causal_mask_from_positions, kv_decode_append, positions_for
            from ..common import apply_pe_to_scores as _apply_scores

            offset = cache_position
            positions = positions_for(offset, seq_len, x.device)
            q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe, position_offset=offset)
            # Continue the running cumulative log-forget sum from the cache.
            cached = slot.read()
            c_prev = cached.get("extras", {}).get("cum_log_f")
            if c_prev is not None and c_prev.numel():
                c_last = c_prev[:, :, -1:]
            else:
                c_last = torch.zeros(bsz, self.num_heads, 1, device=x.device, dtype=log_f.dtype)
            cum_new = c_last + torch.cumsum(log_f, dim=-1)  # (B, H, S) query-side c_i
            K, V, kv_positions, extras = kv_decode_append(
                slot, k, v, positions, extras={"cum_log_f": cum_new}
            )
            c_j = extras["cum_log_f"]  # (B, H, T) every cached token
            bias = cum_new.unsqueeze(-1) - c_j.unsqueeze(-2)  # D_ij = c_i - c_j
            mask = causal_mask_from_positions(positions, kv_positions)
            bias = bias.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = (q @ K.transpose(-2, -1)) * self.scale + bias
            attn = _apply_scores(
                pe,
                self.pe_type,
                attn,
                q,
                self.use_pe,
                kv_len=K.size(-2),
                q_offset=0 if offset is None else int(offset),
                kv_positions=kv_positions,
            )
            attn = module_softmax(self, attn, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ V).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.total_dim)
            return self.out_proj(out)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        cum_log_f = torch.cumsum(log_f, dim=-1)

        bias = cum_log_f.unsqueeze(-1) - cum_log_f.unsqueeze(-2)
        if self.mode == "decoder":
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            bias = bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = (q @ k.transpose(-2, -1)) * self.scale + bias
        attn = apply_pe_to_scores(pe, self.pe_type, attn, q, self.use_pe)
        attn = module_softmax(self, attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.total_dim)
        return self.out_proj(out)
