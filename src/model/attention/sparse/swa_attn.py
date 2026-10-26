"""Sliding-window attention with attention sinks (SWA; arXiv:2608.28444).

Implements the simple, training-free attention-mask intervention studied by
Jolicoeur-Martineau et al. (2026): every query attends to the first
``num_sinks`` tokens (the *attention sinks*) plus the ``window_size -
num_sinks`` most recent tokens. The mechanism introduces no new parameters
and no new math in the query/key/value path --- it is a pure mask change that
can be trained from scratch or applied at inference time to a pretrained
model.

Why sinks matter: LLMs learn to dump excess attention mass onto the first few
tokens even when they are not semantically relevant (Xiao et al., 2024).
Without them in the mask, performance collapses catastrophically once the
first tokens leave the sliding window. Keeping ``num_sinks`` leading tokens
always visible restores most of the full-attention quality.

Reference:
    Jolicoeur-Martineau, A., Sukthanker, R. S., Cameron, P., & Gervais, E.
    (2026). "Sliding Window Beats Linear Attention." arXiv:2608.28444.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear, apply_pe_to_qk, apply_pe_to_scores


class SlidingWindowAttention(nn.Module):
    """Sliding-window attention with ``num_sinks`` leading attention sinks.

    Each query attends to at most ``window_size`` keys: the first
    ``num_sinks`` tokens (sinks) plus a local window of ``window_size -
    num_sinks`` tokens. In decoder mode the local window is causal (the most
    recent tokens up to and including the query); in encoder mode it is
    symmetric around the query, mirroring the Longformer convention.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores
            (``1 / sqrt(head_dim)``).
        window_size (int): Total number of attended tokens ``w`` (sinks plus
            sliding window). The paper evaluates ``w`` in ``{64, 128, 256,
            512}``.
        num_sinks (int): Number of leading sink tokens ``s`` that every query
            always attends to. The paper keeps ``s = 4`` fixed.
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode restricts the
            window and sinks to causal (past) positions.

    Reference:
        Jolicoeur-Martineau, A., Sukthanker, R. S., Cameron, P., & Gervais,
        E. (2026). "Sliding Window Beats Linear Attention." arXiv:2608.28444.
        Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). "Efficient
        Streaming Language Models with Attention Sinks." arXiv:2309.17453.
    """

    def __init__(self, config, pos_encoder=None):
        """Initialize SlidingWindowAttention.

        Args:
            config: Configuration object with the following expected
                attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                swa_window_size (int, optional): Total attended tokens ``w``.
                    Defaults to 256.
                swa_num_sinks (int, optional): Leading sink tokens ``s``.
                    Defaults to 4.
                mode (str, optional): ``"encoder"`` or ``"decoder"``.
                    Defaults to ``"encoder"``.
            pos_encoder: Optional shared positional encoding module.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``,
                or if the window/sink hyperparameters are inconsistent.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for SlidingWindowAttention"
            )

        self.window_size = int(getattr(config, "swa_window_size", 256))
        self.num_sinks = int(getattr(config, "swa_num_sinks", 4))
        if self.window_size < 2:
            raise ValueError(f"swa_window_size must be >= 2, got {self.window_size}")
        if self.num_sinks < 1:
            raise ValueError(f"swa_num_sinks must be >= 1, got {self.num_sinks}")
        if self.num_sinks >= self.window_size:
            raise ValueError(
                "swa_num_sinks must be smaller than swa_window_size, got "
                f"num_sinks={self.num_sinks}, window_size={self.window_size}"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "swa_attn_use_pe", True))

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the sliding-window plus attention-sink mask.

        Decoder mode mirrors the paper exactly: query ``i`` attends to the
        causal sinks ``j < num_sinks`` and the most recent
        ``window_size - num_sinks`` positions
        ``i - (w - s) + 1 <= j <= i``. Encoder mode extends the pattern to a
        symmetric local window ``|i - j| <= (w - s) // 2`` plus the sinks.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device on which to create the mask tensor.

        Returns:
            torch.Tensor: Boolean mask of shape ``(seq_len, seq_len)`` where
                ``True`` indicates allowed attention.
        """
        idx = torch.arange(seq_len, device=device)
        row = idx[:, None]
        col = idx[None, :]
        span = self.window_size - self.num_sinks

        sinks = col < self.num_sinks
        if self.mode == "decoder":
            causal = col <= row
            recent = causal & (col > row - span)
            # Sinks are causal-clipped: early queries cannot see future sinks.
            return (sinks & causal) | recent

        distance = (row - col).abs()
        return sinks | (distance <= span // 2)

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
    ) -> torch.Tensor:
        """Compute sliding-window attention with attention sinks.

        Projects queries, keys, and values, then applies the combined sink +
        sliding-window mask. Decoder mode builds the mask causally so no
        future token (including future sinks) is ever visible.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index for
                layer-specific positional encoding. Defaults to 0.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.

        Returns:
            torch.Tensor: Output tensor of shape
                ``(batch, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._build_mask(seq_len, x.device)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        scores = apply_pe_to_scores(pe, self.pe_type, scores, q, self.use_pe)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
