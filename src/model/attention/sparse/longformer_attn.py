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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


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

    def __init__(self, config):
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

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute sliding-window attention with global tokens.

        Projects queries, keys, and values, then applies the combined
        window + global-token mask. In decoder mode, the mask is further
        intersected with a causal (lower-triangular) mask.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index for
                potential layer-specific behavior. Not used by this
                implementation.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._build_mask(seq_len, x.device)
        if self.mode == "decoder":
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = mask & causal
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
