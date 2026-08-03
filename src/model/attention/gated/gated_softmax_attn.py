"""Gated Softmax Attention.

Implements the Gated Softmax Attention mechanism from Qiu et al. (2025),
arXiv:2505.06708 (NeurIPS 2025 Best Paper). Gated Softmax Attention
applies a learned sigmoid gate after the standard scaled dot-product
attention (SDPA) output to introduce multiplicative non-linearity and
attenuate low-utility attention channels. The computation is:

    Y' = SDPA(Q, K, V) ⊙ σ(X W_g)

where σ is the sigmoid function and W_g is a learned gate projection.
This simple modification eliminates the attention sink phenomenon
observed in standard Transformers, where early tokens receive
disproportionate attention weights regardless of relevance.

Reference:
    Qiu, Z., Zeng, A., Liu, X., Sun, M., Shen, Y., & Yang, S. (2025).
    Gated Attention for Large Language Models: Non-linearity is
    All You Need? arXiv:2505.06708. NeurIPS 2025 Best Paper.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GatedSoftmaxAttention(nn.Module):
    """Gated softmax attention with post-SDPA sigmoid gate.

    Computes standard scaled dot-product attention and then applies a
    learned sigmoid gate element-wise to the SDPA output. The gate
    provides multiplicative non-linearity that attenuates low-utility
    channels and eliminates the attention sink effect. In decoder mode,
    a causal mask is applied before softmax.

    Args:
        config: Model configuration object with the following relevant
            attributes:
            hidden_size (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads. Must divide
                hidden_size evenly.
            dropout (float): Dropout probability applied to attention
                weights.
            use_bitnet (bool): If True, uses BitLinear for Q/K/V/gate/O
                projections instead of nn.Linear.
            mode (str, optional): ``"encoder"`` or ``"decoder"``.
                Defaults to ``"encoder"``. In decoder mode, a causal
                mask is applied.

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
        gate_proj (nn.Module): Gate projection for post-SDPA sigmoid
            modulation.
        out_proj (nn.Module): Output projection.
        dropout (nn.Dropout): Dropout layer applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Qiu, Z., Zeng, A., Liu, X., Sun, M., Shen, Y., & Yang, S. (2025).
        Gated Attention for Large Language Models: Non-linearity is
        All You Need? arXiv:2505.06708. NeurIPS 2025 Best Paper.
    """

    def __init__(self, config):
        """Initialize GatedSoftmaxAttention.

        Args:
            config: Model configuration object. See class docstring for
                required attributes.

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
            raise ValueError("hidden_size must be divisible by num_heads for GatedSoftmaxAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.gate_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute gated softmax attention over the input sequence.

        Computes standard scaled dot-product attention, then applies a
        learned sigmoid gate element-wise to the SDPA output. In decoder
        mode, a causal mask is applied before softmax.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface
                compatibility with other attention mixers.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        sdpa_out = (attn @ v).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.total_dim)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(sdpa_out * gate)
