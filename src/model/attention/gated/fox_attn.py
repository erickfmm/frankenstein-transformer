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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


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

    def __init__(self, config):
        """Initialize ForgettingAttention.

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
            raise ValueError("hidden_size must be divisible by num_heads for ForgettingAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.f_proj = proj_cls(self.hidden_size, self.num_heads, bias=True)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Forgetting Transformer attention over the input sequence.

        Constructs a cumulative log-forget bias matrix D from per-head
        forget gates f_t, adds it to the scaled dot-product attention
        logits, and applies softmax. In decoder mode, a causal mask is
        applied on top of the forget bias.

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

        f = torch.sigmoid(self.f_proj(x)).permute(0, 2, 1)
        log_f = torch.log(f + 1e-6)
        cum_log_f = torch.cumsum(log_f, dim=-1)

        bias = cum_log_f.unsqueeze(-1) - cum_log_f.unsqueeze(-2)
        if self.mode == "decoder":
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            bias = bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = (q @ k.transpose(-2, -1)) * self.scale + bias
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.total_dim)
        return self.out_proj(out)
