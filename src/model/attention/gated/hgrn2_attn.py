"""HGRN2 Attention.

Implements the HGRN2 (Hierarchically Gated Recurrent Network v2) attention
mechanism from Qin et al. (2024), arXiv:2404.07904 (COLM 2024). HGRN2
uses outer-product state expansion with hierarchically lower-bounded
forget gates to combine recurrent memory efficiency with richer
matrix-valued state updates. The state update follows:

    S_t = diag(g_t) · S_{t-1} + v_t k_t^T

where g_t ∈ [lower_bound, 1]^d is a per-channel forget gate with a
configurable lower bound that prevents complete memory erasure.

Reference:
    Qin, Z., Yang, S., Zhong, Y., Shen, Y., & Sun, M. (2024).
    HGRN2: Gated Linear RNNs with State Expansion.
    arXiv:2404.07904. COLM 2024.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class HGRN2Attention(nn.Module):
    """HGRN2 attention with lower-bounded forget gates.

    Maintains a per-head matrix-valued recurrent state S_t ∈ R^{d×d}.
    At each timestep, a per-channel forget gate g_t (sigmoid-activated
    and lower-bounded) scales the previous state element-wise before
    adding the outer product v_t k_t^T. The lower bound prevents
    complete memory erasure, enabling hierarchical memory retention.
    A silu-gated output projection provides channel-wise modulation
    after the recurrent readout.

    Args:
        config: Model configuration object with the following relevant
            attributes:
            hidden_size (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads. Must divide
                hidden_size evenly.
            dropout (float): Dropout probability applied after the
                output gate.
            use_bitnet (bool): If True, uses BitLinear for Q/K/V/G/O
                projections instead of nn.Linear.
            hgrn2_lower_bound (float, optional): Minimum value for the
                forget gate. Defaults to 0.0.
            mode (str, optional): ``"encoder"`` or ``"decoder"``.
                Defaults to ``"encoder"``.

    Attributes:
        hidden_size (int): Input embedding dimensionality.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality per head (hidden_size // num_heads).
        total_dim (int): Total Q/K/V dimensionality (head_dim * num_heads).
        lower_bound (float): Minimum forget gate value.
        q_proj (nn.Module): Query projection.
        k_proj (nn.Module): Key projection.
        v_proj (nn.Module): Value projection.
        forget_proj (nn.Linear): Per-channel forget gate projection.
        g_proj (nn.Module): Output gate projection (silu-gated).
        out_proj (nn.Module): Output projection.
        norm (nn.LayerNorm): Layer normalization applied to the
            recurrent readout before the output gate.
        dropout (nn.Dropout): Dropout layer.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Qin, Z., Yang, S., Zhong, Y., Shen, Y., & Sun, M. (2024).
        HGRN2: Gated Linear RNNs with State Expansion.
        arXiv:2404.07904. COLM 2024.
    """

    def __init__(self, config):
        """Initialize HGRN2Attention.

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

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for HGRN2Attention")

        self.lower_bound = float(getattr(config, "hgrn2_lower_bound", 0.0))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.forget_proj = proj_cls(self.hidden_size, self.total_dim, bias=True)
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute HGRN2 attention over the input sequence.

        Processes the sequence token-by-token with a lower-bounded
        forget-gated recurrent matrix state. At each step t, the
        per-channel forget gate g_t (clamped to [lower_bound, 1])
        scales the previous state before adding v_t k_t^T.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface
                compatibility with other attention mixers.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        fg = torch.sigmoid(self.forget_proj(x)).view(bsz, seq_len, self.num_heads, self.head_dim)
        fg = self.lower_bound + (1 - self.lower_bound) * fg

        state = torch.zeros(
            bsz,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []
        for t in range(seq_len):
            gate = fg[:, t]
            state = state * gate.unsqueeze(-1) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
            out_t = (state * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)
