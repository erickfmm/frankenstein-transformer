"""Gated Linear Attention (GLA).

Implements the Gated Linear Attention mechanism from Yang et al. (2023),
arXiv:2312.06635. GLA introduces data-dependent diagonal gating on linear
attention, replacing purely additive accumulation with a controlled memory
retention scheme. The matrix-valued recurrent state evolves as:

    S_t = G_t ⊙ S_{t-1} + v_t k_t^T

where G_t is a per-head, per-channel gate derived from the input via a
low-rank projection followed by logsigmoid activation. This enables
sub-quadratic training complexity and O(d²) inference memory, making it
suitable for long-sequence modeling.

Reference:
    Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2023).
    Gated Linear Attention Transformers with Hardware-Efficient Training.
    arXiv:2312.06635.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GatedLinearAttention(nn.Module):
    """Gated Linear Attention with data-dependent diagonal gating.

    Maintains a per-head matrix-valued recurrent state S_t ∈ R^{d×d} that
    is updated at each timestep with a learned forget gate. The gate is
    produced by a low-rank projection (gk_proj) followed by logsigmoid,
    scaled by 1/16 for numerical stability. A silu-gated output projection
    (g_proj) provides additional channel-wise modulation after the recurrent
    readout.

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
            gla_gate_low_rank (int, optional): Rank of the low-rank
                bottleneck in the gate projection. Defaults to 16.
            mode (str, optional): ``"encoder"`` or ``"decoder"``.
                Defaults to ``"encoder"``.

    Attributes:
        hidden_size (int): Input embedding dimensionality.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality per head (hidden_size // num_heads).
        total_dim (int): Total Q/K/V dimensionality (head_dim * num_heads).
        q_proj (nn.Module): Query projection.
        k_proj (nn.Module): Key projection.
        v_proj (nn.Module): Value projection.
        g_proj (nn.Module): Output gate projection (silu-gated).
        gk_proj (nn.Sequential): Low-rank gate projection producing
            per-head, per-channel forget logits.
        out_proj (nn.Module): Output projection.
        norm (nn.LayerNorm): Layer normalization applied to the
            recurrent readout before the output gate.
        dropout (nn.Dropout): Dropout layer.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2023).
        Gated Linear Attention Transformers with Hardware-Efficient Training.
        arXiv:2312.06635.
    """

    def __init__(self, config):
        """Initialize GatedLinearAttention.

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
            raise ValueError("hidden_size must be divisible by num_heads for GatedLinearAttention")

        gate_low_rank = max(1, int(getattr(config, "gla_gate_low_rank", 16)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.gk_proj = nn.Sequential(
            proj_cls(self.hidden_size, gate_low_rank, bias=False),
            proj_cls(gate_low_rank, self.total_dim, bias=True),
        )
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Gated Linear Attention over the input sequence.

        Processes the sequence token-by-token with a recurrent matrix
        state. At each step t, the state is gated element-wise by
        exp(gk_t) before adding the outer product v_t k_t^T. The query
        reads from the state via inner product, and the result is
        modulated by a silu-gated output projection.

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
        gk = F.logsigmoid(self.gk_proj(x)).view(bsz, seq_len, self.num_heads, self.head_dim) / 16.0

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
            gate = torch.exp(gk[:, t])
            state = state * gate.unsqueeze(-1) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
            out_t = (state * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)
