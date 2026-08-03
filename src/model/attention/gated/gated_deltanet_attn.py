"""Gated DeltaNet Attention.

Implements the Gated DeltaNet attention mechanism from Yang et al. (2024),
arXiv:2412.06464 (ICLR 2025). Gated DeltaNet combines a decay gate α_t
with a write gate β_t to jointly control memory erasure and targeted
updates in a recurrent linear-attention state. The state update follows:

    S_t = α_t S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T

where α_t ∈ (0, 1) is a per-head decay factor and β_t ∈ (0, 1) is a
per-head write strength, both learned from the input. This formulation
has been integrated into Qwen3-Next, demonstrating strong performance
at scale.

Reference:
    Yang, S., Zhang, Y., Shen, Y., & Kim, Y. (2024).
    Gated Delta Networks: Improving Mamba2 and DeltaNet with
    Gating. arXiv:2412.06464. ICLR 2025.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GatedDeltaNetAttention(nn.Module):
    """Gated DeltaNet attention with decay and write gating.

    Maintains a per-head matrix-valued recurrent state S_t ∈ R^{d×d}.
    At each timestep, a decay gate α_t scales the previous state before
    the delta-rule update, and a write gate β_t controls the strength of
    both the removal term and the new key-value association. Queries and
    keys are L2-normalized; values are activated with silu. A silu-gated
    output projection provides additional channel-wise modulation.

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
            mode (str, optional): ``"encoder"`` or ``"decoder"``.
                Defaults to ``"encoder"``.

    Attributes:
        hidden_size (int): Input embedding dimensionality.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality per head (hidden_size // num_heads).
        total_dim (int): Total Q/K/V dimensionality (head_dim * num_heads).
        q_proj (nn.Module): Query projection.
        k_proj (nn.Module): Key projection.
        v_proj (nn.Module): Value projection (silu-activated).
        alpha_proj (nn.Linear): Per-head decay gate projection.
        beta_proj (nn.Linear): Per-head write gate projection.
        g_proj (nn.Module): Output gate projection (silu-gated).
        out_proj (nn.Module): Output projection.
        norm (nn.LayerNorm): Layer normalization applied to the
            recurrent readout before the output gate.
        dropout (nn.Dropout): Dropout layer.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Yang, S., Zhang, Y., Shen, Y., & Kim, Y. (2024).
        Gated Delta Networks: Improving Mamba2 and DeltaNet with
        Gating. arXiv:2412.06464. ICLR 2025.
    """

    def __init__(self, config):
        """Initialize GatedDeltaNetAttention.

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
            raise ValueError("hidden_size must be divisible by num_heads for GatedDeltaNetAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.alpha_proj = proj_cls(self.hidden_size, self.num_heads, bias=True)
        self.beta_proj = proj_cls(self.hidden_size, self.num_heads, bias=True)
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Gated DeltaNet attention over the input sequence.

        Processes the sequence token-by-token with a gated delta-rule
        recurrent matrix state. At each step t, the previous state is
        scaled by the decay gate α_t, then the delta-rule update removes
        the projection onto k_t (weighted by β_t) and adds the new
        association β_t · v_t k_t^T.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface
                compatibility with other attention mixers.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape

        q = F.normalize(self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim), dim=-1)
        k = F.normalize(self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim), dim=-1)
        v = F.silu(self.v_proj(x)).view(bsz, seq_len, self.num_heads, self.head_dim)
        alpha = torch.sigmoid(self.alpha_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))

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
            a_t = alpha[:, t, :, None, None]
            b_t = beta[:, t, :, None, None]
            k_t = k[:, t]
            v_t = v[:, t]
            kk = k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            state = a_t * state * (1 - b_t * kk) + b_t * (v_t.unsqueeze(-1) * k_t.unsqueeze(-2))
            out_t = (state * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)
