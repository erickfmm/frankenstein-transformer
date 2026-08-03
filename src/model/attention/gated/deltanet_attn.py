"""DeltaNet Attention.

Implements the DeltaNet attention mechanism from Yang et al. (2024),
arXiv:2406.06484. DeltaNet applies a delta learning rule to the recurrent
linear attention state, performing targeted error-correcting writes instead
of purely additive memory updates. The state update follows:

    S_t = S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T

where β_t ∈ (0, 1) is a per-head write strength learned from the input,
and k_t is L2-normalized. This formulation achieves perfect recall on the
Multi-Query Associative Recall (MQAR) task by removing conflicting
key-value associations before writing new ones.

Reference:
    Yang, S., Kailash, B., Zhang, Y., & Kim, Y. (2024).
    Parallelizing Linear Transformers with the Delta Rule over
    Sequence Length. arXiv:2406.06484.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class DeltaNetAttention(nn.Module):
    """DeltaNet attention with delta-rule state updates.

    Maintains a per-head matrix-valued recurrent state S_t ∈ R^{d×d}.
    At each timestep, a learned write strength β_t controls how much of
    the existing key-associated content is removed before the new
    key-value pair is written. Queries and keys are L2-normalized to
    ensure stable inner products. A silu-gated output projection provides
    channel-wise modulation after the recurrent readout.

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
        v_proj (nn.Module): Value projection.
        beta_proj (nn.Linear): Per-head write strength projection.
        g_proj (nn.Module): Output gate projection (silu-gated).
        out_proj (nn.Module): Output projection.
        norm (nn.LayerNorm): Layer normalization applied to the
            recurrent readout before the output gate.
        dropout (nn.Dropout): Dropout layer.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Yang, S., Kailash, B., Zhang, Y., & Kim, Y. (2024).
        Parallelizing Linear Transformers with the Delta Rule over
        Sequence Length. arXiv:2406.06484.
    """

    def __init__(self, config):
        """Initialize DeltaNetAttention.

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
            raise ValueError("hidden_size must be divisible by num_heads for DeltaNetAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.beta_proj = proj_cls(self.hidden_size, self.num_heads, bias=False)
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute DeltaNet attention over the input sequence.

        Processes the sequence token-by-token with a delta-rule recurrent
        matrix state. At each step t, the state is updated by removing
        the projection onto k_t (weighted by β_t) and then adding the
        new association β_t · v_t k_t^T. Queries and keys are L2-normalized
        before use.

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
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
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
            b_t = beta[:, t, :, None, None]
            k_t = k[:, t]
            v_t = v[:, t]
            kk = k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            state = state * (1 - b_t * kk) + b_t * (v_t.unsqueeze(-1) * k_t.unsqueeze(-2))
            out_t = (state * q[:, t].unsqueeze(-2)).sum(-1)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)
