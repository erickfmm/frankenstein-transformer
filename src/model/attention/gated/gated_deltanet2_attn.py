"""Gated DeltaNet-2 Attention.

Implements the Gated DeltaNet-2 attention mechanism from Hatamizadeh et al.
(2026), arXiv:2605.22791 (NVIDIA). Gated DeltaNet-2 decouples the single
scalar delta gate of Gated DeltaNet / KDA into two independent channel-wise
gates: a *key-axis* erase gate ``b_t`` and a *value-axis* write gate ``w_t``,
while inheriting the channel-wise decay ``alpha_t`` from KDA. The state
update follows:

    S_t = (I - k_t (b_t ⊙ k_t)^T) D_t S_{t-1} + k_t (w_t ⊙ v_t)^T

where ``D_t = diag(alpha_t)`` with ``alpha_t`` a per-key-channel decay
computed via a log-decay parameterization, and ``b_t``, ``w_t`` are produced
by independent linear projections followed by sigmoid. This formulation
reduces to KDA when ``b_t = w_t = beta_t * 1`` and to Gated DeltaNet when
``alpha_t`` additionally collapses to a scalar.

At 1.3B parameters trained on 100B FineWeb-Edu tokens, Gated DeltaNet-2
outperforms Mamba-2, Mamba-3, Gated DeltaNet, and KDA on language modeling,
commonsense reasoning, and long-context retrieval, with the largest gains on
multi-key needle-in-a-haystack (RULER). Its throughput stays near-flat as
sequence length grows from 2K to 16K.

Reference:
    Hatamizadeh, A., Choi, Y., & Kautz, J. (2026).
    Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention.
    arXiv:2605.22791. NVIDIA. Code: NVlabs/GatedDeltaNet-2.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GatedDeltaNet2Attention(nn.Module):
    """Gated DeltaNet-2 attention with decoupled erase/write gates.

    Maintains a per-head matrix-valued recurrent state
    ``S_t ∈ R^{head_dim × head_dim}``. At each timestep, a channel-wise
    decay ``alpha_t`` (per key channel, KDA-style) scales the previous
    state, then the delta-rule update removes the projection onto the
    erase direction ``e_t = b_t ⊙ k_t`` (key-axis gate ``b_t``) and adds
    the new association along the write target ``z_t = w_t ⊙ v_t``
    (value-axis gate ``w_t``). Queries and keys are L2-normalized; values
    are activated with silu. A silu-gated output projection provides
    additional channel-wise modulation.

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
        erase_proj (nn.Linear): Per-channel erase gate projection
            (key axis, sigmoid-activated). Shape ``hidden_size ->
            total_dim``.
        write_proj (nn.Linear): Per-channel write gate projection
            (value axis, sigmoid-activated). Shape ``hidden_size ->
            total_dim``.
        log_decay_base (nn.Parameter): Broadcast base vector of the
            channel-wise log-decay, shape ``(total_dim,)``.
        log_decay_delta (nn.Parameter): Per-key-channel bias of the
            log-decay, shape ``(total_dim,)``. The per-channel decay is
            ``alpha = exp(base - softplus(delta))``, computed in fp32 to
            avoid precision loss in long cumulative products.
        g_proj (nn.Module): Output gate projection (silu-gated).
        out_proj (nn.Module): Output projection.
        norm (nn.LayerNorm): Layer normalization applied to the
            recurrent readout before the output gate.
        dropout (nn.Dropout): Dropout layer.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Hatamizadeh, A., Choi, Y., & Kautz, J. (2026).
        Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention.
        arXiv:2605.22791. NVIDIA.
    """

    def __init__(self, config):
        """Initialize GatedDeltaNet2Attention.

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
            raise ValueError(
                "hidden_size must be divisible by num_heads for GatedDeltaNet2Attention"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        # Channel-wise erase (key axis) and write (value axis) gates
        self.erase_proj = proj_cls(self.hidden_size, self.total_dim, bias=True)
        self.write_proj = proj_cls(self.hidden_size, self.total_dim, bias=True)
        # Channel-wise log-decay parameterization (KDA-style): g = a - softplus(δ)
        self.log_decay_base = nn.Parameter(torch.zeros(self.total_dim))
        self.log_decay_delta = nn.Parameter(torch.zeros(self.total_dim))
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Gated DeltaNet-2 attention over the input sequence.

        Processes the sequence token-by-token with a gated delta-rule
        recurrent matrix state with *decoupled* channel-wise erase and
        write gates. At each step t, the previous state is scaled by the
        per-key-channel decay ``alpha_t = exp(g_t)`` (KDA-style log-decay
        computed in fp32), then the delta-rule update removes the
        projection onto the erase direction ``e_t = b_t ⊙ k_t`` (key-axis
        gate) and adds the new association along the write target
        ``z_t = w_t ⊙ v_t`` (value-axis gate).

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface
                compatibility with other attention mixers.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape

        q = F.normalize(
            self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim), dim=-1
        )
        k = F.normalize(
            self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim), dim=-1
        )
        v = F.silu(self.v_proj(x)).view(bsz, seq_len, self.num_heads, self.head_dim)
        b = torch.sigmoid(self.erase_proj(x)).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        w = torch.sigmoid(self.write_proj(x)).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )

        # Channel-wise log-decay in fp32: g = base - softplus(delta); alpha = exp(g)
        log_decay = (
            self.log_decay_base - F.softplus(self.log_decay_delta)
        ).to(torch.float32)
        alpha = torch.exp(log_decay).view(1, 1, self.num_heads, self.head_dim)
        # Back to input dtype for the recurrent loop
        alpha = alpha.to(x.dtype)

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
            k_t = k[:, t]  # [B, H, dk]
            v_t = v[:, t]  # [B, H, dv]
            b_t = b[:, t]  # [B, H, dk]
            w_t = w[:, t]  # [B, H, dv]
            a_t = alpha[0, 0]  # [H, dk]

            # Erase direction e_t = b_t ⊙ k_t ; write target z_t = w_t ⊙ v_t
            e_t = b_t * k_t  # [B, H, dk]
            z_t = w_t * v_t  # [B, H, dv]

            # Decay the state along the key axis: D_t @ S_{t-1}
            #   S *= alpha per key channel: shape [B, H, dk, dv] *= [H, dk]
            state = state * a_t.unsqueeze(0).unsqueeze(-1)

            # Delta rule with decoupled gates:
            #   read = S^T @ e_t  (sum over key axis) -> [B, H, dv]
            read = (state * e_t.unsqueeze(-1)).sum(-2)
            # Subtract k_t (read) along the key axis: S -= k_t ⊗ read
            state = state - k_t.unsqueeze(-1) * read.unsqueeze(-2)
            # Add k_t z_t^T along the key axis: S += k_t ⊗ z_t
            state = state + k_t.unsqueeze(-1) * z_t.unsqueeze(-2)

            # Output: o_t = S^T @ q_t  (sum over key axis) -> [B, H, dv]
            out_t = (state * q[:, t].unsqueeze(-1)).sum(-2)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)