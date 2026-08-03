"""Kimi Delta Attention (KDA).

Implements Kimi Delta Attention, arXiv:2510.26692 (Kimi Team, 2025).
KDA is the core attention module of Kimi Linear -- a hybrid linear
attention architecture that, for the first time, outperforms full
attention under fair comparisons across short-context, long-context and
reinforcement-learning scaling regimes. KDA extends Gated DeltaNet with
a *finer-grained* gating mechanism, enabling more effective use of
limited finite-state RNN memory.

Concretely, KDA replaces Gated DeltaNet's *scalar* per-head write gate
``beta_t`` with a **channel-wise** decay ``alpha_t`` (per key channel)
that is computed via a log-decay parameterisation
``g = a - softplus(delta)`` and applied as a diagonal matrix
``D_t = diag(alpha_t)`` before the delta-rule update. The delta rule
itself keeps a scalar write gate ``beta_t`` controlling the erase-write
strength along the key axis:

    S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t v_t k_t^T
    o_t = S_t^T q_t

A bespoke chunkwise algorithm achieves high hardware efficiency through
a specialised variant of the Diagonal-Plus-Low-Rank (DPLR) transition
matrices, which substantially reduces computation versus the general
DPLR formulation while remaining consistent with the classical delta
rule. KDA reduces to Gated DeltaNet when ``alpha_t`` collapses to a
scalar per head.

The Kimi Linear model -- 3B activated / 48B total parameters, a
layerwise hybrid of KDA and Multi-Head Latent Attention (MLA) --
outperforms full MLA across all evaluated tasks while reducing KV cache
usage by up to 75% and achieving up to 6x decoding throughput for a 1M
context.

Reference:
    Kimi Team (2025). "Kimi Linear: An Expressive, Efficient Attention
    Architecture". arXiv:2510.26692.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class KDAAttention(nn.Module):
    """Kimi Delta Attention with channel-wise decay and scalar write gate.

    Maintains a per-head matrix-valued recurrent state
    ``S_t ∈ R^{head_dim × head_dim}``. At each timestep, the previous
    state is scaled by a per-key-channel decay ``alpha_t`` (KDA-style
    log-decay computed in fp32 to avoid precision loss over long
    cumulative products), then the scalar-gated delta-rule update
    removes the projection onto ``k_t`` (weighted by ``beta_t``) and
    adds the new association ``beta_t · v_t k_t^T``. Queries and keys
    are L2-normalised; values are silu-activated; a silu-gated output
    projection provides additional channel-wise modulation.

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
        beta_proj (nn.Linear): Scalar per-head write gate projection
            (sigmoid-activated).
        log_decay_base (nn.Parameter): Broadcast base vector of the
            channel-wise log-decay, shape ``(total_dim,)``.
        log_decay_delta (nn.Parameter): Per-key-channel bias of the
            log-decay, shape ``(total_dim,)``. The per-channel decay is
            ``alpha = exp(base - softplus(delta))``, computed in fp32.
        g_proj (nn.Module): Output gate projection (silu-gated).
        out_proj (nn.Module): Output projection.
        norm (nn.LayerNorm): Layer normalization applied to the
            recurrent readout before the output gate.
        dropout (nn.Dropout): Dropout layer.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.

    Reference:
        Kimi Team (2025). "Kimi Linear: An Expressive, Efficient
        Attention Architecture". arXiv:2510.26692.
    """

    def __init__(self, config):
        """Initialize KDAAttention.

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
            raise ValueError("hidden_size must be divisible by num_heads for KDAAttention")

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.beta_proj = proj_cls(self.hidden_size, self.num_heads, bias=True)
        self.log_decay_base = nn.Parameter(torch.zeros(self.total_dim))
        self.log_decay_delta = nn.Parameter(torch.zeros(self.total_dim))
        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute KDA attention over the input sequence.

        Processes the sequence token-by-token with a gated delta-rule
        recurrent matrix state that combines a per-key-channel decay
        ``alpha_t`` (KDA-style log-decay computed in fp32) with a scalar
        per-head write gate ``beta_t``. At each step t, the previous
        state is scaled by ``D_t = diag(alpha_t)``, then the delta-rule
        update removes the projection onto ``k_t`` (weighted by
        ``beta_t``) and adds the new association
        ``beta_t · v_t k_t^T``.

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
        beta = torch.sigmoid(self.beta_proj(x))

        log_decay = (
            self.log_decay_base - F.softplus(self.log_decay_delta)
        ).to(torch.float32)
        alpha = torch.exp(log_decay).view(1, 1, self.num_heads, self.head_dim)
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
            k_t = k[:, t]
            v_t = v[:, t]
            b_t = beta[:, t, :, None, None]
            a_t = alpha[0, 0]

            state = state * a_t.unsqueeze(0).unsqueeze(-1)
            read = (state * k_t.unsqueeze(-1)).sum(-2)
            state = state - b_t * k_t.unsqueeze(-1) * read.unsqueeze(-2)
            state = state + b_t * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            out_t = (state * q[:, t].unsqueeze(-1)).sum(-2)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.total_dim)
        out = self.norm(out)
        out = out * F.silu(self.g_proj(x))
        out = self.dropout(out)
        return self.out_proj(out)