"""Kimi Delta Attention (KDA).

Implements Kimi Delta Attention, arXiv:2510.26692 (Kimi Team, 2025), and the
Kimi K3 refinements of arXiv:2607.24653 (§2.1.1).

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

Kimi K3 (arXiv:2607.24653) refines KDA with an **input-dependent bounded
decay** (``g = g_min·σ(e^{A_h} z)``, ``g_min = −5``), a **causal short
convolution** (kernel 4 + SiLU) before the Q/K/V projections, and a
**per-head RMSNorm** before the full-rank output gate. Both the Kimi
Linear (``decay_mode: kimilinear``) and the K3 (``decay_mode: k3``)
parameterisations are supported; the former remains the default so
existing presets keep their exact numerics.

References:
    Kimi Team (2025). "Kimi Linear: An Expressive, Efficient Attention
    Architecture". arXiv:2510.26692.
    Kimi Team (2026). "Kimi K3: Open Frontier Intelligence".
    arXiv:2607.24653.
"""

from __future__ import annotations

from ..params import mixer_params, use_pe_for

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear
from ...norm import RMSNorm


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
            num_heads (int): Default attention-head count.
            dropout (float): Dropout probability applied after the
                output gate.
            bitnet.enabled (bool): If True, uses BitLinear for the
                projections instead of nn.Linear.
            mode (str, optional): ``"encoder"`` or ``"decoder"``.
        pos_encoder: Optional shared positional encoding module.
        params: Bound ``attention.kda`` parameters (head geometry and the
            K3 refinements). See
            ``src/schema/_model/_attention_kda.yaml``.

    Attributes:
        hidden_size (int): Input embedding dimensionality.
        num_heads (int): Number of attention heads (mixer override or global).
        head_dim (int): Per-head width (``hidden_size // num_heads`` by
            default; explicit override allowed).
        total_dim (int): Total Q/K/V dimensionality (``head_dim * num_heads``).
        decay_mode (str): ``"kimilinear"`` or ``"k3"``.
        short_conv (nn.Conv1d | None): Causal depthwise short convolution.
        norm (nn.Module): Readout normalization (head-wise RMSNorm in K3
            mode, LayerNorm otherwise).
        dropout (nn.Dropout): Dropout layer.
        mode (str): ``"encoder"`` or ``"decoder"``.

    Raises:
        ValueError: If ``hidden_size % num_heads != 0`` and no explicit
            ``head_dim`` is configured.

    Reference:
        Kimi Team (2025). "Kimi Linear: An Expressive, Efficient
        Attention Architecture". arXiv:2510.26692.
        Kimi Team (2026). "Kimi K3: Open Frontier Intelligence".
        arXiv:2607.24653.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        """Initialize KDAAttention.

        Args:
            config: Model configuration object. See class docstring for
                required attributes.
            pos_encoder: Optional shared positional encoding module.
            params: Bound mixer parameters.

        Raises:
            ValueError: If the head geometry is not expressible.
        """
        params = mixer_params(config, "kda", params)
        super().__init__()
        self.hidden_size = config.dims.hidden_size
        default_heads = max(1, int(config.dims.num_heads))
        self.num_heads = int(params.value("num_heads", default_heads) or default_heads)
        explicit_head_dim = params.value("head_dim", None)
        if explicit_head_dim:
            # Kimi K3 decouples the head geometry from hidden_size (96 × 128
            # with hidden_size 7168), so the divisibility invariant is waived.
            self.head_dim = int(explicit_head_dim)
        else:
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_heads for "
                    "KDAAttention (or set attention.kda.head_dim)"
                )
            self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads

        self.decay_mode = str(params.value("decay_mode", "kimilinear")).lower()
        self.decay_min_log = float(params.value("decay_min_log", -5.0))
        self.short_conv_kernel = int(params.value("short_conv_kernel", 0) or 0)
        self.use_full_rank_gate = bool(params.value("use_full_rank_gate", True))

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        if self.short_conv_kernel > 1:
            self.short_conv = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=self.short_conv_kernel,
                groups=self.hidden_size,
                bias=True,
            )
        else:
            self.short_conv = None

        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.beta_proj = proj_cls(self.hidden_size, self.num_heads, bias=True)

        if self.decay_mode == "k3":
            rank = int(
                params.value("decay_low_rank", None)
                or max(1, self.hidden_size // 16)
            )
            self.decay_down = proj_cls(self.hidden_size, rank, bias=False)
            self.decay_up = proj_cls(rank, self.total_dim, bias=False)
            self.decay_bias = nn.Parameter(torch.zeros(self.total_dim))
            # Per-head log-scale A_h (init 0 ⇒ e^{A_h} = 1).
            self.decay_log_scale = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.log_decay_base = nn.Parameter(torch.zeros(self.total_dim))
            self.log_decay_delta = nn.Parameter(torch.zeros(self.total_dim))

        self.g_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        if not self.use_full_rank_gate:
            self.g_scalar_proj = proj_cls(self.hidden_size, self.num_heads, bias=True)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        if self.decay_mode == "k3":
            self.norm = RMSNorm(self.head_dim)
        else:
            self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(config.dims.dropout)
        self.mode = getattr(config.dims, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "kda", False))

    def _short_conv_forward(self, x: torch.Tensor, slot, cache_position=None) -> torch.Tensor:
        """Apply the causal depthwise short convolution + SiLU.

        Args:
            x: Input ``(B, T, hidden_size)``.
            slot: Optional state slot holding the last ``kernel − 1`` pre-conv
                rows under ``"conv_tail"``.
            cache_position: Unused; accepted for interface symmetry.

        Returns:
            Convolved and SiLU-activated input, same shape as ``x``.
        """
        if self.short_conv is None:
            return x
        pad = self.short_conv_kernel - 1
        tail = None
        if slot is not None:
            tail = slot.read().get("conv_tail")
        if pad > 0:
            if tail is None:
                tail = x.new_zeros((x.size(0), pad, x.size(2)))
            xin = torch.cat([tail, x], dim=1)
        else:
            xin = x
        y = self.short_conv(xin.transpose(1, 2)).transpose(1, 2)
        if pad > 0:
            y = y[:, -x.size(1) :]
            if slot is not None:
                combined = torch.cat([tail, x], dim=1)
                slot.update(conv_tail=combined[:, -pad:].detach())
        return F.silu(y)

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute KDA attention over the input sequence.

        Processes the sequence token-by-token with a gated delta-rule
        recurrent matrix state that combines a per-key-channel decay
        ``alpha_t`` with a scalar per-head write gate ``beta_t``. At each
        step t, the previous state is scaled by ``D_t = diag(alpha_t)``,
        then the delta-rule update removes the projection onto ``k_t``
        (weighted by ``beta_t``) and adds the new association
        ``beta_t · v_t k_t^T``.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface
                compatibility with other attention mixers.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.
            cache: Optional state slot (``kind == "state"``). Holds the
                recurrent state and, with a short convolution, the conv tail.
            cache_position: Unused; accepted for interface symmetry.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, _ = x.shape
        slot = cache if getattr(cache, "kind", "") == "state" else None

        x_conv = self._short_conv_forward(x, slot, cache_position)

        q = F.normalize(
            self.q_proj(x_conv).view(bsz, seq_len, self.num_heads, self.head_dim),
            dim=-1,
        )
        k = F.normalize(
            self.k_proj(x_conv).view(bsz, seq_len, self.num_heads, self.head_dim),
            dim=-1,
        )
        v = F.silu(self.v_proj(x_conv)).view(bsz, seq_len, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.beta_proj(x_conv))

        if self.use_pe and pe is not None:
            from ..common import apply_pe_to_qk
            q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)
            k = F.normalize(k, dim=-1)

        if self.decay_mode == "k3":
            # Input-dependent bounded decay (arXiv:2607.24653 §2.1.1):
            # z_t = W_up(W_down x_t) + b, g = g_min·σ(e^{A_h}·z), α = e^g.
            z = self.decay_up(F.silu(self.decay_down(x_conv))) + self.decay_bias
            z = z.view(bsz, seq_len, self.num_heads, self.head_dim)
            scale = torch.exp(self.decay_log_scale).view(1, 1, self.num_heads, 1)
            g = self.decay_min_log * torch.sigmoid(z * scale)
            alpha = torch.exp(g.to(torch.float32)).to(x.dtype)
        else:
            log_decay = (
                self.log_decay_base - F.softplus(self.log_decay_delta)
            ).to(torch.float32)
            alpha = torch.exp(log_decay).view(1, 1, self.num_heads, self.head_dim)
            alpha = alpha.to(x.dtype)

        state = None
        if slot is not None:
            state = slot.read().get("state")
        if state is None:
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
            if self.decay_mode == "k3":
                a_t = alpha[:, t].unsqueeze(-1)  # (B, H, D, 1)
            else:
                # (H, D) → (1, H, D, 1) broadcasts over the batch.
                a_t = alpha[0, 0].unsqueeze(0).unsqueeze(-1)

            state = state * a_t
            read = (state * k_t.unsqueeze(-1)).sum(-2)
            state = state - b_t * k_t.unsqueeze(-1) * read.unsqueeze(-2)
            state = state + b_t * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            out_t = (state * q[:, t].unsqueeze(-1)).sum(-2)
            outputs.append(out_t)
        if slot is not None:
            slot.update(state=state)

        out = torch.stack(outputs, dim=1)
        if self.decay_mode == "k3":
            out = self.norm(out)  # head-wise RMSNorm over head_dim
        else:
            out = self.norm(out.reshape(bsz, seq_len, self.total_dim))
        out = out.reshape(bsz, seq_len, self.total_dim)
        if self.use_full_rank_gate:
            out = out * F.silu(self.g_proj(x_conv))
        else:
            gate = torch.sigmoid(self.g_scalar_proj(x_conv))
            out = out * gate.unsqueeze(-1)
        out = self.dropout(out)
        return self.out_proj(out)


__all__ = ["KDAAttention"]
