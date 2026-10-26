"""Multi-Scale Retention mechanism (RetNet).

Implements the parallel representation of RetNet's multi-scale retention,
which replaces softmax attention with an exponential decay matrix D. Supports
three computational paradigms: parallel (training), recurrent (O(1) inference),
and chunkwise (hybrid). This implementation uses the parallel formulation
with per-head decay rates (gamma) and a Swish-gated output pathway.

Reference:
    Sun et al. (2023), "Retentive Network: A Successor to Transformer for
    Large Language Models", arXiv:2307.08621.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .common import BitLinear, apply_pe_to_qk
from ..norm import get_norm


class MultiScaleRetention(nn.Module):
    """Multi-scale retention with exponential decay (Sun et al. 2023).

    Projects the input into query, key, value, and gate tensors. Computes
    scaled dot-product scores, then multiplies element-wise by a per-head
    exponential decay matrix ``D`` where ``D_{nm} = gamma^{|n-m|}`` for
    encoder mode (or ``gamma^{n-m}`` for decoder mode with causal masking).
    The decay rates ``gamma`` are log-uniformly spaced across heads, providing
    multi-scale temporal receptive fields. A Swish-gated pathway modulates the
    output.

    Complexity:
        Training: O(n^2 * d) parallel. Inference: O(1) recurrent, no KV cache.

    Reference:
        Sun et al. (2023), "Retentive Network: A Successor to Transformer for
        Large Language Models", arXiv:2307.08621.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``retention_heads``, ``dropout``, ``use_bitnet``, ``norm_type``,
            and optionally ``mode`` (``"encoder"`` or ``"decoder"``).

    Attributes:
        dim: Dimensionality of the input and output embeddings.
        heads: Number of retention heads.
        head_dim: Dimensionality of each retention head (``dim // heads``).
        scale: Scaling factor ``1 / sqrt(head_dim)`` applied to dot products.
        q_proj: Linear (or BitLinear) projection for queries.
        k_proj: Linear (or BitLinear) projection for keys.
        v_proj: Linear (or BitLinear) projection for values.
        g_proj: Linear (or BitLinear) projection for the gate pathway.
        out_proj: Linear (or BitLinear) output projection.
        swish: SiLU (Swish) activation for the gate pathway.
        norm: Normalization layer applied after retention aggregation.
        decay_mask: Precomputed per-head gamma decay rates (buffer).
        mode: ``"encoder"`` for bidirectional, ``"decoder"`` for causal.
    """

    def __init__(self, config, pos_encoder=None):
        super().__init__()
        self.dim = config.hidden_size
        self.heads = config.retention_heads
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.dim, self.dim, bias=False)
        self.k_proj = proj_cls(self.dim, self.dim, bias=False)
        self.v_proj = proj_cls(self.dim, self.dim, bias=False)
        self.g_proj = proj_cls(self.dim, self.dim, bias=False)
        self.out_proj = proj_cls(self.dim, self.dim, bias=False)
        self.swish = nn.SiLU()
        self.norm = get_norm(config)

        self.register_buffer("decay_mask", self._build_decay_mask(config.hidden_size, 2048))
        self.mode = getattr(config, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "retnet_use_pe", False))

    def _build_decay_mask(self, dim, max_len=2048):
        """Build per-head gamma decay rates for the exponential decay matrix.

        Gamma values are log-uniformly spaced in [1/512, 1/32], converted to
        the form ``1 - exp(log(gamma))`` for numerical stability.

        Args:
            dim: Hidden size (unused; kept for interface compatibility).
            max_len: Maximum sequence length (unused; kept for interface
                compatibility).

        Returns:
            Tensor of shape ``(heads, 1, 1)`` with per-head gamma values.
        """
        gammas = 1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), self.heads))
        return gammas.view(self.heads, 1, 1)

    def forward(self, x, logical_layer_idx: Optional[int] = None, pos_encoder=None, cache=None, cache_position: Optional[int] = None):
        """Compute multi-scale retention in parallel mode.

        Cached decoding switches to the paper's recurrent form
        ``S_n = gamma S_{n-1} + K_n^T V_n``, ``o_n = Q_n S_n`` — algebraically
        identical to the parallel retention sum for the newest row — with
        the recurrent matrix stored in a
        :class:`~src.model.cache.slots.StateSlot` (O(1) per token).

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index for layer-dependent PE.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.
            cache: Optional state slot (``None`` runs the legacy path, D4).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        bsz, seq_len, dim = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

        slot = cache if getattr(cache, "kind", "") == "state" else None
        if slot is not None:
            q, k = apply_pe_to_qk(
                pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe,
                position_offset=cache_position,
            )
            state = slot.read().get("state")
            if state is None:
                state = torch.zeros(
                    bsz, self.heads, self.head_dim, self.head_dim,
                    device=x.device, dtype=x.dtype,
                )
            gammas = self.decay_mask.to(device=x.device, dtype=x.dtype)  # (H, 1, 1)
            outputs = []
            for t in range(seq_len):
                k_t = k[:, :, t]  # (B, H, D)
                v_t = v[:, :, t]
                q_t = q[:, :, t]
                state = gammas * state + v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
                out_t = (state * q_t.unsqueeze(-1)).sum(-2) * self.scale
                outputs.append(out_t)
            slot.update(state=state)
            y = torch.stack(outputs, dim=1).transpose(1, 2).reshape(bsz, seq_len, dim)
            y = self.norm(y)
            g = self.swish(self.g_proj(x))
            return self.out_proj(y * g)

        q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        gammas = 1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), self.heads, device=x.device))
        n = torch.arange(seq_len, device=x.device).unsqueeze(1)
        m = torch.arange(seq_len, device=x.device).unsqueeze(0)
        dist = n - m

        decay_matrix = gammas.view(self.heads, 1, 1) ** dist.abs().unsqueeze(0)
        if self.mode == "decoder":
            causal_mask = (dist >= 0).float().unsqueeze(0)
        else:
            causal_mask = torch.ones_like(dist, dtype=torch.float).unsqueeze(0)

        retention_scores = attn * decay_matrix * causal_mask

        y = retention_scores @ v
        y = y.transpose(1, 2).reshape(bsz, seq_len, dim)
        y = self.norm(y)

        g = self.swish(self.g_proj(x))
        out = y * g

        return self.out_proj(out)
