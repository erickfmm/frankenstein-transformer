"""Compressed Convolutional Attention (CCA) and CCGQA.

Implements Compressed Convolutional Attention (CCA) and Compressed
Convolutional Grouped Query Attention (CCGQA) from arXiv:2510.04476
(Figliola, Alonso, Iyer, Anthony & Millidge, 2025, Zyphra).

Both variants **down-project** queries, keys, and values into a shared
compressed latent space and perform the *entire* attention operation
inside that latent -- there are **no** Q/K/V up-projection matrices
(unlike MLA, which up-projects back to full width before attending).
Only a single output up-projection ``W̃_O`` maps the latent output back
to the residual stream.  This simultaneously reduces parameters,
KV-cache size, **and** attention FLOPs by the compression factor ``C``
(MLA only shrinks the cache).

To make attention in the fully compressed latent space viable, CCA
introduces three innovations, all toggleable via config:

1. **Two convolutions** on the packed q/k tensor: a depth-wise *causal
   sequence* convolution (mixes across positions) followed by a
   head-wise *grouped channel* convolution (mixes across channels
   within each head).  The paper's ablation shows two conv layers is
   optimal.
2. **q-k-mean**: adds the pre-convolution mean of q and k to the
   post-convolution values, increasing attention-diagonal sparsity
   when combined with QK-norm.
3. **Value-shift**: each attention head receives half its values from
   the current token and half from the *previous* token (a token-shift
   inductive bias borrowed from RWKV), implemented via two independent
   value projections.

After down-projection + convolutions + qk-mean + value-shift, QK
L2-normalisation and a learnable key temperature ``β`` are applied,
RoPE is applied *directly in the latent* (no separate RoPE head/cache
needed, unlike MLA), and standard softmax attention is computed.

**CCGQA** extends CCA with GQA-style key/value head sharing applied
*inside* the compressed latent, and **decouples** the query and KV
compression rates: ``C₁`` (query) and ``C₂`` (KV) with ``C₂ ≥ C₁``.
The per-head latent dimension ``d_h`` must match between query and key
heads, which enforces ``C₂ / C₁ = num_heads / num_kv_heads``.

Reference:
    Figliola, T., Alonso, N., Iyer, R., Anthony, Q., & Millidge, B.
    (2025). "Compressed Convolutional Attention: Efficient Attention
    in a Compressed Latent Space". arXiv:2510.04476.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


def _apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply rotary positional embeddings to the last dimension of ``x``.

    Args:
        x: Tensor of shape ``(B, H, S, D)`` with ``D`` even.
        base: RoPE base frequency. Default: 10000.0.

    Returns:
        Tensor of the same shape as ``x`` with RoPE applied.
    """
    bsz, heads, seq, dim = x.shape
    half = dim // 2
    pos = torch.arange(seq, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) * 2.0 / dim))
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cos(freqs).view(1, 1, seq, half)
    sin = torch.sin(freqs).view(1, 1, seq, half)
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.to(x.dtype)


def _apply_causal_convs(
    qk: torch.Tensor,
    conv0: Optional[nn.Conv1d],
    kernel0: int,
    conv1: Optional[nn.Conv1d],
    kernel1: int,
) -> torch.Tensor:
    """Apply causal depthwise + grouped convolutions to a packed qk tensor.

    Both convolutions are **causal** (left-padded) so they are safe in
    decoder (autoregressive) mode.  When a conv layer is ``None`` it is
    skipped.

    Args:
        qk: Packed qk tensor of shape ``(B, S, C)``.
        conv0: Depth-wise causal sequence convolution (or ``None``).
        kernel0: Kernel size of ``conv0``.
        conv1: Grouped channel convolution (or ``None``).
        kernel1: Kernel size of ``conv1``.

    Returns:
        Convolved tensor of shape ``(B, S, C)``.
    """
    if conv0 is None and conv1 is None:
        return qk
    qk = qk.permute(0, 2, 1)  # (B, C, S)
    if conv0 is not None:
        qk = F.pad(qk, (kernel0 - 1, 0))
        qk = conv0(qk)
    if conv1 is not None:
        qk = F.pad(qk, (kernel1 - 1, 0))
        qk = conv1(qk)
    return qk.permute(0, 2, 1)  # (B, S, C)


class CCAAttention(nn.Module):
    """Compressed Convolutional Attention (Figliola et al. 2025).

    Down-projects q, k, v into a shared latent of dimension
    ``cca_latent_rank`` (``ẽ = E / C``), performs attention entirely
    in the latent with optional convolutions, qk-mean, and value-shift,
    then up-projects the output back to the residual stream via a single
    ``W̃_O``.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, ``mode``, and the
            optional CCA-specific fields:

            * ``cca_latent_rank`` -- latent width ẽ (default
              ``hidden_size // 4``, i.e. compression ``C = 4``).  Must be
              divisible by ``num_heads``.
            * ``cca_num_conv_layers`` -- 0, 1, or 2 convolution layers
              (default 2; the paper's recommended setting).
            * ``cca_conv_kernel_seq`` -- kernel size ``k_seq`` of the
              depth-wise causal sequence convolution (default 4).
            * ``cca_conv_kernel_ch`` -- kernel size ``k_ch`` of the
              head-wise grouped channel convolution (default 3).
            * ``cca_qk_mean`` -- enable the q-k-mean bias (default True).
            * ``cca_value_shift`` -- enable value-shift with two value
              projections (default True; requires ``num_heads`` even and
              ``cca_latent_rank`` even).
            * ``rope_base`` -- RoPE base frequency (default 10000.0).

    Attributes:
        hidden_size: Input embedding dimensionality.
        num_heads: Number of attention heads.
        latent_dim: Latent width ẽ.
        latent_head_dim: Per-head latent dimensionality ``d_h = ẽ / num_heads``.
        num_conv_layers: Number of convolution layers (0, 1, or 2).
        conv_kernel_seq: Sequence-conv kernel size.
        conv_kernel_ch: Channel-conv kernel size.
        qk_mean: Whether q-k-mean is enabled.
        value_shift: Whether value-shift is enabled.
        rope_base: RoPE base frequency.
        linear_qk: Packed q/k down-projection ``E -> 2ẽ``.
        val_proj1, val_proj2: Value projections for value-shift
            (each ``E -> ẽ/2``).  Present only when ``value_shift`` is True.
        val_proj: Single value projection ``E -> ẽ``.  Present only when
            ``value_shift`` is False.
        out_proj: Output up-projection ``ẽ -> E`` (``W̃_O``).
        conv_qk0: Depth-wise causal sequence Conv1d (or None).
        conv_qk1: Head-wise grouped channel Conv1d (or None).
        temp: Learnable key temperature ``β`` (scalar, init 0).
        dropout: Dropout layer.
        mode: ``"encoder"`` or ``"decoder"``.
        scale: Attention softmax scale ``1/sqrt(d_h)``.

    Raises:
        ValueError: If ``hidden_size`` not divisible by ``num_heads``;
            if ``cca_latent_rank`` not divisible by ``num_heads``;
            if ``cca_num_conv_layers`` not in {0, 1, 2};
            if ``cca_value_shift`` is True but ``num_heads`` is odd or
            ``cca_latent_rank`` is odd;
            if ``latent_head_dim`` is odd (RoPE requires even).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for CCAAttention")

        self.latent_dim = int(getattr(config, "cca_latent_rank", max(1, self.hidden_size // 4)))
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("cca_latent_rank must be divisible by num_heads for CCAAttention")
        self.latent_head_dim = self.latent_dim // self.num_heads
        if self.latent_head_dim % 2 != 0:
            raise ValueError("cca_latent_rank / num_heads must be even for RoPE (got odd latent_head_dim)")

        self.num_conv_layers = int(getattr(config, "cca_num_conv_layers", 2))
        if self.num_conv_layers not in (0, 1, 2):
            raise ValueError("cca_num_conv_layers must be 0, 1, or 2")
        self.conv_kernel_seq = int(getattr(config, "cca_conv_kernel_seq", 4))
        self.conv_kernel_ch = int(getattr(config, "cca_conv_kernel_ch", 3))

        self.qk_mean = bool(getattr(config, "cca_qk_mean", True))
        self.value_shift = bool(getattr(config, "cca_value_shift", True))
        if self.value_shift:
            if self.num_heads % 2 != 0:
                raise ValueError("cca_value_shift requires num_heads to be even")
            if self.latent_dim % 2 != 0:
                raise ValueError("cca_value_shift requires cca_latent_rank to be even")

        self.rope_base = float(getattr(config, "rope_base", 10000.0))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        packed_dim = 2 * self.latent_dim
        self.linear_qk = proj_cls(self.hidden_size, packed_dim, bias=False)

        if self.value_shift:
            half = self.latent_dim // 2
            self.val_proj1 = proj_cls(self.hidden_size, half, bias=False)
            self.val_proj2 = proj_cls(self.hidden_size, half, bias=False)
        else:
            self.val_proj = proj_cls(self.hidden_size, self.latent_dim, bias=False)

        self.out_proj = proj_cls(self.latent_dim, self.hidden_size, bias=False)

        if self.num_conv_layers >= 1:
            self.conv_qk0 = nn.Conv1d(
                packed_dim, packed_dim,
                kernel_size=self.conv_kernel_seq,
                groups=packed_dim, bias=False,
            )
        else:
            self.conv_qk0 = None
        if self.num_conv_layers >= 2:
            self.conv_qk1 = nn.Conv1d(
                packed_dim, packed_dim,
                kernel_size=self.conv_kernel_ch,
                groups=2 * self.num_heads, bias=False,
            )
        else:
            self.conv_qk1 = None

        self.temp = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.latent_head_dim ** -0.5

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Compressed Convolutional Attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        H = self.num_heads
        dh = self.latent_head_dim
        L = self.latent_dim

        # ---- Down-projection (packed q and k) ----
        qk_packed = self.linear_qk(x)                       # (B, S, 2L)
        q_pre = qk_packed[..., :L]                           # (B, S, L)
        k_pre = qk_packed[..., L:]                           # (B, S, L)

        # ---- Causal convolutions on packed q/k ----
        qk_conv = _apply_causal_convs(
            qk_packed, self.conv_qk0, self.conv_kernel_seq,
            self.conv_qk1, self.conv_kernel_ch,
        )
        q_conv = qk_conv[..., :L]
        k_conv = qk_conv[..., L:]

        # ---- q-k-mean bias ----
        q_pre_h = q_pre.view(bsz, seq_len, H, dh)
        k_pre_h = k_pre.view(bsz, seq_len, H, dh)
        if self.qk_mean:
            qk_mean = (q_pre_h + k_pre_h) * 0.5
            q = q_conv.view(bsz, seq_len, H, dh) + qk_mean
            k = k_conv.view(bsz, seq_len, H, dh) + qk_mean
        else:
            q = q_conv.view(bsz, seq_len, H, dh)
            k = k_conv.view(bsz, seq_len, H, dh)

        # ---- Value projection (with optional value-shift) ----
        if self.value_shift:
            x_shifted = F.pad(x[:, :-1], (0, 0, 1, 0))      # (B, S, E) shifted right by 1
            v1 = self.val_proj1(x)                           # (B, S, L/2)
            v2 = self.val_proj2(x_shifted)                   # (B, S, L/2)
            v = torch.cat([v1, v2], dim=-1).view(bsz, seq_len, H, dh)
        else:
            v = self.val_proj(x).view(bsz, seq_len, H, dh)

        # ---- Reshape to (B, H, S, dh) ----
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ---- QK L2-norm + learnable key temperature ----
        q_norm = q.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        k_norm = k.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        sqrt_dh = math.sqrt(dh)
        q = q * (sqrt_dh / q_norm)
        k = k * (sqrt_dh / k_norm) * torch.exp(self.temp)

        # ---- RoPE applied directly in the latent ----
        q = _apply_rope(q, self.rope_base)
        k = _apply_rope(k, self.rope_base)

        # ---- Standard softmax attention ----
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, L)
        return self.out_proj(out)


class CCGQAAttention(nn.Module):
    """Compressed Convolutional Grouped Query Attention (Figliola et al. 2025).

    Extends CCA with GQA-style key/value head sharing applied *inside*
    the compressed latent, and **decouples** the query and KV
    compression rates.  The query latent has width ``E / C₁`` and the
    KV latent has width ``E / C₂`` with ``C₂ ≥ C₁``.  The per-head
    latent dimension ``d_h`` must be the same for query and key heads,
    which enforces the constraint::

        ccgqa_query_latent_rank / num_heads
            == ccgqa_kv_latent_rank / ccgqa_num_kv_heads

    i.e. ``C₂ / C₁ == num_heads / ccgqa_num_kv_heads`` (the GQA group
    size).

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, ``mode``, and the
            optional CCGQA-specific fields:

            * ``ccgqa_query_latent_rank`` -- query latent width ``E / C₁``
              (default ``hidden_size // 2``, i.e. ``C₁ = 2``).  Must be
              divisible by ``num_heads``.
            * ``ccgqa_kv_latent_rank`` -- KV latent width ``E / C₂``
              (default ``hidden_size // 8``, i.e. ``C₂ = 8``).  Must be
              divisible by ``ccgqa_num_kv_heads`` and ≤
              ``ccgqa_query_latent_rank``.
            * ``ccgqa_num_kv_heads`` -- number of KV (group) heads
              (default ``num_heads // 4``; must divide ``num_heads``).
            * ``ccgqa_num_conv_layers`` -- 0, 1, or 2 (default 2).
            * ``ccgqa_conv_kernel_seq`` -- sequence-conv kernel (default 4).
            * ``ccgqa_conv_kernel_ch`` -- channel-conv kernel (default 3).
            * ``ccgqa_qk_mean`` -- enable q-k-mean with B_group/E_group
              (default True).
            * ``ccgqa_value_shift`` -- enable value-shift (default True;
              requires ``ccgqa_num_kv_heads`` even and
              ``ccgqa_kv_latent_rank`` even).
            * ``rope_base`` -- RoPE base frequency (default 10000.0).

    Attributes:
        hidden_size: Input embedding dimensionality.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV (group) heads.
        group_size: ``num_heads // num_kv_heads``.
        latent_head_dim: Per-head latent dimensionality ``d_h``.
        query_latent_dim: Query latent width ``E / C₁``.
        kv_latent_dim: KV latent width ``E / C₂``.
        num_conv_layers: Number of convolution layers (0, 1, or 2).
        qk_mean: Whether q-k-mean is enabled.
        value_shift: Whether value-shift is enabled.
        rope_base: RoPE base frequency.
        linear_qk: Packed q/k down-projection
            ``E -> (query_latent + kv_latent)``.
        val_proj1, val_proj2: Value projections for value-shift.
        val_proj: Single value projection (when value_shift is False).
        out_proj: Output up-projection ``query_latent -> E`` (``W̃_O``).
        conv_qk0, conv_qk1: Convolution layers (or None).
        temp: Learnable key temperature ``β``.
        dropout, mode, scale: As in CCA.

    Raises:
        ValueError: If any divisibility or constraint check fails (see
            Args above for the full list).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for CCGQAAttention")

        num_kv = int(getattr(config, "ccgqa_num_kv_heads", max(1, self.num_heads // 4)))
        if self.num_heads % num_kv != 0:
            raise ValueError("ccgqa_num_kv_heads must divide num_heads")
        self.num_kv_heads = num_kv
        self.group_size = self.num_heads // num_kv

        self.query_latent_dim = int(
            getattr(config, "ccgqa_query_latent_rank", max(1, self.hidden_size // 2))
        )
        if self.query_latent_dim % self.num_heads != 0:
            raise ValueError("ccgqa_query_latent_rank must be divisible by num_heads")
        self.kv_latent_dim = int(
            getattr(config, "ccgqa_kv_latent_rank", max(1, self.hidden_size // 8))
        )
        if self.kv_latent_dim % self.num_kv_heads != 0:
            raise ValueError("ccgqa_kv_latent_rank must be divisible by ccgqa_num_kv_heads")
        if self.kv_latent_dim > self.query_latent_dim:
            raise ValueError("ccgqa_kv_latent_rank must be <= ccgqa_query_latent_rank (C2 >= C1)")

        q_dh = self.query_latent_dim // self.num_heads
        kv_dh = self.kv_latent_dim // self.num_kv_heads
        if q_dh != kv_dh:
            raise ValueError(
                "Per-head latent dim mismatch: query d_h="
                f"{q_dh} != kv d_h={kv_dh}. Constraint: "
                "ccgqa_query_latent_rank / num_heads == "
                "ccgqa_kv_latent_rank / ccgqa_num_kv_heads"
            )
        self.latent_head_dim = q_dh
        if self.latent_head_dim % 2 != 0:
            raise ValueError("latent_head_dim must be even for RoPE (got odd)")

        self.num_conv_layers = int(getattr(config, "ccgqa_num_conv_layers", 2))
        if self.num_conv_layers not in (0, 1, 2):
            raise ValueError("ccgqa_num_conv_layers must be 0, 1, or 2")
        self.conv_kernel_seq = int(getattr(config, "ccgqa_conv_kernel_seq", 4))
        self.conv_kernel_ch = int(getattr(config, "ccgqa_conv_kernel_ch", 3))

        self.qk_mean = bool(getattr(config, "ccgqa_qk_mean", True))
        self.value_shift = bool(getattr(config, "ccgqa_value_shift", True))
        if self.value_shift:
            if self.num_kv_heads % 2 != 0:
                raise ValueError("ccgqa_value_shift requires ccgqa_num_kv_heads to be even")
            if self.kv_latent_dim % 2 != 0:
                raise ValueError("ccgqa_value_shift requires ccgqa_kv_latent_rank to be even")

        self.rope_base = float(getattr(config, "rope_base", 10000.0))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        packed_dim = self.query_latent_dim + self.kv_latent_dim
        self.linear_qk = proj_cls(self.hidden_size, packed_dim, bias=False)

        if self.value_shift:
            half = self.kv_latent_dim // 2
            self.val_proj1 = proj_cls(self.hidden_size, half, bias=False)
            self.val_proj2 = proj_cls(self.hidden_size, half, bias=False)
        else:
            self.val_proj = proj_cls(self.hidden_size, self.kv_latent_dim, bias=False)

        # Output up-projection: query_latent -> E
        self.out_proj = proj_cls(self.query_latent_dim, self.hidden_size, bias=False)

        # Convolutions on packed q/k
        n_conv_groups = self.num_heads + self.num_kv_heads
        if self.num_conv_layers >= 1:
            self.conv_qk0 = nn.Conv1d(
                packed_dim, packed_dim,
                kernel_size=self.conv_kernel_seq,
                groups=packed_dim, bias=False,
            )
        else:
            self.conv_qk0 = None
        if self.num_conv_layers >= 2:
            self.conv_qk1 = nn.Conv1d(
                packed_dim, packed_dim,
                kernel_size=self.conv_kernel_ch,
                groups=n_conv_groups, bias=False,
            )
        else:
            self.conv_qk1 = None

        self.temp = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")
        self.scale = self.latent_head_dim ** -0.5

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute Compressed Convolutional Grouped Query Attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Unused; accepted for interface compatibility.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        Hq = self.num_heads
        Hkv = self.num_kv_heads
        gs = self.group_size
        dh = self.latent_head_dim
        Lq = self.query_latent_dim
        Lkv = self.kv_latent_dim

        # ---- Down-projection (packed q and k with different widths) ----
        qk_packed = self.linear_qk(x)                       # (B, S, Lq + Lkv)
        q_pre = qk_packed[..., :Lq]                          # (B, S, Lq)
        k_pre = qk_packed[..., Lq:]                          # (B, S, Lkv)

        # ---- Causal convolutions on packed q/k ----
        qk_conv = _apply_causal_convs(
            qk_packed, self.conv_qk0, self.conv_kernel_seq,
            self.conv_qk1, self.conv_kernel_ch,
        )
        q_conv = qk_conv[..., :Lq]
        k_conv = qk_conv[..., Lq:]

        # ---- q-k-mean with B_group / E_group ----
        q_pre_h = q_pre.view(bsz, seq_len, Hq, dh)          # (B, S, Hq, dh)
        k_pre_h = k_pre.view(bsz, seq_len, Hkv, dh)         # (B, S, Hkv, dh)
        q_conv_h = q_conv.view(bsz, seq_len, Hq, dh)
        k_conv_h = k_conv.view(bsz, seq_len, Hkv, dh)
        if self.qk_mean:
            # B_group: replicate each kv head to group_size query heads
            k_expanded = k_pre_h.repeat_interleave(gs, dim=2)   # (B, S, Hq, dh)
            qk_mean_q = (q_pre_h + k_expanded) * 0.5             # (B, S, Hq, dh)
            # E_group: average query heads within each group
            qk_mean_k = qk_mean_q.view(bsz, seq_len, Hkv, gs, dh).mean(dim=3)  # (B, S, Hkv, dh)
            q = q_conv_h + qk_mean_q
            k = k_conv_h + qk_mean_k
        else:
            q = q_conv_h
            k = k_conv_h

        # ---- Value projection (with optional value-shift) ----
        if self.value_shift:
            x_shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
            v1 = self.val_proj1(x)                           # (B, S, Lkv/2)
            v2 = self.val_proj2(x_shifted)                   # (B, S, Lkv/2)
            v = torch.cat([v1, v2], dim=-1).view(bsz, seq_len, Hkv, dh)
        else:
            v = self.val_proj(x).view(bsz, seq_len, Hkv, dh)

        # ---- Reshape to (B, H, S, dh) and replicate KV heads ----
        q = q.transpose(1, 2)                                # (B, Hq, S, dh)
        k = k.transpose(1, 2).repeat_interleave(gs, dim=1)   # (B, Hq, S, dh)
        v = v.transpose(1, 2).repeat_interleave(gs, dim=1)   # (B, Hq, S, dh)

        # ---- QK L2-norm + learnable key temperature ----
        q_norm = q.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        k_norm = k.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        sqrt_dh = math.sqrt(dh)
        q = q * (sqrt_dh / q_norm)
        k = k * (sqrt_dh / k_norm) * torch.exp(self.temp)

        # ---- RoPE applied directly in the latent ----
        q = _apply_rope(q, self.rope_base)
        k = _apply_rope(k, self.rope_base)

        # ---- Standard softmax attention ----
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, Hq * dh)
        return self.out_proj(out)
