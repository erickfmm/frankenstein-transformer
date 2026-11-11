"""Gated Multi-Head Latent Attention (DeepSeek-style MLA, NoPE).

Implements the Gated MLA used by Kimi K3 (arXiv:2607.24653 §2.1.2), following
the DeepSeek-V2/V3 parameterisation (arXiv:2405.04434): a low-rank query
bottleneck (``q_lora_rank``), decoupled NoPE/RoPE key widths, a joint KV
latent (``c_kv``) cached alone, and a full-rank input-dependent output gate.

Formulation (per token ``x``):

    q      = W_UQ(W_DQ x)                  # low-rank queries, heads*qk_dim
    c_KV   = W_DKV x                       # latent, shape r_kv
    k_nope = W_UKN(c_KV)                   # heads*qk_nope_head_dim
    k_rope = W_UKR(c_KV)                   # heads*qk_rope_head_dim (unless NoPE)
    v      = W_UV(c_KV)                    # heads*v_head_dim
    out    = SDPA(q, cat(k_nope, k_rope), v) ⊙ silu(W_g x)
    y      = W_O out

With ``use_nope: true`` (Kimi K3) the rotary slice is dropped, so every MLA
layer is position-independent and extrapolates without RoPE retuning. With
``use_nope: false`` plus a rotary PE, only the rope slice is rotated.

Reference:
    Kimi Team (2026). "Kimi K3: Open Frontier Intelligence".
    arXiv:2607.24653.
    DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model". arXiv:2405.04434.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..params import mixer_params, use_pe_for
from ..common import BitLinear
from ...norm import attach_qk_norm
from ...numerics import module_softmax


class GatedMLAAttention(nn.Module):
    """DeepSeek-style MLA with a low-rank query path and a full-rank gate.

    Args:
        config: Model configuration (``hidden_size``, ``num_heads``,
            ``dropout``, ``bitnet.enabled``, ``mode``, PE settings).
        pos_encoder: Shared positional encoding module (only consumed when
            ``use_nope`` is false and the model-wide PE is enabled).
        params: Bound ``attention.gmla`` parameters.

    Attributes:
        hidden_size: Input embedding dimensionality.
        num_heads: Attention-head count.
        head_dim: Query/key width per head (``qk_nope + qk_rope``, or just
            the NoPE width when ``use_nope``).
        latent_rank: Joint KV latent rank ``r_kv``.
        q_lora_rank: Query bottleneck width (0 disables the low-rank path).
        v_head_dim: Value width per head.
        scale: Score scaling ``head_dim ** -0.5``.
        dropout: Attention-weight dropout.
    """

    def __init__(self, config, pos_encoder=None, params=None):
        params = mixer_params(config, "gmla", params)
        super().__init__()
        self.hidden_size = int(config.dims.hidden_size)
        default_heads = max(1, int(config.dims.num_heads))
        self.num_heads = int(params.value("num_heads", default_heads) or default_heads)
        self.qk_nope_head_dim = int(params.value("qk_nope_head_dim", 128) or 128)
        self.qk_rope_head_dim = int(params.value("qk_rope_head_dim", 64) or 64)
        self.v_head_dim = int(params.value("v_head_dim", 128) or 128)
        self.latent_rank = int(params.value("latent_rank", 512) or 512)
        q_lora = params.value("q_lora_rank", None)
        self.q_lora_rank = int(q_lora) if q_lora else 0
        self.use_nope = bool(params.value("use_nope", True))
        self.use_output_gate = bool(params.value("use_output_gate", True))

        self.qk_dim = self.qk_nope_head_dim + (
            0 if self.use_nope else self.qk_rope_head_dim
        )
        self.head_dim = self.qk_dim
        self.scale = self.qk_dim ** -0.5

        proj_cls = BitLinear if config.bitnet.enabled else nn.Linear
        if self.q_lora_rank > 0:
            self.q_down = proj_cls(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_up = proj_cls(
                self.q_lora_rank, self.num_heads * self.qk_dim, bias=False
            )
        else:
            self.q_proj = proj_cls(
                self.hidden_size, self.num_heads * self.qk_dim, bias=False
            )

        self.dkv_proj = proj_cls(self.hidden_size, self.latent_rank, bias=False)
        self.uk_nope_proj = proj_cls(
            self.latent_rank, self.num_heads * self.qk_nope_head_dim, bias=False
        )
        if not self.use_nope:
            self.uk_rope_proj = proj_cls(
                self.latent_rank, self.num_heads * self.qk_rope_head_dim, bias=False
            )
        self.uv_proj = proj_cls(
            self.latent_rank, self.num_heads * self.v_head_dim, bias=False
        )
        if self.use_output_gate:
            self.g_proj = proj_cls(
                self.hidden_size, self.num_heads * self.v_head_dim, bias=False
            )
        self.out_proj = proj_cls(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        self.dropout = nn.Dropout(config.dims.dropout)
        self.mode = getattr(config.dims, "mode", "encoder")

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(use_pe_for(config, "gmla", False)) and not self.use_nope
        self._rotary_pe = self.pe_type in ("rope", "hope", "sinusoidal_rotary")
        if self.use_pe and not self._rotary_pe:
            raise ValueError(
                "gmla supports NoPE (use_nope: true) or rotary positional "
                f"encodings; got positional_encoding={self.pe_type!r}."
            )
        attach_qk_norm(self, config, self.qk_dim)

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------
    def _queries(self, x: torch.Tensor) -> torch.Tensor:
        """Return queries ``(B, H, T, qk_dim)``."""
        bsz, seq_len, _ = x.shape
        if self.q_lora_rank > 0:
            q = self.q_up(self.q_down(x))
        else:
            q = self.q_proj(x)
        return q.view(bsz, seq_len, self.num_heads, self.qk_dim).transpose(1, 2)

    def _expand(self, c_kv: torch.Tensor, rope_offset: Optional[int] = None):
        """Expand the latent into keys/values for the given token block.

        Args:
            c_kv: Latents ``(B, T, r_kv)``.
            rope_offset: Absolute position of the first latent (rotary PE
                only; ``None`` positions from 0).

        Returns:
            ``(k, v)`` with shapes ``(B, H, T, qk_dim)`` and
            ``(B, H, T, v_head_dim)``.
        """
        bsz, t, _ = c_kv.shape
        k_nope = self.uk_nope_proj(c_kv).view(
            bsz, t, self.num_heads, self.qk_nope_head_dim
        )
        if self.use_nope:
            k = k_nope
        else:
            k_rope = self.uk_rope_proj(c_kv).view(
                bsz, t, self.num_heads, self.qk_rope_head_dim
            )
            if self.use_pe and self.pos_encoder is not None:
                k_rope = self.pos_encoder(
                    k_rope.transpose(1, 2),
                    logical_layer_idx=0,
                    position_offset=rope_offset,
                ).transpose(1, 2)
            k = torch.cat([k_nope, k_rope], dim=-1)
        v = self.uv_proj(c_kv).view(bsz, t, self.num_heads, self.v_head_dim)
        if self.qk_norm is not None:
            k = self.qk_norm.normalize_k(k.transpose(1, 2)).transpose(1, 2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def make_cache_slot(self, meta, cache):
        """Cache only the joint latent ``c_kv`` (MLA contract)."""
        from ...cache.slots import LatentKVSlot

        return LatentKVSlot(meta, axes={"c_kv": "token"}, **cache.codec_slot_kwargs())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute gated latent attention.

        Args:
            x: Input ``(B, S, hidden_size)``.
            logical_layer_idx: Logical layer index (rotary PE only).
            pos_encoder: Optional PE override.
            cache: Optional :class:`~src.model.cache.slots.LatentKVSlot`
                (``kind == "latent_kv"``) caching ``c_kv``.
            cache_position: Absolute index of the first incoming token.

        Returns:
            Output ``(B, S, hidden_size)``.
        """
        if pos_encoder is not None:
            self.pos_encoder = pos_encoder
        bsz, seq_len, _ = x.shape
        q = self._queries(x)
        c_kv = self.dkv_proj(x)

        slot = cache if getattr(cache, "kind", "") == "latent_kv" else None
        if slot is not None:
            prev = slot.read()
            c_prev = prev.get("c_kv")
            prev_pos = prev.get("positions")
            positions = torch.arange(
                int(cache_position or 0),
                int(cache_position or 0) + seq_len,
                device=x.device,
                dtype=torch.long,
            )
            if c_prev is None:
                c_all, kv_positions = c_kv, positions
            else:
                c_all = torch.cat([c_prev, c_kv], dim=1)
                if prev_pos is not None and prev_pos.numel():
                    kv_positions = torch.cat([prev_pos, positions], dim=0)
                else:
                    kv_positions = positions
            slot.append(c_kv=c_kv, positions=positions)
            k, v = self._expand(c_all, rope_offset=0)
        else:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            kv_positions = positions
            k, v = self._expand(c_kv, rope_offset=0)

        if self.use_pe and not self.use_nope and self._rotary_pe:
            # Rotate only the rope slice of the incoming queries with their
            # absolute positions; cached keys are expanded from position 0.
            q_nope, q_rope = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            q_rope = self.pos_encoder(
                q_rope,
                logical_layer_idx=logical_layer_idx or 0,
                position_offset=(cache_position if slot is not None else None),
            )
            q = torch.cat([q_nope, q_rope], dim=-1)

        if self.qk_norm is not None:
            q = self.qk_norm.normalize_q(q)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = positions[:, None] < kv_positions[None, :]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = module_softmax(self, scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        if self.use_output_gate:
            out = out * F.silu(self.g_proj(x))
        return self.out_proj(out)


__all__ = ["GatedMLAAttention"]
