"""Shared cache helpers used by attention mixers.

Keeps mixer-side cached branches tiny and uniform: metadata construction,
position tensors, append+read, causal/pattern masking, and the PE offset
shim. Everything here is additive — the legacy paths of the mixers are
untouched when ``cache is None`` (D4).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..numerics import fixed_softmax
from .base import CacheSlot, SlotMeta

__all__ = [
    "build_slot_meta",
    "positions_for",
    "kv_decode_append",
    "read_kv",
    "causal_mask_from_positions",
    "pattern_mask",
    "cached_qk_attention",
    "cached_latent_attention",
]


def build_slot_meta(
    mixer,
    config,
    logical_layer_idx: int,
    num_kv_heads: Optional[int] = None,
) -> SlotMeta:
    """Build the :class:`SlotMeta` for a mixer layer.

    Args:
        mixer: The mixer module (used for ``num_heads``/``head_dim`` when
            present, falling back to the config).
        config: :class:`~src.model.config.FrankensteinModelConfig`.
        logical_layer_idx: Global logical layer index (slot identity, D6).
        num_kv_heads: Optional override (GQA mixers).
    """
    num_heads = int(getattr(mixer, "num_heads", config.num_heads))
    head_dim = int(getattr(mixer, "head_dim", config.hidden_size // num_heads))
    if num_kv_heads is None:
        num_kv_heads = int(getattr(mixer, "num_kv_heads", num_heads))
    pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
    rope_scaling = float(getattr(config, "rope_scaling", 1.0) or 1.0)
    return SlotMeta(
        mixer_key=str(getattr(mixer, "__mixer_key__", "")),
        logical_layer_idx=int(logical_layer_idx),
        num_heads=num_heads,
        num_kv_heads=int(num_kv_heads),
        head_dim=head_dim,
        dtype=torch.float32,
        pe_type=pe_type,
        rope_scaling=rope_scaling,
    )


def positions_for(
    cache_position: Optional[int],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Absolute int64 positions of the incoming tokens.

    ``cache_position`` is the absolute index of the FIRST incoming token
    (``None`` ⇒ a full sequence starting at 0).
    """
    start = 0 if cache_position is None else int(cache_position)
    return torch.arange(start, start + seq_len, device=device, dtype=torch.long)


def kv_decode_append(
    slot: CacheSlot,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    extras: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Append new K/V (post-RoPE) and return the full cache views.

    Args:
        slot: The layer's :class:`KVSlot` (or ``RingSlot``).
        k: New keys ``(B, H_kv, T, D)``, post-RoPE.
        v: New values ``(B, H_kv, T, D)``.
        positions: Absolute positions ``(T,)``.
        extras: Optional per-token metadata (FoX ``c_j``, ...).

    Returns:
        ``(K, V, kv_positions, extras_dict)`` with ``K``/``V`` covering the
        whole cached prefix plus the new tokens.
    """
    slot.append(k=k, v=v, positions=positions, extras=extras)
    out = slot.read()
    return out["k"], out["v"], out["positions"], out.get("extras", {})


def read_kv(slot: CacheSlot) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Read the full cache views without appending."""
    out = slot.read()
    return out["k"], out["v"], out["positions"], out.get("extras", {})


def causal_mask_from_positions(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
) -> torch.Tensor:
    """Boolean causal mask from absolute positions (True = masked).

    ``mask[i, j] = q_positions[i] < kv_positions[j]`` — a query may only
    attend to keys at the same or earlier absolute positions. For a full
    sequence this reproduces the classic ``triu(diagonal=1)`` mask exactly.
    """
    return q_positions[:, None] < kv_positions[None, :]


def pattern_mask(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    pattern,
) -> torch.Tensor:
    """Evaluate a sparse pattern against absolute positions (B5 fix).

    Args:
        q_positions: Query absolute positions ``(Q,)``.
        kv_positions: Key absolute positions ``(T,)``.
        pattern: Callable ``(q_pos: Tensor, kv_pos: Tensor) -> Bool (Q, T)``
            returning True where the pair is attendable.

    Returns:
        Boolean mask ``(Q, T)`` with True = **masked** (attendable pairs
        are False), ready for ``masked_fill``.
    """
    allowed = pattern(q_positions[:, None], kv_positions[None, :])
    causal = q_positions[:, None] < kv_positions[None, :]
    return ~(allowed & ~causal)


def cached_qk_attention(
    slot,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_position: Optional[int],
    scale: float,
    pe=None,
    pe_type: str = "",
    use_pe: bool = False,
    x_for_pe: Optional[torch.Tensor] = None,
    logical_layer_idx: int = 0,
    normalize=None,
    dropout=None,
    causal: bool = True,
    rotate: bool = True,
    use_fixed_softmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generic cached QK^T attention over a KV/Ring slot (F1 recipe).

    Appends the (post-RoPE) new keys/values to the slot, reads the full
    cache, computes scaled scores with the PE bias on the accumulated
    length, and normalizes. Covers every quadratic-softmax-style mixer
    (standard, sigmoid, gated softmax, titan, ...) without per-mixer
    cache bookkeeping.

    Args:
        slot: The layer's ``KVSlot``/``RingSlot``.
        q: Queries ``(B, H, S, D)`` (not yet RoPE'd when a rotation PE is
            used; rotated here with the absolute offset).
        k: Keys ``(B, H_kv, S, D)``.
        v: Values ``(B, H_kv, S, D)``.
        cache_position: Absolute index of the first incoming token.
        scale: Score scaling (``head_dim ** -0.5``).
        pe: PE module (or ``None``).
        pe_type: PE type string.
        use_pe: Whether the mixer consumes the PE.
        x_for_pe: Hidden state ``(B, S, C)`` for content-dependent PEs.
        logical_layer_idx: Logical layer index for layer-dependent PEs.
        normalize: Score normalizer (default softmax over the last dim).
            Sigmoid attention passes its sigmoid+L1 normalizer — its
            denominator must be re-summed over the whole cache each step
            (sigmoid is not shift-invariant, plan §6.2).
        dropout: Optional attention-weight dropout module.
        causal: Apply the position-derived causal mask (True for decoder
            semantics; cached attention is causal by construction).
        rotate: When False, ``q``/``k`` arrive already rotated (mixers like
            IHA that rotate real heads before head-space mixing); the PE
            score bias is still applied.
        use_fixed_softmax: Use the cancellation-free softmax of
            arXiv:2610.31642 for the default normalizer (mirrors the mixer's
            ``use_fixed_softmax`` attribute).

    Returns:
        ``(output, K, kv_positions)`` where ``output`` is the attended
        values ``(B, H, S, D)`` (pre out-projection / gating).
    """
    import torch.nn.functional as F

    from ..attention.common import apply_pe_to_qk, apply_pe_to_scores

    offset = cache_position
    positions = positions_for(offset, q.size(-2), q.device)
    if rotate and pe is not None and use_pe and pe_type in ("rope", "hope", "sinusoidal_rotary"):
        q, k = apply_pe_to_qk(pe, pe_type, q, k, x_for_pe, logical_layer_idx, use_pe, position_offset=offset)
    K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)

    attn_scores = (q @ K.transpose(-2, -1)) * scale
    attn_scores = apply_pe_to_scores(
        pe,
        pe_type,
        attn_scores,
        q,
        use_pe,
        kv_len=K.size(-2),
        q_offset=0 if offset is None else int(offset),
        kv_positions=kv_positions,
    )
    if causal:
        mask = causal_mask_from_positions(positions, kv_positions)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    if normalize is None:
        attn_weights = (
            fixed_softmax(attn_scores, dim=-1)
            if use_fixed_softmax
            else F.softmax(attn_scores, dim=-1)
        )
    else:
        attn_weights = normalize(attn_scores)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    out = attn_weights @ V
    return out, K, kv_positions


def cached_latent_attention(
    slot,
    q: torch.Tensor,
    new_latents: Dict[str, torch.Tensor],
    cache_position: Optional[int],
    scale: float,
    expand_fn,
    pe=None,
    pe_type: str = "",
    use_pe: bool = False,
    x_for_pe: Optional[torch.Tensor] = None,
    logical_layer_idx: int = 0,
    dropout=None,
    causal: bool = True,
    use_fixed_softmax: bool = False,
) -> torch.Tensor:
    """Generic cached latent attention (latent family, F1 recipe).

    Only the low-rank latent(s) are cached (MLA: "only the latent is
    cached", arXiv:2506.09342); expansion to full K/V happens per step via
    ``expand_fn`` and rotation PEs apply post-expansion with the absolute
    offset (this repo's MLA variant applies RoPE to the decompressed q/k,
    so DeepSeek's absorb trick is unavailable — documented cost).

    Args:
        slot: The layer's :class:`~src.model.cache.slots.LatentKVSlot`.
        q: Queries ``(B, H, S, D)``.
        new_latents: Named latent tensors for the new tokens, e.g.
            ``{"c_kv": (B, S, r)}`` (or GTA's ``{"k_g": ..., "v_lat": ...}``).
        cache_position: Absolute index of the first incoming token.
        scale: Score scaling (``head_dim ** -0.5``).
        expand_fn: Callable mapping the concatenated latent dict to
            ``(k, v)`` of shapes ``(B, H, T, D)``.
        pe / pe_type / use_pe / x_for_pe / logical_layer_idx: PE plumbing
            (rotation applied with the absolute offset).
        dropout: Optional attention-weight dropout module.
        causal: Apply the position-derived causal mask.
        use_fixed_softmax: Use the cancellation-free softmax of
            arXiv:2610.31642 (mirrors the mixer's ``use_fixed_softmax``).

    Returns:
        Attended values ``(B, H, S, D)`` (pre out-projection).
    """
    import torch.nn.functional as F

    from ..attention.common import apply_pe_to_qk, apply_pe_to_scores

    offset = cache_position
    positions = positions_for(offset, q.size(-2), q.device)
    cached = slot.read()
    c_all: Dict[str, torch.Tensor] = {}
    for name, tensor in new_latents.items():
        prev = cached.get(name)
        if prev is not None:
            dim = 2 if tensor.dim() >= 4 else 1
            c_all[name] = torch.cat([prev, tensor], dim=dim)
        else:
            c_all[name] = tensor
    slot.append(positions=positions, **new_latents)
    cached_pos = cached.get("positions")
    if cached_pos is not None and cached_pos.numel():
        kv_positions = torch.cat([cached_pos, positions], dim=0)
    else:
        kv_positions = positions

    k, v = expand_fn(c_all)
    q, k = apply_pe_to_qk(pe, pe_type, q, k, x_for_pe, logical_layer_idx, use_pe, position_offset=offset)

    attn_scores = (q @ k.transpose(-2, -1)) * scale
    attn_scores = apply_pe_to_scores(
        pe,
        pe_type,
        attn_scores,
        q,
        use_pe,
        kv_len=k.size(-2),
        q_offset=0 if offset is None else int(offset),
        kv_positions=kv_positions,
    )
    if causal:
        mask = causal_mask_from_positions(positions, kv_positions)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_weights = (
        fixed_softmax(attn_scores, dim=-1)
        if use_fixed_softmax
        else F.softmax(attn_scores, dim=-1)
    )
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return attn_weights @ v
