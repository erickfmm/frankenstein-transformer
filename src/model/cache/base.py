"""Core protocols and metadata for the KV-cache subsystem.

This module defines the typed contracts shared by every cache component:

* :class:`SlotMeta` — static per-slot description fixed at slot creation.
* :class:`CacheCodec` — lossless/lossy transform applied at slot write/read
  time (quantization: ``none``, ``int8``, ``int4_kivi``).
* :class:`CachePolicy` — eviction/compression policy operating on a slot
  across decode steps (``none``, ``window_sinks``).
* :class:`CacheSlot` — per-layer storage (``KVSlot``, ``LatentKVSlot``,
  ``StateSlot``, ``RingSlot``, ``ConvTailSlot``, ``TokenTailSlot``).

The layering follows the KV-cache-management literature: quantization and
normalization are **write-time** transforms (KIVI, arXiv:2402.02750;
KVQuant, arXiv:2401.18079) while sparsity and eviction are **read-time**
transforms (Quest, arXiv:2406.10774; StreamingLLM, arXiv:2309.17453).
Real systems keep them orthogonal; this package exposes both, per slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, runtime_checkable

import torch

QAxis = Literal["channel", "token"]
"""Quantization granularity axis.

KIVI (arXiv:2402.02750) shows that keys carry persistent per-CHANNEL
outliers (a RoPE-frequency structure) while values carry per-TOKEN
outliers, so keys are quantized per channel(-group) and values per token.
"""


@dataclass
class SlotMeta:
    """Static per-slot description, fixed at slot creation from config + mixer.

    Attributes:
        mixer_key: Schema mixer enum value owning the slot (e.g.
            ``"standard_attn"``).
        logical_layer_idx: Global logical layer index (0 .. num_layers *
            num_loops - 1). The slot identity key (D6).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA); equals ``num_heads``
            for MHA-style mixers.
        head_dim: Per-head dimensionality.
        dtype: Floating dtype used for full-precision storage (post
            dequantization the tensors are cast back to this dtype).
        pe_type: Positional-encoding type of the owning layer (``"rope"``,
            ``"hope"``, ``"alibi"``, ...). Visibility for codecs: post-RoPE
            storage works with per-channel K quantization, while pre-RoPE
            or codebook schemes (future codecs) would need different
            treatment (see §9.6 rule 3 of the KV-cache plan).
        rope_scaling: RoPE scaling factor versioned at slot creation. Keys
            are stored POST-RoPE (D7); if the schedule changes at runtime
            (YaRN-style) already-rotated keys are invalid. The manager
            detects the mismatch and raises a configuration error at
            generation start — never mid-decode.
    """

    mixer_key: str
    logical_layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    pe_type: str = "rope"
    rope_scaling: float = 1.0


@runtime_checkable
class CacheCodec(Protocol):
    """Lossless (``none``) or lossy transform applied at slot WRITE/READ time.

    Contract:

    * ``encode_write``/``decode_read`` are pure with respect to the slot.
    * ``update`` appends already-encoded rows (dequant -> concat -> requant
      semantics for lossy codecs).
    * Codecs must never raise on unknown tensors: on any internal error
      they fall back to identity and record a one-time warning.
    * Recurrent states are NOT quantized in v1 (D12 of the plan) — the
      literature is thinnest there; the fallback to fp is automatic
      because state slots simply do not invoke a codec.
    """

    name: str

    def encode_write(self, name: str, tensor: torch.Tensor, axis: QAxis, meta: SlotMeta) -> Any:
        """Encode ``tensor`` for storage. Returns an opaque stored payload."""
        ...

    def decode_read(self, stored: Any, meta: SlotMeta) -> torch.Tensor:
        """Dequantize a stored payload back to a floating tensor."""
        ...

    def update(self, stored: Any, new_stored: Any, meta: SlotMeta) -> Any:
        """Merge two stored payloads along the token dimension."""
        ...


@runtime_checkable
class CachePolicy(Protocol):
    """Eviction / compression policy operating on a slot across decode steps.

    Policies apply only to position-indexed slots (``KVSlot``, ``RingSlot``,
    ``LatentKVSlot``). ``StateSlot``\\ s are constant-size — nothing to
    evict. Compression policies never run on ``mtla_attn`` slots (D13):
    MTLA manages its own dynamic merging schedule internally.
    """

    name: str

    def on_append(self, slot: "CacheSlot", cache_position: int) -> None:
        """Called after ``slot.append`` (may evict immediately)."""
        ...

    def on_step_end(self, slot: "CacheSlot", cache_position: int) -> None:
        """Called by the manager at the end of each decode step."""
        ...

    def budget(self, logical_layer_idx: int) -> Optional[int]:
        """Max retained tokens for this layer; ``None`` = unlimited."""
        ...


@runtime_checkable
class CacheSlot(Protocol):
    """Per-layer cache storage.

    Attributes:
        kind: Slot type discriminator (``"kv"``, ``"latent_kv"``, ``"state"``,
            ``"ring"``, ``"conv_tail"``, ``"token_tail"``).
        meta: The :class:`SlotMeta` fixed at creation.
        length: Number of cached positions (0 for state slots).
    """

    kind: str
    meta: SlotMeta
    length: int

    def append(self, **tensors: torch.Tensor) -> None:
        """Append new tensors (per-mixer schema, e.g. ``k=``, ``v=``, ``positions=``)."""
        ...

    def read(self) -> Dict[str, Any]:
        """Return dequantized views of the stored tensors for the mixer."""
        ...

    def truncate(self, num_tokens: int) -> None:
        """Cheap rollback: drop positions beyond a token boundary
        (LayerSkip-style verification, arXiv:2404.16710). No-op for
        position-less slots (states)."""
        ...

    def to(self, device: torch.device) -> None:
        """Move every stored tensor to ``device``."""
        ...


__all__ = [
    "QAxis",
    "SlotMeta",
    "CacheCodec",
    "CachePolicy",
    "CacheSlot",
]
