"""Identity (full-precision passthrough) cache codec.

Baseline for exactness tests (T2/T3): stores tensors cast to the slot
dtype and returns them untouched. Round-trip is bit-exact.
"""

from __future__ import annotations

import torch

from ..base import QAxis, SlotMeta

__all__ = ["NoneCodec"]


class NoneCodec:
    """Lossless passthrough codec (fp16/bf16/fp32 from :class:`SlotMeta`).

    Attributes:
        name: Codec identifier (``"none"``).
    """

    name = "none"
    groupwise = False

    def encode_write(self, name: str, tensor: torch.Tensor, axis: QAxis, meta: SlotMeta) -> torch.Tensor:
        """Store ``tensor`` as-is (cast to the slot dtype)."""
        del name, axis, meta
        return tensor

    def decode_read(self, stored: torch.Tensor, meta: SlotMeta) -> torch.Tensor:
        """Return the stored tensor unchanged."""
        del meta
        return stored

    def update(self, stored: torch.Tensor, new_stored: torch.Tensor, meta: SlotMeta) -> torch.Tensor:
        """Concatenate two stored payloads along the token dimension."""
        del meta
        dim = 2 if stored.dim() >= 3 else 1
        return torch.cat([stored, new_stored], dim=dim)
