"""Group-wise int4 KIVI-style codec with a protected full-precision window.

Combines KIVI group-wise 4-bit quantization (arXiv:2402.02750) with the
two-region protection pattern of StreamingLLM sinks (arXiv:2309.17453)
and SKVQ (arXiv:2405.06219):

* Signed 4-bit levels ``[-7, 7]``; ``group_size`` tokens per group along T
  (default 64); granularity axes identical to int8 (K per-channel-per-
  group, V per-token).
* **Protected region**: the most recent ``protected_window`` tokens plus
  ``num_sinks`` leading tokens are kept full precision — never quantized,
  never evicted (KVQuant dense-and-sparse outliers, arXiv:2401.18079;
  KVSink, arXiv:2508.04257). Reading concatenates
  ``[quantized history | fp protected window]``.

The full-precision window itself lives in the slot
(:class:`~src.model.cache.slots.KVSlot` manages the append/flush cycle
through the ``groupwise`` flag); this codec supplies the group-wise
quantizer and the region parameters. ``group_size`` may be ``1`` for
per-token int4 (append-time quantization with no history grouping).
"""

from __future__ import annotations

import torch

from ..base import QAxis, SlotMeta
from .int8 import Int8Codec

__all__ = ["Kivi4Codec"]

_LEVELS = {
    2: 1,  # 2-bit signed levels [-1, 1]
    4: 7,  # 4-bit signed levels [-7, 7]
    8: 127,
}


class Kivi4Codec(Int8Codec):
    """Group-wise low-bit codec with a protected fp window.

    Args:
        k_bits: Bits for key streams (2, 4 or 8; default 4).
        v_bits: Bits for value streams (2, 4 or 8; default 4).
        group_size: Tokens per quantization group along T (default 64).
        protected_window: Most recent tokens kept in full precision
            (default 32).
        num_sinks: Leading sink tokens kept in full precision and pinned
            against eviction (default 4). Applied by the slot.

    Attributes:
        name: Codec identifier (``"int4_kivi"``).
        groupwise: Always True — declares the two-region storage to slots.
    """

    name = "int4_kivi"
    groupwise = True

    def __init__(
        self,
        k_bits: int = 4,
        v_bits: int = 4,
        group_size: int = 64,
        protected_window: int = 32,
        num_sinks: int = 4,
    ):
        if k_bits not in _LEVELS or v_bits not in _LEVELS:
            raise ValueError(f"k_bits/v_bits must be one of {sorted(_LEVELS)}")
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.group_size = max(1, int(group_size))
        self.protected_window = max(0, int(protected_window))
        self.num_sinks = max(0, int(num_sinks))

    def encode_write(self, name: str, tensor: torch.Tensor, axis: QAxis, meta: SlotMeta) -> dict:
        """Group-wise quantize one block (a complete group or a decode step).

        Keys (and, by convention, every stream whose quantization axis is
        ``"channel"``) use ``k_bits``; token-axis streams use ``v_bits``.
        """
        qmax = _LEVELS[self.k_bits if axis == "channel" else self.v_bits]
        x = tensor.to(torch.float32)
        if axis == "channel":
            scale = x.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8) / qmax
        else:
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.round(x / scale).clamp(-qmax, qmax)
        # Store as int8 container (values fit any supported bit width).
        return {
            "q": q.to(torch.int8),
            "scale": scale.to(meta.dtype),
            "axis": axis,
            "dtype": meta.dtype,
            "qmax": qmax,
        }

    def decode_read(self, stored: dict, meta: SlotMeta) -> torch.Tensor:
        del meta
        return stored["q"].to(stored["dtype"]) * stored["scale"].to(stored["dtype"])

    def update(self, stored: dict, new_stored: dict, meta: SlotMeta) -> dict:
        """Merge by dequant -> concat -> requant within one group."""
        a = self.decode_read(stored, meta).to(torch.float32)
        b = self.decode_read(new_stored, meta).to(torch.float32)
        dim = 2 if a.dim() >= 3 else 1
        merged = torch.cat([a, b], dim=dim)
        return self.encode_write("k", merged, stored.get("axis", "token"), meta)
