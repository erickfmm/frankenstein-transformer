"""Symmetric int8 cache codec with asymmetric granularity axes (KIVI).

Follows KIVI (arXiv:2402.02750): keys have persistent per-CHANNEL outliers
(a RoPE-frequency structure) while values have per-TOKEN outliers, so

* K (``axis="channel"``) is quantized per (head, channel) within each
  stored token block — ``s_{h,c} = max_t |K[t,h,c]| / 127``;
* V (``axis="token"``) is quantized per (token, head) —
  ``s_{t,h} = max_c |V[t,h,c]| / 127``.

Each stored block carries its own scales (a block is a token group);
token-by-token decode blocks therefore get exact per-token scales, keeping
the relative rounding error bounded by 1/127 (~0.8%) on any block size.
"""

from __future__ import annotations

import torch

from ..base import QAxis, SlotMeta

__all__ = ["Int8Codec"]

_QMAX = 127


class Int8Codec:
    """Symmetric int8 codec with per-axis scale granularity.

    Attributes:
        name: Codec identifier (``"int8"``).
    """

    name = "int8"
    groupwise = False

    def encode_write(self, name: str, tensor: torch.Tensor, axis: QAxis, meta: SlotMeta) -> dict:
        """Quantize ``tensor`` symmetrically to int8.

        Args:
            name: Stream name (``"k"``, ``"v"``, latent names, ...).
            tensor: Floating tensor ``(B, H, T, D)`` or ``(B, T, R)``.
            axis: Granularity — ``"channel"`` computes scales per
                (head, channel) broadcast over T; ``"token"`` computes
                scales per (token, head) broadcast over D.
            meta: Slot metadata (dtype used for dequantized output).

        Returns:
            Stored payload ``{"q": int8 tensor, "scale": float tensor,
            "axis": axis, "dtype": meta.dtype}``.
        """
        del name
        x = tensor.to(torch.float32)
        if axis == "channel":
            # Scales per (..., channel): reduce over the token dim (dim -2).
            scale = x.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8) / _QMAX
        else:
            # Scales per (..., token): reduce over the channel dim (dim -1).
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / _QMAX
        q = torch.round(x / scale).clamp(-_QMAX, _QMAX).to(torch.int8)
        return {
            "q": q,
            "scale": scale.to(meta.dtype),
            "axis": axis,
            "dtype": meta.dtype,
        }

    def decode_read(self, stored: dict, meta: SlotMeta) -> torch.Tensor:
        """Dequantize ``K_hat * scale`` back to the slot dtype."""
        del meta
        return stored["q"].to(stored["dtype"]) * stored["scale"].to(stored["dtype"])

    def update(self, stored: dict, new_stored: dict, meta: SlotMeta) -> dict:
        """Merge by dequant -> concat -> requant (keeps bounded error)."""
        a = self.decode_read(stored, meta).to(torch.float32)
        b = self.decode_read(new_stored, meta).to(torch.float32)
        dim = 2 if a.dim() >= 3 else 1
        merged = torch.cat([a, b], dim=dim)
        return self.encode_write("k", merged, stored.get("axis", "token"), meta)
