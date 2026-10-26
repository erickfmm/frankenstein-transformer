"""KV-cache quantization codecs.

Quantization is a **write-time** transform (literature-mandated layering:
KIVI arXiv:2402.02750, KVQuant arXiv:2401.18079): slots encode on append
and dequantize on read. Codecs are orthogonal to mixers — the same codec
serves ``KVSlot`` and ``LatentKVSlot`` (latent quantized per-token,
``k_rope`` per-channel). Recurrent states are NOT quantized in v1 (D12).

Registry (v1, decision D3):

* ``none`` — full-precision passthrough (baseline for exactness tests).
* ``int8`` — symmetric int8, asymmetric axes (K per-channel, V per-token).
* ``int4_kivi`` — group-wise int4 (k_bits/v_bits configurable) with a
  protected full-precision window + pinned sinks.

Future codecs stay interface-ready but are intentionally NOT implemented
in v1: ``int2_kivi`` (KVarN, arXiv:2606.03458), ``palu_lowrank`` (Palu,
arXiv:2407.21118), ``commvq`` (RoPE-commutative codebooks, arXiv:2506.18879),
``qjl_sketch`` (QJL, arXiv:2406.03482), ``block_gtq`` (Block-GTQ,
arXiv:2606.24033). Unknown names raise ``ValueError`` from
:func:`build_cache_codec` — a *configuration* error at generation start,
never a mid-decode failure.
"""

from __future__ import annotations

from typing import Optional

from ..base import CacheCodec
from .int8 import Int8Codec
from .kivi4 import Kivi4Codec
from .none import NoneCodec

__all__ = [
    "NoneCodec",
    "Int8Codec",
    "Kivi4Codec",
    "CACHE_CODECS",
    "build_cache_codec",
]

#: Codec registry (v1: D3). Future plugin names are listed in the module
#: docstring; adding one is a matter of appending a class + entry here.
CACHE_CODECS = {
    "none": NoneCodec,
    "int8": Int8Codec,
    "int4_kivi": Kivi4Codec,
}


def build_cache_codec(
    method: str,
    k_bits: int = 4,
    v_bits: int = 4,
    group_size: int = 64,
    protected_window: int = 32,
    num_sinks: int = 4,
) -> Optional[CacheCodec]:
    """Build a cache codec from resolved inference configuration.

    Args:
        method: One of :data:`CACHE_CODECS` keys (``"none"`` disables
            encoding but still returns a codec so slots have a uniform
            path; use ``None`` (the argument) for codec-free slots).
        k_bits: Bits for key streams (``int4_kivi`` only).
        v_bits: Bits for value streams (``int4_kivi`` only).
        group_size: Tokens per quantization group (``int4_kivi`` only).
        protected_window: Recent tokens kept full precision (``int4_kivi``).
        num_sinks: Leading sink tokens pinned full precision (``int4_kivi``).

    Returns:
        A codec instance, or ``None`` when ``method is None`` (slots then
        store plain fp tensors).

    Raises:
        ValueError: If ``method`` is not a registered codec.
    """
    if method is None:
        return None
    key = str(method).lower()
    if key == "none":
        return NoneCodec()
    if key == "int8":
        return Int8Codec()
    if key == "int4_kivi":
        return Kivi4Codec(
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            protected_window=protected_window,
            num_sinks=num_sinks,
        )
    raise ValueError(
        f"Unknown kv-cache quantization method {method!r}; "
        f"supported: {sorted(CACHE_CODECS)}"
    )
