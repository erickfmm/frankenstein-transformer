"""KV-cache subsystem for incremental decoding.

Public exports (plan §5.1): the central :class:`FrankensteinCache`
manager, the declarative :data:`CACHE_SPECS` capability registry, the
typed slots, and the quantization/compression registries.

Typical use (inference script)::

    from src.model.cache import FrankensteinCache

    cache = FrankensteinCache.new(model, inference_config)
    out = decoder.generate(input_ids, use_cache=cache.active,
                           past_key_values=cache)
"""

from .base import CacheCodec, CachePolicy, CacheSlot, QAxis, SlotMeta
from .capabilities import CACHE_SPECS, CacheSpec, IMPLEMENTED_MIXERS, is_cacheable, spec_for
from .manager import FrankensteinCache
from .slots import (
    ConvTailSlot,
    KVSlot,
    LatentKVSlot,
    RingSlot,
    StateSlot,
    TokenTailSlot,
)
from . import compress, hooks, quant

__all__ = [
    "FrankensteinCache",
    "CacheSpec",
    "CACHE_SPECS",
    "IMPLEMENTED_MIXERS",
    "is_cacheable",
    "spec_for",
    "CacheSlot",
    "CacheCodec",
    "CachePolicy",
    "QAxis",
    "SlotMeta",
    "KVSlot",
    "LatentKVSlot",
    "StateSlot",
    "RingSlot",
    "ConvTailSlot",
    "TokenTailSlot",
    "compress",
    "hooks",
    "quant",
]
