"""Central cache manager: :class:`FrankensteinCache`.

A HF ``past_key_values``-style manager (vLLM-inspired slot abstraction,
D1) created by the inference script and threaded through
``decoder → encoder → HybridLayer → mixer`` as an optional keyword
argument. It owns:

* the per-layer slot dict keyed by ``logical_layer_idx`` (D6 — correct
  under ``num_loops``/Hyperloop where the same physical layer executes
  with distinct logical roles);
* the global token tail (FIFO of recent token ids for token-level
  consumers such as ``engram_attn``);
* the write-time codec (quantization) and the read-time policy
  (eviction), built from the resolved :class:`InferenceConfig`;
* the per-mixer enable/disable overrides;
* the ``force_dense_prefill`` flag used by Mixture-of-Depths (D8).

Slot creation is lazy and declarative: :meth:`resolve_slot` consults
:class:`~src.model.cache.capabilities.CACHE_SPECS` (plus the mixer's
optional ``make_cache_slot`` hook for mixer-specific layouts) on first use
per logical layer. Anything unsupported or unimplemented receives
``cache=None`` and runs the legacy path — **inference never fails because
of the cache** (D5).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import CacheSlot, SlotMeta
from .capabilities import is_cacheable, spec_for
from .compress import build_cache_policy
from .hooks import build_slot_meta
from .quant import build_cache_codec
from .slots import KVSlot, LatentKVSlot, RingSlot, StateSlot, TokenTailSlot

__all__ = ["FrankensteinCache"]


class FrankensteinCache:
    """Central KV-cache manager for one generation run.

    Attributes:
        config: The :class:`~src.model.config.FrankensteinModelConfig`.
        inference_config: The resolved
            :class:`~src.deploy.inference_config.InferenceConfig` (or
            ``None`` for default off-quant/off-compress behavior).
        slots: Mapping ``logical_layer_idx -> CacheSlot``.
        token_capacity: Capacity of the token tail FIFO.
        codec: Active quantization codec (``None`` = full precision).
        policy: Active compression policy (``None`` = keep everything).
    """

    def __init__(
        self,
        config,
        inference_config=None,
        device: Optional[torch.device] = None,
    ):
        """Build the manager. Prefer the :meth:`new` classmethod."""
        self.config = config
        self.inference_config = inference_config
        self.device = torch.device(device) if device is not None else None
        self.slots: Dict[int, CacheSlot] = {}
        self.token_capacity = 16
        self._token_tail = TokenTailSlot(capacity=self.token_capacity)
        self._position = 0
        # An explicitly constructed cache is active by default; when an
        # InferenceConfig is supplied, its ``kv_cache_enabled`` flag governs
        # (opt-in D4 lives at the config layer, the object itself is the
        # mechanism layer).
        if inference_config is not None:
            self._active = bool(getattr(inference_config, "kv_cache_enabled", True))
        else:
            self._active = True

        quant = inference_config
        sink_count = int(getattr(quant, "kv_num_sinks", 4) or 0) if quant is not None else 0
        if self._active:
            self.codec = build_cache_codec(
                getattr(quant, "kv_quant_method", "none") or "none",
                k_bits=int(getattr(quant, "kv_k_bits", 4)),
                v_bits=int(getattr(quant, "kv_v_bits", 4)),
                group_size=int(getattr(quant, "kv_group_size", 64)),
                protected_window=int(getattr(quant, "kv_protected_window", 32)),
                num_sinks=sink_count,
            )
            self.policy = build_cache_policy(
                getattr(quant, "kv_compress_method", "none") or "none",
                num_sinks=sink_count,
                window=int(getattr(quant, "kv_window", 512)),
                budget=getattr(quant, "kv_budget", None),
                per_layer_budgets=list(getattr(quant, "kv_per_layer_budgets", []) or []),
            )
            self.per_mixer_enabled: Dict[str, bool] = {}
            for entry in getattr(quant, "kv_per_mixer", []) or []:
                if isinstance(entry, dict) and "mixer" in entry:
                    self.per_mixer_enabled[str(entry["mixer"])] = bool(entry.get("enabled", True))
        else:
            self.codec = None
            self.policy = None
            self.per_mixer_enabled = {}

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def new(cls, model, inference_config=None, device: Optional[torch.device] = None) -> "FrankensteinCache":
        """Build a cache for ``model`` (any Frankenstein model class).

        Args:
            model: Model exposing ``.config``
                (:class:`~src.model.config.FrankensteinModelConfig`).
            inference_config: Resolved
                :class:`~src.deploy.inference_config.InferenceConfig`.
            device: Device for the token tail (resolved lazily otherwise).
        """
        return cls(getattr(model, "config", model), inference_config, device)

    @property
    def active(self) -> bool:
        """Whether the cache is enabled (opt-in D4)."""
        return self._active

    @property
    def force_dense_prefill(self) -> bool:
        """Mixture-of-Depths runs the dense path at prefill when caching
        (D8: ragged caches are out of scope for v1)."""
        return self._active

    @property
    def length(self) -> int:
        """Total cached positions so far."""
        return self._position

    def next_position(self) -> int:
        """Absolute index of the next token to write."""
        return self._position

    def advance(self, num_tokens: int = 1) -> None:
        """Advance the write cursor after a forward that consumed tokens."""
        self._position += int(num_tokens)

    # ------------------------------------------------------------------ #
    # Token tail (token-level consumers, e.g. engram_attn)
    # ------------------------------------------------------------------ #
    def push_tokens(self, token_ids: torch.Tensor) -> None:
        """Append the latest token ids ``(B, T)`` to the token tail FIFO."""
        if token_ids is not None:
            self._token_tail.append(token_ids)

    @property
    def token_tail(self) -> Optional[torch.Tensor]:
        """Most recent token ids ``(B, T')`` or ``None`` when empty."""
        return self._token_tail.read()

    # ------------------------------------------------------------------ #
    # Slot resolution (called by HybridLayer)
    # ------------------------------------------------------------------ #
    def resolve_slot(
        self,
        logical_layer_idx: int,
        layer,
    ) -> Optional[CacheSlot]:
        """Return the cache slot for one logical layer, or ``None``.

        The layer receives ``None`` whenever caching does not apply: the
        cache is inactive, a ``per_mixer`` override disabled the mixer,
        the mixer's spec says unsupported, the incremental branch is not
        implemented, or the layer's PE configuration is PaPE (D10/D11).
        Never raises (D5).

        Args:
            logical_layer_idx: Global logical layer index (slot identity).
            layer: The :class:`~src.model.hybrid_layer.HybridLayer` asking
                for its slot (reads ``layer_type`` / ``mixer``).
        """
        if not self._active:
            return None
        layer_type = str(getattr(layer, "layer_type", ""))
        enabled = self.per_mixer_enabled.get(layer_type, True)
        if not enabled or not is_cacheable(layer_type):
            return None
        spec = spec_for(layer_type)
        if logical_layer_idx in self.slots:
            return self.slots[logical_layer_idx]

        config = self.config
        # PaPE-configured layers fall back per-layer (D10/D11): the
        # content-dependent q/k augmentation has unproven caching exactness.
        pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        use_pe = getattr(config, f"{layer_type}_use_pe", True)
        if pe_type in ("pape", "pape_efficient", "pape_ri") and use_pe:
            return None

        num_kv_heads = int(getattr(layer.mixer, "num_kv_heads", 0)) or None
        meta = build_slot_meta(layer.mixer, config, logical_layer_idx, num_kv_heads)
        meta.mixer_key = layer_type
        if self.device is not None:
            meta.dtype = torch.float32

        # Mixer-specific layout hook (declared on the mixer class).
        factory = getattr(layer.mixer, "make_cache_slot", None)
        if factory is not None:
            try:
                slot = factory(meta, self)
            except Exception as exc:  # D5: never raise — but make it visible
                import warnings

                warnings.warn(
                    f"cache slot factory failed for mixer '{layer_type}' "
                    f"({type(exc).__name__}: {exc}); running uncached."
                )
                return None
        else:
            slot = self._default_slot(spec, meta)
        if slot is None:
            return None
        self.slots[logical_layer_idx] = slot
        return slot

    def _default_slot(self, spec, meta: SlotMeta) -> Optional[CacheSlot]:
        """Build the default slot for a spec (mixers may override via the
        ``make_cache_slot`` hook)."""
        kind = spec.slot_type
        sinks = 0
        if self.policy is not None and getattr(self.policy, "name", "") == "window_sinks":
            sinks = int(getattr(self.policy, "num_sinks", 0))
        if self.codec is not None and getattr(self.codec, "groupwise", False):
            sinks = max(sinks, int(getattr(self.codec, "num_sinks", 0)))
        if kind == "kv":
            return KVSlot(
                meta,
                codec=self.codec,
                policy=self.policy,
                num_sinks=sinks,
                protected_window=int(getattr(self.codec, "protected_window", 0)) if self.codec is not None else 0,
                group_size=int(getattr(self.codec, "group_size", 64)) if self.codec is not None else 64,
            )
        if kind == "ring":
            capacity = None
            if self.policy is not None and hasattr(self.policy, "budget"):
                capacity = self.policy.budget(meta.logical_layer_idx)
            if capacity is None:
                capacity = 512
            return RingSlot(meta, capacity=capacity, num_sinks=sinks, codec=self.codec, policy=self.policy)
        if kind == "state":
            return StateSlot(meta)
        if kind == "latent_kv":
            # Heterogeneous layouts are mixer-specific; without the mixer
            # hook we cannot know the stream axes — fall back (D5).
            return None
        if kind == "token_tail":
            # Manager-level: consumers read `cache.token_tail` directly.
            return None
        return None

    # ------------------------------------------------------------------ #
    # Inspection / lifecycle
    # ------------------------------------------------------------------ #
    def slot(self, logical_layer_idx: int) -> Optional[CacheSlot]:
        """Return an already-created slot (or ``None``)."""
        return self.slots.get(logical_layer_idx)

    def codec_slot_kwargs(self) -> Dict[str, Any]:
        """Common codec/policy kwargs for mixer-defined ``make_cache_slot``
        hooks (quantization regions, sinks, policy)."""
        sinks = 0
        if self.policy is not None and getattr(self.policy, "name", "") == "window_sinks":
            sinks = int(getattr(self.policy, "num_sinks", 0))
        if self.codec is not None and getattr(self.codec, "groupwise", False):
            sinks = max(sinks, int(getattr(self.codec, "num_sinks", 0)))
        return {
            "codec": self.codec,
            "policy": self.policy,
            "num_sinks": sinks,
            "protected_window": int(getattr(self.codec, "protected_window", 0)) if self.codec is not None else 0,
            "group_size": int(getattr(self.codec, "group_size", 64)) if self.codec is not None else 64,
        }

    def total_tokens(self) -> int:
        """Maximum cached position count across position-indexed slots."""
        counts = [s.length for s in self.slots.values() if getattr(s, "kind", "") in ("kv", "ring", "latent_kv")]
        return max(counts) if counts else self._position

    def truncate(self, num_tokens: int) -> None:
        """Roll the cache back to ``num_tokens`` positions (LayerSkip-style
        verification, arXiv:2404.16710). State slots are no-ops."""
        for slot in self.slots.values():
            slot.truncate(num_tokens)
        self._position = min(self._position, int(num_tokens))

    def to(self, device: torch.device) -> None:
        """Move every slot and the token tail to ``device``."""
        self.device = torch.device(device)
        for slot in self.slots.values():
            slot.to(device)
        self._token_tail.to(device)

    def reset(self) -> None:
        """Drop all slots and restart the cursor (new generation)."""
        self.slots.clear()
        self._token_tail = TokenTailSlot(capacity=self.token_capacity)
        self._position = 0
