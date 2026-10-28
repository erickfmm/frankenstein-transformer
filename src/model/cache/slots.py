"""Typed per-layer cache slots.

Every slot stores the state of one logical layer (identity key:
``logical_layer_idx``) across decode steps. Six slot kinds cover the 45
dispatchable layer types (see ``capabilities.py``):

* :class:`KVSlot` — post-RoPE K/V (+ per-token ``extras`` metadata) for
  quadratic softmax mixers.
* :class:`LatentKVSlot` — heterogeneous named streams (latent ``c_kv``,
  optional decoupled ``k_rope``) for the latent family.
* :class:`StateSlot` — fixed-size recurrent matrix updated in place.
* :class:`RingSlot` — capacity-bounded K/V ring buffer with pinned sinks
  (sliding-window family / ``window_sinks`` policy).
* :class:`ConvTailSlot` — FIFO of the last ``kernel - 1`` pre-conv rows.
* :class:`TokenTailSlot` — FIFO of the last ``n`` token ids.

Quantization hooks live on the write path (:meth:`KVSlot.append`) and the
read path (:meth:`KVSlot.read`) via a
:class:`~src.model.cache.base.CacheCodec`; eviction policies are invoked
on append / step end via a :class:`~src.model.cache.base.CachePolicy`.

Storage layout (``KVSlot`` / ``RingSlot`` / ``LatentKVSlot``): an ordered
list of *parts*. A part is either full precision (``fp``) or an encoded
payload produced by the codec. Pinned sink parts (``protected=True``) are
never evicted and — for group-wise codecs — never quantized (KVQuant
dense-and-sparse outliers, arXiv:2401.18079; KVSink, arXiv:2508.04257).
Reading concatenates ``[sinks | history | protected window]``.

v1 keeps decode+concat-per-step simple; a persistent decoded buffer
invalidated on append (and fused quantized-attention kernels à la
KIVI/QJL) is documented future work — do NOT build kernels in v1.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import QAxis, SlotMeta

__all__ = [
    "KVSlot",
    "LatentKVSlot",
    "StateSlot",
    "RingSlot",
    "ConvTailSlot",
    "TokenTailSlot",
]


def _detach_cast(t: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Detach a tensor from the autograd graph (caches are inference-only)."""
    out = t.detach()
    if dtype is not None and out.dtype != dtype and out.is_floating_point():
        out = out.to(dtype)
    return out


def _tdim(t: torch.Tensor) -> int:
    """Index of the token dimension.

    ``(B, H, T, D)`` streams (K/V, grouped keys) carry T at dim 2;
    ``(B, T, R)`` latent streams carry T at dim 1.
    """
    if t.dim() >= 4:
        return 2
    if t.dim() == 3:
        return 1
    return 0


def _token_count(t: torch.Tensor) -> int:
    return int(t.shape[_tdim(t)])


def _slice_tokens(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
    d = _tdim(t)
    return t[:, :, start:end] if d == 2 else t[:, start:end]


def _slice_extras(t: torch.Tensor, start: int, end: int, dim: int) -> torch.Tensor:
    """Slice an extras tensor along its (normalized) token axis."""
    if dim < 0:
        dim = t.dim() + dim
    return t[:, :, start:end] if dim == 2 else t[:, start:end] if dim == 1 else t[..., start:end]


def _extras_count(t: torch.Tensor, dim: int) -> int:
    """Token count of an extras tensor along its (normalized) token axis."""
    if dim < 0:
        dim = t.dim() + dim
    return int(t.shape[dim])


class _Part:
    """One contiguous token block inside a position-indexed slot."""

    __slots__ = ("fp", "protected", "n", "positions", "streams", "extras")

    def __init__(
        self,
        n: int,
        positions: torch.Tensor,
        streams: Dict[str, Any],
        fp: bool,
        protected: bool = False,
        extras: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.fp = fp
        self.protected = protected
        self.n = n
        self.positions = positions
        self.streams = streams
        self.extras = extras or {}


class _PositionedSlotBase:
    """Shared machinery for position-indexed slots (KV / latent / ring).

    Implements the parts list, decode-on-read, truncation, oldest-first
    eviction with protected sinks, and device movement. Sub-classes declare
    their per-stream quantization axes and fp-region semantics.
    """

    kind = "kv"

    def __init__(
        self,
        meta: SlotMeta,
        axes: Dict[str, QAxis],
        codec=None,
        policy=None,
        num_sinks: int = 0,
        protected_window: int = 0,
        group_size: int = 64,
    ):
        self.meta = meta
        self.axes = dict(axes)
        self.codec = codec
        self.policy = policy
        # FP-region configuration (only meaningful for group-wise codecs
        # that declare a protected window; see Kivi4Codec).
        self.groupwise = bool(getattr(codec, "groupwise", False))
        self.fp_prefix = int(num_sinks) if (self.groupwise or num_sinks) else 0
        self.fp_suffix = int(protected_window) if self.groupwise else 0
        self.group_size = max(1, int(group_size)) if self.groupwise else 0
        self.num_sinks = int(num_sinks)
        self.extras_tdim = -1
        self._parts: List[_Part] = []
        self._sinks_filled = 0
        # Full-precision protected window (single concatenated tensors).
        self._tail_streams: Optional[Dict[str, torch.Tensor]] = None
        self._tail_positions: Optional[torch.Tensor] = None
        self._tail_extras: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def length(self) -> int:
        """Number of cached positions (sinks + history + protected window)."""
        total = sum(p.n for p in self._parts)
        if self._tail_positions is not None:
            total += int(self._tail_positions.shape[0])
        return int(total)

    @property
    def positions(self) -> torch.Tensor:
        """Concatenated absolute int64 positions of every cached token."""
        chunks = [p.positions for p in self._parts]
        if self._tail_positions is not None and self._tail_positions.numel():
            chunks.append(self._tail_positions)
        if not chunks:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def _encode_stream(self, name: str, tensor: torch.Tensor) -> Tuple[bool, Any]:
        """Encode one stream. Returns ``(is_fp, stored)``."""
        if self.codec is None:
            return True, _detach_cast(tensor, self.meta.dtype)
        try:
            stored = self.codec.encode_write(
                name, _detach_cast(tensor, self.meta.dtype), self.axes[name], self.meta
            )
            return False, stored
        except Exception:  # pragma: no cover - codec contract: never raise
            import warnings

            warnings.warn(
                f"cache codec {getattr(self.codec, 'name', '?')} failed on stream "
                f"'{name}'; falling back to full precision for this block."
            )
            return True, _detach_cast(tensor, self.meta.dtype)

    def _decode_stream(self, stored: Any) -> torch.Tensor:
        if isinstance(stored, torch.Tensor):
            return stored
        if self.codec is None:  # pragma: no cover - defensive
            raise TypeError("non-tensor stored payload without a codec")
        return self.codec.decode_read(stored, self.meta)

    def append(
        self,
        positions: torch.Tensor,
        extras: Optional[Dict[str, torch.Tensor]] = None,
        **streams: torch.Tensor,
    ) -> None:
        """Append new tokens.

        Args:
            positions: Absolute int64 positions of the new tokens, shape ``(T,)``.
            extras: Optional per-token metadata tensors (FoX cumulative
                log-forget sums, MTLA position ranges, ...). Stored in full
                precision; keyed by name.
            **streams: Named tensors following the slot's ``axes`` layout,
                e.g. ``k=(B,H,T,D), v=(B,H,T,D)`` for KV slots or
                ``c_kv=(B,T,r)`` for latent slots.
        """
        positions = _detach_cast(positions, torch.long)
        n_total = int(positions.shape[0])
        extras = extras or {}
        pos = 0
        # 1) Fill pinned full-precision sinks first.
        while self.fp_prefix and self._sinks_filled < self.fp_prefix and pos < n_total:
            take = min(self.fp_prefix - self._sinks_filled, n_total - pos)
            sl = slice(pos, pos + take)
            raw = {
                name: _slice_tokens(t, sl.start, sl.stop) for name, t in streams.items()
            }
            if self.groupwise:
                # Sinks stay full precision forever (KIVI/KVSink protected
                # slot classes, §9.6 rule 4).
                stored = {name: _detach_cast(t, self.meta.dtype) for name, t in raw.items()}
                fp = True
            else:
                stored = {}
                fp = True
                for name, t in raw.items():
                    is_fp, st = self._encode_stream(name, t)
                    stored[name] = st
                    fp = fp and is_fp
            self._parts.append(
                _Part(
                    take,
                    positions[sl],
                    stored,
                    fp=fp,
                    protected=True,
                    extras={
                        k: _slice_extras(v, sl.start, sl.stop, self.extras_tdim)
                        for k, v in extras.items()
                    },
                )
            )
            self._sinks_filled += take
            pos += take
        remaining = n_total - pos
        if remaining > 0:
            sl = slice(pos, n_total)
            chunk_positions = positions[sl]
            chunk_streams = {
                name: _slice_tokens(t, sl.start, sl.stop) for name, t in streams.items()
            }
            chunk_extras = {
                k: _slice_extras(v, sl.start, sl.stop, self.extras_tdim) for k, v in extras.items()
            }
            if self.groupwise:
                # Group-wise codecs (KIVI-style): keep a full-precision
                # protected window; flush completed groups into history.
                self._append_tail(chunk_streams, chunk_positions, chunk_extras)
                self._flush_groups()
            else:
                stored = {}
                fp_all = True
                for name, t in chunk_streams.items():
                    is_fp, st = self._encode_stream(name, t)
                    stored[name] = st
                    fp_all = fp_all and is_fp
                self._parts.append(
                    _Part(
                        remaining,
                        _detach_cast(chunk_positions, torch.long),
                        stored,
                        fp=fp_all,
                        extras={k: _detach_cast(v) for k, v in chunk_extras.items()},
                    )
                )
        if self.policy is not None and n_total:
            self.policy.on_append(self, int(positions[-1]))

    # ------------------------------------------------------------------ #
    # Group-wise (KIVI) protected window
    # ------------------------------------------------------------------ #
    def _append_tail(
        self,
        streams: Dict[str, torch.Tensor],
        positions: torch.Tensor,
        extras: Dict[str, torch.Tensor],
    ) -> None:
        if self._tail_positions is None or self._tail_positions.numel() == 0:
            self._tail_streams = {k: _detach_cast(v, self.meta.dtype) for k, v in streams.items()}
            self._tail_positions = _detach_cast(positions, torch.long)
            self._tail_extras = {k: _detach_cast(v) for k, v in extras.items()}
            return
        for name in list(self._tail_streams):
            self._tail_streams[name] = torch.cat(
                [self._tail_streams[name], _detach_cast(streams[name], self.meta.dtype)],
                dim=_tdim(self._tail_streams[name]),
            )
        self._tail_positions = torch.cat([self._tail_positions, _detach_cast(positions, torch.long)], dim=0)
        for k, v in extras.items():
            prev = self._tail_extras.get(k)
            self._tail_extras[k] = (
                _detach_cast(v) if prev is None else torch.cat([prev, _detach_cast(v)], dim=_tdim(prev))
            )

    def _flush_groups(self) -> None:
        """Move completed groups from the fp protected window into history.

        The trailing ``fp_suffix`` tokens stay full precision (never
        quantized, rule 4 of §9.6); the oldest chunk that completes a
        ``group_size`` group is quantized and moved to the history parts.
        Mirrors KIVI's residual window and avoids per-token scale
        recomputation. Prefill quantizes whole groups immediately.
        """
        if self._tail_positions is None:
            return
        tail_len = int(self._tail_positions.shape[0])
        flushable = tail_len - self.fp_suffix
        while flushable >= self.group_size:
            take = self.group_size
            stored = {}
            fp_all = True
            for name, t in self._tail_streams.items():
                chunk = _slice_tokens(t, 0, take)
                is_fp, st = self._encode_stream(name, chunk)
                stored[name] = st
                fp_all = fp_all and is_fp
            positions = self._tail_positions[:take]
            chunk_extras = {
                k: _slice_extras(v, 0, take, self.extras_tdim) for k, v in self._tail_extras.items()
            }
            self._parts.append(
                _Part(take, positions, stored, fp=fp_all, extras=chunk_extras)
            )
            # Consume the emitted rows from the front of the window.
            self._tail_streams = {
                name: _slice_tokens(t, take, _token_count(t)) for name, t in self._tail_streams.items()
            }
            self._tail_positions = self._tail_positions[take:]
            self._tail_extras = {
                k: _slice_extras(v, take, _extras_count(v, self.extras_tdim), self.extras_tdim) for k, v in self._tail_extras.items()
            }
            flushable -= take
        if self._tail_positions is not None and self._tail_positions.numel() == 0:
            self._tail_streams = None
            self._tail_positions = None
            self._tail_extras = {}

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def read(self) -> Dict[str, Any]:
        """Return dequantized views: ``{name: tensor, ..., 'positions': ...,
        'extras': {...}}``."""
        out: Dict[str, Any] = {}
        for name in self.axes:
            chunks = []
            for p in self._parts:
                t = p.streams.get(name)
                if t is None:
                    continue
                chunks.append(t if p.fp else self._decode_stream(t))
            if self._tail_streams is not None and name in self._tail_streams:
                chunks.append(self._tail_streams[name])
            if not chunks:
                continue
            dim = _tdim(chunks[0])
            out[name] = torch.cat(chunks, dim=dim) if len(chunks) > 1 else chunks[0]
        out["positions"] = self.positions
        merged: Dict[str, List[torch.Tensor]] = {}
        for p in self._parts:
            for k, v in p.extras.items():
                merged.setdefault(k, []).append(v)
        for k, v in self._tail_extras.items():
            merged.setdefault(k, []).append(v)
        # Extras layouts are mixer-defined (FoX c_j is (B, H, T)); concatenate
        # along the token axis declared by ``extras_tdim`` (default: last).
        out["extras"] = {
            k: (torch.cat(vs, dim=self.extras_tdim) if len(vs) > 1 else vs[0])
            for k, vs in merged.items()
        }
        return out

    # ------------------------------------------------------------------ #
    # Eviction / rollback / device
    # ------------------------------------------------------------------ #
    def evict_oldest(self, num_tokens: int) -> None:
        """Drop the oldest non-protected tokens (StreamingLLM eviction).

        Pinned sink parts are never evicted; the trailing fp protected
        window is preserved too (rule 4 of §9.6: protected slot classes
        stay fp and un-evicted).
        """
        remaining = int(num_tokens)
        while remaining > 0:
            candidates = [i for i, p in enumerate(self._parts) if not p.protected]
            if not candidates:
                # Last resort: drop the oldest tokens of the fp window.
                if self._tail_positions is not None and self._tail_positions.numel():
                    tail_len = int(self._tail_positions.shape[0])
                    protected_tail = min(
                        self.fp_suffix,
                        max(tail_len - remaining, 0),
                    )
                    drop = tail_len - protected_tail
                    if drop > 0:
                        self._tail_positions = self._tail_positions[drop:]
                        self._tail_streams = {
                            name: _slice_tokens(t, drop, _token_count(t))
                            for name, t in self._tail_streams.items()
                        }
                        self._tail_extras = {
                            k: _slice_extras(v, drop, _extras_count(v, self.extras_tdim), self.extras_tdim)
                            for k, v in self._tail_extras.items()
                        }
                        remaining -= drop
                break
            i = candidates[0]
            part = self._parts[i]
            if part.n <= remaining:
                remaining -= part.n
                self._parts.pop(i)
            else:
                # Encoded payloads cannot be sliced without decode/re-encode;
                # only fp parts are cut in place.
                if part.fp:
                    self._cut_part_front(part, remaining)
                    remaining = 0
                else:
                    # Split: decode the block, cut, re-encode as one fp part.
                    decoded = {n: self._decode_stream(t) for n, t in part.streams.items()}
                    keep_part = _Part(
                        part.n - remaining,
                        part.positions[remaining:],
                        {n: _slice_tokens(t, remaining, _token_count(t)) for n, t in decoded.items()},
                        fp=True,
                        extras={
                            k: _slice_extras(v, remaining, _extras_count(v, self.extras_tdim), self.extras_tdim)
                            for k, v in part.extras.items()
                        },
                    )
                    self._parts[i] = keep_part
                    remaining = 0

    def _cut_part_front(self, part: _Part, count: int) -> None:
        part.n -= count
        part.positions = part.positions[count:]
        for name, t in list(part.streams.items()):
            part.streams[name] = _slice_tokens(t, count, _token_count(t))
        for k, v in list(part.extras.items()):
            part.extras[k] = _slice_extras(v, count, _extras_count(v, self.extras_tdim), self.extras_tdim)

    def truncate(self, num_tokens: int) -> None:
        """Keep only the first ``num_tokens`` cached positions."""
        target = max(0, int(num_tokens))
        # Drop from the fp tail first.
        if self._tail_positions is not None:
            while self._tail_positions is not None and self.length > target:
                tail_len = int(self._tail_positions.shape[0])
                if self.length - tail_len >= target:
                    self._tail_positions = None
                    self._tail_streams = None
                    self._tail_extras = {}
                else:
                    need = self.length - target
                    self._tail_positions = self._tail_positions[:-need]
                    self._tail_streams = {
                        name: _slice_tokens(t, 0, _token_count(t) - need)
                        for name, t in self._tail_streams.items()
                    }
                    self._tail_extras = {
                        k: _slice_extras(v, 0, _extras_count(v, self.extras_tdim) - need, self.extras_tdim)
                        for k, v in self._tail_extras.items()
                    }
        while self._parts and self.length > target:
            part = self._parts[-1]
            if self.length - part.n >= target:
                self._parts.pop()
                continue
            need = self.length - target
            if part.fp:
                part.n -= need
                part.positions = part.positions[:-need]
                for name, t in list(part.streams.items()):
                    part.streams[name] = _slice_tokens(t, 0, _token_count(t) - need)
                for k, v in part.extras.items():
                    part.extras[k] = _slice_extras(v, 0, _extras_count(v, self.extras_tdim) - need, self.extras_tdim)
            else:
                decoded = {n: self._decode_stream(t) for n, t in part.streams.items()}
                self._parts[-1] = _Part(
                    part.n - need,
                    part.positions[:-need],
                    {n: _slice_tokens(t, 0, _token_count(t) - need) for n, t in decoded.items()},
                    fp=True,
                    extras={
                        k: _slice_extras(v, 0, _extras_count(v, self.extras_tdim) - need, self.extras_tdim)
                        for k, v in part.extras.items()
                    },
                )

    def to(self, device: torch.device) -> None:
        """Move every stored tensor to ``device``."""
        for p in self._parts:
            p.positions = p.positions.to(device)
            for name, t in p.streams.items():
                p.streams[name] = t.to(device) if isinstance(t, torch.Tensor) else t
            for k, v in p.extras.items():
                p.extras[k] = v.to(device)
        if self._tail_positions is not None:
            self._tail_positions = self._tail_positions.to(device)
        if self._tail_streams is not None:
            for name, t in self._tail_streams.items():
                self._tail_streams[name] = t.to(device)
        for k, v in self._tail_extras.items():
            self._tail_extras[k] = v.to(device)


class KVSlot(_PositionedSlotBase):
    """Post-RoPE key/value cache for quadratic softmax mixers.

    Stores ``k`` and ``v`` of shape ``(B, H_kv, T, D)`` (keys are cached
    POST-RoPE, D7), absolute int64 ``positions`` of shape ``(T,)`` and
    optional per-token ``extras`` (FoX cumulative log-forget sums, MTLA
    position ranges, BigBird selected block ids, ...).
    """

    kind = "kv"

    def __init__(
        self,
        meta: SlotMeta,
        codec=None,
        policy=None,
        num_sinks: int = 0,
        protected_window: int = 0,
        group_size: int = 64,
    ):
        super().__init__(
            meta,
            axes={"k": "channel", "v": "token"},
            codec=codec,
            policy=policy,
            num_sinks=num_sinks,
            protected_window=protected_window,
            group_size=group_size,
        )

    def append(self, *, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor, extras=None, **_):
        super().append(positions=positions, extras=extras, k=k, v=v)


class RingSlot(KVSlot):
    """Capacity-bounded K/V ring buffer with a pinned sink region.

    Fixed-capacity sliding window used by the bounded-context mixers
    (``swa_attn``, ``longformer_attn``, ``bigbird_attn``, bounded
    ``sparse_transformer_attn``) and by the ``window_sinks`` policy. When
    the buffer overflows, the oldest non-protected tokens are evicted —
    position remapping on wrap is not needed for correctness because the
    ``positions`` metadata is stored, and Mistral-style unrolled
    re-rotation is unnecessary since keys are stored post-RoPE (D7).
    """

    kind = "ring"

    def __init__(
        self,
        meta: SlotMeta,
        capacity: int,
        num_sinks: int = 0,
        codec=None,
        policy=None,
    ):
        super().__init__(meta, codec=codec, policy=policy, num_sinks=num_sinks)
        self.capacity = max(1, int(capacity))

    def append(self, *, k, v, positions, extras=None, **_):
        super().append(k=k, v=v, positions=positions, extras=extras)
        overflow = self.length - self.capacity
        if overflow > 0:
            self.evict_oldest(overflow)


class LatentKVSlot(_PositionedSlotBase):
    """Heterogeneous named-stream cache for the latent attention family.

    Stores per-mixer tensors (e.g. ``c_kv: (B, T, r_kv)`` plus an optional
    decoupled ``k_rope: (B, H_kv, T, d_rope)``). Quantization granularity
    is declared per stream: latents are per-token (``"token"``), RoPE keys
    per-channel (``"channel"``).
    """

    kind = "latent_kv"

    def __init__(
        self,
        meta: SlotMeta,
        axes: Dict[str, QAxis],
        codec=None,
        policy=None,
        num_sinks: int = 0,
        protected_window: int = 0,
        group_size: int = 64,
    ):
        super().__init__(
            meta,
            axes=axes,
            codec=codec,
            policy=policy,
            num_sinks=num_sinks,
            protected_window=protected_window,
            group_size=group_size,
        )


class StateSlot:
    """Fixed-size recurrent state cache (updated in place per token).

    Holds named state tensors (e.g. the ``S: (B, H, d_k, d_v)`` recurrent
    matrix plus optional gate accumulators). Constant memory; never
    quantized in v1 (D12); :meth:`truncate` is a no-op because states are
    not position-indexed.
    """

    kind = "state"

    def __init__(self, meta: SlotMeta):
        self.meta = meta
        self._state: Dict[str, torch.Tensor] = {}

    @property
    def length(self) -> int:
        """States are position-less: always 0."""
        return 0

    def update(self, **tensors: torch.Tensor) -> None:
        """Replace/insert state tensors (in-place semantics for decode)."""
        for name, t in tensors.items():
            self._state[name] = _detach_cast(t)

    # CacheSlot protocol alias.
    append = update

    def read(self) -> Dict[str, torch.Tensor]:
        return dict(self._state)

    def truncate(self, num_tokens: int) -> None:  # noqa: ARG002
        """No-op: recurrent states are not position-indexed."""

    def to(self, device: torch.device) -> None:
        self._state = {k: v.to(device) for k, v in self._state.items()}


class ConvTailSlot:
    """FIFO of the last ``kernel - 1`` pre-conv rows (causal short conv).

    Used by the falcon family, ``cca_attn``/``ccgqa_attn`` and the gated
    DeltaNet family when a causal short convolution is present.
    Compositional: a mixer can own one ``KVSlot``/``StateSlot`` AND one
    ``ConvTailSlot``.
    """

    kind = "conv_tail"

    def __init__(self, meta: SlotMeta, capacity: int):
        self.meta = meta
        self.capacity = max(1, int(capacity))
        self._buf: deque = deque(maxlen=self.capacity)

    @property
    def length(self) -> int:
        return len(self._buf)

    def append(self, tensor: torch.Tensor) -> None:
        """Push new rows ``(B, T, C)``; only the tail ``capacity`` survive."""
        self._buf.append(_detach_cast(tensor))

    def read(self) -> Optional[torch.Tensor]:
        """Concatenated FIFO contents ``(B, T', C)`` or ``None`` when empty."""
        if not self._buf:
            return None
        return torch.cat(list(self._buf), dim=1)

    def truncate(self, num_tokens: int) -> None:
        """Drop entries beyond the ``num_tokens``-th newest row."""
        total = sum(int(t.shape[1]) for t in self._buf)
        drop = total - max(0, int(num_tokens))
        while drop > 0 and self._buf:
            front = self._buf[0]
            cnt = int(front.shape[1])
            if cnt <= drop:
                self._buf.popleft()
                drop -= cnt
            else:
                self._buf[0] = front[:, drop:]
                drop = 0

    def to(self, device: torch.device) -> None:
        self._buf = deque((t.to(device) for t in self._buf), maxlen=self.capacity)


class TokenTailSlot:
    """FIFO of the last ``n`` token ids (int64).

    Fed from the generation loop's token stream (not from activations);
    consumed by ``engram_attn``'s n-gram-suffix lookup (needs the previous
    ``max_ngram - 1`` ids) and other token-level consumers.
    """

    kind = "token_tail"

    def __init__(self, meta: Optional[SlotMeta] = None, capacity: int = 8):
        self.meta = meta
        self.capacity = max(1, int(capacity))
        self._buf: deque = deque(maxlen=self.capacity)

    @property
    def length(self) -> int:
        return len(self._buf)

    def append(self, token_ids: torch.Tensor) -> None:
        """Push token ids ``(B, T)`` (batch-broadcast rows share storage)."""
        flat = _detach_cast(token_ids, torch.long)
        self._buf.append(flat)

    def read(self) -> Optional[torch.Tensor]:
        """Concatenated ids ``(B, T')`` or ``None`` when empty."""
        if not self._buf:
            return None
        return torch.cat(list(self._buf), dim=-1)

    def truncate(self, num_tokens: int) -> None:
        """Keep at most ``num_tokens`` rows in the FIFO."""
        while len(self._buf) > max(0, int(num_tokens)):
            self._buf.popleft()

    def to(self, device: torch.device) -> None:
        self._buf = deque((t.to(device) for t in self._buf), maxlen=self.capacity)
