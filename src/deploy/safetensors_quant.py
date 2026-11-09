"""Block quantization codecs for HuggingFace ``safetensors`` exports.

The functions in this module implement the legacy ``Q8_0``/``Q4_0`` block
quantization layouts plus the BitNet ``i2_s`` ternary layout, but they are
**container agnostic**: the resulting byte buffers are ordinary
``torch.uint8`` tensors and can be stored in any safetensors file. No
container-specific writer or reader is involved.

Layouts
-------
``q8_0``
    32 weights per block, ``float16`` scale + 32 ``int8`` quants
    (34 bytes per block, 8.5 bits per weight). Symmetric scale
    ``d = max(|w|) / 127``.

``q4_0``
    32 weights per block, ``float16`` scale + 16 packed nibbles
    (18 bytes per block, 4.5 bits per weight). Follows the llama.cpp
    reference exactly: the scale uses the *signed* value with the largest
    magnitude (``d = max_signed / -8``), bins are ``[-8, 7]`` and each byte
    stores element ``j`` in the low nibble and element ``j + 16`` in the high
    nibble.

``i2_s``
    BitNet b1.58 ternary ``{-1, 0, 1}`` packed four values per byte
    (2 bits per weight) plus one ``float32`` per-tensor scale.

The module is copied verbatim into exported HF directories as
``quantization_frankenstein.py`` so the generated ``modeling_*.py`` code can
dequantize packed blocks without importing the Frankenstein ``src`` tree.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

__all__ = [
    "Q8_0",
    "Q4_0",
    "I2_S",
    "SUPPORTED_FORMATS",
    "BLOCK_SIZE",
    "PACKED_SUFFIX",
    "SCALE_SUFFIX",
    "bits_per_weight",
    "packed_numel",
    "quantize_weight",
    "dequantize_weight",
]

Q8_0 = "q8_0"
Q4_0 = "q4_0"
I2_S = "i2_s"
SUPPORTED_FORMATS: Tuple[str, ...] = (Q8_0, Q4_0, I2_S)
NONE = "none"

#: Weights per affine block for the Q8_0/Q4_0 layouts.
BLOCK_SIZE = 32
#: State-dict suffixes used by the packed representation.
PACKED_SUFFIX = ".__packed__"
SCALE_SUFFIX = ".__scale__"

_BLOCK_BYTES = {Q8_0: 34, Q4_0: 18}
_BITS_PER_WEIGHT = {Q8_0: 8.5, Q4_0: 4.5, I2_S: 2.0}


def bits_per_weight(fmt: str) -> float:
    """Return the nominal bits-per-weight of a storage format.

    Args:
        fmt: One of ``q8_0``, ``q4_0`` or ``i2_s``.

    Returns:
        Bits per weight, including block/scale overhead.

    Raises:
        ValueError: If ``fmt`` is unknown.
    """
    try:
        return _BITS_PER_WEIGHT[fmt]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown quantization format: {fmt!r}") from exc


def packed_numel(shape: Iterable[int], fmt: str) -> int:
    """Return the number of ``uint8`` elements in a packed payload.

    Args:
        shape: Original weight shape.
        fmt: ``q8_0``, ``q4_0`` or ``i2_s``.

    Returns:
        Flat payload length in bytes.

    Raises:
        ValueError: If ``fmt`` is unknown.
    """
    n = int(math.prod(tuple(shape)))
    if fmt == Q8_0:
        return (n // BLOCK_SIZE) * _BLOCK_BYTES[Q8_0]
    if fmt == Q4_0:
        return (n // BLOCK_SIZE) * _BLOCK_BYTES[Q4_0]
    if fmt == I2_S:
        return (n + 3) // 4
    raise ValueError(f"Unknown quantization format: {fmt!r}")


def _as_block_rows(weight: torch.Tensor) -> np.ndarray:
    """Validate and flatten a 2-D weight into ``(num_blocks, 32)`` rows.

    Args:
        weight: Floating-point 2-D weight tensor.

    Returns:
        A contiguous ``float32`` array of shape ``(n_blocks, 32)``.

    Raises:
        ValueError: If the tensor is not 2-D or its last dimension is not a
            multiple of :data:`BLOCK_SIZE`.
    """
    w = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() != 2:
        raise ValueError(f"block quantization needs a 2-D weight, got {tuple(w.shape)}")
    if w.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(
            "block quantization needs last_dim % 32 == 0, got "
            f"{tuple(w.shape)}"
        )
    return w.numpy().reshape(-1, BLOCK_SIZE)


def _round_half_away(values: np.ndarray) -> np.ndarray:
    """Round like C ``roundf`` (half away from zero), not banker's rounding."""
    return np.trunc(values + np.copysign(0.5, values))


def _f16_bytes(values: np.ndarray) -> np.ndarray:
    """Little-endian ``float16`` bytes for one value per row."""
    return values.astype("<f2").view(np.uint8).reshape(-1, 2)


def _f16_values(raw: np.ndarray) -> np.ndarray:
    """Decode little-endian ``float16`` bytes of shape ``(n, 2)``."""
    return raw.copy().view("<f2").reshape(-1).astype(np.float32)


def _quantize_q8_0(weight: torch.Tensor) -> torch.Tensor:
    rows = _as_block_rows(weight)
    n = rows.shape[0]
    amax = np.abs(rows).max(axis=1)
    d = np.where(amax > 0.0, amax / 127.0, 0.0).astype(np.float32)
    inv = np.where(d > 0.0, 1.0 / np.where(d > 0.0, d, 1.0), 0.0)
    q = _round_half_away(rows * inv[:, None]).clip(-127, 127).astype(np.int8)

    out = np.empty((n, _BLOCK_BYTES[Q8_0]), dtype=np.uint8)
    out[:, :2] = _f16_bytes(d)
    out[:, 2:] = q.view(np.uint8)
    return torch.from_numpy(out.reshape(-1))


def _dequantize_q8_0(packed: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    raw = packed.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    blocks = raw.reshape(-1, _BLOCK_BYTES[Q8_0])
    d = _f16_values(blocks[:, :2])
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    weights = q * d[:, None]
    return torch.from_numpy(weights.reshape(tuple(shape)))


def _quantize_q4_0(weight: torch.Tensor) -> torch.Tensor:
    rows = _as_block_rows(weight)
    n = rows.shape[0]
    # llama.cpp picks the signed value with the largest magnitude, then
    # d = max_signed / -8 so that the [-8, 7] bins are centred on zero.
    absmax_index = np.abs(rows).argmax(axis=1)
    max_signed = rows[np.arange(n), absmax_index]
    d = np.where(max_signed != 0.0, max_signed / -8.0, 0.0).astype(np.float32)
    inv = np.where(d != 0.0, 1.0 / np.where(d != 0.0, d, 1.0), 0.0)
    scaled = rows * inv[:, None]
    q = np.trunc(scaled + 8.5).astype(np.int32).clip(0, 15).astype(np.uint8)

    low = q[:, : BLOCK_SIZE // 2]
    high = q[:, BLOCK_SIZE // 2 :]
    quants = (low | (high << 4)).astype(np.uint8)

    out = np.empty((n, _BLOCK_BYTES[Q4_0]), dtype=np.uint8)
    out[:, :2] = _f16_bytes(d)
    out[:, 2:] = quants
    return torch.from_numpy(out.reshape(-1))


def _dequantize_q4_0(packed: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    raw = packed.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    blocks = raw.reshape(-1, _BLOCK_BYTES[Q4_0])
    d = _f16_values(blocks[:, :2])
    quants = blocks[:, 2:]
    low = (quants & 0x0F).astype(np.int32)
    high = ((quants >> 4) & 0x0F).astype(np.int32)
    q = np.concatenate([low, high], axis=1) - 8
    weights = q.astype(np.float32) * d[:, None]
    return torch.from_numpy(weights.reshape(tuple(shape)))


def _is_ternary(weight: torch.Tensor, atol: float = 1e-6) -> bool:
    """Return True when every nonzero value equals ``+/-max(|weight|)``."""
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    if w.numel() == 0:
        return True
    gamma = w.abs().max().item()
    if gamma < atol:
        return True
    nonzero = w[w.abs() > atol]
    if nonzero.numel() == 0:
        return True
    return ((nonzero.abs() - gamma).abs().max().item() / gamma) < atol


def _quantize_i2_s(weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    if _is_ternary(w):
        scale = float(w.abs().max().item())
        if scale < 1e-5:
            scale = 1.0
        ternary = torch.sign(w).clamp(-1, 1)
    else:
        scale = float(w.abs().mean().item())
        if scale < 1e-5:
            scale = 1.0
        ternary = torch.round(w / scale).clamp(-1, 1)

    values = (ternary.to(torch.int32) + 1).numpy().astype(np.uint8).reshape(-1)
    n = values.size
    packed = np.zeros((n + 3) // 4, dtype=np.uint8)
    full = n - (n % 4)
    if full:
        reshaped = values[:full].reshape(-1, 4)
        packed[: full // 4] = (
            reshaped[:, 0]
            | (reshaped[:, 1] << 2)
            | (reshaped[:, 2] << 4)
            | (reshaped[:, 3] << 6)
        ).astype(np.uint8)
    for index in range(full, n):
        packed[index // 4] |= np.uint8(
            (values[index] & 0b11) << ((index % 4) * 2)
        )
    return torch.from_numpy(packed), scale


def _dequantize_i2_s(
    packed: torch.Tensor,
    scale: torch.Tensor | float | None,
    shape: Iterable[int],
) -> torch.Tensor:
    raw = packed.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    n = int(math.prod(tuple(shape)))
    padded = np.zeros(max((n + 3) // 4, raw.size), dtype=np.uint8)
    padded[: raw.size] = raw
    out = np.zeros(n, dtype=np.int32)
    full = n - (n % 4)
    if full:
        block = padded[: full // 4].astype(np.int32)
        fields = np.empty((full // 4, 4), dtype=np.int32)
        fields[:, 0] = block & 0b11
        fields[:, 1] = (block >> 2) & 0b11
        fields[:, 2] = (block >> 4) & 0b11
        fields[:, 3] = (block >> 6) & 0b11
        out[:full] = fields.reshape(-1) - 1
    for index in range(full, n):
        value = int((padded[index // 4] >> ((index % 4) * 2)) & 0b11)
        out[index] = value - 1

    if scale is None:
        scale_value = 1.0
    elif isinstance(scale, torch.Tensor):
        scale_value = float(scale.detach().to(device="cpu", dtype=torch.float32).reshape(-1)[0])
    else:
        scale_value = float(scale)
    weights = out.astype(np.float32) * scale_value
    return torch.from_numpy(weights.reshape(tuple(shape)))


def quantize_weight(weight: torch.Tensor, fmt: str) -> Dict[str, torch.Tensor]:
    """Quantize one weight tensor into packed safetensors payload tensors.

    Args:
        weight: 2-D floating-point weight (``[out_features, in_features]``).
        fmt: ``q8_0``, ``q4_0`` or ``i2_s``.

    Returns:
        Mapping with ``__packed__`` (``uint8``) and, for ``i2_s``, an extra
        ``__scale__`` scalar tensor.

    Raises:
        ValueError: If ``fmt`` is unknown.
    """
    if fmt == Q8_0:
        return {PACKED_SUFFIX: _quantize_q8_0(weight)}
    if fmt == Q4_0:
        return {PACKED_SUFFIX: _quantize_q4_0(weight)}
    if fmt == I2_S:
        packed, scale = _quantize_i2_s(weight)
        return {
            PACKED_SUFFIX: packed,
            SCALE_SUFFIX: torch.tensor(scale, dtype=torch.float32),
        }
    raise ValueError(f"Unknown quantization format: {fmt!r}")


def dequantize_weight(
    packed: torch.Tensor,
    scale: torch.Tensor | float | None,
    shape: Iterable[int],
    fmt: str,
) -> torch.Tensor:
    """Reconstruct a floating-point weight from a packed payload.

    Args:
        packed: ``uint8`` payload produced by :func:`quantize_weight`.
        scale: Per-tensor scale (``i2_s`` only).
        shape: Original weight shape.
        fmt: ``q8_0``, ``q4_0`` or ``i2_s``.

    Returns:
        ``float32`` tensor with the original shape.

    Raises:
        ValueError: If ``fmt`` is unknown.
    """
    if fmt == Q8_0:
        return _dequantize_q8_0(packed, shape)
    if fmt == Q4_0:
        return _dequantize_q4_0(packed, shape)
    if fmt == I2_S:
        return _dequantize_i2_s(packed, scale, shape)
    raise ValueError(f"Unknown quantization format: {fmt!r}")
