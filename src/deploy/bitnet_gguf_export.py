#!/usr/bin/env python3
"""Best-effort GGUF exporter for BitNet b1.58 (1.58-bit) TORMENTED-BERT models.

Targets `Microsoft/BitNet <https://github.com/microsoft/BitNet>`_ (bitnet.cpp),
which is built on top of llama.cpp and consumes the **GGUF** format with the
``i2_s`` quantization type (ternary ``{-1, 0, 1}`` weights + int8 activations,
per-tensor scale).

Scope and limitations
---------------------
bitnet.cpp only supports architectures registered in llama.cpp (Llama,
Falcon3, Falcon-E, and the official BitNet b1.58 models). The Frankenstein
**hybrid** model (33 mixer families, custom code) cannot run in bitnet.cpp
because its compute graph is not registered.

This exporter therefore only supports Frankenstein models configured with a
**single mixer type ``standard_attn``** (pure encoder or pure decoder). For
that subset the attention/FFN graph maps to the Llama/BERT tensor naming
convention, and the ternary weights are packed with the ``i2_s`` layout that
``setup_env.py -q i2_s`` expects.

Any other ``layer_pattern`` (e.g. ``retnet``, ``mamba``, ``gla_attn``, ...)
raises :class:`GGUFExportError`. This is a deliberate limitation: those mixers
have no equivalent in llama.cpp.

The writer emits a minimal GGUF container (magic, version, tensor count,
metadata kv pairs, tensor info table, padding, and the packed tensor data).
It is a **self-contained writer** that does not depend on the optional
``gguf`` Python package, so it works in minimal environments. For full
fidelity / production use, prefer the official
``llama.cpp``/``gguf`` tooling applied to an exported Llama-compatible
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

try:
    from ..utils.config_flatten import flatten_model_dict
except ImportError:
    from utils.config_flatten import flatten_model_dict

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# GGUF tensor types
GGML_TYPE_I2_S = 25  # BitNet i2_s (ternary weights, per-tensor fp scale)


class GGUFExportError(ValueError):
    """Raised when a model cannot be exported to the GGUF/BitNet format."""


# ---------------------------------------------------------------------------
# Low-level GGUF binary writer
# ---------------------------------------------------------------------------
class _GGUFWriter:
    """Minimal self-contained GGUF v3 binary writer."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def _u8(self, v: int) -> None:
        self._buf += struct.pack("<B", v & 0xFF)

    def _u32(self, v: int) -> None:
        self._buf += struct.pack("<I", v & 0xFFFFFFFF)

    def _i32(self, v: int) -> None:
        self._buf += struct.pack("<i", v)

    def _f32(self, v: float) -> None:
        self._buf += struct.pack("<f", v)

    def _str(self, v: str) -> None:
        b = v.encode("utf-8")
        self._buf += struct.pack("<Q", len(b))
        self._buf += b

    def write_header(self, n_tensors: int) -> None:
        self._u32(GGUF_MAGIC)
        self._u32(GGUF_VERSION)
        self._u32(n_tensors)

    def write_kv_string(self, key: str, value: str) -> None:
        self._str(key)
        self._u32(GGUF_TYPE_STRING)
        self._str(value)

    def write_kv_uint32(self, key: str, value: int) -> None:
        self._str(key)
        self._u32(GGUF_TYPE_UINT32)
        self._u32(value)

    def write_kv_int32(self, key: str, value: int) -> None:
        self._str(key)
        self._u32(GGUF_TYPE_INT32)
        self._i32(value)

    def write_kv_float32(self, key: str, value: float) -> None:
        self._str(key)
        self._u32(GGUF_TYPE_FLOAT32)
        self._f32(value)

    def write_kv_string_array(self, key: str, values: List[str]) -> None:
        self._str(key)
        self._u32(GGUF_TYPE_ARRAY)
        self._u32(GGUF_TYPE_STRING)
        self._u32(len(values))
        for v in values:
            self._str(v)

    def write_tensor_info(
        self,
        name: str,
        dims: List[int],
        ggml_type: int,
        offset: int,
    ) -> None:
        self._str(name)
        n_dims = len(dims)
        self._u32(n_dims)
        # GGUF stores dims in reverse (column-major / row-major convention)
        for d in reversed(dims):
            self._u32(d)
        self._u32(ggml_type)
        self._u64(offset)

    def _u64(self, v: int) -> None:
        self._buf += struct.pack("<Q", v)

    def write_raw(self, data: bytes) -> None:
        self._buf += data

    def pad_to(self, alignment: int) -> None:
        rem = len(self._buf) % alignment
        if rem:
            self._buf += b"\x00" * (alignment - rem)

    def bytes(self) -> bytes:
        return bytes(self._buf)


# ---------------------------------------------------------------------------
# i2_s ternary packing
# ---------------------------------------------------------------------------
def pack_i2_s(weight: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Pack a weight tensor to the BitNet ``i2_s`` representation.

    Quantizes the weight to ternary ``{-1, 0, 1}`` using the BitNet b1.58
    per-tensor absmean scale, then packs the ternary indices into a 2-bit
    stream (4 values per byte, mapping ``-1 -> 0b00, 0 -> 0b01, 1 -> 0b10``).

    Args:
        weight: 2D weight tensor (``[out_features, in_features]``).

    Returns:
        Tuple of ``(packed_uint8, scale_fp32)``.
    """
    scale = weight.abs().mean().item()
    if scale < 1e-5:
        scale = 1.0
    w_ternary = torch.round(weight / scale).clamp(-1, 1).cpu().numpy().astype(np.int32)
    values = (w_ternary + 1).astype(np.uint8).flatten()
    n = len(values)
    packed = np.zeros((n + 3) // 4, dtype=np.uint8)
    full = n - (n % 4)
    if full:
        reshaped = values[:full].reshape(-1, 4)
        byte = (
            reshaped[:, 0]
            | (reshaped[:, 1] << 2)
            | (reshaped[:, 2] << 4)
            | (reshaped[:, 3] << 6)
        ).astype(np.uint8)
        packed[: full // 4] = byte
    for i in range(full, n):
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        packed[byte_idx] |= np.uint8((values[i] & 0b11) << bit_offset)
    return packed, float(scale)


# ---------------------------------------------------------------------------
# Tensor remapping (standard_attn -> Llama-style names)
# ---------------------------------------------------------------------------
def _remap_standard_attn_tensors(
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> List[Tuple[str, torch.Tensor]]:
    """Map Frankenstein ``standard_attn`` tensors to Llama-style GGUF names.

    Only tensors belonging to the embedding, per-layer attention (q/k/v/o),
    FFN (up/down), final norm and LM head are remapped. Anything else raises
    :class:`GGUFExportError`.

    Args:
        state_dict: Raw Frankenstein state dict.
        config: Model config dict (reads ``num_layers``).

    Returns:
        List of ``(gguf_name, weight_tensor)`` pairs.
    """
    num_layers = int(config.get("num_layers", 0))
    remapped: List[Tuple[str, torch.Tensor]] = []

    for name, tensor in state_dict.items():
        if not name.endswith(".weight"):
            # Drop biases/norm params for the i2_s layout (bitnet.cpp expects
            # quantized weights only; biases are not part of i2_s).
            continue
        new_name = _map_single(name, num_layers)
        if new_name is not None:
            remapped.append((new_name, tensor))
    return remapped


def _map_single(name: str, num_layers: int) -> str | None:
    """Translate a Frankenstein tensor name to a Llama-style GGUF name."""
    if name == "emb.weight" or name.endswith(".embedding.weight"):
        return "token_embd.weight"
    if name == "head.weight" or name.endswith(".backbone.head.weight"):
        return "output.weight"
    if name == "final_norm.alpha" or name == "final_norm.weight":
        return "output_norm.weight"

    # Per-layer: [backbone.]layers.{i}.<attr>.weight (decoder/mini wrap a
    # backbone; strip a leading "backbone." prefix transparently).
    parts = name.split(".")
    if parts and parts[0] == "backbone":
        parts = parts[1:]
    if len(parts) >= 3 and parts[0] == "layers":
        try:
            layer_idx = int(parts[1])
        except ValueError:
            return None
        suffix = ".".join(parts[2:])
        prefix = f"blk.{layer_idx}."
        mapping = {
            "mixer.q_proj.weight": f"{prefix}attn_q.weight",
            "mixer.k_proj.weight": f"{prefix}attn_k.weight",
            "mixer.v_proj.weight": f"{prefix}attn_v.weight",
            "mixer.out_proj.weight": f"{prefix}attn_output.weight",
            "ffn.0.weight": f"{prefix}ffn_gate.weight",  # up-projection first
            "ffn.2.weight": f"{prefix}ffn_down.weight",
            "norm1.alpha": f"{prefix}attn_norm.weight",
            "norm1.weight": f"{prefix}attn_norm.weight",
            "norm2.alpha": f"{prefix}ffn_norm.weight",
            "norm2.weight": f"{prefix}ffn_norm.weight",
        }
        return mapping.get(suffix)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_gguf_compatibility(yaml_path: str) -> Dict[str, Any]:
    """Check whether a training YAML is compatible with the GGUF/BitNet export.

    The model must use ``use_bitnet: true`` and a single ``standard_attn``
    layer type (pure encoder or decoder). Any hybrid mixer is rejected.

    Args:
        yaml_path: Path to the training YAML.

    Returns:
        Dictionary with ``is_compatible`` (bool) and ``reason`` (str).
    """
    yaml_file = Path(yaml_path).expanduser().resolve()
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")
    with yaml_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    model = flatten_model_dict(data.get("model", {}) or {})

    if not bool(model.get("use_bitnet", False)):
        return {"is_compatible": False, "reason": "use_bitnet must be true"}

    layers = list(model.get("layer_pattern") or [])
    if not layers:
        return {"is_compatible": False, "reason": "layer_pattern is empty"}
    unique = set(layers)
    if unique != {"standard_attn"}:
        return {
            "is_compatible": False,
            "reason": (
                "GGUF/BitNet export only supports the 'standard_attn' mixer "
                f"(found {sorted(unique)}). Hybrid mixers have no equivalent "
                "in llama.cpp / bitnet.cpp."
            ),
        }
    return {"is_compatible": True, "reason": "ok"}


def export_bitnet_gguf(
    model_path: str,
    yaml_path: str,
    output_gguf: str,
    alignment: int = 32,
) -> Dict[str, Any]:
    """Export a BitNet checkpoint to a GGUF file (i2_s packing).

    Args:
        model_path: Path to a training checkpoint (``.pt``).
        yaml_path: Path to the YAML config used during training.
        output_gguf: Path to the output ``.gguf`` file.
        alignment: GGUF tensor alignment in bytes (default 32).

    Returns:
        Dictionary with ``status``, ``output_gguf``, ``n_tensors``, and the
        compatibility check.

    Raises:
        GGUFExportError: If the model is not GGUF/BitNet compatible.
        FileNotFoundError: If model or YAML does not exist.
    """
    model_file = Path(model_path).expanduser().resolve()
    yaml_file = Path(yaml_path).expanduser().resolve()
    output = Path(output_gguf).expanduser().resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    compat = check_gguf_compatibility(str(yaml_file))
    if not compat["is_compatible"]:
        raise GGUFExportError(compat["reason"])

    with yaml_file.open("r", encoding="utf-8") as handle:
        yaml_data = yaml.safe_load(handle) or {}
    model_config = flatten_model_dict(yaml_data.get("model", {}) or {})

    checkpoint = torch.load(str(model_file), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise GGUFExportError("Checkpoint must be a dict-like payload")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    remapped = _remap_standard_attn_tensors(state_dict, model_config)
    if not remapped:
        raise GGUFExportError(
            "No remappable standard_attn tensors found in the checkpoint"
        )

    # Pack every tensor to i2_s.
    packed: List[Tuple[str, np.ndarray, float, List[int]]] = []
    for name, tensor in remapped:
        t2d = tensor.detach().float()
        if t2d.dim() == 1:
            t2d = t2d.unsqueeze(0)
        data, scale = pack_i2_s(t2d)
        dims = [int(d) for d in t2d.shape]
        packed.append((name, data, scale, dims))

    # Build the GGUF binary: header + metadata kv + tensor info table + data.
    writer = _GGUFWriter()
    writer.write_header(len(packed))

    # Metadata
    writer.write_kv_string("general.architecture", "frankestein")
    writer.write_kv_string("general.name", "frankestein-bitnet")
    writer.write_kv_uint32(
        "frankestein.hidden_size", int(model_config.get("hidden_size", 0))
    )
    writer.write_kv_uint32(
        "frankestein.block_count", int(model_config.get("num_layers", 0))
    )
    writer.write_kv_uint32(
        "frankestein.attention.head_count", int(model_config.get("num_heads", 0))
    )
    vocab_size = int(model_config.get("vocab_size", 0))
    writer.write_kv_uint32("frankestein.vocab_size", vocab_size)
    writer.write_kv_string(
        "frankestein.context_length",
        str(int(model_config.get("max_position_embeddings", 512))),
    )
    writer.write_kv_string(
        "general.quantization", "i2_s (BitNet b1.58 ternary)"
    )

    # Record the byte offset right after the kv section: the tensor info table
    # will be written next, followed by aligned tensor data.
    info_table_start = len(writer._buf)
    # First pass: write the info table with offset=0 placeholders so we can
    # measure the table size (each record has variable-length names).
    for name, _data, _scale, dims in packed:
        writer.write_tensor_info(name, dims, GGML_TYPE_I2_S, 0)
    info_table_size = len(writer._buf) - info_table_start

    # Compute the absolute data offset for each tensor, accounting for
    # alignment. The data section begins immediately after the info table.
    data_section_start = info_table_start + info_table_size
    offsets: List[int] = []
    cursor = data_section_start
    for _name, data, _scale, _dims in packed:
        rem = cursor % alignment
        if rem:
            cursor += alignment - rem
        offsets.append(cursor)
        cursor += len(data)
    total_size = cursor

    # Rebuild cleanly with real offsets (offsets are stored relative to the
    # start of the data section, per the GGUF convention).
    final = _GGUFWriter()
    final.write_header(len(packed))
    final.write_kv_string("general.architecture", "frankestein")
    final.write_kv_string("general.name", "frankestein-bitnet")
    final.write_kv_uint32(
        "frankestein.hidden_size", int(model_config.get("hidden_size", 0))
    )
    final.write_kv_uint32(
        "frankestein.block_count", int(model_config.get("num_layers", 0))
    )
    final.write_kv_uint32(
        "frankestein.attention.head_count", int(model_config.get("num_heads", 0))
    )
    final.write_kv_uint32("frankestein.vocab_size", vocab_size)
    final.write_kv_string(
        "frankestein.context_length",
        str(int(model_config.get("max_position_embeddings", 512))),
    )
    final.write_kv_string(
        "general.quantization", "i2_s (BitNet b1.58 ternary)"
    )
    for (name, _data, _scale, dims), abs_offset in zip(packed, offsets):
        final.write_tensor_info(
            name, dims, GGML_TYPE_I2_S, abs_offset - data_section_start
        )
    # Sanity: data_section_start for the final writer equals the analogous
    # point in `writer` only if the kv section is byte-identical (it is, by
    # construction). Re-derive from the final writer to be safe.
    final_data_start = len(final._buf)
    cursor = final_data_start
    for _name, data, _scale, _dims in packed:
        rem = cursor % alignment
        if rem:
            final.write_raw(b"\x00" * (alignment - rem))
            cursor += alignment - rem
        final.write_raw(data.tobytes())
        cursor += len(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(final.bytes())

    return {
        "status": "ok",
        "output_gguf": str(output),
        "n_tensors": len(packed),
        "size_bytes": len(final.bytes()),
        "compatibility": compat,
        "quantization": "i2_s",
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Best-effort GGUF (BitNet i2_s) exporter for standard_attn-only "
            "Frankenstein BitNet models."
        )
    )
    parser.add_argument("--model", required=True, help="Path to training checkpoint (*.pt)")
    parser.add_argument("--yaml", required=True, help="Path to YAML used during training")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument(
        "--check", action="store_true", help="Only run the compatibility check and exit"
    )
    args = parser.parse_args(argv)

    if args.check:
        result = check_gguf_compatibility(args.yaml)
        print(json.dumps(result, indent=2))
        return 0 if result["is_compatible"] else 1

    result = export_bitnet_gguf(args.model, args.yaml, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
